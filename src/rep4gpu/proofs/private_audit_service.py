"""Run an external prover and a four-server Rep4 verification committee."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

from ..protocol.runtime import Party
from ..protocol.setup import correlated_seeds
from ..protocol.transport import SocketTransport
from .audit_model import AuditModel
from .audit_protocol import receive_message, send_message, MAX_FRAME_BYTES
from .audit_service import check_peer, init_example, tls_context, write_json
from .private_model import provision_private_model, load_private_model, validate_manifest
from .private_commitments import PrivateTrace, validate_commitment, verify_private_opening
from .rep4_audit import agreement, joint_random, verify_shared_trace

VERSION = 3
PRIVACY = ('The prover knows its trace. Each verifier receives three replicated components, '
           'never plaintext weights, biases, or intermediate states. Public information includes model '
           'commitment, architecture, input, '
           'shapes, sampled indices, acceptance, and accepted output. Rep4 tolerates at most '
           'one malicious verifier; two colluding verifiers can reconstruct the trace.')



class CountingSocket:
    """Count application bytes after TLS encryption/decryption boundaries."""
    def __init__(self, connection, meter, channel):
        self.connection, self.meter, self.channel = connection, meter, channel

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def sendall(self, data):
        self.connection.sendall(data)
        self.meter.add(self.channel, len(data), 0)

    def recv(self, size):
        data = self.connection.recv(size)
        self.meter.add(self.channel, 0, len(data))
        return data


class TrafficMeter:
    def __init__(self, report=None):
        # Mesh workers update counters concurrently, one worker per TLS socket.
        import threading
        self.lock = threading.Lock()
        self.report = report if report is not None else {}
        self.report.update(units='bytes', includes='application payloads and framing',
                           excludes='TLS records and handshakes, TCP/IP, certificates and file provisioning',
                           channels={})

    def add(self, channel, sent, received):
        with self.lock:
            counts = self.report['channels'].setdefault(channel, {'sent': 0, 'received': 0})
            counts['sent'] += sent
            counts['received'] += received

    def wrap(self, connection, channel):
        self.add(channel, 0, 0)
        return CountingSocket(connection, self, channel)

    def snapshot(self, channel):
        return dict(self.report['channels'][channel])

    def record_check(self, before):
        after = self.snapshot('verifier_mesh')
        self.report['transition_checks_and_output'] = {key: after[key] - before[key] for key in after}


def copy_role_setup(source, destination, role):
    """Distribute only one role's keys and model material."""
    allowed = {'deployment.json', 'input.json', 'model-manifest.json', 'ca.pem',
               'prover.pem', *(f'party{p}.pem' for p in range(4)), f'{role}.key',
               'model.json' if role == 'prover' else f'{role}-model.json'}
    for name in sorted(allowed):
        target = Path(destination) / name
        target.write_bytes((Path(source) / name).read_bytes())
        if name.endswith('.key') or name == 'model.json' or name.endswith('-model.json'):
            target.chmod(0o600)


def provisioning_sizes(directory):
    """File payload sizes for one-time model delivery, not network measurements."""
    directory = Path(directory)
    bundles = {f'party{p}': (directory / f'party{p}-model.json').stat().st_size for p in range(4)}
    manifest = (directory / 'model-manifest.json').stat().st_size
    return {'measurement': 'serialized file sizes; excludes transport overhead',
            'verifier_share_files_bytes': bundles, 'manifest_bytes_per_recipient': manifest,
            'total_to_four_verifiers_bytes': sum(bundles.values()) + 4 * manifest}


def aggregate_traffic(output):
    reports = {role: json.loads((output / role / 'traffic.json').read_text())
               for role in ('prover', 'party0', 'party1', 'party2', 'party3')}
    channels = {}
    for report in reports.values():
        for channel, counts in report['channels'].items():
            totals = channels.setdefault(channel, {'sent': 0, 'received': 0})
            for key in totals:
                totals[key] += counts[key]
    if any(counts['sent'] != counts['received'] for counts in channels.values()):
        raise ValueError('global traffic sent/received accounting mismatch')
    return {'units': 'bytes', 'counting': 'global outgoing bytes, each transmission counted once',
            'includes': reports['prover']['includes'], 'excludes': reports['prover']['excludes'],
            'channels': channels, 'total_sent': sum(c['sent'] for c in channels.values()),
            'transition_checks_and_output_sent': sum(
                r.get('transition_checks_and_output', {}).get('sent', 0) for r in reports.values()),
            'per_role_sent': {role: sum(c['sent'] for c in r['channels'].values())
                              for role, r in reports.items()}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def message(kind, **fields):
    return {'version': VERSION, 'type': kind, **fields}


def receive(connection, kind, fields):
    value = receive_message(connection)
    if (set(value) != {'version', 'type', *fields} or type(value['version']) is not int
            or value['version'] != VERSION or value['type'] != kind):
        raise ValueError(f'expected private {kind} message')
    return value


def sample_indices(seed, count, samples):
    """Select without replacement using a SHA-256 stream and rejection sampling."""
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError('challenge seed must contain 32 bytes')
    if type(count) is not int or type(samples) is not int or not 1 <= samples <= count:
        raise ValueError('samples must be between one and the layer count')
    remaining, chosen, counter = list(range(count)), [], 0
    for _ in range(samples):
        size = len(remaining)
        limit = (1 << 64) - ((1 << 64) % size)
        while True:
            digest = hashlib.sha256(b'siv.private-audit.sample.v1\0' + seed + counter.to_bytes(8, 'big')).digest()
            counter += 1
            value = int.from_bytes(digest[:8], 'big')
            if value < limit:
                break
        chosen.append(remaining.pop(value % size))
    return sorted(chosen)


def boundaries(indices, count):
    return sorted({0, count, *(b for i in indices for b in (i, i + 1))})


def report_for(commitment, indices, output):
    return {'accepted': True, 'protocol': 'rep4-private-audit-v2',
            'model_id': commitment['model_id'], 'run_id': commitment['run_id'],
            'commitment': commitment, 'checked_transitions': indices,
            'total_transitions': len(commitment['layer_names']),
            'verification_mode': 'full' if len(indices) == len(commitment['layer_names']) else 'sampled',
            'output': {'x': output.tolist()}, 'privacy': PRIVACY}


def load_setup(directory, party_id=None):
    directory = Path(directory)
    config = json.loads((directory / 'deployment.json').read_text())
    if (not isinstance(config, dict)
            or set(config) != {'version', 'prover', 'verifiers', 'samples', 'timeout', 'checkpoint_interval'}
            or type(config['version']) is not int or config['version'] != 1
            or not isinstance(config['verifiers'], list) or len(config['verifiers']) != 4):
        raise ValueError('invalid deployment configuration')
    for address in [config['prover'], *config['verifiers']]:
        if (not isinstance(address, list) or len(address) != 2 or not isinstance(address[0], str)
                or not address[0] or type(address[1]) is not int or not 1 <= address[1] <= 65535):
            raise ValueError('addresses must be [host, port] pairs')
    if len({tuple(a) for a in [config['prover'], *config['verifiers']]}) != 5:
        raise ValueError('each process needs a distinct listening address')
    timeout = config['timeout']
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if type(config['checkpoint_interval']) is not int or config['checkpoint_interval'] < 1:
        raise ValueError('checkpoint interval must be positive')
    manifest = validate_manifest(json.loads((directory / 'model-manifest.json').read_text()))
    if party_id is None:
        model = AuditModel.load(directory / 'model.json')
        if model.width != manifest['width'] or len(model.layers) != manifest['layer_count']:
            raise ValueError('prover model architecture differs from the approved manifest')
        model.model_id = manifest['model_id']
    else:
        model = load_private_model(manifest,
            json.loads((directory / f'party{party_id}-model.json').read_text()), party_id)
    if config['samples'] is not None and (type(config['samples']) is not int
            or not 1 <= config['samples'] <= len(model.layers)):
        raise ValueError('samples must be null or between one and the layer count')
    with (directory / 'input.json').open('rb') as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError('input file exceeds 1 MiB')
    data = json.loads(payload)
    if (not isinstance(data, dict) or set(data) != {'x'} or not isinstance(data['x'], list)
            or any(not isinstance(row, list) or any(type(x) is not int for x in row) for row in data['x'])):
        raise ValueError('input must contain an integer matrix named x')
    inputs = {'x': torch.tensor(data['x'], dtype=torch.int64)}
    model.validate_state(inputs)
    return config, model, inputs


def contexts(directory, role):
    directory = Path(directory)
    args = SimpleNamespace(ca=str(directory / 'ca.pem'), cert=str(directory / f'{role}.pem'),
                           key=str(directory / f'{role}.key'))
    return tls_context(args, server=True), tls_context(args, server=False)


def run_prover(directory, traffic=None):
    """Serve one committee request; keep shares and keys out of the report."""
    directory = Path(directory)
    config, model, inputs = load_setup(directory)
    meter = TrafficMeter(traffic)
    server, _ = contexts(directory, 'prover')
    with ExitStack() as stack:
        listener = stack.enter_context(socket.create_server(tuple(config['prover'])))
        listener.settimeout(config['timeout'])
        peers = {}
        for _ in range(4):
            raw, _ = listener.accept()
            stack.enter_context(raw)
            raw.settimeout(config['timeout'])
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection = stack.enter_context(server.wrap_socket(raw, server_side=True))
            connection = meter.wrap(connection, 'prover_links')
            hello = receive(connection, 'hello', {'party'})
            party = hello['party']
            if type(party) is not int or party not in range(4) or party in peers:
                raise ValueError('unexpected or duplicate verifier identity')
            check_peer(connection, directory / f'party{party}.pem')
            peers[party] = connection
        requests = [receive(peers[p], 'request', {'model_id', 'run_id', 'input_digest', 'samples'})
                    for p in range(4)]
        if any(request != requests[0] for request in requests):
            raise ValueError('verifiers disagree on the request')
        request = requests[0]
        samples = config['samples'] if config['samples'] is not None else len(model.layers)
        if (type(request['samples']) is not int or not isinstance(request['run_id'], str)
                or len(request['run_id']) != 64 or any(c not in '0123456789abcdef' for c in request['run_id'])
                or request['model_id'] != model.model_id or request['input_digest'] != hashlib.sha256(
                canonical({'x': inputs['x'].tolist()})).hexdigest() or request['samples'] != samples):
            raise ValueError('request differs from configured model, input, or policy')
        trace = PrivateTrace(model, inputs, run_id=request['run_id'],
                             checkpoint_interval=config['checkpoint_interval'])
        for connection in peers.values():
            send_message(connection, message('commitment', commitment=trace.commitment))
        challenges = [receive(peers[p], 'challenge', {'indices'})['indices'] for p in range(4)]
        indices = challenges[0]
        if (any(value != indices for value in challenges) or not isinstance(indices, list)
                or len(indices) != samples or any(type(i) is not int or not 0 <= i < len(model.layers) for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError('invalid or inconsistent challenge')
        for boundary in boundaries(indices, len(model.layers)):
            for p, connection in peers.items():
                send_message(connection, message('share_opening', opening=trace.opening(boundary, p)))
        reports = [receive(peers[p], 'result', {'report'})['report'] for p in range(4)]
        expected = report_for(trace.commitment, indices, trace.output['x'])
        if any(canonical(report) != canonical(expected) for report in reports):
            raise ValueError('committee did not return a matching accepted result')
        for connection in peers.values():
            send_message(connection, message('complete', report=expected))
        return expected


def connect_prover(address, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            connection = socket.create_connection(tuple(address), timeout=max(0.001, deadline - time.monotonic()))
            connection.settimeout(timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return connection
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def run_verifier(directory, party_id, traffic=None):
    directory = Path(directory)
    config, model, inputs = load_setup(directory, party_id)
    meter = TrafficMeter(traffic)
    samples = config['samples'] if config['samples'] is not None else len(model.layers)
    server, client = contexts(directory, f'party{party_id}')
    with ExitStack() as stack:
        transport = SocketTransport(party_id, [tuple(a) for a in config['verifiers']],
            timeout=config['timeout'], server_context=server, client_context=client,
            peer_hostnames={p: f'party{p}.test' for p in range(4)}, max_frame_bytes=MAX_FRAME_BYTES)
        stack.callback(transport.close)
        for p, connection in transport.peers.items():
            check_peer(connection, directory / f'party{p}.pem')
        transport.peers = {p: meter.wrap(c, 'verifier_mesh') for p, c in transport.peers.items()}
        # SocketTransport exchanges one role byte per connection before wrapping.
        meter.add('mesh_identity', party_id, 3 - party_id)
        public_context = canonical({'model_id': model.model_id, 'input': inputs['x'].tolist(), 'config': config})
        agreement(transport, party_id, public_context)
        run_id = joint_random(transport, party_id, b'siv.private-audit.run.v1\0' + public_context).hex()
        seeds = correlated_seeds(party_id, transport, context=b'siv.private-audit.seeds.v1\0' + run_id.encode('ascii'))
        party = Party(party_id, transport, seeds, bits=64, binary_bits=64, prg='aes')
        raw = stack.enter_context(connect_prover(config['prover'], config['timeout']))
        connection = stack.enter_context(client.wrap_socket(raw, server_hostname='prover.test'))
        check_peer(connection, directory / 'prover.pem')
        connection = meter.wrap(connection, 'prover_links')
        send_message(connection, message('hello', party=party_id))
        send_message(connection, message('request', model_id=model.model_id, run_id=run_id,
            input_digest=hashlib.sha256(canonical({'x': inputs['x'].tolist()})).hexdigest(), samples=samples))
        value = receive(connection, 'commitment', {'commitment'})['commitment']
        commitment = validate_commitment(value, model=model, run_id=run_id, shape=list(inputs['x'].shape))
        agreement(transport, party_id, canonical(commitment))
        seed = joint_random(transport, party_id, b'siv.private-audit.challenge.v1\0' + canonical(commitment))
        indices = sample_indices(seed, len(model.layers), samples)
        send_message(connection, message('challenge', indices=indices))
        states = {}
        for boundary in boundaries(indices, len(model.layers)):
            value = receive(connection, 'share_opening', {'opening'})['opening']
            states[boundary] = verify_private_opening(commitment, value, party=party_id,
                expected_boundary=boundary, model=model, run_id=run_id, shape=list(inputs['x'].shape))
        agreement(transport, party_id, b'siv.private-audit.shares-validated.v1\0' + canonical(commitment))
        before_check = meter.snapshot('verifier_mesh')
        output = verify_shared_trace(party, model, states, inputs['x'], indices)
        meter.record_check(before_check)
        report = report_for(commitment, indices, output)
        agreement(transport, party_id, canonical(report))
        send_message(connection, message('result', report=report))
        complete = receive(connection, 'complete', {'report'})['report']
        if canonical(complete) != canonical(report):
            raise ValueError('prover completion differs from the verified result')
        return report


def init_private(directory):
    init_example(directory, roles=('prover', 'party0', 'party1', 'party2', 'party3'))
    directory = Path(directory)
    manifest, bundles = provision_private_model(AuditModel.load(directory / 'model.json'))
    write_json(directory / 'model-manifest.json', manifest)
    (directory / 'model.json').chmod(0o600)
    for party_id, bundle in enumerate(bundles):
        path = directory / f'party{party_id}-model.json'
        write_json(path, bundle)
        path.chmod(0o600)
    write_json(directory / 'deployment.json', {
        'version': 1, 'prover': ['127.0.0.1', 19443],
        'verifiers': [['127.0.0.1', 19444 + p] for p in range(4)],
        'samples': None, 'timeout': 60, 'checkpoint_interval': 8})


def demo(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    codes, processes, error = {}, {}, None
    with tempfile.TemporaryDirectory(prefix='siv-private-audit-') as temporary, ExitStack() as stack:
        setup = Path(temporary) / 'setup'
        init_private(setup)
        # Reserve all five ports together, then release them just before launch.
        reservations = [socket.socket() for _ in range(5)]
        try:
            for sock in reservations:
                sock.bind(('127.0.0.1', 0))
            addresses = [list(sock.getsockname()) for sock in reservations]
        finally:
            for sock in reservations:
                sock.close()
        config = json.loads((setup / 'deployment.json').read_text())
        config.update(prover=addresses[0], verifiers=addresses[1:], samples=args.samples, timeout=args.timeout)
        write_json(setup / 'deployment.json', config)
        provisioning = provisioning_sizes(setup)
        try:
            for role in ('prover', 'party0', 'party1', 'party2', 'party3'):
                # Local demo isolation mirrors file distribution on separate hosts.
                role_setup = Path(temporary) / role
                role_setup.mkdir(mode=0o700)
                copy_role_setup(setup, role_setup, role)
                log = stack.enter_context((output / f'{role}.log').open('w'))
                command = [sys.executable, '-m', 'rep4gpu.proofs.private_audit_service',
                           'prover' if role == 'prover' else 'verifier', '--setup', str(role_setup),
                           '--output-dir', str(output / role)]
                if role != 'prover':
                    command += ['--party', role[-1]]
                processes[role] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.timeout + 30
            while any(process.poll() is None for process in processes.values()):
                if any(process.poll() not in (None, 0) for process in processes.values()):
                    raise RuntimeError('a protocol process failed; see role logs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('private audit exceeded its execution deadline')
                time.sleep(0.05)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            for role, process in processes.items():
                if process.poll() is None:
                    process.kill()
                codes[role] = process.wait()
    passed = error is None and len(codes) == 5 and all(code == 0 for code in codes.values())
    summary = {'passed': passed, 'exit_codes': codes, 'error': error,
               'model_provisioning': provisioning}
    if passed:
        summary['traffic'] = aggregate_traffic(output)
    write_json(output / 'summary.json', summary)
    print(f'{"Passed" if passed else "Failed"}: {output / "summary.json"}', flush=True)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--directory', required=True)
    local = sub.add_parser('demo')
    local.add_argument('--output-dir', required=True)
    local.add_argument('--samples', type=int)
    local.add_argument('--timeout', type=float, default=60)
    for name in ('prover', 'verifier'):
        child = sub.add_parser(name)
        child.add_argument('--setup', required=True)
        child.add_argument('--output-dir', required=True)
        if name == 'verifier':
            child.add_argument('--party', type=int, choices=range(4), required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == 'demo' and (not math.isfinite(args.timeout) or args.timeout <= 0
                                  or (args.samples is not None and not 1 <= args.samples <= 4)):
        parser.error('demo requires a positive timeout and samples between one and four')
    output = None
    try:
        if args.command == 'init':
            init_private(args.directory)
            print(f'Created private audit setup in {args.directory}')
            return 0
        if args.command == 'demo':
            return demo(args)
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        traffic = {}
        try:
            report = (run_prover(args.setup, traffic) if args.command == 'prover'
                      else run_verifier(args.setup, args.party, traffic))
        finally:
            write_json(output / 'traffic.json', traffic)
        write_json(output / 'result.json', report)
        print('Accepted: Rep4 checks passed; only final output opened', flush=True)
        return 0
    except Exception as exc:
        failure = {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
        if output is not None and output.exists() and not (output / 'result.json').exists():
            write_json(output / 'failure.json', failure)
        print(failure['error'], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
