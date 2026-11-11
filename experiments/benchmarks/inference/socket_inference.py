"""Run a checked four-process Rep4 toy network over mutually authenticated TLS.

Run from the repository root with ``python -m experiments.benchmarks.inference.socket_inference``.
Each role receives only its assigned toy inputs. Temporary TLS private keys and
role input files are removed after the run; JSON reports contain final outputs.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from experiments.benchmarks.inference.inference import TruncationPolicy, fixture, infer, synchronize
from rep4gpu.protocol.runtime import Party
from rep4gpu.protocol.setup import correlated_seeds
from rep4gpu.protocol.transport import SocketTransport


class MeteredTransport(SocketTransport):
    def __init__(self, *args, **kwargs):
        self.rounds = 0
        super().__init__(*args, **kwargs)

    def exchange(self, outgoing):
        self.rounds += 1
        return super().exchange(outgoing)


def certificates(directory):
    """Create one ephemeral CA and four distinct TLS identities."""
    directory = Path(directory)
    def openssl(*args):
        subprocess.run(['openssl', *map(str, args)], check=True, capture_output=True, text=True)
    ca, ca_key = directory / 'ca.pem', directory / 'ca.key'
    openssl('req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1',
            '-nodes', '-keyout', ca_key, '-out', ca, '-days', '1', '-subj', '/CN=Rep4 toy CA')
    identities = []
    for p in range(4):
        key, csr, cert, ext = [directory / f'party{p}.{suffix}' for suffix in ('key', 'csr', 'pem', 'ext')]
        ext.write_text(f'subjectAltName=DNS:party{p}\nextendedKeyUsage=serverAuth,clientAuth\n')
        openssl('req', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1',
                '-nodes', '-keyout', key, '-out', csr, '-subj', f'/CN=party{p}')
        openssl('x509', '-req', '-in', csr, '-CA', ca, '-CAkey', ca_key, '-CAcreateserial',
                '-out', cert, '-days', '1', '-extfile', ext)
        identities.append(dict(certificate=str(cert), key=str(key)))
    return str(ca), identities


def reserve_addresses():
    sockets = []
    try:
        for _ in range(4):
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            sockets.append(sock)
        return [list(sock.getsockname()) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def child(config_path, result_path):
    config = json.loads(Path(config_path).read_text())
    p = config['party']
    transport = None
    torch.set_num_threads(1)
    try:
        cert, key = config['identity']['certificate'], config['identity']['key']
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(cert, key)
        server.load_verify_locations(config['ca'])
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=config['ca'])
        client.load_cert_chain(cert, key)
        start = time.perf_counter()
        transport = MeteredTransport(p, [tuple(a) for a in config['addresses']],
                                     timeout=config['timeout'], server_context=server,
                                     client_context=client, peer_hostnames={q: f'party{q}' for q in range(4)})
        handshake_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        seeds = correlated_seeds(p, transport, context=b'Rep4-toy-socket-inference-v1')
        setup_ms = (time.perf_counter()-start)*1000
        base = Party
        party_transport = transport
        if config['fused']:
            from rep4gpu.protocol.fused_runtime import FusedParty
            base = FusedParty
        if config.get('device_protocol', False):
            from rep4gpu.models.device_inference import DeviceInferenceParty
            from rep4gpu.protocol.device_transport import DeviceByteBridge
            base = DeviceInferenceParty
            party_transport = DeviceByteBridge(transport, config['device'])
        options = {}
        if config.get('device_protocol', False):
            runtime_type = base
            options['optimized'] = config.get('optimized', False)
            options['gpu_open'] = config.get('gpu_open', False)
        else:
            runtime_type = type('SocketInferenceParty', (TruncationPolicy, base), {'truncation': 'compact'})
        runtime = runtime_type(p, party_transport, seeds, device=config['device'], bits=config['bits'],
                               binary_bits=config['bits'] if config.get('device_protocol', False) else 64,
                               **options)
        start = time.perf_counter()
        values = []
        for index, (owner, shape) in enumerate(zip([0, 1, 1, 1, 1], config['shapes'])):
            own = config['inputs'].get(str(index))
            value = torch.tensor(own, dtype=torch.int64) if p == owner else None
            values.append(runtime.input(owner, value, shape))
        synchronize(config['device'])
        input_ms = (time.perf_counter()-start)*1000
        input_bytes, input_rounds = transport.bytes_sent, transport.rounds
        start = time.perf_counter()
        output = infer(runtime, values, config['fractional_bits'])
        synchronize(config['device'])
        inference_ms = (time.perf_counter()-start)*1000
        result = dict(party=p, passed=True, device=config['device'], bits=config['bits'],
                      device_protocol=config.get('device_protocol', False),
                      optimized=config.get('optimized', False),
                      gpu_open=config.get('gpu_open', False),
                      authenticated_tls=transport.authenticated, distributed_setup=True,
                      handshake_ms=handshake_ms, correlated_setup_ms=setup_ms,
                      input_ms=input_ms, checked_inference_ms=inference_ms,
                      payload_bytes=transport.bytes_sent, rounds=transport.rounds,
                      inference_payload_bytes=transport.bytes_sent-input_bytes,
                      inference_rounds=transport.rounds-input_rounds,
                      output=output.cpu().tolist())
        Path(result_path).write_text(json.dumps(result, indent=2)+'\n')
        print(f'Party {p}: checked inference completed; payload={transport.bytes_sent} bytes', flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        Path(result_path).write_text(json.dumps(dict(party=p, passed=False, error=f'{type(error).__name__}: {error}'), indent=2)+'\n')
        return 1
    finally:
        if transport is not None:
            transport.close()


def run(output, device='cpu', bits=64, width=8, batch=1, fused=False, timeout=120, fractional_bits=12,
        device_protocol=False, optimized=False, gpu_open=False):
    """Return the saved report; raise on invalid fixture/device before launch."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if width < 1 or batch < 1 or timeout <= 0:
        raise ValueError('width, batch, and timeout must be positive')
    if bits not in (32, 64):
        raise ValueError('bits must be 32 or 64')
    if (optimized or gpu_open) and not device_protocol:
        raise ValueError('--optimized and --gpu-open require --device-protocol')
    if (fused or device_protocol) and torch.device(device).type != 'cuda':
        raise ValueError('--fused and --device-protocol each requires a CUDA device')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    _, integers, expected = fixture(batch, width, width, 4, fractional_bits)
    x, w1, b1, w2, _ = integers
    raw1 = x @ w1 + (b1 << fractional_bits)
    h = (raw1 >> fractional_bits).clamp_min(0)
    raw2 = h @ w2
    maximum = max(int(raw1.abs().max()), int(raw2.abs().max()))
    if maximum >= 1 << (bits-1):
        raise ValueError(f'toy accumulator {maximum} exceeds signed {bits}-bit range')
    logs = output.parent / (output.stem + '_logs')
    logs.mkdir(parents=True, exist_ok=True)
    for p in range(4):
        (logs / f'party{p}.json').unlink(missing_ok=True)
    processes, handles = [], []
    timed_out = False
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='rep4-socket-toy-') as temporary:
        os.chmod(temporary, 0o700)
        temporary = Path(temporary)
        ca, identities = certificates(temporary)
        addresses = reserve_addresses()
        shapes = [list(t.shape) for t in integers]
        try:
            for p in range(4):
                owned = {str(i): t.tolist() for i, (owner, t) in enumerate(zip([0, 1, 1, 1, 1], integers)) if p == owner}
                config = dict(party=p, ca=ca, identity=identities[p], addresses=addresses,
                              inputs=owned, shapes=shapes, bits=bits, device=device, fused=fused,
                              device_protocol=device_protocol,
                              optimized=optimized,
                              gpu_open=gpu_open,
                              timeout=timeout, fractional_bits=fractional_bits)
                config_path = temporary / f'party{p}.json'
                config_path.write_text(json.dumps(config))
                log = open(logs / f'party{p}.log', 'w')
                handles.append(log)
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                processes.append(subprocess.Popen([sys.executable, '-m', 'experiments.benchmarks.inference.socket_inference',
                    '--party-config', str(config_path), '--party-result', str(logs / f'party{p}.json')],
                    cwd=Path(__file__).resolve().parents[3], stdout=log, stderr=subprocess.STDOUT, env=env))
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            for handle in handles:
                handle.close()
            raise
        deadline = time.monotonic() + timeout
        try:
            for process in processes:
                process.wait(timeout=max(0.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for handle in handles:
                handle.close()
    results = []
    for p in range(4):
        path = logs / f'party{p}.json'
        results.append(json.loads(path.read_text()) if path.exists() else dict(party=p, passed=False, error='no result'))
    codes = [process.returncode for process in processes]
    matched = all(result.get('passed') and result.get('output') == expected.tolist() for result in results)
    report = dict(passed=matched and all(code == 0 for code in codes) and not timed_out,
                  method='Four isolated processes; distinct mutual-TLS identities; distributed triple-seed commit/reveal; owner0 inputs, owner1 weights and biases; compact exact truncation; checked final opening.',
                  device=device, bits=bits, fused=fused, width=width, hidden=width, out=4, batch=batch,
                  device_protocol=device_protocol,
                  optimized=optimized,
                  gpu_open=gpu_open,
                  online_prg='chacha20' if device_protocol else 'shake',
                  online_hash='blake3' if device_protocol else 'blake2b-256',
                  staging='host',
                  fractional_bits=fractional_bits, maximum_accumulator=maximum, range_validated=True,
                  return_codes=codes, timed_out=timed_out, exact_integer_match=matched,
                  expected=expected.tolist(), parties=results, total_wall_ms=(time.perf_counter()-start)*1000,
                  global_payload_bytes=sum(result.get('payload_bytes', 0) for result in results),
                  global_inference_payload_bytes=sum(result.get('inference_payload_bytes', 0) for result in results))
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bits', type=int, choices=(32, 64), default=64)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--device-protocol', action='store_true',
                        help='experimental CUDA ChaCha20/BLAKE3 protocol through the TLS byte bridge')
    parser.add_argument('--optimized', action='store_true',
                        help='batch GPU masks and fuse carry/conversion kernels in the device protocol')
    parser.add_argument('--gpu-open', action='store_true', help='check output opening on CUDA with BLAKE3')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/transport/socket_inference.json'))
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config is not None:
        return child(args.party_config, args.party_result)
    try:
        report = run(args.output, args.device, args.bits, args.width, args.batch, args.fused, args.timeout,
                     device_protocol=args.device_protocol, optimized=args.optimized, gpu_open=args.gpu_open)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ('passed', 'return_codes', 'exact_integer_match', 'global_payload_bytes', 'total_wall_ms')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
