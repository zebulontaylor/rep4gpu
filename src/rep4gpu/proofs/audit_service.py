"""Run a plaintext layer audit between mutually authenticated server processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import time

import torch

from .audit_model import AuditModel, example_spec
from .audit_protocol import serve_session, verify_session


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def init_example(directory, *, roles=('prover', 'verifier')):
    """Create a new demo directory and role certificates; discard the CA key."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'SIV audit demo CA')])

    def builder(subject, public_key):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
                .public_key(public_key).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=5))
                .not_valid_after(now + timedelta(days=7)))

    ca = (builder(ca_name, ca_key.public_key())
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    (directory / 'ca.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    for role in roles:
        usages = ([ExtendedKeyUsageOID.SERVER_AUTH] if role == 'prover' else
                  [ExtendedKeyUsageOID.CLIENT_AUTH] if role == 'verifier' else
                  [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH])
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, role + '.test')])
        cert = (builder(subject, key.public_key())
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName(role + '.test')]), critical=False)
                .add_extension(x509.ExtendedKeyUsage(usages), critical=False)
                .sign(ca_key, hashes.SHA256()))
        (directory / f'{role}.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path = directory / f'{role}.key'
        with key_path.open('xb') as stream:
            key_path.chmod(0o600)
            stream.write(key.private_bytes(serialization.Encoding.PEM,
                                          serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    write_json(directory / 'model.json', example_spec())
    write_json(directory / 'input.json', {'x': [[1, -2, 3, 4]]})


def tls_context(args, *, server):
    if server:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(args.ca)
    else:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=args.ca)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(args.cert, args.key)
    return context


def check_peer(sock, certificate):
    expected = ssl.PEM_cert_to_DER_cert(Path(certificate).read_text())
    if not hmac.compare_digest(hashlib.sha256(sock.getpeercert(binary_form=True)).digest(),
                               hashlib.sha256(expected).digest()):
        raise ValueError('peer certificate does not match the configured role identity')


def serve(args):
    model = AuditModel.load(args.model)
    context = tls_context(args, server=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    failed = False
    with socket.create_server((args.host, args.port)) as listener:
        address = listener.getsockname()
        if args.ready_file:
            write_json(args.ready_file, {'host': address[0], 'port': address[1]})
        print(f'Prover listening on {address[0]}:{address[1]}', flush=True)
        count = 0
        while args.sessions == 0 or count < args.sessions:
            raw, _ = listener.accept()
            count += 1
            try:
                with raw:
                    raw.settimeout(args.timeout)
                    with context.wrap_socket(raw, server_side=True) as connection:
                        check_peer(connection, args.peer_cert)
                        report = serve_session(connection, model, checkpoint_interval=args.checkpoint_interval)
                print(f'Session {count}: verifier accepted', flush=True)
            except Exception as exc:
                failed = True
                report = {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
                print(f'Session {count}: {report["error"]}', file=sys.stderr, flush=True)
            write_json(Path(args.output_dir) / f'session-{count:06d}.json', report)
    return int(failed)


def verify(args):
    try:
        model = AuditModel.load(args.model)
        with Path(args.input).open('rb') as stream:
            payload = stream.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError('input file exceeds 1 MiB')
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {'x'} or not isinstance(value['x'], list):
            raise ValueError('input must contain x as an integer matrix')
        if any(not isinstance(row, list) or any(type(x) is not int for x in row) for row in value['x']):
            raise ValueError('input values must be integers')
        inputs = {'x': torch.tensor(value['x'], dtype=torch.int64)}
        model.validate_state(inputs)
        context = tls_context(args, server=False)
        with socket.create_connection((args.host, args.port), timeout=args.timeout) as raw:
            with context.wrap_socket(raw, server_hostname=args.server_name) as connection:
                check_peer(connection, args.peer_cert)
                report = verify_session(connection, model, inputs, sample_count=args.samples)
        write_json(args.output, report)
        print(json.dumps(report), flush=True)
        return 0
    except Exception as exc:
        report = {'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
        write_json(args.output, report)
        print(report['error'], file=sys.stderr, flush=True)
        return 1


def demo(args):
    """Launch two real TLS processes, retaining logs and both exit codes."""
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    codes = {'prover': None, 'verifier': None}
    error = None
    with tempfile.TemporaryDirectory(prefix='siv-audit-') as temporary:
        setup = Path(temporary) / 'setup'
        init_example(setup)
        ready = Path(temporary) / 'ready.json'
        base = [sys.executable, '-m', 'rep4gpu.proofs.audit_service']

        def role_args(role):
            peer = 'verifier' if role == 'prover' else 'prover'
            return ['--model', str(setup / 'model.json'), '--ca', str(setup / 'ca.pem'),
                    '--cert', str(setup / f'{role}.pem'), '--key', str(setup / f'{role}.key'),
                    '--peer-cert', str(setup / f'{peer}.pem'), '--timeout', str(args.timeout)]

        prover = None
        with (output / 'prover.log').open('w') as prover_log, (output / 'verifier.log').open('w') as verifier_log:
            try:
                prover = subprocess.Popen(base + ['serve', *role_args('prover'), '--port', '0',
                    '--sessions', '1', '--ready-file', str(ready), '--output-dir', str(output / 'prover')],
                    stdout=prover_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.timeout
                while not ready.exists():
                    if prover.poll() is not None:
                        raise RuntimeError('prover exited before becoming ready')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('prover startup timed out')
                    time.sleep(0.05)
                address = json.loads(ready.read_text())
                command = base + ['verify', *role_args('verifier'), '--port', str(address['port']),
                    '--input', str(setup / 'input.json'), '--output', str(output / 'verifier.json')]
                if args.samples is not None:
                    command += ['--samples', str(args.samples)]
                completed = subprocess.run(command, stdout=verifier_log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout + 10)
                codes['verifier'] = completed.returncode
                codes['prover'] = prover.wait(timeout=args.timeout + 10)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                if prover is not None:
                    if prover.poll() is None:
                        prover.kill()
                    codes['prover'] = prover.wait()
    passed = error is None and all(code == 0 for code in codes.values())
    write_json(output / 'summary.json', {'passed': passed, 'exit_codes': codes, 'error': error})
    print(f'{"Passed" if passed else "Failed"}: {output / "summary.json"}', flush=True)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='create a fresh toy model, input, and seven-day demo certificates')
    init.add_argument('--directory', required=True)
    local = commands.add_parser('demo', help='run prover and verifier in separate local TLS processes')
    local.add_argument('--output-dir', required=True)
    local.add_argument('--samples', type=int)
    local.add_argument('--timeout', type=float, default=60)
    for name in ('serve', 'verify'):
        child = commands.add_parser(name)
        for flag in ('model', 'ca', 'cert', 'key', 'peer-cert'):
            child.add_argument('--' + flag, required=True)
        child.add_argument('--host', default='127.0.0.1')
        child.add_argument('--port', type=int, default=9443)
        child.add_argument('--timeout', type=float, default=60)
        if name == 'serve':
            child.add_argument('--sessions', type=int, default=0, help='connections to handle; 0 runs until stopped')
            child.add_argument('--checkpoint-interval', type=int, default=8)
            child.add_argument('--output-dir', required=True)
            child.add_argument('--ready-file', help=argparse.SUPPRESS)
        else:
            child.add_argument('--server-name', default='prover.test')
            child.add_argument('--input', required=True)
            child.add_argument('--samples', type=int, help='sample this many layers; omission checks every layer')
            child.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if hasattr(args, 'timeout') and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be finite and positive')
    if hasattr(args, 'sessions') and (args.sessions < 0 or args.checkpoint_interval < 1):
        parser.error('--sessions must be nonnegative and --checkpoint-interval must be positive')
    torch.set_num_threads(1)
    try:
        if args.command == 'init':
            init_example(args.directory)
            print(f'Created setup in {args.directory}')
            return 0
        return {'serve': serve, 'verify': verify, 'demo': demo}[args.command](args)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
