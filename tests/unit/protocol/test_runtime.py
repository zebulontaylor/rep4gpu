import concurrent.futures
import torch
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.transports.byte import IdealNetwork


def run_four(function, device="cpu"):
    seeds = make_seeds()
    network = IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p}, device=device) for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(function, parties))


def test_full_ring_product():
    x = torch.tensor([-(1 << 63), -1, 2**62, 17, 0], dtype=torch.int64)
    y = torch.tensor([-1, -(1 << 63), 7, -91, 0], dtype=torch.int64)
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape)
        b = p.input(2, y if p.party == 2 else None, y.shape)
        before = p.transport.bytes_sent
        result = p.mul(a, b)
        sent = p.transport.bytes_sent - before
        return p.open(result), sent
    outputs = run_four(work)
    for result, _ in outputs:
        assert torch.equal(result, x * y)
    assert sum(n for _, n in outputs) == 6 * x.numel() * 8 + 6 * 32


def test_binary_product():
    x = torch.tensor([-(1 << 63), -1, 71], dtype=torch.int64)
    y = torch.tensor([-1, 71, -91], dtype=torch.int64)
    def work(p):
        a = p.input(1, x if p.party == 1 else None, x.shape, binary=True)
        b = p.input(3, y if p.party == 3 else None, y.shape, binary=True)
        return p.open(p.mul(a, b))
    for result in run_four(work):
        assert torch.equal(result, x & y)


def test_public_broadcast():
    def work(p):
        x = p.public(torch.tensor([1, 2, 3], dtype=torch.int64))
        return p.open(p.add(x, p.public(7, like=x)))
    for result in run_four(work):
        assert torch.equal(result, torch.tensor([8, 9, 10]))


def _socket_worker(party, addresses, seeds, queue, tls_files=None):
    from rep4gpu.transports.byte import SocketTransport
    options = {}
    if tls_files is not None:
        import ssl
        certificate, key = tls_files
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(certificate, key)
        server.load_verify_locations(certificate)
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=certificate)
        client.load_cert_chain(certificate, key)
        options = dict(server_context=server, client_context=client,
                       peer_hostnames={p: 'localhost' for p in range(4)})
    transport = SocketTransport(party, addresses, timeout=15, **options)
    try:
        p = Party(party, transport, seeds)
        x = torch.tensor([[2**62, -1], [71, -(1 << 63)]], dtype=torch.int64)
        y = torch.tensor([[7, -91], [-1, 2**62]], dtype=torch.int64)
        a = p.input(0, x if party == 0 else None, x.shape)
        b = p.input(1, y if party == 1 else None, y.shape)
        result = p.open(p.matmul(a, b))
        queue.put((party, torch.equal(result, x @ y), None))
    except Exception as error:
        queue.put((party, False, repr(error)))
        raise
    finally:
        transport.close()


def test_four_process_socket_matmul(tmp_path):
    _run_socket_regression()


def test_four_process_tls_matmul(tmp_path):
    import shutil
    import subprocess
    import pytest
    if shutil.which('openssl') is None:
        pytest.skip('openssl required for TLS test certificates')
    certificate = str(tmp_path / 'localhost.pem')
    key = str(tmp_path / 'localhost.key')
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', key, '-out', certificate, '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost'],
                   check=True, capture_output=True)
    _run_socket_regression((certificate, key))


def _run_socket_regression(tls_files=None):
    import multiprocessing
    import socket
    context = multiprocessing.get_context('spawn')
    reservations = []
    addresses = []
    for _ in range(4):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        reservations.append(sock)
        addresses.append(sock.getsockname())
    for sock in reservations:
        sock.close()
    seeds = make_seeds()
    queue = context.Queue()
    processes = [context.Process(target=_socket_worker, args=(p, addresses,
                 {i: s for i, s in seeds.items() if i != p}, queue, tls_files)) for p in range(4)]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=15)
        assert all(process.exitcode == 0 for process in processes)
        assert all(ok for _, ok, _ in results), results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def test_joint_input_tamper_rejected():
    import pytest
    def work(p):
        original_exchange = p.transport.exchange
        def corrupt(outgoing):
            if p.party == 0 and 3 in outgoing:
                data = bytearray(outgoing[3])
                data[0] ^= 1
                outgoing = dict(outgoing)
                outgoing[3] = bytes(data)
            return original_exchange(outgoing)
        p.transport.exchange = corrupt
        x = p.public(torch.tensor([17], dtype=torch.int64))
        try:
            p.mul(x, x)
            return False
        except ValueError as error:
            assert 'hash mismatch' in str(error)
            return True
    assert run_four(work) == [False, False, False, True]


def test_joint_transcript_matches_independent_oracle():
    import hashlib
    from rep4gpu.protocol.runtime import SCHEDULE, Share
    mask64 = (1 << 64) - 1

    def signed(value):
        return value if value < 1 << 63 else value - (1 << 64)

    def oracle_multiply(xparts, yparts, binary):
        op = (lambda a, b: a ^ b) if binary else (lambda a, b: (a + b) & mask64)
        sub = (lambda a, b: a ^ b) if binary else (lambda a, b: (a - b) & mask64)
        mul = (lambda a, b: a & b) if binary else (lambda a, b: (a * b) & mask64)
        contribs = [
            op(mul(op(xparts[2], xparts[3]), yparts[2]), mul(xparts[2], yparts[3])),
            op(mul(op(xparts[3], xparts[0]), yparts[3]), mul(xparts[3], yparts[0])),
            op(mul(op(xparts[0], xparts[1]), yparts[0]), mul(xparts[0], yparts[1])),
            op(mul(op(xparts[1], xparts[2]), yparts[1]), mul(xparts[1], yparts[2])),
            op(mul(xparts[1], yparts[3]), mul(xparts[3], yparts[1])),
            op(mul(xparts[0], yparts[2]), mul(xparts[2], yparts[0])),
        ]
        out_components = [0, 0, 0, 0]
        payloads, hashes = {}, {}
        for k, (s, b, r, o) in enumerate(SCHEDULE):
            m = int.from_bytes(hashlib.blake2b(f'mask/{binary}/{k}'.encode(), digest_size=32).digest()[:8], 'little')
            masked = sub(contribs[k], m)
            out_components[r] = op(out_components[r], m)
            out_components[o] = op(out_components[o], masked)
            raw = masked.to_bytes(8, 'little')
            payloads.setdefault((s, r), []).append(raw)
            hashes.setdefault((b, r), []).append((s, hashlib.blake2b(raw, digest_size=32).digest()))
        edges = []
        for s in range(4):
            for r in range(4):
                if (s, r) in payloads or (s, r) in hashes:
                    frame = b''.join(payloads.get((s, r), ())) + b''.join(h for _, h in sorted(hashes.get((s, r), ())))
                    edges.append({'sender': s, 'receiver': r, 'hex': frame.hex()})
        return {
            'edges': edges,
            'output_shares': [[out_components[(p + j + 1) % 4] for j in range(3)] for p in range(4)],
        }

    for binary in (False, True):
        xparts = [mask64, 7, 0, 3]
        yparts = [11, 0, 1 << 63, 9]
        expected = oracle_multiply(xparts, yparts, binary)
        def work(p):
            calls = {0: 0, 1: 0, 2: 0, 3: 0}
            schedule_indices = {0: (1,), 1: (2,), 2: (3, 5), 3: (0, 4)}
            def masks(excluded, shape, device, binary=False):
                k = schedule_indices[excluded][calls[excluded]]
                calls[excluded] += 1
                value = int.from_bytes(hashlib.blake2b(f'mask/{binary}/{k}'.encode(), digest_size=32).digest()[:8], 'little')
                return torch.full(shape, signed(value), dtype=torch.int64, device=device)
            p._random = masks
            captured = {}
            exchange = p.transport.exchange
            def capture(outgoing):
                captured.update(outgoing)
                return exchange(outgoing)
            p.transport.exchange = capture
            a = Share(tuple(torch.tensor([signed(xparts[(p.party+j+1)%4])]) for j in range(3)), binary)
            b = Share(tuple(torch.tensor([signed(yparts[(p.party+j+1)%4])]) for j in range(3)), binary)
            result = p.mul(a, b)
            return captured, [int(t.item()) & mask64 for t in result.parts]
        actual = run_four(work)
        for edge in expected['edges']:
            assert actual[edge['sender']][0][edge['receiver']].hex() == edge['hex']
        for party in range(4):
            assert actual[party][1] == expected['output_shares'][party]


def test_cuda_fused_and_prepared_matmul():
    import pytest
    from rep4gpu.protocol.runtime import prepare_weight_share
    if not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    x = torch.tensor([[2**62, -1], [71, -(1 << 63)]], dtype=torch.int64, device='cuda')
    y = torch.tensor([[7, -91], [-1, 2**62]], dtype=torch.int64, device='cuda')
    expected = (x.cpu() @ y.cpu()).to('cuda')
    def work(p):
        a = p.input(0, x if p.party == 0 else None, x.shape)
        b = p.input(1, y if p.party == 1 else None, y.shape)
        plain = p.open(p.matmul(a, b))
        packed = p.open(p.matmul(a, prepare_weight_share(b)))
        return torch.equal(plain, expected) and torch.equal(packed, expected)
    assert all(run_four(work, device='cuda'))


def test_recording_and_replay():
    import pytest
    from rep4gpu.transports.byte import RecordingTransport, ReplayTransport
    seeds = make_seeds()
    network = IdealNetwork()
    recordings = [RecordingTransport(network.endpoint(p)) for p in range(4)]
    def operation(p):
        a = p.public(torch.tensor([17, -91], dtype=torch.int64))
        return p.open(p.mul(a, a))
    parties = [Party(p, recordings[p], {i:s for i,s in seeds.items() if i != p}) for p in range(4)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        expected = list(pool.map(operation, parties))
    replay = ReplayTransport(recordings[0].transcript)
    p = Party(0, replay, {i:s for i,s in seeds.items() if i != 0})
    assert torch.equal(operation(p), expected[0])
    replay.assert_consumed()
    with pytest.raises(ValueError, match='exhausted'):
        replay.exchange({})
    replay = ReplayTransport(recordings[0].transcript)
    with pytest.raises(ValueError, match='consumed 0'):
        replay.assert_consumed()
    with pytest.raises(ValueError, match='outgoing mismatch'):
        replay.exchange({})
    assert replay.round == 0


def test_replay_phase_timing():
    from rep4gpu.transports.byte import ReplayTransport
    ticks = iter([10.0, 10.2, 11.0, 11.4, 12.0, 12.8])
    synchronized = []
    replay = ReplayTransport([({}, {}), ({}, {})])
    replay.start(clock=lambda: next(ticks), synchronize=lambda: synchronized.append(True))
    replay.exchange({})
    replay.exchange({})
    phases = replay.finish()
    assert len(synchronized) == 4
    assert all(abs(a - b) < 1e-9 for a, b in zip(phases, [.2, .4, .8]))
