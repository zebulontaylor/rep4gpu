import concurrent.futures
import pytest
from rep4gpu.protocol.setup import SetupError, correlated_seeds
from rep4gpu.protocol.transport import IdealNetwork


def _run_setup(tamper=None):
    network = IdealNetwork()
    captured = [[] for _ in range(4)]
    def work(party):
        transport = network.endpoint(party)
        exchange = transport.exchange
        def record(outgoing):
            outgoing = dict(outgoing)
            if tamper:
                outgoing = tamper(party, transport.round, outgoing)
            captured[party].append(dict(outgoing))
            return exchange(outgoing)
        transport.exchange = record
        try:
            return correlated_seeds(party, transport, context=b'unit-test',
                                    _allow_insecure_test_transport=True)
        except SetupError as error:
            return error
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    return results, captured


def test_seed_ownership_and_agreement():
    results, captured = _run_setup()
    for party, seeds in enumerate(results):
        assert set(seeds) == set(range(4)) - {party}
        assert all(len(seed) == 32 for seed in seeds.values())
        for round_index in (0, 1, 3, 4):
            for receiver, payload in captured[party][round_index].items():
                # Each private record names an exclusion different from both
                # endpoints, so excluded parties receive no seed transcript.
                assert set(payload[4::33]) == set(range(4)) - {party, receiver}
    for excluded in range(4):
        assert len({results[p][excluded] for p in range(4) if p != excluded}) == 1
    other, _ = _run_setup()
    assert all(results[0][e] != other[0][e] for e in results[0])


def test_setup_requires_authenticated_transport():
    with pytest.raises(ValueError, match='authenticated confidential'):
        correlated_seeds(0, IdealNetwork().endpoint(0))


@pytest.mark.parametrize('fault_round', [0, 3])
def test_equivocation_and_commitment_mismatch_abort(fault_round):
    def tamper(party, round_index, outgoing):
        if party == 0 and round_index == fault_round:
            payload = bytearray(outgoing[1])
            payload[5] ^= 1
            outgoing[1] = bytes(payload)
        return outgoing
    results, _ = _run_setup(tamper)
    assert all(isinstance(result, SetupError) for result in results)


def test_malformed_commitment_aborts_before_reveal():
    def tamper(party, round_index, outgoing):
        if party == 0 and round_index == 0:
            outgoing[1] = b'wrong'
        return outgoing
    results, captured = _run_setup(tamper)
    assert all(isinstance(result, SetupError) for result in results)
    assert all(len(rounds) == 3 for rounds in captured)


def _tls_setup_worker(party, addresses, certificate_files, queue):
    import ssl
    import torch
    from rep4gpu.protocol.runtime import Party
    from rep4gpu.protocol.transport import SocketTransport
    ca, identities = certificate_files
    certificate, key = identities[party]
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.verify_mode = ssl.CERT_REQUIRED
    server.load_cert_chain(certificate, key)
    server.load_verify_locations(ca)
    client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
    client.load_cert_chain(certificate, key)
    transport = SocketTransport(party, addresses, timeout=20, server_context=server,
                                client_context=client,
                                peer_hostnames={p: f'party{p}' for p in range(4)})
    try:
        seeds = correlated_seeds(party, transport, context=b'TLS-setup-test')
        runtime = Party(party, transport, seeds)
        value = torch.tensor([17, -91, -(1 << 63)], dtype=torch.int64)
        shared = runtime.input(0, value if party == 0 else None, value.shape)
        output = runtime.open(runtime.mul(shared, shared))
        queue.put((party, seeds, torch.equal(output, value * value)))
    finally:
        transport.close()


def test_tls_setup_and_online_product(tmp_path):
    import multiprocessing
    import shutil
    import socket
    import subprocess
    if shutil.which('openssl') is None:
        pytest.skip('openssl required for TLS certificate fixtures')
    def openssl(*args):
        subprocess.run(['openssl', *map(str, args)], check=True, capture_output=True)
    ca = tmp_path / 'ca.pem'
    ca_key = tmp_path / 'ca.key'
    openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', ca_key,
            '-out', ca, '-days', '1', '-subj', '/CN=Rep4 test CA')
    identities = []
    for party in range(4):
        key = tmp_path / f'party{party}.key'
        csr = tmp_path / f'party{party}.csr'
        certificate = tmp_path / f'party{party}.pem'
        extensions = tmp_path / f'party{party}.ext'
        extensions.write_text(f'subjectAltName=DNS:party{party}\nextendedKeyUsage=serverAuth,clientAuth\n')
        openssl('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', csr,
                '-subj', f'/CN=party{party}')
        openssl('x509', '-req', '-in', csr, '-CA', ca, '-CAkey', ca_key,
                '-CAcreateserial', '-out', certificate, '-days', '1', '-extfile', extensions)
        identities.append((str(certificate), str(key)))
    reservations = []
    addresses = []
    for _ in range(4):
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        reservations.append(sock)
        addresses.append(sock.getsockname())
    for sock in reservations:
        sock.close()
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    processes = [context.Process(target=_tls_setup_worker,
                 args=(p, addresses, (str(ca), identities), queue)) for p in range(4)]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=40) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        assert all(ok for _, _, ok in results)
        seeds = {party: seed for party, seed, _ in results}
        for excluded in range(4):
            assert len({seeds[p][excluded] for p in range(4) if p != excluded}) == 1
            assert excluded not in seeds[excluded]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
