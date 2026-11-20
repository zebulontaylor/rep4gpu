"""Exercise GPU mailboxes across four independent CUDA contexts on one device."""
import multiprocessing
import shutil
import ssl
import sys
import time
import traceback

import pytest
import torch

from examples._mtls import (
    MeteredSocketTransport as MeteredTransport,
    mtls_certificates as certificates,
    reserve_addresses,
)


class ControlOnlyMesh(MeteredTransport):
    corrupt_uuid = False

    def exchange(self, outgoing):
        # Cap control frames independently of the GPU payload size.
        assert all(len(frame) <= 4096 for frame in outgoing.values())
        if self.corrupt_uuid:
            outgoing = {peer: bytes([frame[0] ^ 1]) + frame[1:] for peer, frame in outgoing.items()}
        return super().exchange(outgoing)


def _ipc_worker(rank, addresses, ca, identity, results):
    from rep4gpu.transports.cuda_ipc import CudaIpcTransport

    try:
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(identity['certificate'], identity['key'])
        server.load_verify_locations(ca)
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
        client.load_cert_chain(identity['certificate'], identity['key'])
        control = ControlOnlyMesh(rank, [tuple(address) for address in addresses],
                                  timeout=30, server_context=server,
                                  client_context=client,
                                  peer_hostnames={p: f'party{p}' for p in range(4)})
        transport = CudaIpcTransport(control, device='cuda')
        with pytest.raises(RuntimeError, match='device messages'):
            transport.exchange({})
        sender, receiver = (rank - 1) % 4, (rank + 1) % 4
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        held = []
        sizes = (8192, 131072, (2 << 20) + 13, 0, 32771, 65536, 131072)
        for iteration, count in enumerate(sizes):
            with torch.cuda.stream(streams[iteration % 2]):
                value = rank * 32 + iteration
                source = torch.full((count,), value, dtype=torch.uint8, device='cuda')
                target = torch.empty_like(source)
                result = transport.exchange_device_into({receiver: source}, {sender: count},
                                                        {sender: target})
                assert result[sender] is target
                # Reuse the caller's source immediately and retain the received
                # data through subsequent exchanges on alternating streams.
                source.fill_(255)
                held.append((target, sender * 32 + iteration))
        torch.cuda.synchronize()
        for target, value in held:
            assert torch.all(target == value).item()
        assert transport.device_payload_bytes_sent == sum(sizes)
        assert transport.host_payload_bytes_sent == 0
        assert 0 < transport.control_bytes_sent == control.bytes_sent < sum(sizes)
        with pytest.raises(ValueError, match='length differs from the public schedule'):
            transport.exchange_device({receiver: source}, {sender: source.numel() + 1})
        control.corrupt_uuid = True
        with pytest.raises(ValueError, match='same physical GPU'):
            transport.exchange_device({receiver: source}, {sender: source.numel()})
        transport.close()
        results.put((rank, None))
    except BaseException:
        results.put((rank, traceback.format_exc()))


@pytest.mark.skipif(sys.platform != 'linux' or not torch.cuda.is_available()
                    or shutil.which('openssl') is None, reason='Linux, CUDA and OpenSSL required')
def test_ipc_payloads_and_reused_mailboxes_on_alternating_streams(tmp_path):
    addresses = reserve_addresses()
    ca, identities = certificates(tmp_path)
    ctx = multiprocessing.get_context('spawn')
    results = ctx.Queue()
    processes = [ctx.Process(target=_ipc_worker, args=(i, addresses, ca, identities[i], results))
                 for i in range(4)]
    for process in processes:
        process.start()
    try:
        deadline = time.monotonic() + 90
        for _ in processes:
            rank, error = results.get(timeout=max(0, deadline - time.monotonic()))
            assert error is None, f'CUDA IPC party {rank} failed\n{error}'
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


def test_ipc_handles_require_authenticated_control_mesh():
    from rep4gpu.transports.cuda_ipc import CudaIpcTransport
    from rep4gpu.transports.byte import IdealNetwork

    with pytest.raises(ValueError, match='authenticated control mesh'):
        CudaIpcTransport(IdealNetwork().endpoint(0))
