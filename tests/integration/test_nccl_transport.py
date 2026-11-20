"""Exercise sparse P2P routing on CPU (Gloo) for NcclDeviceTransport."""
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist


def _routing_worker(rank, rendezvous):
    from rep4gpu.transports.nccl import NcclDeviceTransport

    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4)
    try:
        adapter = object.__new__(NcclDeviceTransport)
        adapter.device, adapter.group, adapter.party = torch.device('cpu'), dist.group.WORLD, rank
        adapter.rounds = adapter.bytes_sent = adapter.messages_sent = 0
        with patch('torch.cuda.device', lambda _: nullcontext()):
            for sender in range(4):
                receiver = (sender + 1) % 4
                outgoing = {receiver: torch.tensor([sender, receiver], dtype=torch.uint8)} if rank == sender else {}
                sizes = {sender: 2} if rank == receiver else {}
                buffers = {p: torch.empty(n, dtype=torch.uint8) for p, n in sizes.items()}
                assert adapter.exchange_device_into(outgoing, sizes, buffers) is buffers
                if rank == receiver:
                    assert buffers[sender].tolist() == [sender, receiver]
            empty = torch.empty(0, dtype=torch.uint8)
            received = adapter.exchange_device({(rank + 1) % 4: empty}, {(rank - 1) % 4: 0})
            assert received[(rank - 1) % 4].numel() == 0
            assert (adapter.rounds, adapter.bytes_sent, adapter.messages_sent) == (5, 2, 2)
            with pytest.raises(RuntimeError, match='device messages'):
                adapter.exchange({})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required for CPU routing test')
def test_sparse_device_round_schedule(tmp_path):
    torch.multiprocessing.spawn(_routing_worker, args=((tmp_path / 'store').as_uri(),), nprocs=4)
