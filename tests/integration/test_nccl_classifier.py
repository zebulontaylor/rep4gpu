"""Exercise sparse P2P routing on CPU and the full torchrun path on four GPUs."""
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist


def _routing_worker(rank, rendezvous):
    from rep4gpu.protocol.nccl_transport import NcclDeviceTransport

    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4)
    try:
        # Run the NCCL adapter's P2P scheduling against a real CPU process group.
        # Construction and the CUDA stream wait are covered by the four-GPU test.
        adapter = object.__new__(NcclDeviceTransport)
        adapter.device, adapter.group, adapter.party = torch.device('cpu'), dist.group.WORLD, rank
        adapter.rounds = adapter.bytes_sent = adapter.messages_sent = 0
        with patch('torch.cuda.device', lambda _: nullcontext()):
            for sender in range(4):
                receiver = (sender+1) % 4
                outgoing = {receiver: torch.tensor([sender, receiver], dtype=torch.uint8)} if rank == sender else {}
                sizes = {sender: 2} if rank == receiver else {}
                buffers = {p: torch.empty(n, dtype=torch.uint8) for p, n in sizes.items()}
                assert adapter.exchange_device_into(outgoing, sizes, buffers) is buffers
                if rank == receiver:
                    assert buffers[sender].tolist() == [sender, receiver]
            empty = torch.empty(0, dtype=torch.uint8)
            received = adapter.exchange_device({(rank+1) % 4: empty}, {(rank-1) % 4: 0})
            assert received[(rank-1) % 4].numel() == 0
            assert (adapter.rounds, adapter.bytes_sent, adapter.messages_sent) == (5, 2, 2)
            with pytest.raises(RuntimeError, match='device messages'):
                adapter.exchange({})
    finally:
        dist.destroy_process_group()


def test_worker_checks_only_local_owner_arrays(tmp_path):
    from experiments.benchmarks.inference.nccl_text_classifier import worker_config

    manifest = dict(architecture='linear', fractional_bits=0, weight_shapes={'weights': [3, 1]},
                    input_shape=[4, 3], addresses=[['localhost', port] for port in range(20000, 20004)])
    for rank in (2, 3):
        assert 'input' not in worker_config(manifest, tmp_path, rank, 0)
    np.savez(tmp_path/'values.npz', x=np.ones((4, 3), dtype=np.int64))
    assert worker_config(manifest, tmp_path, 0, 0)['input'] == str(tmp_path/'values.npz')
    np.savez(tmp_path/'values.npz', weights=np.ones((3, 1), dtype=np.int64))
    assert worker_config(manifest, tmp_path, 1, 0)['input'] == str(tmp_path/'values.npz')
    np.savez(tmp_path/'values.npz', weights=np.ones((1, 3), dtype=np.int64))
    with pytest.raises(ValueError, match='shape'):
        worker_config(manifest, tmp_path, 1, 0)


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo required for CPU routing test')
def test_sparse_device_round_schedule(tmp_path):
    torch.multiprocessing.spawn(_routing_worker, args=((tmp_path/'store').as_uri(),), nprocs=4)


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 4,
                    reason='four separate NCCL GPUs required')
def test_four_gpu_classifier(tmp_path):
    from experiments.benchmarks.inference.socket_inference import certificates, reserve_addresses

    ca, identities = certificates(tmp_path)
    for rank, identity in enumerate(identities):
        directory = tmp_path/f'party{rank}'
        directory.mkdir()
        shutil.copyfile(ca, directory/'ca.pem')
        shutil.copyfile(identity['certificate'], directory/'party.pem')
        shutil.copyfile(identity['key'], directory/'party.key')
    x = np.array([[2, 3], [3, 2], [0, 0]], dtype=np.int64)
    w1 = np.eye(2, dtype=np.int64)
    w2 = np.array([[1], [-1]], dtype=np.int64)
    np.savez(tmp_path/'party0/values.npz', x=x)
    np.savez(tmp_path/'party1/values.npz', w1=w1, w2=w2, b1=np.zeros(2, dtype=np.int64), b2=np.array([0]))
    manifest = dict(architecture='square_unscaled', fractional_bits=0, weight_shapes={'w1': [2, 2], 'b1': [2],
                    'w2': [2, 1], 'b2': [1]}, input_shape=[3, 2], addresses=reserve_addresses(), repeats=2, warmup=1)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4', '--module',
               'experiments.benchmarks.inference.nccl_text_classifier', '--manifest', str(tmp_path/'manifest.json'),
               '--party-dir', str(tmp_path/'party{rank}'), '--output-dir', str(tmp_path/'results')]
    subprocess.run(command, check=True, timeout=300, cwd=Path(__file__).resolve().parents[2])
    for rank in range(4):
        result = json.loads((tmp_path/f'results/party{rank}.json').read_text())
        assert result['passed'] and result['predictions'] == [0, 1, 1]
        assert all(m['host_payload_bytes'] == 0 and m['socket_application_bytes'] == 0 for m in result['measurements'])
