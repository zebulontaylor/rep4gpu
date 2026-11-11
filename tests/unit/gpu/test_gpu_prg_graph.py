import pytest
import torch

from rep4gpu.gpu.crypto.gpu_prg import DomainKeys, generate
from rep4gpu.gpu.crypto.gpu_prg_graph import generate_rows_into

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("count", [1, 17, 1025])
def test_graph_replay_reads_fresh_nonces(bits, count):
    key = DomainKeys(bytes(range(32))).key(b"graph-replay")
    keys = (key, None, key, key, None, key)
    nonces = torch.arange(6, device="cuda").to(torch.uint64)
    output = torch.empty((6, count), device="cuda", dtype=torch.int64)
    # Compile and warm up before capturing allocations and kernel launches.
    generate_rows_into(keys, nonces, output, bits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        generate_rows_into(keys, nonces, output, bits)
    previous = None
    for start in (0, 2**64-6):
        draws = [start+i for i in range(6)]
        nonces.copy_(torch.tensor(draws, device="cuda", dtype=torch.uint64))
        graph.replay()
        expected = torch.stack([
            generate(key, nonce, count, bits) if active is not None
            else torch.zeros(count, device="cuda", dtype=torch.int64)
            for active, nonce in zip(keys, draws)
        ])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(output[0], previous)
        previous = output[0].clone()
