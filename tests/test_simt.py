"""Full-ring correctness of the shared-memory SIMT CUDA candidate."""
import numpy as np
import pytest
import torch

from rep4gpu.simt_kernels import local_products, ring_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def random_ring(shape, seed):
    return np.random.default_rng(seed).bit_generator.random_raw(np.prod(shape)).view(np.int64).reshape(shape)


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("shape", [(1, 71, 39), (17, 67, 65), (128, 128, 128), (3, 0, 7), (0, 3, 4)])
def test_matmul(bits, shape):
    m, k, n = shape
    a, b = random_ring((m, k), 11), random_ring((k, n), 19)
    expected = a @ b
    if bits == 32:
        expected &= 0xffffffff
    actual = ring_matmul(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(), bits)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("bits", [32, 64])
def test_fused_local_products(bits):
    xs = [torch.from_numpy(random_ring((33, 71), i)).cuda() for i in range(3)]
    ys = [torch.from_numpy(random_ring((71, 69), i+3)).cuda() for i in range(3)]
    xc, yc = [x.cpu() for x in xs], [y.cpu() for y in ys]
    expected = [(xc[0]+xc[1])@yc[0]+xc[0]@yc[1],
                (xc[1]+xc[2])@yc[1]+xc[1]@yc[2],
                xc[0]@yc[2]+xc[2]@yc[0]]
    if bits == 32:
        expected = [x & 0xffffffff for x in expected]
    for actual, reference in zip(local_products(*xs, *ys, bits), expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


def test_cuda_graph_capture():
    a = torch.tensor([[-1, -2**63], [2**63-1, 0]], dtype=torch.int64, device="cuda")
    expected = a.cpu() @ a.cpu()
    ring_matmul(a, a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = ring_matmul(a, a)
    graph.replay()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
