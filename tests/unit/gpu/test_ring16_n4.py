"""The narrow classifier matrix product preserves every ring8 and ring16 bit."""
import pytest
import torch

from rep4gpu.gpu.matmul.kernels import local_products, prepare_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('bits,storage', [(8, torch.int64), (8, torch.uint8),
                                         (16, torch.int64), (16, torch.uint16)])
@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('rows', [17, 129])
def test_n4_full_ring_oracle(bits, storage, prepared, rows):
    generator = torch.Generator().manual_seed(1604 + rows)
    modulus = 1 << bits
    xs = [torch.randint(0, modulus, (rows, 512), generator=generator) for _ in range(3)]
    ys = [torch.randint(0, modulus, (512, 4), generator=generator) for _ in range(3)]
    # Include both canonical extremes and signed wide representatives.
    for value in (*xs, *ys):
        value.flatten()[:4] = torch.tensor([0, modulus//2-1, modulus//2, modulus-1])
        if storage == torch.int64:
            value[::2] -= modulus
    expected = [((xs[0] + xs[1]) @ ys[0] + xs[0] @ ys[1]) & (modulus-1),
                ((xs[1] + xs[2]) @ ys[1] + xs[1] @ ys[2]) & (modulus-1),
                (xs[0] @ ys[2] + xs[2] @ ys[0]) & (modulus-1)]
    gpu_x = [x.cuda().to(storage) for x in xs]
    gpu_y = [y.cuda().to(storage) for y in ys]
    if prepared:
        gpu_y = [prepare_weight(y, bits=bits) for y in gpu_y]
    actual = local_products(*gpu_x, *gpu_y, bits=bits)
    for a, e in zip(actual, expected):
        assert a.dtype == storage
        assert torch.equal(a.cpu().to(torch.int64), e)
