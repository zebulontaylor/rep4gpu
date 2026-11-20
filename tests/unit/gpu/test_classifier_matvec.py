import pytest
import torch

from rep4gpu.gpu.matmul.classifier_matvec import local_products_classifier
from rep4gpu.gpu.matmul.kernels import local_products


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('batch,width', [(0, 513), (7, 0), (1, 1), (17, 71),
                                        (128, 513), (1024, 513), (3, 2049)])
def test_full_width_shares_against_cpu_and_generic(batch, width):
    rng = torch.Generator().manual_seed(3901 + batch + width)
    xs = [torch.randint(-2**63, 2**63-1, (batch, width), generator=rng,
                        dtype=torch.int64) for _ in range(3)]
    ys = [torch.randint(-2**63, 2**63-1, (width, 1), generator=rng,
                        dtype=torch.int64) for _ in range(3)]
    reference = [((xs[0] + xs[1]) @ ys[0] + xs[0] @ ys[1]) & 0xffffffff,
                 ((xs[1] + xs[2]) @ ys[1] + xs[1] @ ys[2]) & 0xffffffff,
                 (xs[0] @ ys[2] + xs[2] @ ys[0]) & 0xffffffff]
    inputs = [value.cuda() for value in (*xs, *ys)]
    result = local_products_classifier(*inputs)
    generic = local_products(*inputs, bits=32)
    for got, want, previous in zip(result, reference, generic):
        torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)
        torch.testing.assert_close(got, previous, atol=0, rtol=0)


def test_noncontiguous_and_extreme_shares():
    xs = [torch.full((31, 14), value, dtype=torch.int64, device='cuda')[:, ::2]
          for value in [-(2**63), 2**63-1, 0xffffffff]]
    ys = [torch.full((7, 2), value, dtype=torch.int64, device='cuda')[:, ::2]
          for value in [2**63-1, 0xffffffff, -(2**63)]]
    for got, want in zip(local_products_classifier(*xs, *ys),
                         local_products(*xs, *ys, bits=32)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_rejects_invalid_contracts():
    x = torch.zeros((2, 7), dtype=torch.int64, device='cuda')
    y = torch.zeros((7, 1), dtype=torch.int64, device='cuda')
    with pytest.raises(ValueError, match='ring32'):
        local_products_classifier(x, x, x, y, y, y, bits=64)
    with pytest.raises(ValueError, match='weight shares'):
        local_products_classifier(x, x, x, y, y, x)
    with pytest.raises(ValueError, match='input share shapes'):
        local_products_classifier(x, x[:1], x, y, y, y)
    with pytest.raises(ValueError, match='int64 CUDA'):
        local_products_classifier(x, x, x, y, y, y.float())
