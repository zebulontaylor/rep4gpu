"""Classifier-shaped tiles retain exact full-ring local products."""
import pytest
import torch

from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(128, 512, 32), (1024, 512, 32),
                                  (4097, 512, 17), (4097, 512, 64),
                                  (1024, 513, 32), (17, 512, 33)])
def test_classifier_shapes_against_full_ring_cpu(shape):
    m, k, n = shape
    generator = torch.Generator().manual_seed(9032 + m + k + n)
    xs = [torch.empty((m, k), dtype=torch.int64).random_(generator=generator) for _ in range(3)]
    ys = [torch.empty((k, n), dtype=torch.int64).random_(generator=generator) for _ in range(3)]
    for value in (*xs, *ys):
        value[::2] = -value[::2] - 1
    expected = [((xs[0] + xs[1]) @ ys[0] + xs[0] @ ys[1]) & 0xffffffff,
                ((xs[1] + xs[2]) @ ys[1] + xs[1] @ ys[2]) & 0xffffffff,
                (xs[0] @ ys[2] + xs[2] @ ys[0]) & 0xffffffff]
    prepared = [prepare_weight(y.cuda(), bits=32, storage='words32') for y in ys]
    actual = local_products_tuned(*(x.cuda() for x in xs), *prepared, bits=32)
    for value, reference in zip(actual, expected):
        assert torch.equal(value.cpu(), reference)
