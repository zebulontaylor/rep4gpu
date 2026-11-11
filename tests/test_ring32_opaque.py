"""Exact ring32 reconstruction across the opaque tensor-core accumulator barrier."""
import numpy as np
import pytest
import torch

from rep4gpu.kernels import prepare_weight
from rep4gpu.ring32_opaque import local_products_opaque

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _exact_products(xs, ys):
    # Python integers keep the oracle independent of signed-int64 overflow.
    x0,x1,x2 = [x.numpy().astype(object) for x in xs]
    y0,y1,y2 = [y.numpy().astype(object) for y in ys]
    values = ((x0+x1)@y0+x0@y1, (x1+x2)@y1+x1@y2, x0@y2+x2@y0)
    return [torch.from_numpy(np.asarray(value & 0xffffffff, dtype=np.int64)) for value in values]


def _check(xs, ys, *, tile=(32,128,32), separate=False):
    expected = _exact_products(xs, ys)
    actual = local_products_opaque(*(x.cuda() for x in xs),
        *(prepare_weight(y.cuda(),32,storage='words32') for y in ys),
        tile=tile,separate=separate)
    for got,want in zip(actual,expected):
        assert got.dtype == torch.int64
        torch.testing.assert_close(got.cpu(),want,rtol=0,atol=0)
        assert bool(((got>=0)&(got<=0xffffffff)).all())


@pytest.mark.parametrize('shape', [(1,71,39), (17,35,137), (33,65,129)])
@pytest.mark.parametrize('tile,separate', [((32,128,32),False), ((32,128,64),True)])
def test_full_ring_inputs_and_all_tile_tails(shape,tile,separate):
    m,k,n = shape
    generator = torch.Generator().manual_seed(9981+m+k+n)
    xs = [torch.randint(-2**63,2**63-1,(m,k),generator=generator) for _ in range(3)]
    ys = [torch.randint(-2**63,2**63-1,(k,n),generator=generator) for _ in range(3)]
    xs[0][0,0] = -2**63
    ys[0][0,0] = 2**63-1
    _check(xs,ys,tile=tile,separate=separate)


@pytest.mark.parametrize('tile', [(32,128,32), (32,128,64)])
def test_saturating_accumulator_fusion_regression(tile):
    generator = torch.Generator().manual_seed(3280909)
    xs = [torch.randint(-2**63,2**63-1,(128,71),generator=generator) for _ in range(3)]
    ys = [torch.randint(-2**63,2**63-1,(71,39),generator=generator) for _ in range(3)]
    _check(xs,ys,tile=tile)


def _balanced_word(digits):
    return sum(digit << (8*i) for i,digit in enumerate(digits)) & 0xffffffff


@pytest.mark.parametrize('digits', [(-128,)*4, (127,)*4, (-128,127,-128,127)])
@pytest.mark.parametrize('tile', [(32,128,32), (32,128,64)])
def test_worst_signed_limb_values_wrap_between_zero_based_dots(digits,tile):
    word = _balanced_word(digits)
    # x0+x1 and x1+x2 retain the chosen balanced digits. Repeated blocks
    # force modular output accumulation across the signed-int32 boundary.
    xs = [torch.full((3,257),word,dtype=torch.int64), torch.zeros((3,257),dtype=torch.int64),
          torch.full((3,257),word,dtype=torch.int64)]
    ys = [torch.full((257,7),word,dtype=torch.int64) for _ in range(3)]
    _check(xs,ys,tile=tile)


@pytest.mark.parametrize('shape', [(5,0,7), (0,3,7), (5,3,0)])
def test_zero_inner_dimension_and_empty_outputs(shape):
    m,k,n = shape
    xs = [torch.empty((m,k),dtype=torch.int64) for _ in range(3)]
    ys = [torch.empty((k,n),dtype=torch.int64) for _ in range(3)]
    _check(xs,ys)
