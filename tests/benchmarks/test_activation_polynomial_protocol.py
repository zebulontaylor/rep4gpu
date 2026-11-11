"""Check the bounded GPU benchmark's integer schedule and rounding oracle."""
import pytest
import torch

from experiments.benchmarks.activations.activation_approximation_quality import fit
from experiments.benchmarks.activations.activation_polynomial_protocol import interval_oracle, polynomial, precision


class Oracle:
    def __init__(self, ceil=False):
        self.ceil=ceil
    def public(self,value,like):
        return torch.full_like(like,value)
    def trunc(self,x,shift):
        return (x+((1 << shift)-1 if self.ceil else 0)) >> shift
    def mul(self,x,y):return x*y
    def mul_public(self,x,y):return x*y
    def add(self,x,y):return x+y


@pytest.mark.parametrize('profile',['uniform','mixed'])
@pytest.mark.parametrize('kind',['gelu','silu'])
@pytest.mark.parametrize('degree',[4,6,8,10])
def test_normalized_schedule_matches_exact_integer_oracle(kind,degree,profile):
    coefficients=fit(kind,degree,profile);f=precision(coefficients)
    x=torch.arange(-65536,65537,dtype=torch.int64)
    lo,hi,maximum=interval_oracle(x,coefficients,f,probabilistic=False)
    assert torch.equal(polynomial(Oracle(),x,coefficients,f),lo)
    assert torch.equal(lo,hi)
    assert maximum < 1 << 30


@pytest.mark.parametrize('profile',['uniform','mixed'])
@pytest.mark.parametrize('kind',['gelu','silu'])
@pytest.mark.parametrize('degree',[4,6,8,10])
def test_probabilistic_interval_covers_floor_and_ceil(kind,degree,profile):
    coefficients=fit(kind,degree,profile);f=precision(coefficients)
    x=torch.arange(-65536,65537,19,dtype=torch.int64)
    lo,hi,maximum=interval_oracle(x,coefficients,f)
    for ceil in (False,True):
        actual=polynomial(Oracle(ceil),x,coefficients,f)
        assert bool(((actual>=lo)&(actual<=hi)).all())
    assert maximum < 1 << 30
