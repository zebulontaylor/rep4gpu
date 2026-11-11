"""The quadratic benchmark encloses every allowed integer rounding outcome."""
from itertools import product

import pytest
import torch

from experiments.benchmarks.activations.activation_common import quadratic_fixture_bounds, quadratic_interval
from rep4gpu.arithmetic.activations import quadratic_plan


def rounded(value, shift):
    return sorted({value >> shift, (value+(1 << shift)-1) >> shift})


@pytest.mark.parametrize('kind', ['relu','gelu','silu'])
def test_fixture_bounds_include_all_rounding_choices(kind):
    values = [torch.tensor([[-17]]), torch.tensor([[31,-31]]), torch.tensor([0,0]),
              torch.tensor([[5,-9],[-7,3]]), torch.tensor([2,-3])]
    proof, low, high = quadratic_fixture_bounds(values,32,4,'probabilistic',kind,4)
    plan = quadratic_plan(kind,4,fractional_bits=4,bits=32)
    raw = (values[0] @ values[1]).reshape(-1).tolist()
    outputs = []
    for normalized in product(*(rounded(x,4+plan.prescale) for x in raw)):
        nums = [u*u+plan.linear_coeff*u+plan.constant_numerator for u in normalized]
        for hidden in product(*(rounded(n,plan.shift) for n in nums)):
            final_raw = (torch.tensor([hidden]) @ values[3]).reshape(-1).tolist()
            for result in product(*(rounded(n,4) for n in final_raw)):
                outputs.append(torch.tensor([result])+values[4])
    actual = torch.stack(outputs)
    assert bool(((actual >= low) & (actual <= high)).all())
    assert torch.equal(actual.amin(0), low)
    assert torch.equal(actual.amax(0), high)
    assert proof['activation_is_approximation']


def test_convex_interval_includes_negative_vertex():
    plan = quadratic_plan('relu',4,fractional_bits=4,bits=32)
    lower, upper = quadratic_interval(torch.tensor([-40]),torch.tensor([-20]),plan)
    reference = torch.arange(-40,-19,dtype=torch.int64)
    expected = reference**2+plan.linear_coeff*reference+plan.constant_numerator
    assert lower.item() == expected.min().item()
    assert upper.item() == expected.max().item()
    assert lower.item() < 0


def test_fixture_rejects_undeclared_range():
    values = [torch.tensor([[100]]),torch.tensor([[100]]),torch.tensor([0]),
              torch.tensor([[1]]),torch.tensor([0])]
    with pytest.raises(ValueError,match='declared quadratic'):
        quadratic_fixture_bounds(values,32,4,'probabilistic','relu',4)


@pytest.mark.parametrize('low,high',[(-20,-5),(-11,-10),(-11,-11),(-10,-10),(-64,-12),(-10,30)])
def test_local_gelu_interval_includes_integer_vertex(low,high):
    plan=quadratic_plan('gelu',4,fractional_bits=4,bits=32,profile='local')
    assert plan.quadratic_coeff==3
    assert plan.linear_coeff==64
    lower,upper=quadratic_interval(torch.tensor([low]),torch.tensor([high]),plan)
    values=torch.arange(low,high+1,dtype=torch.int64)
    expected=3*values*values+64*values
    assert lower.item()==expected.min().item()
    assert upper.item()==expected.max().item()
