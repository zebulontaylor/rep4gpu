"""All negative reduced Taylor inputs obey the proved denominator bounds."""
import pytest
import torch

from benchmarks.silu_denominator_bound import certificate, grid_case, encoded_negative_exp


@pytest.mark.parametrize('fractional_bits',range(20,25))
def test_coefficient_order_certifies_all_reduced_encodings(fractional_bits):
    proof=certificate(fractional_bits)
    scale=proof['scale']
    assert proof['denominator_bounds']==[scale,2*scale]
    for rounding in ('floor','ceil'):
        output=encoded_negative_exp(torch.tensor([-scale,-scale+1,-1,0]),fractional_bits,rounding)
        assert bool(((output>=0)&(output<=scale)).all())


@pytest.mark.parametrize('domain',[4,16])
@pytest.mark.parametrize('rounding',['floor','ceil'])
def test_q24_grid_matches_independent_recurrence(domain,rounding):
    torch.set_num_threads(1)
    result=grid_case(domain=domain,rounding=rounding)
    assert result['independent_recurrence_matches']
    assert result['denominator_min']>=1<<24
    assert result['denominator_max']<=2<<24
