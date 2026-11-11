"""The rational SiLU program verifies its declared fixed-point schedule."""
import pytest
import torch

from rep4gpu.proofs.silu_rational import (
    PRODUCT_NAMES, rational_silu_program, rational_silu_witnesses, rational_silu_plain,
)
from rep4gpu.proofs.witness_activations import InvalidWitness, WitnessSession
from tests.unit.proofs.test_silu_output_witness import run_four, secret


def test_product_witness_builder_matches_plain():
    scale=1 << 24
    x=torch.round(torch.linspace(-16,16,1001,dtype=torch.float64)*scale).long()
    actual=rational_silu_witnesses({'x':x})
    expected=rational_silu_plain(x)
    assert torch.equal(actual['output'],expected)
    assert set(actual)=={'output',*PRODUCT_NAMES}
    program=rational_silu_program()
    for name in PRODUCT_NAMES:
        lower,upper=program.witness_bounds[name]
        assert bool(((actual[name] >= lower*scale) & (actual[name] <= upper*scale)).all())


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_product_witness_secure_matches_plain(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA required')
    x=torch.tensor([-16*(1 << 24),-25231595,-1234567,-1,0,1,1234567,25231595,16*(1 << 24)])
    claims=rational_silu_witnesses({'x':x})
    def work(party):
        shared_x=secret(party,x)
        shared={name:secret(party,value,owner=1) for name,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(rational_silu_program(),{'x':shared_x},shared)
        output,=session.finish()
        return party.open(output).cpu()
    assert all(torch.equal(result,claims['output']) for result in run_four(work,device))


@pytest.mark.parametrize('name',PRODUCT_NAMES)
@pytest.mark.parametrize('delta',[1,-(1 << 63)])
def test_product_witness_rejects_each_tampered_product(name,delta):
    x=torch.tensor([0,1 << 24])
    claims=rational_silu_witnesses({'x':x})
    claims[name][0]+=delta
    def work(party):
        shared_x=secret(party,x)
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(rational_silu_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)


@pytest.mark.parametrize('bad',[-(1 << 63),-(16 << 24)-1,(16 << 24)+1,(1 << 63)-1])
def test_product_witness_rejects_invalid_input(bad):
    claims=rational_silu_witnesses({'x':torch.tensor([0])})
    def work(party):
        shared_x=secret(party,[bad])
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(rational_silu_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)


@pytest.mark.parametrize('bad',[-(1 << 63),(1 << 63)-1,-(1 << 24)-1,31 << 24])
def test_product_witness_rejects_output_range_aliases(bad):
    x=torch.tensor([0])
    claims=rational_silu_witnesses({'x':x})
    claims['output'][0]=bad
    # At x=0, B is exactly 2**21. Match the forged output product.
    claims[PRODUCT_NAMES[-1]]=claims['output']*(2 << 21)
    def work(party):
        shared_x=secret(party,x)
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(rational_silu_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)


def test_rational_quality_at_worst_exhaustive_input():
    x=torch.tensor([-16*(1 << 24),-25231595,0,25231595,16*(1 << 24)])
    y=rational_silu_plain(x).double()/(1 << 24)
    reference=torch.nn.functional.silu(x.double()/(1 << 24))
    assert float((y-reference).abs().max()) < 1e-6
