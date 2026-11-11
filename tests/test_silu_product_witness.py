"""Batched raw products preserve the existing direct SiLU integer contract."""
import pytest
import torch

from rep4gpu.silu_product_witness import (
    PRODUCT_NAMES, silu_product_witness_program, silu_product_witnesses,
)
from rep4gpu.witness_activations import InvalidWitness, WitnessSession
from rep4gpu.witness_prover import numerical_witnesses
from test_silu_output_witness import run_four, secret


@pytest.mark.parametrize('fractional_bits',[20,24])
def test_product_witness_builder_matches_default(fractional_bits):
    scale=1 << fractional_bits
    x=torch.round(torch.linspace(-16,16,1001,dtype=torch.float64)*scale).long()
    actual=silu_product_witnesses({'x':x},fractional_bits)
    expected=numerical_witnesses('silu',{'x':x},fractional_bits)
    assert torch.equal(actual['output'],expected['output'])
    assert set(actual)=={'output',*PRODUCT_NAMES}
    program=silu_product_witness_program()
    for name in PRODUCT_NAMES:
        lower,upper=program.witness_bounds[name]
        assert bool(((actual[name] >= lower*scale) & (actual[name] <= upper*scale)).all())


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_product_witness_secure_matches_default(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA required')
    x=torch.tensor([-16*(1 << 24),-1234567,-1,0,1,1234567,16*(1 << 24)])
    claims=silu_product_witnesses({'x':x})
    def work(party):
        shared_x=secret(party,x)
        shared={name:secret(party,value,owner=1) for name,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(silu_product_witness_program(),{'x':shared_x},shared)
        output,=session.finish()
        return party.open(output).cpu()
    assert all(torch.equal(result,claims['output']) for result in run_four(work,device))


@pytest.mark.parametrize('name',PRODUCT_NAMES)
@pytest.mark.parametrize('delta',[1,-(1 << 63)])
def test_product_witness_rejects_each_tampered_product(name,delta):
    x=torch.tensor([0,1 << 24])
    claims=silu_product_witnesses({'x':x})
    claims[name][0]+=delta
    def work(party):
        shared_x=secret(party,x)
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(silu_product_witness_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)


@pytest.mark.parametrize('bad',[-(1 << 63),-(16 << 24)-1,(16 << 24)+1,(1 << 63)-1])
def test_product_witness_rejects_invalid_input(bad):
    claims=silu_product_witnesses({'x':torch.tensor([0])})
    def work(party):
        shared_x=secret(party,[bad])
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(silu_product_witness_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)


@pytest.mark.parametrize('bad',[-(1 << 63),(1 << 63)-1,-(1 << 24)-1,31 << 24])
def test_product_witness_rejects_output_range_aliases(bad):
    x=torch.tensor([0])
    claims=silu_product_witnesses({'x':x})
    claims['output'][0]=bad
    claims[PRODUCT_NAMES[-1]]=claims['output']*(2 << 24)
    def work(party):
        shared_x=secret(party,x)
        shared={key:secret(party,value,owner=1) for key,value in claims.items()}
        session=WitnessSession(party)
        session.evaluate(silu_product_witness_program(),{'x':shared_x},shared)
        with pytest.raises(InvalidWitness): session.finish()
    run_four(work)
