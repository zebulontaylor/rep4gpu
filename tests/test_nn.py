"""The Sequential frontend keeps model ownership and fixed-point semantics."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.nn import sequential_spec, share_sequential
from rep4gpu.runtime import Party, Share, make_seeds
from rep4gpu.transport import IdealNetwork


def test_private_sequential_exact_outputs_and_reuse():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    module = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.ReLU(),
                                 torch.nn.Identity(), torch.nn.Linear(4,2,bias=False))
    spec = sequential_spec(module)
    assert set(spec['layers'][0]) == {'type','in_features','out_features','bias'}
    scale, shift = 4096, 12
    x = torch.tensor([[1000,-2000,3000],[-4000,5000,6000]])
    w1 = torch.round(module[0].weight.detach().T.double()*scale).long()
    b1 = torch.round(module[0].bias.detach().double()*scale).long()
    w2 = torch.round(module[3].weight.detach().T.double()*scale).long()
    expected = (((x@w1 + b1*scale) >> shift).clamp_min(0) @ w2) >> shift
    network, seeds = IdealNetwork(), make_seeds()
    parties = [Party(p, network.endpoint(p), {i:s for i,s in seeds.items() if i != p}) for p in range(4)]
    def work(p):
        try:
            original_input = p.input
            calls=[]
            def tracked_input(owner,value=None,shape=None,binary=False):
                calls.append((owner,value is not None,shape))
                return original_input(owner,value,shape,binary)
            p.input=tracked_input
            model = share_sequential(p,spec,module if p.party == 1 else None)
            assert len(calls)==3
            assert all(owner==1 and present==(p.party==1) for owner,present,_ in calls)
            value = p.input(0,x if p.party == 0 else None,x.shape)
            before = p.transport.round
            output = model(value)
            assert isinstance(output,Share)
            assert p.transport.round > before
            # Public architecture construction and forward retain no plaintext model.
            assert not any(isinstance(layer.weight,torch.Tensor) for layer in model.layers)
            first = p.open(output)
            second = p.open(model(value))
            return first,second
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,parties))
    assert all(torch.equal(a,expected) and torch.equal(b,expected) for a,b in outputs)


@pytest.mark.parametrize('layer',[torch.nn.Dropout(),torch.nn.Softmax(-1),torch.nn.Sequential()])
def test_unsupported_modules(layer):
    with pytest.raises(TypeError,match='unsupported'):
        sequential_spec(torch.nn.Sequential(layer))


def test_validation_precedes_collective():
    class NoCollective:
        bits,device,party = 32,'cpu',1
        def input(self,*args,**kwargs):
            raise AssertionError('validation must precede collective')
    module = torch.nn.Sequential(torch.nn.Linear(2,3))
    spec = sequential_spec(module)
    with pytest.raises(ValueError,match='does not match'):
        share_sequential(NoCollective(),spec,torch.nn.Sequential(torch.nn.Linear(2,4)))
    NoCollective.party = 0
    with pytest.raises(ValueError,match='only the model owner'):
        share_sequential(NoCollective(),spec,module)


@pytest.mark.parametrize('activation,tolerance',[(torch.nn.GELU,.02),(torch.nn.SiLU,.08)])
def test_private_smooth_activation_frontend(activation,tolerance):
    module=torch.nn.Sequential(activation())
    spec=sequential_spec(module)
    inputs=torch.tensor([-5.,-1.,0.,1.,5.])
    encoded=(inputs*4096).long()
    network,seeds=IdealNetwork(),make_seeds()
    def work(party):
        p=Party(party,network.endpoint(party),{i:s for i,s in seeds.items() if i!=party})
        try:
            shared=share_sequential(p,spec,module if party==1 else None)
            value=p.input(0,encoded if party==0 else None,encoded.shape)
            return p.open(shared(value)).double()/4096
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs=list(pool.map(work,range(4)))
    reference=module(inputs)
    assert all(torch.equal(outputs[0],result) for result in outputs)
    assert (outputs[0]-reference).abs().max() < tolerance
