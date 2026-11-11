"""Bounded quadratics use one private square and explicit integer rescaling."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.activations import FixedPoint,quadratic_plan
from rep4gpu.runtime import Share,make_seeds
from rep4gpu.transport import IdealNetwork


OFFSETS = {'relu':1/4,'gelu':951/4096,'silu':427/4096}


@dataclass
class Value:
    value: torch.Tensor
    @property
    def device(self):
        return self.value.device


class IntegerOracle:
    def __init__(self,bits):
        self.bits,self.squares,self.shifts=bits,0,[]
    def public(self,value,like):
        return Value(torch.as_tensor(value,dtype=torch.int64).expand_as(like.value))
    def add(self,x,y):
        return Value(x.value+y.value)
    def mul_public(self,x,value):
        return Value(x.value*value)
    def mul(self,x,y):
        self.squares+=1
        return Value(x.value*y.value)
    def matmul(self,x,y):
        return Value(x.value@y.value)
    def trunc(self,x,shift):
        assert bool(((x.value>-(1<<(self.bits-2))) & (x.value<(1<<(self.bits-2)))).all())
        self.shifts.append(shift)
        return Value(x.value>>shift)


@pytest.mark.parametrize('kind',OFFSETS)
@pytest.mark.parametrize('bits,f,prescale,shift',[(32,12,0,15),(32,14,2,13),(64,14,0,17)])
def test_floor_quadratic_matches_integer_formula(kind,bits,f,prescale,shift):
    rt=IntegerOracle(bits)
    fp=FixedPoint(rt,f)
    x=fp.encode(torch.linspace(-4,4,1031,dtype=torch.float64))
    actual=fp.bounded_quadratic(Value(x),kind=kind).value
    u=x>>prescale
    expected=(u*u+(u<<(f+2-prescale))+(round(OFFSETS[kind]*(1<<f))<<shift))>>shift
    assert torch.equal(actual,expected)
    assert rt.squares==1
    assert rt.shifts==([prescale] if prescale else [])+[shift]
    plan=fp.quadratic_plan(kind)
    assert (plan.prescale,plan.shift)==(prescale,shift)
    assert plan.numerator_abs_bound<1<<(bits-2)
    assert torch.equal(plan.encoded_numerator(u),u*u+plan.linear_coeff*u+plan.constant_numerator)


@pytest.mark.parametrize('kind',OFFSETS)
def test_matmul_quadratic_combines_input_rescaling(kind):
    rt=IntegerOracle(32)
    fp=FixedPoint(rt,14)
    x=fp.encode([[.25,-.5],[.375,.125]])
    weight=fp.encode([[1.,-.25],[.125,.5]])
    bias=fp.encode([.0625,-.125])
    actual=fp.matmul_quadratic(Value(x),Value(weight),Value(bias),kind=kind).value
    raw=x@weight+bias*(1<<14)
    u=raw>>16
    expected=(u*u+(u<<14)+(round(OFFSETS[kind]*(1<<14))<<13))>>13
    assert torch.equal(actual,expected)
    assert rt.squares==1
    assert rt.shifts==[16,13]


@pytest.mark.parametrize('kind,error',[('relu',.2506),('gelu',.233),('silu',.177)])
def test_quadratic_error_on_declared_grid(kind,error):
    rt=IntegerOracle(32)
    fp=FixedPoint(rt,14)
    x=torch.linspace(-4,4,16001,dtype=torch.float64)
    actual=fp.decode(fp.bounded_quadratic(Value(fp.encode(x)),kind=kind).value)
    reference={'relu':torch.relu,'gelu':torch.nn.functional.gelu,'silu':torch.nn.functional.silu}[kind](x)
    assert float((actual-reference).abs().max())<error


@pytest.mark.parametrize('kwargs',[{'kind':'other'},{'bound':0},{'bound':4.1},
    {'bound':float('nan')},{'fractional_bits':1},{'fractional_bits':30},{'bits':16}])
def test_quadratic_rejects_invalid_public_plan(kwargs):
    with pytest.raises(ValueError):
        quadratic_plan(**kwargs)


def test_plaintext_numerator_helper_checks_declared_range():
    plan=quadratic_plan()
    with pytest.raises(ValueError,match='exceeds'):
        plan.encoded_numerator(torch.tensor([plan.normalized_bound+1]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('kind',OFFSETS)
@pytest.mark.parametrize('rounding',['floor','probabilistic'])
def test_device_quadratic_two_fresh_calls_match_exact_integer_oracle(bits,kind,rounding,fused=False,profile='uniform'):
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.device_transport import DeviceIdealEndpoint
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    f=14
    scale=1<<f
    x=torch.tensor([-4*scale,-scale-1,-1,0,1,scale+1,4*scale])
    if fused:
        x=torch.tensor([-scale,-scale-1,-1,0,1,scale+1,scale]).reshape(-1,1)
    generator=torch.Generator().manual_seed(511)
    parts=[torch.randint(0,1<<30,x.shape,generator=generator) for _ in range(3)]
    parts.append(x-sum(parts))
    if bits==32:
        parts=[part&0xffffffff for part in parts]
    def work(party):
        try:
            rt=DeviceInferenceParty(party,DeviceIdealEndpoint(network.endpoint(party)),
                {e:s for e,s in seeds.items() if e!=party},bits=bits,gpu_open=True,
                rounding=rounding,sparse_products=True)
            masks=[]
            original=rt._random
            def remember(excluded,shape,device,binary=False,**kwargs):
                result=original(excluded,shape,device,binary,**kwargs)
                if rounding=='probabilistic' and party==0 and not binary and tuple(shape)==tuple(x.shape):
                    masks.append((excluded,result.clone()))
                return result
            rt._random=remember
            value=Share(tuple(parts[(party+j+1)%4].to('cuda') for j in range(3)))
            fp=FixedPoint(rt,f)
            outputs=[]
            previous=dict(rt.counters)
            for _ in range(2):
                result=(fp.matmul_quadratic(value,rt.public(torch.tensor([[scale]],device='cuda')),
                        rt.public(torch.tensor([scale//16],device='cuda')),kind=kind,profile=profile)
                        if fused else fp.bounded_quadratic(value,kind=kind,profile=profile))
                outputs.append(rt.open(result).cpu())
                assert all(rt.counters[e]>previous[e] for e in previous)
                previous=dict(rt.counters)
            return outputs,[(e,v.cpu().to(torch.int64)) for e,v in masks]
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs=list(pool.map(work,range(4)))
    plan=quadratic_plan(kind,fractional_bits=f,bits=bits,profile=profile)
    draws=iter(outputs[0][1])
    def rescale(value,shift):
        if not shift:
            return value
        if rounding=='floor':
            return value>>shift
        e2,r2=next(draws)
        e3,r3=next(draws)
        assert (e2,e3)==(2,3)
        return (value+((r2+r3)&((1<<shift)-1)))>>shift
    for call in range(2):
        raw=x*scale+(scale//16)*scale if fused else x
        u=rescale(raw,f+plan.prescale if fused else plan.prescale)
        m=3 if profile=='local' and kind=='gelu' else 1
        q=2 if profile=='local' and kind=='silu' else 3
        offset=0. if profile=='local' else OFFSETS[kind]
        numerator=m*u*u+(u<<(f+q-plan.prescale-1))+(round(offset*scale)<<plan.shift)
        assert bool(((numerator>-(1<<(bits-2))) & (numerator<(1<<(bits-2)))).all())
        expected=rescale(numerator,plan.shift)
        assert all(torch.equal(role[0][call],expected) for role in outputs)
    assert next(draws,None) is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
def test_device_fused_quadratic_matches_exact_integer_oracle(bits):
    test_device_quadratic_two_fresh_calls_match_exact_integer_oracle(bits,'gelu','probabilistic',fused=True)


@pytest.mark.parametrize('kind,m,q,t,d',[('gelu',3,3,3,11),('silu',1,2,2,12)])
def test_local_quadratic_preserves_zero_and_integer_formula(kind,m,q,t,d):
    rt=IntegerOracle(32)
    fp=FixedPoint(rt,14)
    x=fp.encode(torch.linspace(-4,4,1031,dtype=torch.float64))
    actual=fp.bounded_quadratic(Value(x),kind=kind,profile='local').value
    u=x>>t
    expected=(m*u*u+(u<<(14+q-t-1)))>>d
    assert torch.equal(actual,expected)
    plan=fp.quadratic_plan(kind,profile='local')
    assert (plan.prescale,plan.shift,plan.quadratic_coeff,plan.denominator_power2)==(t,d,m,q)
    assert plan.coefficients==(0.,.5,m/(1<<q))
    assert plan.numerator_abs_bound<1<<30
    assert torch.equal(plan.encoded_numerator(u),m*u*u+plan.linear_coeff*u)
    assert rt.squares==1
    assert rt.shifts==[t,d]
    assert int(fp.bounded_quadratic(Value(torch.tensor([0])),kind=kind,profile='local').value)==0


def test_local_profile_has_explicit_scope_and_uniform_default_is_unchanged():
    with pytest.raises(ValueError,match='supports gelu and silu'):
        quadratic_plan('relu',profile='local')
    with pytest.raises(ValueError,match='profile'):
        quadratic_plan('gelu',profile='other')
    for kind in OFFSETS:
        plan=quadratic_plan(kind)
        assert plan.profile=='uniform'
        assert plan.coefficients==(OFFSETS[kind],.5,.125)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('kind',['gelu','silu'])
def test_device_local_quadratic_matches_exact_integer_oracle(bits,kind):
    test_device_quadratic_two_fresh_calls_match_exact_integer_oracle(
        bits,kind,'probabilistic',fused=True,profile='local')
