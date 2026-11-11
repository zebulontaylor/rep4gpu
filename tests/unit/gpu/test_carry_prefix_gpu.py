"""Check shortened carry prefixes against integer addition."""
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.gpu.bits.carry_prefix_gpu import carry_power_plus_one


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,binary_bits,position',[(32,32,17),(64,64,33),(32,32,9),(32,32,2),(32,64,17)])
@pytest.mark.parametrize('count',[31,32,33,257])
def test_prefix_carry(bits,binary_bits,position,count):
    network=IdealNetwork(timeout=120)
    seeds=make_seeds()
    generator=torch.Generator().manual_seed(318)
    a=torch.randint(0,1<<30,(count,),generator=generator)
    b=torch.randint(0,1<<30,(count,),generator=generator)
    low=(1<<(position-1))-1
    high=1<<(position-1)
    # Exercise the final generate and propagate gates with both lower carries.
    a[:8]=torch.tensor([0,low,1,low,high,high|low,high,high|low])
    b[:8]=torch.tensor([0,1,1,low,high,1,low,high|low])
    def work(party):
        try:
            rt=Party(party,network.endpoint(party),{e:s for e,s in seeds.items() if e!=party},
                     device='cuda',bits=bits,binary_bits=binary_bits)
            left=rt.input(0,a.to('cuda') if party==0 else None,a.shape,binary=True)
            right=rt.input(1,b.to('cuda') if party==1 else None,b.shape,binary=True)
            return rt.open(carry_power_plus_one(rt,left,right,position)).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(work,range(4)))
    mask=(1<<position)-1
    expected=(((a&mask)+(b&mask))>>position)&1
    for result in results:
        assert torch.equal(result,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,binary_bits,shift',[(32,32,14),(32,32,1),(32,32,30),(64,64,14),(64,64,62),(32,64,14)])
@pytest.mark.parametrize('count',[5,257])
def test_posttrunc_sign_endpoints(bits,binary_bits,shift,count):
    from rep4gpu.gpu.bits.bounded_activation import posttrunc_sign
    limit=1<<(bits-shift-2)
    values=torch.tensor([-limit,-1,0,1,limit],dtype=torch.int64).repeat((count+4)//5)[:count]
    network=IdealNetwork(timeout=120)
    seeds=make_seeds()
    def work(party):
        try:
            rt=Party(party,network.endpoint(party),{e:s for e,s in seeds.items() if e!=party},
                     device='cuda',bits=bits,binary_bits=binary_bits)
            value=rt.input(0,values.to('cuda') if party==0 else None,values.shape)
            return rt.open(posttrunc_sign(rt,value,shift)).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(work,range(4)))
    for result in results:
        assert torch.equal(result,(values<0).to(torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,sign_bits',[(32,18),(64,50)])
def test_sign_conversion_exact_transcript(bits,sign_bits):
    from rep4gpu.arithmetic.compact_conversion import weighted_b2a
    from rep4gpu.gpu.bits.compact_conversion_gpu import sign_b2a_gpu
    from rep4gpu.protocol.fused_runtime import FusedParty
    from rep4gpu.protocol.runtime import Share
    from rep4gpu.protocol.transport import RecordingTransport, ReplayTransport
    generator=torch.Generator().manual_seed(432)
    components=[torch.randint(0,1<<60,(4,257),generator=generator) for _ in range(2)]
    carry=torch.randint(0,2,(4,257),generator=generator)
    sign=((components[0]^components[1])>>(sign_bits-1))&1
    sign^=carry
    seeds,network=make_seeds(),IdealNetwork()
    records=[RecordingTransport(network.endpoint(p)) for p in range(4)]
    def share(parts,p,device):
        return Share(tuple(parts[(p+j+1)%4].to(device) for j in range(3)),True)
    def reference(p):
        rt=Party(p,records[p],{e:s for e,s in seeds.items() if e!=p},bits=bits)
        planes=share(sign,p,'cpu')
        planes=Share(tuple(part.unsqueeze(0) for part in planes.parts),True)
        result=weighted_b2a(rt,planes,[1])
        return result,dict(rt.counters)
    with ThreadPoolExecutor(4) as pool:
        expected=list(pool.map(reference,range(4)))
    for p in range(4):
        replay=ReplayTransport(records[p].transcript)
        rt=FusedParty(p,replay,{e:s for e,s in seeds.items() if e!=p},bits=bits,device='cuda')
        result=sign_b2a_gpu(rt,share(components[0],p,'cuda'),
            share(components[1],p,'cuda'),share(carry,p,'cuda'),sign_bits)
        assert all(torch.equal(a.cpu(),b) for a,b in zip(result.parts,expected[p][0].parts))
        assert rt.counters==expected[p][1]
        replay.assert_consumed()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shape',[(),(0,),(2,0),(3,37)])
def test_posttrunc_sign_shapes_mixed_words(shape):
    from rep4gpu.gpu.bits.bounded_activation import posttrunc_sign
    values=torch.zeros(shape,dtype=torch.int64)
    if values.numel():
        values.reshape(-1)[::2]=-1
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    def work(party):
        try:
            rt=Party(party,network.endpoint(party),{e:s for e,s in seeds.items() if e!=party},
                     device='cuda',bits=32,binary_bits=64)
            value=rt.input(0,values.to('cuda') if party==0 else None,shape)
            return rt.open(posttrunc_sign(rt,value,14)).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(work,range(4)))
    for result in results:
        assert result.shape==values.shape
        assert torch.equal(result,(values<0).to(torch.int64))
