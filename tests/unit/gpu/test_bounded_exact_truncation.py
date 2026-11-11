"""Quarter-ring exact floor agrees with signed integer division and full floor."""
from concurrent.futures import ThreadPoolExecutor
import random

import pytest
import torch

from rep4gpu.arithmetic.exact_truncation import trunc_bounded
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def run_four(bits, device, work, *, binary_bits=None, sparse=True):
    torch.set_num_threads(1)
    seeds, network = make_seeds(), IdealNetwork(timeout=180)
    def run(role):
        try:
            endpoint=network.endpoint(role)
            keys={i:s for i,s in seeds.items() if i!=role}
            if device=='cpu':
                rt=Party(role,endpoint,keys,bits=bits,binary_bits=binary_bits or bits)
            else:
                from rep4gpu.models.device_inference import DeviceInferenceParty
                from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
                rt=DeviceInferenceParty(role,DeviceIdealEndpoint(endpoint,'cuda:0'),keys,
                    device='cuda:0',bits=bits,binary_bits=binary_bits or bits,
                    optimized=True,gpu_open=True,rounding='floor',sparse_products=sparse)
            return work(rt,endpoint)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run,range(4)))


def values(bits, count):
    bound=1 << (bits-2)
    rng=random.Random(bits*count+517)
    edges=[-bound+1,-bound+2,-(1<<24)-1,-(1<<24),-1,0,1,
           (1<<24)-1,1<<24,(1<<24)+1,bound-2,bound-1]
    edges=[x for x in edges if abs(x)<bound]
    return torch.tensor((edges+[rng.randrange(-bound+1,bound) for _ in range(count)])[:count],dtype=torch.int64)


@pytest.mark.parametrize('bits',[32,64])
def test_bounded_floor_cpu(bits):
    x=values(bits,41)
    shifts=sorted({0,1,14,24,bits-2})
    def work(rt,ep):
        share=rt.input(0,x if rt.party==0 else None,x.shape)
        return [rt.open(trunc_bounded(rt,share,shift)).cpu() for shift in shifts]
    for outputs in run_four(bits,'cpu',work):
        assert all(torch.equal(out,x>>shift) for out,shift in zip(outputs,shifts))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,binary_bits',[(32,32),(64,64),(32,64)])
@pytest.mark.parametrize('count',[1,65,257])
def test_bounded_floor_cuda_full_integer_range(bits,binary_bits,count):
    x=values(bits,count)
    shifts=sorted({0,1,2,4,14,24,bits-2})
    def work(rt,ep):
        share=rt.input(0,x if rt.party==0 else None,x.shape)
        rows=[]
        for shift in shifts:
            bounded=trunc_bounded(rt,share,shift)
            original=rt.trunc(share,shift)
            rows.append((rt.open(bounded).cpu(),rt.open(original).cpu()))
        return rows
    for rows in run_four(bits,'cuda',work,binary_bits=binary_bits):
        for (bounded,full),shift in zip(rows,shifts):
            assert torch.equal(bounded,x>>shift)
            assert torch.equal(bounded,full)


def test_bounded_floor_rejects_unsupported_shift_and_binary():
    def work(rt,ep):
        x=rt.public(torch.tensor([1]))
        with pytest.raises(ValueError,match='bits-2'):
            trunc_bounded(rt,x,rt.bits-1)
        with pytest.raises(ValueError,match='arithmetic'):
            trunc_bounded(rt,rt.public(torch.tensor([1]),binary=True),1)
    run_four(64,'cpu',work)


def measure(bits=64, elements=14336, shift=24):
    x=values(bits,elements)
    def work(rt,ep):
        share=rt.input(0,x if rt.party==0 else None,x.shape)
        result={}
        for name,fn in [('full',lambda:rt.trunc(share,shift)),
                        ('bounded',lambda:trunc_bounded(rt,share,shift))]:
            start,rounds=ep.bytes_sent,ep.round
            output=fn()
            result[name]={'bytes':ep.bytes_sent-start,'rounds':ep.round-rounds}
            assert torch.equal(rt.open(output).cpu(),x>>shift)
        return result
    results=run_four(bits,'cuda',work)
    return {name:{'bytes':sum(r[name]['bytes'] for r in results),'rounds':results[0][name]['rounds']}
            for name in results[0]}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_bounded_floor_communication_reduction():
    result=measure(elements=257)
    assert result['bounded']['bytes']<result['full']['bytes']


def test_headroom_identity_exhaustive_small_ring():
    bits=8
    modulus=1 << bits
    offset=1 << (bits-2)
    for z in range(-offset+1,offset):
        shifted=z+offset
        for u in range(modulus):
            v=(shifted-u)%modulus
            top_u,top_v=u>>(bits-1),v>>(bits-1)
            assert (u+v)//modulus == (top_u | top_v)
            for shift in range(1,bits-1):
                low_mask=(1 << shift)-1
                carry=((u&low_mask)+(v&low_mask))>>shift
                factor=1 << (bits-shift)
                result=(u>>shift)+(v>>shift)-factor*(top_u+top_v)
                result+=carry+factor*(top_u&top_v)-(offset>>shift)
                assert result == z>>shift
