"""Independent masked-rounding math and CPU/device logical transcript checks."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from rep4gpu.runtime import Party,Share,SCHEDULE,make_seeds
from rep4gpu.transport import IdealNetwork,RecordingTransport
from rep4gpu.probabilistic import trunc_pr as cpu_trunc
from rep4gpu.device_probabilistic import trunc_pr


def draw(excluded,counter,shape,bits,residues=False):
    if residues and counter==0 and excluded in (2,3):
        return torch.arange(shape[-1]).expand(shape).clone() if excluded==2 else torch.zeros(shape,dtype=torch.int64)
    array=np.random.default_rng(1000*excluded+counter).bit_generator.random_raw(np.prod(shape)).view(np.int64).reshape(shape)
    value=torch.from_numpy(array.copy())
    return value&0xffffffff if bits==32 else value


def components(x,bits):
    rng=np.random.default_rng(77)
    parts=[torch.from_numpy(rng.bit_generator.random_raw(x.numel()).view(np.int64).reshape(x.shape).copy()) for _ in range(3)]
    parts.append(x-parts[0]-parts[1]-parts[2])
    return [p&0xffffffff for p in parts] if bits==32 else parts


def run_four(x,bits,shift,gpu=False,residues=False,injected=True):
    network=IdealNetwork(timeout=10)
    seeds=make_seeds()
    parts=components(x,bits)
    def work(party):
        try:
            endpoint=network.endpoint(party)
            if gpu:
                from rep4gpu.device_transport import DeviceByteBridge,DeviceRecorder
                from rep4gpu.device_runtime import DeviceParty
                transport=DeviceRecorder(DeviceByteBridge(endpoint,'cuda'))
                runtime=DeviceParty(party,transport,{e:s for e,s in seeds.items() if e!=party},bits=bits,binary_bits=bits,prg_batch=not injected)
            else:
                transport=RecordingTransport(endpoint)
                runtime=Party(party,transport,{e:s for e,s in seeds.items() if e!=party},bits=bits,binary_bits=bits)
            def controlled(excluded,shape,device,binary=False):
                counter=runtime.counters[excluded]
                runtime.counters[excluded]+=1
                return draw(excluded,counter,shape,bits,residues).to(device)
            if injected: runtime._random=controlled
            value=Share(tuple(parts[(party+j+1)%4].to(runtime.device) for j in range(3)))
            result=(trunc_pr if gpu else cpu_trunc)(runtime,value,shift)
            return [t.cpu() for t in result.parts],transport.transcript,runtime.multiplications,dict(runtime.counters)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(work,range(4)))


def reconstruct(results,bits):
    value=sum(results[0][0])+results[1][0][2]
    return ((value&0xffffffff)^0x80000000)-0x80000000 if bits==32 else value


def expected_device_wires(cpu,bits,count):
    """Translate CPU payloads to independently computed BLAKE3 tags."""
    blake3=pytest.importorskip('blake3').blake3
    stages=[([(0,1,2,3),(1,0,3,2)],count),
            ([(0,1,3,2),(2,3,1,0)],2*count),(SCHEDULE,count)]
    expected=[]
    for stage,(schedule,elements) in enumerate(stages):
        size=elements*(bits//8)
        payloads=[{} for _ in range(4)]
        hashes=[{} for _ in range(4)]
        offsets={}
        for sender,backup,receiver,_ in schedule:
            offset=offsets.get((sender,receiver),0)
            data=cpu[sender][1][stage][0][receiver][offset:offset+size]
            assert len(data)==size
            offsets[sender,receiver]=offset+size
            payloads[sender].setdefault(receiver,[]).append(data)
            hashes[backup].setdefault(receiver,[]).append((sender,blake3(data).digest()))
        expected.append([{peer:b''.join(payloads[p].get(peer,[]))+b''.join(tag for _,tag in sorted(hashes[p].get(peer,[])))
                          for peer in set(payloads[p])|set(hashes[p])} for p in range(4)])
    return expected


@pytest.mark.parametrize('bits',[32,64])
def test_cpu_oracle_floor_ceil_and_three_rounds(bits):
    limit=1<<(bits-2)
    x=torch.tensor([-limit+1,-4097,-16,-1,0,1,16,4097,limit-1])
    shift=12
    result=run_four(x,bits,shift)
    mask=draw(2,0,x.shape,bits)+draw(3,0,x.shape,bits)
    expected=(x+(mask&((1<<shift)-1)))>>shift
    assert torch.equal(reconstruct(result,bits),expected)
    assert all(len(row[1])==3 and row[2]==1 for row in result)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,shift',[(32,1),(32,12),(32,30),(64,1),(64,12),(64,62)])
def test_gpu_matches_cpu_math_shares_and_logical_wires(bits,shift):
    limit=1<<(bits-2)
    x=torch.randint(-limit+1,limit,(257,),generator=torch.Generator().manual_seed(80))
    x[:9]=torch.tensor([-limit+1,-4097,-16,-1,0,1,16,4097,limit-1])
    cpu=run_four(x,bits,shift)
    gpu=run_four(x,bits,shift,gpu=True)
    wires=expected_device_wires(cpu,bits,x.numel())
    mask=draw(2,0,x.shape,bits)+draw(3,0,x.shape,bits)
    assert torch.equal(reconstruct(gpu,bits),(x+(mask&((1<<shift)-1)))>>shift)
    for party in range(4):
        assert all(torch.equal(a,b) for a,b in zip(cpu[party][0],gpu[party][0]))
        assert gpu[party][2]==1 and gpu[party][3]==cpu[party][3]
        assert len(gpu[party][1])==3
        for stage,frame in enumerate(gpu[party][1]):
            assert frame.kind=='device' and frame.outgoing==wires[stage][party]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
def test_gpu_exhaustive_low_mask_rounding(bits):
    residues=torch.arange(16)
    x=torch.tensor([-33,-17,-16,-15,-1,0,1,15,16,17,33])[:,None].expand(-1,16)
    actual=reconstruct(run_four(x,bits,4,gpu=True,residues=True),bits)
    assert torch.equal(actual,(x+residues)>>4)
    assert torch.equal((actual>(x>>4)).sum(1),x[:,0]&15)
    assert torch.equal(actual[x[:,0]%16==0],(x>>4)[x[:,0]%16==0])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('fault',['payload','tag','length'])
def test_masked_opening_rejected_before_resharing(fault):
    from rep4gpu.device_runtime import DeviceParty
    raw=bytes(8)
    tag=pytest.importorskip('blake3').blake3(raw).digest()
    incoming={0:bytearray(raw),1:bytearray(tag)}
    incoming[0 if fault!='tag' else 1][0]^=1
    if fault=='length': incoming[0].append(0)
    class BadTransport:
        def exchange_device(self,outgoing,incoming_sizes):
            return {p:torch.tensor(list(v),dtype=torch.uint8,device='cuda') for p,v in incoming.items()}
    seeds=make_seeds()
    rt=DeviceParty(2,BadTransport(),{e:s for e,s in seeds.items() if e!=2})
    rt.joint_input_batch=lambda *a,**kw:pytest.fail('unchecked opening reached resharing')
    with pytest.raises(ValueError,match='BLAKE3 hash mismatch|frame sizes'):
        trunc_pr(rt,rt.public(torch.tensor([0])),12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
def test_identity_empty_and_invalid_parameters(bits):
    from rep4gpu.device_runtime import DeviceParty
    class NoTransport:
        def exchange_device(self,*a): pytest.fail('identity/validation must not communicate')
    seeds=make_seeds()
    rt=DeviceParty(0,NoTransport(),{e:s for e,s in seeds.items() if e!=0},bits=bits,binary_bits=bits)
    x=rt.public(torch.tensor([-1,0,1]))
    assert trunc_pr(rt,x,0) is x
    for shape in [(0,),(2,0,7)]:
        empty=rt.public(torch.empty(shape,dtype=torch.int64))
        assert trunc_pr(rt,empty,12) is empty
    assert all(counter==0 for counter in rt.counters.values())
    for shift in [-1,bits-1,1.5]:
        with pytest.raises(ValueError): trunc_pr(rt,x,shift)
    with pytest.raises(ValueError,match='arithmetic'):
        trunc_pr(rt,rt.public(0,binary=True),1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
def test_fresh_device_prg_with_batched_product_masks(bits):
    limit=1<<(bits-2)
    x=torch.tensor([-limit+1,-4097,-1,0,1,4097,limit-1])
    result=run_four(x,bits,12,gpu=True,injected=False)
    actual=reconstruct(result,bits)
    floor=x>>12
    assert torch.all((actual==floor)|(actual==floor+((x&4095)!=0).long()))
    for excluded in range(4):
        assert len({result[p][3][excluded] for p in range(4) if p!=excluded})==1
