"""Complementary-support products retain all four nonzero checked terms."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from functools import reduce

from rep4gpu.protocol.runtime import Party, Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


RETAINED = (1, 3, 4, 5)
SPARSE_SCHEDULE = ((2,1,0,3),(3,0,1,2),(0,2,3,1),(1,3,2,0))


def sparse_oracle(rt, x, y):
    """Reshare cross products directly, independently of fused local products."""
    xs = {(rt.party+j+1)%4: value for j,value in enumerate(x.parts)}
    ys = {(rt.party+j+1)%4: value for j,value in enumerate(y.parts)}
    pairs = ((3,0), (2,1), (3,1), (2,0))
    specs = []
    for (sender,backup,receiver,outsider),(i,j) in zip(SPARSE_SCHEDULE,pairs):
        value = ((xs[i]&ys[j]) if x.binary else xs[i]*ys[j]) if rt.party in (sender,backup) else None
        specs.append((sender,backup,receiver,outsider,value))
    result = rt.joint_input_batch(specs,x.shape,x.binary,x.device)
    return rt._share(tuple(reduce(torch.bitwise_xor,(value.parts[j] for value in result))
                           if x.binary else sum(value.parts[j] for value in result) for j in range(3)),x.binary)


def components(bits,count):
    generator = torch.Generator().manual_seed(719)
    x = torch.randint(0,1<<32,(4,count),generator=generator)
    y = torch.randint(0,1<<32,(4,count),generator=generator)
    if bits==64:
        x |= torch.randint(0,1<<32,(4,count),generator=generator)<<32
        y |= torch.randint(0,1<<32,(4,count),generator=generator)<<32
    if count>=5:
        edge = torch.tensor([-1,-(1<<(bits-1)),(1<<(bits-1))-1,0,1])
        if bits==32:
            edge &= 0xffffffff
        x[2,:5],x[3,:5] = edge,edge.roll(1)
        y[0,:5],y[1,:5] = edge.roll(2),edge.roll(3)
    x[:2] = 0
    y[2:] = 0
    return x,y


def share(value,party,device='cpu',binary=False):
    return Share(tuple(value[(party+j+1)%4].to(device) for j in range(3)),binary)


def signed(value,bits):
    return ((value & 0xffffffff)^0x80000000)-0x80000000 if bits==32 else value


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('binary',[False,True])
def test_sparse_oracle_values_and_fresh_masks(bits,binary):
    network,seeds = IdealNetwork(timeout=10),make_seeds()
    x,y = components(bits,31)
    def work(party):
        try:
            rt = Party(party,network.endpoint(party),{e:s for e,s in seeds.items() if e!=party},bits=bits,binary_bits=bits)
            a,b = share(x,party,binary=binary),share(y,party,binary=binary)
            first = sparse_oracle(rt,a,b)
            counters = dict(rt.counters)
            second = sparse_oracle(rt,a,b)
            # Each holder advances every retained mask stream exactly once per
            # occurrence. The excluded party never receives that stream's key.
            for excluded in rt.counters:
                draws = sum(receiver==excluded for _,_,receiver,_ in SPARSE_SCHEDULE)
                assert rt.counters[excluded]-counters[excluded]==draws
            assert any(not torch.equal(a,b) for a,b in zip(first.parts,second.parts))
            return rt.open(first),rt.open(second)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,range(4)))
    expected = ((reduce(torch.bitwise_xor,x.unbind(0)) & reduce(torch.bitwise_xor,y.unbind(0)))
                if binary else signed(x.sum(0)*y.sum(0),bits))
    for first,second in outputs:
        assert torch.equal(first,expected)
        assert torch.equal(second,expected)


class CorruptOneFrame:
    def __init__(self,underlying,target,offset=0):
        self.underlying,self.target,self.offset = underlying,target,offset
    def exchange(self,outgoing):
        outgoing = dict(outgoing)
        if self.target is not None and self.target in outgoing:
            frame = bytearray(outgoing[self.target])
            frame[self.offset] ^= 1
            outgoing[self.target] = bytes(frame)
            self.target = None
        return self.underlying.exchange(outgoing)


@pytest.mark.parametrize('index',RETAINED)
@pytest.mark.parametrize('source',['sender','backup'])
def test_sparse_oracle_detects_each_retained_frame_fault(index,source):
    network,seeds = IdealNetwork(timeout=10),make_seeds()
    x,y = components(32,31)
    sender,backup,receiver,_ = SPARSE_SCHEDULE[RETAINED.index(index)]
    corrupt = sender if source=='sender' else backup
    active = SPARSE_SCHEDULE
    offset = (sum(s==corrupt and r==receiver for s,_,r,_ in active)*31*4
              + sum(b==corrupt and r==receiver and s<sender for s,b,r,_ in active)*32
              if source=='backup' else 0)
    def work(party):
        try:
            transport = CorruptOneFrame(network.endpoint(party),receiver if party==corrupt else None,offset)
            rt = Party(party,transport,{e:s for e,s in seeds.items() if e!=party},bits=32)
            result = sparse_oracle(rt,share(x,party),share(y,party))
            return rt.open(result)
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,range(4)))
    assert all(isinstance(result,BaseException) for result in outputs)
    assert any('hash mismatch' in str(result) for result in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('count',[0,1,257])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('binary',[False,True])
def test_gpu_sparse_matches_independent_oracle(bits,count,reverse,binary):
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    network,seeds = IdealNetwork(timeout=120),make_seeds()
    x,y = components(bits,count)
    if reverse:
        x,y = y,x
    def work(party):
        try:
            rt = DeviceInferenceParty(party,DeviceIdealEndpoint(network.endpoint(party)),
                {e:s for e,s in seeds.items() if e!=party},bits=bits,binary_bits=bits,sparse_products=True,gpu_open=True)
            a,b = share(x,party,'cuda',binary),share(y,party,'cuda',binary)
            first = rt._mul_sparse_pairs(a,b)
            second = rt.mul(a,b)
            return rt.open(first).cpu(),rt.open(second).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,range(4)))
    expected = ((reduce(torch.bitwise_xor,x.unbind(0)) & reduce(torch.bitwise_xor,y.unbind(0)))
                if binary else signed(x.sum(0)*y.sum(0),bits))
    for first,second in outputs:
        assert torch.equal(first,expected)
        assert torch.equal(second,expected)


class CorruptDeviceFrame:
    def __init__(self,underlying,target,offset):
        self.underlying,self.target,self.offset = underlying,target,offset
    def __getattr__(self,name):
        return getattr(self.underlying,name)
    def exchange_device(self,outgoing,incoming_sizes):
        outgoing = dict(outgoing)
        if self.target is not None and self.target in outgoing:
            frame = outgoing[self.target].clone()
            frame[self.offset] ^= 1
            outgoing[self.target] = frame
            self.target = None
        return self.underlying.exchange_device(outgoing,incoming_sizes)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('index',RETAINED)
@pytest.mark.parametrize('source',['sender','backup'])
def test_gpu_sparse_detects_each_retained_frame_fault(index,source):
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    network,seeds = IdealNetwork(timeout=30),make_seeds()
    x,y = components(32,31)
    sender,backup,receiver,_ = SPARSE_SCHEDULE[RETAINED.index(index)]
    corrupt = sender if source=='sender' else backup
    active = SPARSE_SCHEDULE
    offset = (sum(s==corrupt and r==receiver for s,_,r,_ in active)*31*4
              + sum(b==corrupt and r==receiver and s<sender for s,b,r,_ in active)*32
              if source=='backup' else 0)
    def work(party):
        try:
            endpoint = DeviceIdealEndpoint(network.endpoint(party))
            transport = CorruptDeviceFrame(endpoint,receiver if party==corrupt else None,offset)
            rt = DeviceInferenceParty(party,transport,{e:s for e,s in seeds.items() if e!=party},
                bits=32,sparse_products=True,gpu_open=True)
            result = rt._mul_sparse_pairs(share(x,party,'cuda'),share(y,party,'cuda'))
            return rt.open(result)
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work,range(4)))
    assert all(isinstance(result,BaseException) for result in outputs)
    assert any('hash mismatch' in str(result) for result in outputs)
