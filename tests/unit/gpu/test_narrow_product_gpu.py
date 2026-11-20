"""Independent 16-bit sharing oracle and coefficient-annihilation checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import struct

import numpy as np
import pytest
import torch

from rep4gpu.protocol.runtime import Share,make_seeds
from rep4gpu.transports.byte import IdealNetwork


ROUTES=((2,1,0,3),(3,0,1,2),(0,2,3,1),(1,3,2,0))
PAIRS=((3,0),(2,1),(3,1),(2,0))


def inputs(bits,count):
    rng=np.random.default_rng(8817)
    x=torch.from_numpy(rng.bit_generator.random_raw((4,count)).view(np.int64).copy())
    y=torch.from_numpy(rng.bit_generator.random_raw((4,count)).view(np.int64).copy())
    x[:2]=0
    y[2:]=0
    if bits==32:
        x&=0xffffffff
        y&=0xffffffff
    return x,y


def oracle(x,y,seeds,bits,nonce):
    """Compute each balanced joint from CPU ChaCha20 and CPU BLAKE3."""
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms
    from blake3 import blake3
    count=x.shape[-1]
    components=torch.zeros_like(x)
    outgoing=[{} for _ in range(4)]
    if not count:
        return components,outgoing
    for (sender,backup,receiver,outsider),(i,j) in zip(ROUTES,PAIRS):
        domain=b'rep4/trunc-pr/product16/v1\0'+bytes((bits,bits,0,bits,receiver))
        key=hashlib.sha256(seeds[receiver]+b'Rep4GPU-DevicePRGv1\0'+struct.pack('<Q',len(domain))+domain).digest()
        cipher=Cipher(algorithms.ChaCha20(key,struct.pack('<QQ',0,nonce)),mode=None).encryptor()
        data=cipher.update(bytes(4*((count+1)//2)))+cipher.finalize()
        mask=torch.from_numpy(np.frombuffer(data,dtype='<u2').astype(np.int64)[:count].copy())
        value=(x[i]*y[j])&0xffff
        masked=(value-mask)&0xffff
        payload=masked.numpy().astype('<u2').tobytes()
        outgoing[sender][receiver]=outgoing[sender].get(receiver,b'')+payload
        outgoing[backup][receiver]=outgoing[backup].get(receiver,b'')+blake3(payload).digest()
        components[receiver]+=mask
        components[outsider]+=masked
    return components&0xffff,outgoing


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('shift',[1,8,14,16])
def test_narrow_product_lift_is_annihilated(bits,shift):
    x,y=inputs(bits,33)
    seeds=make_seeds()
    components,_=oracle(x,y,seeds,bits,0)
    for n in range(x.shape[-1]):
        full=sum(int(v[n]) for v in x)*sum(int(v[n]) for v in y)
        lifted=sum(int(v[n]) for v in components)
        assert ((lifted-full)*(1<<(bits-shift)))% (1<<bits)==0


def test_product16_is_not_a_general_full_ring_product():
    for bits in (32,64):
        # These valid 16-bit components sum to zero modulo 2**16, but their
        # lift differs by 2**16, which shift17's coefficient does not erase.
        assert ((65535+1)*(1<<(bits-17)))%(1<<bits)!=0


requires_gpu=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@requires_gpu
@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('count',[0,1,2,17,257])
def test_product16_matches_cpu_masks_shares_frames_and_fresh_counters(bits,count):
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceByteBridge,DeviceRecorder
    from rep4gpu.gpu.matmul.narrow_product_gpu import product16
    x,y=inputs(bits,count)
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    def work(party):
        try:
            transport=DeviceRecorder(DeviceByteBridge(network.endpoint(party),'cuda'))
            rt=DeviceInferenceParty(party,transport,{e:s for e,s in seeds.items() if e!=party},
                                    bits=bits,narrow_products=True)
            a=Share(tuple(x[(party+j+1)%4].to('cuda') for j in range(3)))
            b=Share(tuple(y[(party+j+1)%4].to('cuda') for j in range(3)))
            outputs=[tuple(t.cpu() for t in product16(rt,a,b).parts) for _ in range(2)]
            return outputs,dict(rt.counters),rt.joint_inputs,rt.multiplications,[f.outgoing for f in transport.transcript]
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(work,range(4)))
    for nonce in range(2):
        expected,frames=oracle(x,y,seeds,bits,nonce)
        for p,(outputs,counters,joints,mults,transcript) in enumerate(actual):
            assert all(torch.equal(value,expected[(p+j+1)%4]) for j,value in enumerate(outputs[nonce]))
            if count:
                assert transcript[nonce]==frames[p]
                assert sorted(map(len,transcript[nonce].values()))==sorted([2*count,32])
            assert all(value==(2 if count else 0) for value in counters.values())
            assert (joints,mults)==((8,2) if count else (0,0))


@requires_gpu
@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('shift',[1,14,16,17])
def test_narrow_truncation_two_fresh_calls_match_rounding_oracle(bits,shift):
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    limit=1<<(bits-2)
    x=torch.tensor([-limit+1,-4097,-17,-1,0,1,17,4097,limit-1])
    rng=np.random.default_rng(331)
    parts=torch.from_numpy(rng.bit_generator.random_raw((4,x.numel())).view(np.int64).copy())
    parts[3]=x-parts[:3].sum(0)
    if bits==32:
        parts&=0xffffffff
    def work(party):
        try:
            rt=DeviceInferenceParty(party,DeviceIdealEndpoint(network.endpoint(party)),
                {e:s for e,s in seeds.items() if e!=party},bits=bits,rounding='probabilistic',
                sparse_products=True,narrow_products=True,gpu_open=True)
            masks=[]
            original=rt._random
            def remember(excluded,shape,device,binary=False,**kwargs):
                result=original(excluded,shape,device,binary,**kwargs)
                if party==0 and tuple(shape)==tuple(x.shape) and excluded in (2,3):
                    masks.append((excluded,result.cpu().to(torch.int64)))
                return result
            rt._random=remember
            value=Share(tuple(parts[(party+j+1)%4].to('cuda') for j in range(3)))
            outputs=[rt.open(rt.trunc(value,shift)).cpu() for _ in range(2)]
            return outputs,masks
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(work,range(4)))
    for call in range(2):
        (e2,r2),(e3,r3)=actual[0][1][2*call:2*call+2]
        assert (e2,e3)==(2,3)
        expected=(x+((r2+r3)&((1<<shift)-1)))>>shift
        assert all(torch.equal(role[0][call],expected) for role in actual)


@requires_gpu
@pytest.mark.parametrize('bits',[32,64])
def test_narrow_truncation_graph_uses_fresh_domains_and_preserves_outputs(bits):
    from rep4gpu.protocol.phase_graph import PhaseGraphParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    roles=[PhaseGraphParty(p,DeviceIdealEndpoint(network.endpoint(p)),
        {e:s for e,s in seeds.items() if e!=p},bits=bits,narrow_products=True,
        sparse_products=True,rounding='probabilistic',gpu_open=True) for p in range(4)]
    values=[r.public(torch.arange(17,device='cuda')<<14) for r in roles]
    plans=[r.prepare_inference(lambda x,r=r:r.open(r.trunc(x,14)),value,check_at_send=True)
           for r,value in zip(roles,values)]
    def work(p):
        try:
            first=plans[p].run(values[p]);saved=first.clone();counters=dict(roles[p].counters)
            second=plans[p].run(values[p])
            assert all(roles[p].counters[e]>counters[e] for e in counters)
            assert torch.equal(first,saved)
            return first.cpu(),second.cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(work,range(4)))
    for first,second in actual:
        assert torch.equal(first,torch.arange(17))
        assert torch.equal(second,torch.arange(17))


class CorruptFrame:
    def __init__(self,underlying,target):
        self.underlying,self.target=underlying,target
    def __getattr__(self,name):
        return getattr(self.underlying,name)
    def exchange_device(self,outgoing,sizes):
        outgoing=dict(outgoing)
        if self.target is not None and self.target in outgoing:
            frame=outgoing[self.target].clone()
            frame[0]^=1
            outgoing[self.target]=frame
            self.target=None
        return self.underlying.exchange_device(outgoing,sizes)


@requires_gpu
@pytest.mark.parametrize('joint',range(4))
@pytest.mark.parametrize('source',['sender','backup'])
def test_product16_checks_every_sender_and_backup_frame(joint,source):
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.gpu.matmul.narrow_product_gpu import product16
    x,y=inputs(32,17)
    network,seeds=IdealNetwork(timeout=30),make_seeds()
    sender,backup,receiver,_=ROUTES[joint]
    corrupt=sender if source=='sender' else backup
    def work(party):
        try:
            endpoint=DeviceIdealEndpoint(network.endpoint(party))
            rt=DeviceInferenceParty(party,CorruptFrame(endpoint,receiver if party==corrupt else None),
                {e:s for e,s in seeds.items() if e!=party},bits=32,narrow_products=True,gpu_open=True)
            a=Share(tuple(x[(party+j+1)%4].to('cuda') for j in range(3)))
            b=Share(tuple(y[(party+j+1)%4].to('cuda') for j in range(3)))
            # Reaching the next exchange requires the product's tag checks to
            # succeed. A corrupt masked payload or backup tag must abort first.
            result=product16(rt,a,b)
            return rt.open(rt.mul_public(result,1<<18))
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(work,range(4)))
    assert all(isinstance(result,BaseException) for result in actual)
    assert any('hash mismatch' in str(result) for result in actual)


@requires_gpu
@pytest.mark.parametrize('bits',[32,64])
def test_narrow_tiny_ffn_graph_keeps_fresh_masks_and_checked_relu_chain(bits):
    from rep4gpu.arithmetic.activations import FixedPoint
    from rep4gpu.protocol.phase_graph import PhaseGraphParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    f,scale=14,1<<14
    x=torch.tensor([[2,-1,1],[-1,0,2]])*(scale//4)
    w1=torch.tensor([[1,0,-1,1],[-1,1,1,0],[1,-1,0,1]])*(scale//4)
    b1=torch.tensor([2,-2,0,1])*(scale//32)
    w2=torch.tensor([[1,-1],[1,1],[-1,1],[1,0]])*(scale//4)
    b2=torch.tensor([1,-1])*(scale//32)
    def plain(value):
        hidden=((value@w1+b1*scale)>>f).clamp_min(0)
        return (hidden@w2>>f)+b2
    rng=np.random.default_rng(77123)
    def split(value):
        components=torch.from_numpy(rng.bit_generator.random_raw((4,*value.shape)).view(np.int64).copy())
        components[3]=value-components[:3].sum(0)
        return components&0xffffffff if bits==32 else components
    shared=[split(value) for value in (x,-x,w1,b1,w2,b2)]
    network,seeds=IdealNetwork(timeout=120),make_seeds()
    roles=[PhaseGraphParty(p,DeviceIdealEndpoint(network.endpoint(p)),
        {e:s for e,s in seeds.items() if e!=p},bits=bits,narrow_products=True,
        sparse_products=True,bounded_sign=True,rounding='probabilistic',gpu_open=True) for p in range(4)]
    values=[[Share(tuple(value[(p+j+1)%4].to('cuda') for j in range(3)))
             for value in shared] for p in range(4)]
    def prepare(p):
        role=roles[p]
        _,_,a,b,c,d=values[p]
        fp=FixedPoint(role,f)
        return role.prepare_inference(lambda value:role.open(
            fp.add(fp.matmul(fp.matmul_relu(value,a,b),c),d)),values[p][0],check_at_send=True)
    plans=[prepare(p) for p in range(4)]
    def work(p):
        try:
            first=plans[p].run(values[p][0]);saved=first.clone()
            counters=dict(roles[p].counters)
            second=plans[p].run(values[p][1])
            assert all(roles[p].counters[e]>counters[e] for e in counters)
            assert torch.equal(first,saved)
            return first.cpu(),second.cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual=list(pool.map(work,range(4)))
    for first,second in actual:
        assert torch.equal(first,plain(x))
        assert torch.equal(second,plain(-x))
