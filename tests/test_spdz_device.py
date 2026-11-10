"""Independent CPU-crypto transcripts for GPU authenticated openings."""
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from rep4gpu.spdz import TrustedDealer,PRIME,field_matmul
from rep4gpu.spdz_distributed import AuthShare,split_share,split_triple
from rep4gpu.spdz_device import DeviceDistributedParty
from rep4gpu.device_transport import DeviceFrame,DeviceReplay,DeviceIdealEndpoint
from rep4gpu.transport import IdealNetwork


def _hash(data):return hashlib.blake2b(data,digest_size=32).digest()
def _pack(t):return t.numpy().astype('<u4').tobytes()


def _opening_oracle(shares,alpha,session,salts,label=b''):
    from blake3 import blake3
    h=alpha.shape[0]
    shapes=[list(x.values.shape) for x in shares[0]]
    layout=json.dumps(dict(field=PRIME,dtype='uint32-le',keys=h,shapes=shapes),sort_keys=True,separators=(',',':')).encode()
    session_hash=_hash(b'SPDZ31-session-v1'+struct.pack('<Q',len(session))+session)
    context=b'SPDZ31G-open-v1'+session_hash+struct.pack('<Q',0)+_hash(layout)+_hash(label)
    transcripts=[[] for _ in range(4)]
    def broadcast(payloads,round_id,stage_label):
        def frame(stage,p,body):
            return b'SPDZ31G1'+session_hash+struct.pack('<QI',round_id,p)+stage+_hash(stage_label)+struct.pack('<Q',len(body))+body
        frames=[frame(b'V',p,payloads[p]) for p in range(4)]
        view=b''.join(blake3(struct.pack('<I',p)+frames[p]).digest() for p in range(4))
        echoes=[frame(b'E',p,view) for p in range(4)]
        for role in range(4):
            peers=set(range(4))-{role}
            transcripts[role].append(DeviceFrame('device',{p:frames[role] for p in peers},{p:frames[p] for p in peers}))
            transcripts[role].append(DeviceFrame('device',{p:echoes[role] for p in peers},{p:echoes[p] for p in peers}))
    values=[torch.cat([x.values.reshape(-1) for x in role]) for role in shares]
    opened=sum(values).remainder(PRIME)
    residuals=[]
    for role in range(4):
        macs=torch.cat([x.macs.reshape(h,-1) for x in shares[role]],dim=1)
        residuals.append((macs-alpha[:,role,None]*opened).remainder(PRIME))
    assert not torch.any(sum(residuals).remainder(PRIME))
    broadcast([_pack(value) for value in values],0,context+b'values')
    reveals=[salts[p]+_pack(residuals[p]) for p in range(4)]
    commits=[blake3(context+struct.pack('<I',p)+reveals[p]).digest() for p in range(4)]
    broadcast(commits,1,context+b'commit')
    broadcast(reveals,2,context+b'reveal')
    expected=[];offset=0
    for shape in shapes:
        count=shape[0]*shape[1];expected.append(opened[offset:offset+count].reshape(shape));offset+=count
    return transcripts,expected


@pytest.mark.parametrize('keys',[5,10])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('shapes',[[(3,5)],[(1,1),(5,3)],[(0,3)]])
def test_device_open_matches_independent_cpu_crypto(keys,dtype,shapes):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    pytest.importorskip('blake3')
    dealer=TrustedDealer(mac_keys=keys)
    values=[torch.arange(shape[0]*shape[1]).reshape(shape).remainder(PRIME) for shape in shapes]
    locals_by_matrix=[split_share(dealer.share(value)) for value in values]
    shares=[[matrix[p] for matrix in locals_by_matrix] for p in range(4)]
    salts=[bytes([p+1])*32 for p in range(4)]
    transcript,expected=_opening_oracle(shares,dealer.alpha_shares,b'gpu-oracle',salts)
    for p in range(4):
        replay=DeviceReplay(transcript[p],check_outgoing=True)
        runtime=DeviceDistributedParty(p,replay,dealer.alpha_shares[:,p].to('cuda'),session=b'gpu-oracle',
                                       _allow_insecure_test_transport=True)
        runtime._commit_salt=lambda p=p:salts[p]
        inputs=[AuthShare(x.values.to(device='cuda',dtype=dtype),x.macs.to(device='cuda',dtype=dtype)) for x in shares[p]]
        actual=runtime.open_many(inputs)
        assert all(torch.equal(x.cpu(),y) for x,y in zip(actual,expected))
        assert not runtime.poisoned
        replay.assert_consumed()


@pytest.mark.parametrize('keys',[5,10])
def test_device_fresh_four_role_checked_beaver(keys):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    dealer=TrustedDealer(mac_keys=keys)
    x=torch.tensor([[0,PRIME-1,71],[1,17,PRIME-2]])
    y=torch.tensor([[17,PRIME-1],[0,1],[PRIME-2,71]])
    xs,ys=split_share(dealer.share(x)),split_share(dealer.share(y))
    triples=split_triple(dealer.triple(2,3,2))
    network=IdealNetwork(timeout=90)
    device=torch.device('cuda',torch.cuda.current_device())
    def role(p):
        try:
            transport=DeviceIdealEndpoint(network.endpoint(p),device=device)
            runtime=DeviceDistributedParty(p,transport,dealer.alpha_shares[:,p].to(device),session=b'fresh-GPU-matmul',
                                           _allow_insecure_test_transport=True)
            def move(s):return AuthShare(s.values.to(device).int(),s.macs.to(device).int())
            triple=triples[p]
            triple.a,triple.b,triple.c=move(triple.a),move(triple.b),move(triple.c)
            result=runtime.matmul(move(xs[p]),move(ys[p]),triple)
            return runtime.open(result).cpu()
        except BaseException as error:
            network.abort(error);raise
    with ThreadPoolExecutor(4) as pool:
        outputs=list(pool.map(role,range(4)))
    expected=field_matmul(x,y)
    assert all(torch.equal(value,expected) for value in outputs)


def test_device_rejects_cpu_constructor():
    with pytest.raises(ValueError,match='CUDA'):
        DeviceDistributedParty(0,None,torch.zeros(5,dtype=torch.int64),session=b'x')


@pytest.mark.parametrize('noncontiguous',[False,True])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('kind',['square','elementwise'])
def test_device_ten_key_nonlinear_correlations(dtype,kind,noncontiguous):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rep4gpu.spdz import MatrixTriple as CentralTriple,_random
    from rep4gpu.spdz_distributed import split_elementwise_triple
    from rep4gpu.spdz_polynomial import make_square_masks,square
    dealer=TrustedDealer(mac_keys=10)
    x=torch.tensor([[0,PRIME-1,PRIME-2],[1,71,2147483645]])
    y=torch.tensor([[PRIME-1,PRIME-2,1],[71,0,PRIME-1]])
    xs,ys=split_share(dealer.share(x)),split_share(dealer.share(y))
    if kind=='square':
        masks=make_square_masks(dealer,tuple(x.shape))
    else:
        a,b=_random(tuple(x.shape),'cpu'),_random(tuple(x.shape),'cpu')
        masks=split_elementwise_triple(CentralTriple(dealer.share(a),dealer.share(b),dealer.share((a*b).remainder(PRIME))))
    network=IdealNetwork(timeout=60)
    device=torch.device('cuda',torch.cuda.current_device())
    def role(p):
        try:
            runtime=DeviceDistributedParty(p,DeviceIdealEndpoint(network.endpoint(p),device=device),
                dealer.alpha_shares[:,p].to(device),session=b'nonlinear-ten-key',_allow_insecure_test_transport=True)
            def move(s):
                values,macs=s.values.to(device=device,dtype=dtype),s.macs.to(device=device,dtype=dtype)
                if noncontiguous:
                    values=values.T.contiguous().T
                    macs=macs.transpose(1,2).contiguous().transpose(1,2)
                return AuthShare(values,macs)
            if kind=='square':
                mask=masks[p];mask.a,mask.a_squared=move(mask.a),move(mask.a_squared)
                result=square(runtime,move(xs[p]),mask)
            else:
                mask=masks[p];mask.a,mask.b,mask.c=move(mask.a),move(mask.b),move(mask.c)
                result=runtime.mul(move(xs[p]),move(ys[p]),mask)
            assert result.values.dtype==result.macs.dtype==torch.int32
            return runtime.open(result).cpu()
        except BaseException as error:
            network.abort(error);raise
    with ThreadPoolExecutor(4) as pool:
        outputs=list(pool.map(role,range(4)))
    expected=(x*(x if kind=='square' else y)).remainder(PRIME)
    assert all(torch.equal(value,expected) for value in outputs)


@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('party',[0,3])
def test_noncontiguous_local_arithmetic_matches_wide_reference(dtype,party):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    class LocalOnly:
        def exchange_device(self,*args,**kwargs):raise AssertionError('local arithmetic must not communicate')
    generator=torch.Generator().manual_seed(419)
    xv=torch.randint(PRIME,(5,3),generator=generator).T
    yv=torch.randint(PRIME,(5,3),generator=generator).T
    xm=torch.randint(PRIME,(10,5,3),generator=generator).transpose(1,2)
    ym=torch.randint(PRIME,(10,5,3),generator=generator).transpose(1,2)
    xv[0,0]=yv[0,0]=PRIME-1
    xm[0,0,0]=ym[0,0,0]=PRIME-1
    alpha=torch.arange(1,11,dtype=torch.int64)*7919
    runtime=DeviceDistributedParty(party,LocalOnly(),alpha.to('cuda'),session=b'noncontiguous-local',
                                    _allow_insecure_test_transport=True)
    x=AuthShare(xv.to(device='cuda',dtype=dtype),xm.to(device='cuda',dtype=dtype))
    y=AuthShare(yv.to(device='cuda',dtype=dtype),ym.to(device='cuda',dtype=dtype))
    assert not x.values.is_contiguous() and not x.macs.is_contiguous()
    cases=[(runtime.add(x,y),(xv+yv).remainder(PRIME),(xm+ym).remainder(PRIME)),
           (runtime.subtract(x,y),(xv-yv).remainder(PRIME),(xm-ym).remainder(PRIME))]
    public=torch.tensor([[-1,0,1,PRIME-1,17]],dtype=torch.int64).remainder(PRIME)
    cases.append((runtime.add_public(x,public),
                  (xv+public if party==0 else xv).remainder(PRIME),
                  (xm+alpha[:,None,None]*public).remainder(PRIME)))
    for actual,value,mac in cases:
        assert actual.values.is_contiguous() and actual.macs.is_contiguous()
        assert torch.equal(actual.values.cpu(),value)
        assert torch.equal(actual.macs.cpu(),mac)


@pytest.mark.parametrize('counts',[(1,0),(0,3),(3,5),(17,33)])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_direct_residual_matches_wide_reference_without_input_alias(counts,dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    class LocalOnly:
        def exchange_device(self,*args,**kwargs):raise AssertionError('residual arithmetic must not communicate')
    generator=torch.Generator().manual_seed(937)
    h=10
    alpha=torch.randint(PRIME,(h,),generator=generator)
    runtime=DeviceDistributedParty(0,LocalOnly(),alpha.to('cuda'),session=b'residual-parts',
                                    _allow_insecure_test_transport=True)
    shares=[];cpu_macs=[]
    for count in counts:
        mac=torch.randint(PRIME,(h,1,count),generator=generator)
        value=torch.randint(PRIME,(1,count),generator=generator)
        cpu_macs.append(mac.reshape(h,count))
        shares.append(AuthShare(value.to(device='cuda',dtype=dtype),mac.to(device='cuda',dtype=dtype)))
    opened=torch.randint(PRIME,(sum(counts),),generator=generator)
    actual=runtime._residual(shares,opened.to('cuda'))
    expected=(torch.cat(cpu_macs,dim=1)-alpha[:,None]*opened).remainder(PRIME)
    assert torch.equal(actual.cpu(),expected)
    saved=actual.clone()
    for share in shares:share.macs.zero_()
    assert torch.equal(actual,saved)


@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_singleton_value_packing_preserves_bytes_and_source(dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    class LocalOnly:
        def exchange_device(self,*args,**kwargs):raise AssertionError('packing must not communicate')
    runtime=DeviceDistributedParty(0,LocalOnly(),torch.arange(1,11,device='cuda',dtype=torch.int64),
                                    session=b'pack-values',_allow_insecure_test_transport=True)
    values=torch.tensor([[0,PRIME-1,71]],device='cuda',dtype=dtype)
    share=AuthShare(values,torch.zeros((10,1,3),device='cuda',dtype=dtype))
    before=values.clone()
    packed=runtime._packed_values((share,))
    assert packed.cpu().numpy().tobytes()==values.cpu().long().numpy().astype('<u4').tobytes()
    assert torch.equal(values,before)
    if dtype==torch.int32:assert packed.data_ptr()==values.data_ptr()


@pytest.mark.parametrize('deferred',[False,True])
def test_square_identifier_mismatch_rejects_before_candidate(deferred):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rep4gpu.spdz_polynomial import make_square_masks,square
    from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
    dealer=TrustedDealer(mac_keys=10)
    x=torch.tensor([[1,PRIME-1,71]])
    shares=split_share(dealer.share(x))
    masks=make_square_masks(dealer,tuple(x.shape))
    masks[0].identifier=bytes([masks[0].identifier[0]^1])+masks[0].identifier[1:]
    network=IdealNetwork(timeout=15)
    parties=[]
    party_type=DeviceDeferredCompressedParty if deferred else DeviceDistributedParty
    for p in range(4):
        party=party_type(p,DeviceIdealEndpoint(network.endpoint(p)),dealer.alpha_shares[:,p].cuda(),
            session=b'square-bound-identifier',_allow_insecure_test_transport=True)
        parties.append(party)
    import rep4gpu.spdz_device as module
    original=module._sum_values
    def forbidden_sum(*args,**kwargs):raise AssertionError('mismatched identifier reached candidate reconstruction')
    module._sum_values=forbidden_sum
    def role(p):
        try:
            def move(s):return AuthShare(s.values.cuda().int(),s.macs.cuda().int())
            masks[p].a,masks[p].a_squared=move(masks[p].a),move(masks[p].a_squared)
            return square(parties[p],move(shares[p]),masks[p])
        except BaseException as error:
            network.abort(error)
            return error
    try:
        with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    finally:module._sum_values=original
    assert all(isinstance(error,BaseException) and not isinstance(error,AssertionError) for error in results)
    assert all(party.poisoned for party in parties)
    assert all(mask.used for mask in masks)
    if deferred:assert all(party.pending_values==0 and not party._pending for party in parties)
