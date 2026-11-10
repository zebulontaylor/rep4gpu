"""Algebra tests with synthetic correlations, not a cryptographic producer."""
import numpy as np
import pytest
import torch

from rep4gpu.field_kernels import PRIME
from rep4gpu.matrix_pcg_linear import PublicLinearMap,RegularSparseColumns,LinearExpansionAccumulator,sparse_local_product


def mm(a,b):return np.asarray((a.astype(object)@b.astype(object))%PRIME,dtype=np.int64)
def mod(a):return np.asarray(a.astype(object)%PRIME,dtype=np.int64)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_whole_four_party_fusion_and_global_product(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    rng=np.random.default_rng(418931)
    a,l,k,n,t=8,16,4,5,4
    h=rng.integers(0,PRIME,(a,l),dtype=np.int64)
    tensor=lambda x:torch.tensor(x,device=device,dtype=torch.int64)
    plan=PublicLinearMap(tensor(h));identity=plan.identity
    sparse=[];dense=[];right=[];left=[];acc=[];unfused=[]
    for party in range(4):
        positions=np.stack([rng.integers(j*(l//t),(j+1)*(l//t),k) for j in range(t)],axis=1)
        values=rng.integers(1,PRIME,(k,t),dtype=np.int64)
        e=RegularSparseColumns(plan,tensor(positions),tensor(values),map_identity=identity)
        em=e.materialize().cpu().numpy()
        b=rng.integers(0,PRIME,(k,n),dtype=np.int64)
        aa=mm(h,em)
        stream=LinearExpansionAccumulator(plan,n,map_identity=identity)
        stream.add_local(e,tensor(b),map_identity=identity)
        sparse.append(e);dense.append(em);right.append(b);left.append(aa);acc.append(stream);unfused.append(mm(aa,b))
    for sender in range(4):
        for receiver in range(4):
            if sender==receiver:continue
            for inner in range(k):
                delta=mod(dense[sender][:,inner,None].astype(object)*right[receiver][None,inner,:].astype(object))
                y=rng.integers(0,PRIME,(l,n),dtype=np.int64)
                z=mod(y.astype(object)-delta.astype(object))
                acc[sender].add(tensor(y),sign=1,map_identity=identity)
                acc[receiver].add(tensor(z),sign=-1,map_identity=identity)
                unfused[sender]=mod(unfused[sender].astype(object)+mm(h,y).astype(object))
                unfused[receiver]=mod(unfused[receiver].astype(object)-mm(h,z).astype(object))
    outputs=[x.finish(map_identity=identity).cpu().numpy() for x in acc]
    for got,want in zip(outputs,unfused):np.testing.assert_array_equal(got,want)
    total_left=mod(sum(x.astype(object) for x in left))
    total_right=mod(sum(x.astype(object) for x in right))
    np.testing.assert_array_equal(mod(sum(x.astype(object) for x in outputs)),mm(total_left,total_right))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sparse_collisions_signed_extremes_and_streamed_reduction(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    h=torch.tensor([[1,PRIME-1,1,1],[PRIME-1,1,0,1]],device=device,dtype=torch.int64)
    plan=PublicLinearMap(h);identity=plan.identity
    k,t=257,2
    positions=torch.tensor([[0,2]]*k,device=device,dtype=torch.int64)
    values=torch.full((k,t),PRIME-1,device=device,dtype=torch.int64)
    sparse=RegularSparseColumns(plan,positions,values,map_identity=identity)
    b=torch.full((k,3),PRIME-1,device=device,dtype=torch.int64)
    raw=sparse_local_product(plan,sparse,b,map_identity=identity)
    np.testing.assert_array_equal(raw.cpu().numpy(),np.asarray([[257]*3,[0]*3,[257]*3,[0]*3]))
    term=torch.tensor([[-2**63,2**63-1,-1]]*4,device=device,dtype=torch.int64)
    stream=LinearExpansionAccumulator(plan,3,map_identity=identity)
    stream.add_local(sparse,b,map_identity=identity)
    stream.add(term,sign=1,map_identity=identity)
    stream.add(term,sign=-1,map_identity=identity)
    np.testing.assert_array_equal(stream.finish(map_identity=identity).cpu().numpy(),mm(h.cpu().numpy(),raw.cpu().numpy()))


def test_public_map_identity_dimensions_lifetime_and_owned_snapshots():
    source=torch.eye(4,dtype=torch.int64)
    plan=PublicLinearMap(source);identity=plan.identity
    source.zero_()
    assert PublicLinearMap(torch.eye(4,dtype=torch.int32)).identity==identity
    stream=LinearExpansionAccumulator(plan,3,map_identity=identity)
    term=torch.ones((4,3),dtype=torch.int64)
    with pytest.raises(ValueError,match='identity'):stream.add(term,sign=1,map_identity=b'x'*32)
    with pytest.raises(ValueError,match='dimensions'):stream.add(term[:3],sign=1,map_identity=identity)
    with pytest.raises(ValueError,match='sign'):stream.add(term,sign=0,map_identity=identity)
    stream.add(term,sign=1,map_identity=identity)
    torch.testing.assert_close(stream.finish(map_identity=identity),term.int(),rtol=0,atol=0)
    with pytest.raises(ValueError,match='sealed'):stream.finish(map_identity=identity)
    with pytest.raises(ValueError,match='sealed'):stream.add(term,sign=1,map_identity=identity)
    plan._h.add_(1)
    with pytest.raises(ValueError,match='modified'):plan.expand(term,map_identity=identity)


def test_rejects_invalid_regular_blocks_and_other_map():
    plan=PublicLinearMap(torch.eye(4,dtype=torch.int64));identity=plan.identity
    with pytest.raises(ValueError,match='regular block'):
        RegularSparseColumns(plan,torch.tensor([[0,1]]),torch.ones((1,2),dtype=torch.int64),map_identity=identity)
    with pytest.raises(ValueError,match='nonzero'):
        RegularSparseColumns(plan,torch.tensor([[0,2]]),torch.zeros((1,2),dtype=torch.int64),map_identity=identity)
    valid=RegularSparseColumns(plan,torch.tensor([[0,2]]),torch.ones((1,2),dtype=torch.int64),map_identity=identity)
    other=PublicLinearMap(torch.ones((4,4),dtype=torch.int64))
    with pytest.raises(ValueError,match='identity'):
        sparse_local_product(other,valid,torch.ones((1,3),dtype=torch.int64),map_identity=other.identity)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('batch',[1,8,64])
def test_batched_stream_matches_individual_signed_additions(device,batch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    rng=np.random.default_rng(6741+batch)
    h=rng.integers(0,PRIME,(3,8),dtype=np.int64)
    terms=rng.bit_generator.random_raw(batch*8*5).view(np.int64).reshape(batch,8,5)
    terms[0,0,:]=np.asarray([-2**63,2**63-1,-1,0,1],dtype=np.int64)
    plan=PublicLinearMap(torch.tensor(h,device=device));identity=plan.identity
    data=torch.tensor(terms,device=device)
    signs=tuple(1 if i%3 else -1 for i in range(batch))
    first=LinearExpansionAccumulator(plan,5,map_identity=identity)
    second=LinearExpansionAccumulator(plan,5,map_identity=identity)
    for term,sign in zip(data,signs):first.add(term,sign=sign,map_identity=identity)
    second.add_batch(data,signs=signs,map_identity=identity)
    expected=mm(h,mod(sum(int(sign)*term.astype(object) for sign,term in zip(signs,terms))))
    for result in (first.finish(map_identity=identity),second.finish(map_identity=identity)):
        np.testing.assert_array_equal(result.cpu().numpy(),expected)


def test_batch_metadata_failures_preserve_accumulator():
    plan=PublicLinearMap(torch.eye(4,dtype=torch.int64));identity=plan.identity
    state=LinearExpansionAccumulator(plan,3,map_identity=identity)
    terms=torch.ones((8,4,3),dtype=torch.int64)
    with pytest.raises(ValueError,match='sign'):state.add_batch(terms,signs=(1,),map_identity=identity)
    with pytest.raises(ValueError,match='identity'):state.add_batch(terms,signs=(1,)*8,map_identity=b'x'*32)
    with pytest.raises(ValueError,match='batch'):state.add_batch(terms[:,:3],signs=(1,)*8,map_identity=identity)
    torch.testing.assert_close(state.finish(map_identity=identity),torch.zeros((4,3),dtype=torch.int32),rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sparse_atomic_sum_exceeds_int32_before_field_reduction(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    plan=PublicLinearMap(torch.eye(4,dtype=torch.int64,device=device));identity=plan.identity
    k=257
    sparse=RegularSparseColumns(plan,torch.tensor([[0,2]]*k,device=device),torch.ones((k,2),device=device,dtype=torch.int64),map_identity=identity)
    b=torch.full((k,3),PRIME-1,device=device,dtype=torch.int64)
    raw=sparse_local_product(plan,sparse,b,map_identity=identity)
    expected=np.asarray([[k*(PRIME-1)]*3,[0]*3,[k*(PRIME-1)]*3,[0]*3],dtype=np.int64)
    np.testing.assert_array_equal(raw.cpu().numpy(),expected)
    state=LinearExpansionAccumulator(plan,3,map_identity=identity)
    state.add_local(sparse,b,map_identity=identity)
    np.testing.assert_array_equal(state.finish(map_identity=identity).cpu().numpy(),expected%PRIME)
