import random
import numpy as np
import pytest
import torch
import triton

from rep4gpu.field_fp127 import PRIME,MAX_K,field_matmul_fp127,prepare_fp127_left,prepare_fp127_weight,_finish

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def encode(values,compact=False):
    values=np.asarray(values,dtype=object)
    limbs=np.stack([np.asarray((values>>(16*i))&65535,dtype=np.int32) for i in range(8)],axis=-1)
    return torch.tensor(limbs,device='cuda',dtype=torch.uint16 if compact else torch.int32)


def decode(value):
    limbs=value.cpu().numpy().astype(object)
    assert np.all((limbs>=0)&(limbs<65536))
    assert np.all(limbs[...,-1]<32768)
    result=sum(limbs[...,i]<<(16*i) for i in range(8))
    assert np.all(result<PRIME)
    return result


def fixture(m,k,n,seed):
    rng=random.Random(seed)
    a=np.asarray([[rng.randrange(PRIME) for _ in range(k)] for _ in range(m)],dtype=object).reshape(m,k)
    b=np.asarray([[rng.randrange(PRIME) for _ in range(n)] for _ in range(k)],dtype=object).reshape(k,n)
    return a,b


@pytest.mark.parametrize('shape',[(1,0,1),(1,1,1),(1,7,19),(3,33,5),(17,65,33)])
@pytest.mark.parametrize('compact',[False,True])
def test_full_field_bigint_tails(shape,compact):
    a,b=fixture(*shape,seed=127189+sum(shape))
    np.testing.assert_array_equal(decode(field_matmul_fp127(encode(a,compact),encode(b,compact))),(a@b)%PRIME)


@pytest.mark.parametrize('mode',['left','right','both'])
def test_prepared_snapshots_and_mixed_input_types(mode):
    a,b=fixture(3,37,11,127378)
    ta,tb=encode(a,True),encode(b,False)
    left=prepare_fp127_left(ta) if mode in ('left','both') else ta
    right=prepare_fp127_weight(tb) if mode in ('right','both') else tb
    if mode in ('left','both'):
        assert left.storage_bytes==16*3*37+4*16*3
        assert set(vars(left))=={'digits','sums','m','k','_versions'}
        ta.zero_()
    if mode in ('right','both'):
        assert right.storage_bytes==16*37*11+4*16*11
        assert set(vars(right))=={'digits','sums','k','n','_versions'}
        tb.zero_()
    np.testing.assert_array_equal(decode(field_matmul_fp127(left,right)),(a@b)%PRIME)


@pytest.mark.parametrize('k',[1,257,MAX_K])
def test_full_product_width_and_maximum_carry(k):
    a=np.full((1,k),PRIME-1,dtype=object)
    b=np.full((k,3),PRIME-1,dtype=object)
    np.testing.assert_array_equal(decode(field_matmul_fp127(encode(a),encode(b))),np.full((1,3),k,dtype=object))


def test_zero_byte_boundaries_and_canonical_zero():
    edges=np.asarray([0,1,127,128,255,256,65535,65536,2**126,PRIME-1],dtype=object)
    a,b=edges.reshape(-1,1),edges.reshape(1,-1)
    np.testing.assert_array_equal(decode(field_matmul_fp127(encode(a),encode(b))),(a@b)%PRIME)
    a=np.asarray([[PRIME-1,1]],dtype=object)
    b=np.asarray([[1],[1]],dtype=object)
    np.testing.assert_array_equal(decode(field_matmul_fp127(encode(a),encode(b))),np.zeros((1,1),dtype=object))


def test_carry_reducer_at_proven_coefficient_bound():
    rng=random.Random(12743)
    coefficients=[[2**43-1]*8,[65535]*7+[32767],[65536]+[65535]*6+[32767]]
    coefficients.extend([[rng.randrange(2**43) for _ in range(8)] for _ in range(64)])
    source=torch.tensor(coefficients,device='cuda',dtype=torch.int64)
    output=torch.empty(source.shape,device='cuda',dtype=torch.int32)
    _finish[(triton.cdiv(len(coefficients),32),)](source,output,len(coefficients),32,num_warps=4)
    expected=np.asarray([sum(int(v)<<(16*i) for i,v in enumerate(row))%PRIME for row in coefficients],dtype=object)
    np.testing.assert_array_equal(decode(output),expected)


def test_native_little_endian_wire_layout():
    a,b=fixture(1,9,7,12716)
    wire=b''.join(int(x).to_bytes(16,'little') for x in a.flat)
    native_words=np.frombuffer(wire,dtype='<u2').copy().reshape(1,9,8)
    native=torch.tensor(native_words,device='cuda',dtype=torch.uint16)
    torch.testing.assert_close(native.int(),encode(a),rtol=0,atol=0)
    np.testing.assert_array_equal(decode(field_matmul_fp127(native,encode(b))),(a@b)%PRIME)


def test_rejects_dimensions_bounds_and_modified_snapshots():
    a,b=fixture(1,7,3,1277)
    ta,tb=encode(a),encode(b)
    left,right=prepare_fp127_left(ta),prepare_fp127_weight(tb)
    left.digits.add_(1)
    with pytest.raises(ValueError,match='modified'):field_matmul_fp127(left,right)
    right.sums.add_(1)
    with pytest.raises(ValueError,match='modified'):field_matmul_fp127(ta,right)
    with pytest.raises(ValueError,match='8192'):
        prepare_fp127_left(torch.zeros((1,MAX_K+1,8),device='cuda',dtype=torch.int32))
    with pytest.raises(ValueError,match='limbs'):
        field_matmul_fp127(ta[...,:7],tb)
    with pytest.raises(ValueError,match='dimensions'):
        field_matmul_fp127(ta,encode(b[:-1]))
