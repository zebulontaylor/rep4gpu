import random
import numpy as np
import pytest
import torch

from rep4gpu.field_q256 import PRIME,field_matmul_q256
from rep4gpu.field_q256_bytes import field_matmul_q256_bytes as field_matmul_q256_tensor
from test_field_q256 import encode,decode

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('m,k,n',[(1,0,1),(1,1,1),(1,7,19),(2,33,5),(1,129,33)])
@pytest.mark.parametrize('compact',[False,True])
def test_q256_tensor_bigint(m,k,n,compact):
    rng=random.Random(257190+m+k+n)
    a=np.asarray([[rng.randrange(PRIME) for _ in range(k)] for _ in range(m)],dtype=object).reshape(m,k)
    b=np.asarray([[rng.randrange(PRIME) for _ in range(n)] for _ in range(k)],dtype=object).reshape(k,n)
    ta,tb=encode(a),encode(b)
    if compact:ta,tb=ta.to(torch.uint16),tb.to(torch.uint16)
    actual=decode(field_matmul_q256_tensor(ta,tb))
    np.testing.assert_array_equal(actual,(a@b)%PRIME)


@pytest.mark.parametrize('k',[1,257,4096])
def test_q256_tensor_maximum_carry(k):
    a=np.full((1,k),PRIME-1,dtype=object)
    b=np.full((k,3),PRIME-1,dtype=object)
    actual=decode(field_matmul_q256_tensor(encode(a),encode(b)))
    np.testing.assert_array_equal(actual,np.full((1,3),k,dtype=object))


@pytest.mark.parametrize('compact',[False,True])
def test_prepared_weight_is_owned_exact_snapshot(compact):
    from rep4gpu.field_q256_bytes import prepare_q256_bytes as prepare_q256_weight
    rng=random.Random(189256)
    a=np.asarray([[rng.randrange(PRIME) for _ in range(71)]],dtype=object)
    b=np.asarray([[rng.randrange(PRIME) for _ in range(19)] for _ in range(71)],dtype=object)
    ta,tb=encode(a),encode(b)
    if compact:ta,tb=ta.to(torch.uint16),tb.to(torch.uint16)
    prepared=prepare_q256_weight(tb)
    assert set(vars(prepared))=={'digits','sums','k','n'}
    assert prepared.digits.dtype==torch.int8
    assert prepared.digits.numel()==32*71*19
    tb.zero_()
    np.testing.assert_array_equal(decode(field_matmul_q256_tensor(ta,prepared)),(a@b)%PRIME)


def test_byte_centering_zero_and_extremes():
    edges=np.asarray([0,1,127,128,255,256,65535,65536,int.from_bytes(bytes([128])*32,'little'),PRIME-1],dtype=object)
    a,b=edges.reshape(-1,1),edges.reshape(1,-1)
    np.testing.assert_array_equal(decode(field_matmul_q256_tensor(encode(a),encode(b))),(a@b)%PRIME)
