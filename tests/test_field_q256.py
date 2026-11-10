import random
import numpy as np
import pytest
import torch

from rep4gpu.field_q256 import PRIME,field_matmul_q256

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def encode(values):
    values=np.asarray(values,dtype=object)
    limbs=np.stack([np.asarray((values>>(16*i))&65535,dtype=np.int32) for i in range(16)],axis=-1)
    return torch.from_numpy(limbs).cuda()


def decode(tensor):
    limbs=tensor.cpu().numpy().astype(object)
    assert np.all((limbs>=0)&(limbs<65536))
    return sum(limbs[...,i]<<(16*i) for i in range(16))


@pytest.mark.parametrize('m,k,n',[(1,0,1),(1,1,1),(1,3,17),(3,7,19),(2,33,5)])
def test_q256_big_integer(m,k,n):
    rng=random.Random(256189+m+k+n)
    a=np.asarray([[rng.randrange(PRIME) for _ in range(k)] for _ in range(m)],dtype=object).reshape(m,k)
    b=np.asarray([[rng.randrange(PRIME) for _ in range(n)] for _ in range(k)],dtype=object).reshape(k,n)
    actual=decode(field_matmul_q256(encode(a),encode(b)))
    expected=(a@b)%PRIME
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('k',[1,17,4096])
def test_q256_carry_boundaries(k):
    a=np.full((1,k),PRIME-1,dtype=object)
    b=np.full((k,3),PRIME-1,dtype=object)
    actual=decode(field_matmul_q256(encode(a),encode(b)))
    np.testing.assert_array_equal(actual,np.full((1,3),k%PRIME,dtype=object))


def test_q256_zero_one_and_modulus_boundary():
    edge=np.asarray([0,1,65535,65536,PRIME-189,PRIME-1],dtype=object)
    a=edge.reshape(-1,1)
    b=edge.reshape(1,-1)
    np.testing.assert_array_equal(decode(field_matmul_q256(encode(a),encode(b))),(a@b)%PRIME)
