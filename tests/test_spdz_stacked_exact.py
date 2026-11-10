"""Check stacked MAC products against independent Python field arithmetic."""
import pytest
import torch

from rep4gpu.spdz import PRIME
from rep4gpu.spdz_distributed import AuthShare, DistributedParty, MatrixTriple


def _edges(shape, offset):
    values = torch.tensor([0, 1, PRIME-1, PRIME-2, PRIME//2], dtype=torch.int64)
    count = 1
    for size in shape:
        count *= size
    return values[(torch.arange(count)+offset) % len(values)].reshape(shape)


def _python_matmul(a, b):
    left, right = a.tolist(), b.tolist()
    return torch.tensor([[sum(int(left[i][k])*int(right[k][j]) for k in range(a.shape[1])) % PRIME
                          for j in range(b.shape[1])] for i in range(a.shape[0])], dtype=torch.int64)


@pytest.mark.parametrize('shape', [(1,9,7), (3,17,2), (5,4,11)])
@pytest.mark.parametrize('keys', [5,7,10])
@pytest.mark.parametrize('party', [0,3])
def test_stacked_mac_products_match_python_field_oracle(shape, keys, party):
    m,k,n = shape
    runtime = DistributedParty(party, object(), _edges((keys,), 3), session=b'stacked-exact',
                               _allow_insecure_test_transport=True)
    def share(shape, offset):
        return AuthShare(_edges(shape,offset), _edges((keys,*shape),offset+1))
    triple = MatrixTriple(share((m,k),0), share((k,n),2), share((m,n),4))
    d,e = _edges((m,k),1), _edges((k,n),3)
    reference = runtime._matmul_local(triple,d,e)
    runtime.stack_macs = True
    actual = runtime._matmul_local(triple,d,e)
    assert torch.equal(actual.values,reference.values)
    assert torch.equal(actual.macs,reference.macs)
    public = _python_matmul(d,e)
    expected_value = (triple.c.values+_python_matmul(d,triple.b.values)
                      +_python_matmul(triple.a.values,e)+(public if party==0 else 0)).remainder(PRIME)
    assert torch.equal(actual.values,expected_value)
    for key in range(keys):
        expected_mac = (triple.c.macs[key]+_python_matmul(d,triple.b.macs[key])
                        +_python_matmul(triple.a.macs[key],e)
                        +runtime.alpha_share[key]*public).remainder(PRIME)
        assert torch.equal(actual.macs[key],expected_mac)


@pytest.mark.parametrize('keys', [5,10])
@pytest.mark.parametrize('party', [0,3])
def test_elementwise_mac_arithmetic_matches_python_field_oracle(keys, party):
    from rep4gpu.spdz_distributed import ElementwiseTriple

    shape = (3,17)
    runtime = DistributedParty(party, object(), _edges((keys,), 3), session=b'elementwise-exact',
                               _allow_insecure_test_transport=True)
    def share(offset):
        return AuthShare(_edges(shape,offset),_edges((keys,*shape),offset+1))
    triple = ElementwiseTriple(share(0),share(2),share(4),identifier=bytes(32))
    d,e = _edges(shape,1),_edges(shape,3)
    runtime.open_many = lambda shares, *, label: (d,e)
    actual = runtime.mul(share(1),share(3),triple)
    assert triple.used
    assert triple.identifier in runtime.spent_triples
    for row in range(shape[0]):
        for col in range(shape[1]):
            index = (row,col)
            dd,ee = int(d[index]),int(e[index])
            public = dd*ee % PRIME
            expected = (int(triple.c.values[index])+dd*int(triple.b.values[index])
                        +int(triple.a.values[index])*ee+(public if party==0 else 0)) % PRIME
            assert int(actual.values[index]) == expected
            for key in range(keys):
                mac_index = (key,*index)
                expected_mac = (int(triple.c.macs[mac_index])+dd*int(triple.b.macs[mac_index])
                                +int(triple.a.macs[mac_index])*ee
                                +int(runtime.alpha_share[key])*public) % PRIME
                assert int(actual.macs[mac_index]) == expected_mac
