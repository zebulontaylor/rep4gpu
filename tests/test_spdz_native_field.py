"""Cross-field checks for the experimental HE backend's online adapter."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import random

import pytest

from rep4gpu.spdz_prime256 import GOLDILOCKS_PRIME,LOWGEAR_PRIME,PRIME,Prime256Party,QShare,_pack,_unpack
from rep4gpu.transport import IdealNetwork


def run_field_case(prime,*,fault=None):
    rng=random.Random(20261110)
    keys=[rng.randrange(prime) for _ in range(4)]
    def share(values,shape):
        columns=[]
        for clear in (values,[sum(keys)*v%prime for v in values]):
            parts=[[rng.randrange(prime) for _ in clear] for _ in range(3)]
            parts.append([(v-sum(row[j] for row in parts))%prime for j,v in enumerate(clear)])
            columns.append(parts)
        return [QShare(columns[0][i],columns[1][i],shape) for i in range(4)]
    x=[prime-1,2,7,prime-9];y=[4,prime-3,11,5]
    a=[rng.randrange(prime) for _ in x];b=[rng.randrange(prime) for _ in y]
    def multiply(left,right):
        return [sum(left[2*i+k]*right[2*k+j] for k in range(2))%prime for i in range(2) for j in range(2)]
    xs,ys,aa,bb,cc=[share(values,(2,2)) for values in (x,y,a,b,multiply(a,b))]
    sq=[rng.randrange(prime) for _ in x]
    ss,sc=share(sq,(2,2)),share([v*v%prime for v in sq],(2,2))
    if fault=='mac':xs[0].macs[0]=(xs[0].macs[0]+1)%prime
    if fault=='value':xs[0].values[0]=prime
    network=IdealNetwork(timeout=3)
    parties=[Prime256Party(i,network.endpoint(i),keys[i],session=b'native-fields',field_prime=prime,
                          _allow_insecure_test_transport=True) for i in range(4)]
    if fault=='different-field':
        parties[3]=Prime256Party(3,network.endpoint(3),keys[3],session=b'native-fields',field_prime=PRIME,
                                _allow_insecure_test_transport=True)
    def worker(i):
        try:
            triple=SimpleNamespace(a=aa[i],b=bb[i],c=cc[i],identifier=b't'*32,used=False)
            square=SimpleNamespace(a=ss[i],a_squared=sc[i],identifier=b's'*32,used=False)
            result=parties[i].matmul(xs[i],ys[i],triple)
            result=parties[i].add_public(result,[1,prime-1,13,0])
            return parties[i].open(parties[i].square(result,square))
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(worker,range(4)))
    expected=[(v+c)**2%prime for v,c in zip(multiply(x,y),[1,prime-1,13,0])]
    return results,parties,expected


@pytest.mark.parametrize('prime',[PRIME,LOWGEAR_PRIME,GOLDILOCKS_PRIME])
def test_supported_native_fields_match_integer_algebra(prime):
    results,parties,expected=run_field_case(prime)
    assert results==[expected]*4
    assert all(not party.poisoned for party in parties)


@pytest.mark.parametrize('fault',['mac','value','different-field'])
@pytest.mark.parametrize('prime',[LOWGEAR_PRIME,GOLDILOCKS_PRIME])
def test_lowgear_field_rejects_bad_openings_and_cross_field_sessions(fault,prime):
    results,parties,_=run_field_case(prime,fault=fault)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(party.poisoned for party in parties)


@pytest.mark.parametrize('prime,width',[(PRIME,32),(LOWGEAR_PRIME,32),(GOLDILOCKS_PRIME,8)])
def test_wire_width_and_canonical_boundaries(prime,width):
    values=[0,1,prime-1]
    encoded=_pack(values,prime)
    assert encoded==b''.join(v.to_bytes(width,'little') for v in values)
    assert _unpack(encoded,len(values),prime)==values
    with pytest.raises(RuntimeError,match='length'):_unpack(encoded+b'\0',len(values),prime)
    with pytest.raises(RuntimeError,match='noncanonical'):_unpack(prime.to_bytes(width,'little'),1,prime)
    with pytest.raises(ValueError,match='noncanonical'):_pack([prime],prime)


@pytest.mark.parametrize('owner',range(4))
def test_goldilocks_private_input_uses_unsigned_full_width_masks(owner):
    prime=GOLDILOCKS_PRIME
    rng=random.Random(907+owner)
    keys=[rng.randrange(prime) for _ in range(4)]
    mask=[prime-1,2**63+29,0]
    query=[prime-2,7,2**63+19]
    values=[[rng.randrange(prime) for _ in mask] for _ in range(3)]
    values.append([(v-sum(row[j] for row in values))%prime for j,v in enumerate(mask)])
    macs=[[rng.randrange(prime) for _ in mask] for _ in range(3)]
    macs.append([(v*sum(keys)-sum(row[j] for row in macs))%prime for j,v in enumerate(mask)])
    network=IdealNetwork(timeout=3)
    parties=[Prime256Party(i,network.endpoint(i),keys[i],session=b'goldilocks-input',
                          field_prime=prime,_allow_insecure_test_transport=True) for i in range(4)]
    def worker(i):
        try:
            material=SimpleNamespace(share=QShare(values[i],macs[i],(1,3)),owner=owner,
                                     clear=mask if i==owner else None,identifier=b'i'*32,used=False)
            result=parties[i].open(parties[i].input(material,query if i==owner else None))
            assert material.used
            return result
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(worker,range(4)))
    assert results==[query]*4


def test_only_explicit_native_fields_are_accepted():
    network=IdealNetwork()
    with pytest.raises(ValueError,match='unsupported'):
        Prime256Party(0,network.endpoint(0),1,session=b'bad-field',field_prime=2**256-1,
                      _allow_insecure_test_transport=True)
