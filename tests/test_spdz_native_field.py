"""Cross-field checks for the experimental HE backend's online adapter."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import random

import pytest

from rep4gpu.spdz_prime256 import LOWGEAR_PRIME,PRIME,Prime256Party,QShare
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


@pytest.mark.parametrize('prime',[PRIME,LOWGEAR_PRIME])
def test_supported_native_fields_match_integer_algebra(prime):
    results,parties,expected=run_field_case(prime)
    assert results==[expected]*4
    assert all(not party.poisoned for party in parties)


@pytest.mark.parametrize('fault',['mac','value','different-field'])
def test_lowgear_field_rejects_bad_openings_and_cross_field_sessions(fault):
    results,parties,_=run_field_case(LOWGEAR_PRIME,fault=fault)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(party.poisoned for party in parties)


def test_only_explicit_native_fields_are_accepted():
    network=IdealNetwork()
    with pytest.raises(ValueError,match='unsupported'):
        Prime256Party(0,network.endpoint(0),1,session=b'bad-field',field_prime=2**256-1,
                      _allow_insecure_test_transport=True)
