"""Independent bigint algebra and fault checks for the q256 reference."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import secrets
import pytest
from rep4gpu.spdz_prime256 import PRIME,QShare,Prime256Party,clear_matmul
from rep4gpu.transport import IdealNetwork


def _share(values,shape,alpha):
    coordinates=[]
    for vector in (values,[(sum(alpha)*x)%PRIME for x in values]):
        rows=[[secrets.randbelow(PRIME) for _ in vector] for _ in range(3)]
        rows.append([(x-sum(row[i] for row in rows))%PRIME for i,x in enumerate(vector)])
        coordinates.append(rows)
    return [QShare(coordinates[0][p],coordinates[1][p],shape) for p in range(4)]


def _fixture():
    alpha=[secrets.randbelow(PRIME) for _ in range(4)]
    x=[0,PRIME-1,2,PRIME-2,7,13,19,31];w=list(range(16));z=[(i*7)%PRIME for i in range(16)]
    xs,ws,zs=_share(x,(1,8),alpha),_share(w,(8,2),alpha),_share(z,(2,8),alpha)
    def triple(m,k,n):
        a=[secrets.randbelow(PRIME) for _ in range(m*k)];b=[secrets.randbelow(PRIME) for _ in range(k*n)]
        aa,bb,cc=_share(a,(m,k),alpha),_share(b,(k,n),alpha),_share(clear_matmul(a,b,(m,k),(k,n)),(m,n),alpha)
        identifier=secrets.token_bytes(32)
        return [SimpleNamespace(a=aa[p],b=bb[p],c=cc[p],used=False,identifier=identifier) for p in range(4)]
    t1,t2=triple(1,8,2),triple(1,2,8)
    a=[secrets.randbelow(PRIME) for _ in range(2)]
    aa,cc=_share(a,(1,2),alpha),_share([v*v%PRIME for v in a],(1,2),alpha)
    identifier=secrets.token_bytes(32)
    square=[SimpleNamespace(a=aa[p],a_squared=cc[p],used=False,identifier=identifier) for p in range(4)]
    masks=[]
    for owner,shape in ((0,(1,8)),(1,(8,2)),(1,(2,8))):
        clear=[secrets.randbelow(PRIME) for _ in range(shape[0]*shape[1])]
        ss=_share(clear,shape,alpha);identifier=secrets.token_bytes(32)
        masks.append([SimpleNamespace(share=ss[p],owner=owner,clear=clear if p==owner else None,used=False,identifier=identifier) for p in range(4)])
    return alpha,(x,w,z),(xs,ws,zs),(t1,t2,square),masks


def _run(operation,*,mutate=None):
    alpha,clear,shares,corr,masks=_fixture()
    if mutate:mutate(shares,corr)
    network=IdealNetwork(timeout=10)
    parties=[Prime256Party(p,network.endpoint(p),alpha[p],session=b'q256-independent',_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return operation(parties[p],[v[p] for v in shares],[v[p] for v in corr],[v[p] for v in masks],clear)
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    return results,parties,clear


def test_q256_private_square_network_matches_python_oracle():
    def operation(p,shares,corr,masks,clear):
        x,w,z=[p.input(mask,values if p.party==mask.owner else None) for mask,values in zip(masks,clear)]
        y=p.matmul(x,w,corr[0]);y=p.add_public(y,[11,PRIME-1]);y=p.square(y,corr[2])
        return p.open(p.add_public(p.matmul(y,z,corr[1]),list(range(8))))
    results,parties,clear=_run(operation)
    hidden=clear_matmul(clear[0],clear[1],(1,8),(8,2))
    hidden=[((v+b)%PRIME)**2%PRIME for v,b in zip(hidden,[11,PRIME-1])]
    expected=[(v+b)%PRIME for v,b in zip(clear_matmul(hidden,clear[2],(1,2),(2,8)),range(8))]
    assert results==[expected]*4
    assert all(not p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['value','mac','noncanonical','identifier'])
def test_q256_bad_opening_rejected_before_local_matmul(fault,monkeypatch):
    def mutate(shares,corr):
        if fault=='value':shares[0][0].values[0]=(shares[0][0].values[0]+1)%PRIME
        if fault=='mac':shares[0][0].macs[0]=(shares[0][0].macs[0]+1)%PRIME
        if fault=='noncanonical':shares[0][0].values[0]=PRIME
        if fault=='identifier':corr[0][0].identifier=bytes(32)
    def forbidden(*args):raise AssertionError('unchecked D/E reached local matrix arithmetic')
    monkeypatch.setattr('rep4gpu.spdz_prime256.clear_matmul',forbidden)
    def operation(p,shares,corr,masks,clear):return p.matmul(shares[0],shares[1],corr[0])
    results,parties,_=_run(operation,mutate=mutate)
    assert all(isinstance(result,BaseException) and not isinstance(result,AssertionError) for result in results)
    assert all(p.poisoned for p in parties)


def test_q256_correlation_reuse_poison():
    import threading
    ready=threading.Barrier(4)
    def operation(p,shares,corr,masks,clear):
        p.open(p.matmul(shares[0],shares[1],corr[0]))
        ready.wait(timeout=10)
        return p.matmul(shares[0],shares[1],corr[0])
    results,parties,_=_run(operation)
    assert all(isinstance(result,ValueError) for result in results)
    assert all(p.poisoned for p in parties)


@pytest.mark.parametrize('phase',[0,2,4])
def test_q256_consistent_wire_mutation_rejected_before_local_product(phase,monkeypatch):
    def forbidden(*args):raise AssertionError('unchecked wire reached local product')
    monkeypatch.setattr('rep4gpu.spdz_prime256.clear_matmul',forbidden)
    def operation(p,shares,corr,masks,clear):
        exchange=p.transport.exchange
        index=0
        def fault(outgoing):
            nonlocal index
            current=index;index+=1
            if p.party==0 and current==phase:
                changed={}
                for peer,frame in outgoing.items():
                    frame=bytearray(frame)
                    if phase==0:frame[93:125]=PRIME.to_bytes(32,'little')
                    else:frame[-1]^=1
                    changed[peer]=bytes(frame)
                outgoing=changed
            return exchange(outgoing)
        p.transport.exchange=fault
        return p.matmul(shares[0],shares[1],corr[0])
    results,parties,_=_run(operation)
    assert all(isinstance(result,BaseException) and not isinstance(result,AssertionError) for result in results)
    assert all(p.poisoned for p in parties)
