"""Packing/sign tests for the experimental adapter; no native jobs are launched."""
import secrets
from types import MethodType

from rep4gpu.matrix_spdz_reference import Matrix,MatrixParty,random_matrix,PRIME
from rep4gpu.matrix_native_ole import ExperimentalNativeOle
from rep4gpu.transport import IdealNetwork


def _local_provider(width=2,columns=1):
    party=MatrixParty(0,IdealNetwork().endpoint(0),random_matrix(width,columns),session=secrets.token_bytes(32),
        statistical_security_bits=40,_allow_insecure_test_transport=True)
    provider=object.__new__(ExperimentalNativeOle);provider.party=party
    from rep4gpu.matrix_spdz_reference import BackendEvidence
    provider.evidence=BackendEvidence('unit layout fixture',True,True,False)
    return provider


def test_authentication_concatenates_data_and_masks_under_each_fixed_scalar():
    provider=_local_provider(columns=2)
    values=tuple(Matrix.from_rows(rows) for rows in ([[1,2],[3,4]],[[5,6],[7,8]],[[13,14],[15,16]]))
    captured={}
    def pairs(self,**kwargs):
        captured.update(kwargs)
        vectors=kwargs['vectors']
        return {1:vectors},{2:tuple(tuple(2*x%PRIME for x in vector) for vector in vectors)}
    provider._pairs=MethodType(pairs,provider)
    result=provider.authenticate_columns(0,values,provider.party.key,epoch=provider.party.session,batch_id=secrets.token_bytes(32))
    assert captured['length']==6 and captured['calls']==4
    assert captured['vectors']==((1,3,5,7,13,15),(1,3,5,7,13,15),(2,4,6,8,14,16),(2,4,6,8,14,16))
    assert captured['scalars']==provider.party.key.data
    for matrix,u,w in zip(values,result.outgoing,result.incoming):
        sums=[sum(row) for row in matrix.to_rows()]
        assert u.to_rows()==[[value,value] for value in sums]
        assert w.to_rows()==[[2*value,2*value] for value in sums]
    assert result.evidence.test_only and not result.evidence.privacy_reviewed


def test_candidate_outer_products_preserve_role_local_signs_and_row_blocks(monkeypatch):
    provider=_local_provider();m=2
    ahat=Matrix.from_rows([[1,2],[3,4],[5,6],[7,8]])
    b=Matrix.from_rows([[2,1],[4,3]]);mask=Matrix.from_rows([[3,4],[5,6]])
    generated=iter((ahat,b,mask));monkeypatch.setattr('rep4gpu.matrix_native_ole.random_matrix',lambda *shape:next(generated))
    peer_a={p:[[x+10*p for x in row] for row in ahat.to_rows()] for p in (1,2,3)}
    peer_b={p:[[x+p for x in row] for row in b.to_rows()] for p in (1,2,3)}
    def pairs(self,**kwargs):
        assert kwargs['length']==4 and kwargs['calls']==4
        outgoing={p:tuple(tuple(ahat.to_rows()[row][k]*peer_b[p][k][col]%PRIME for row in range(4)) for k in range(m) for col in range(m)) for p in (1,2,3)}
        incoming={p:tuple(tuple(-peer_a[p][row][k]*b.to_rows()[k][col]%PRIME for row in range(4)) for k in range(m) for col in range(m)) for p in (1,2,3)}
        return outgoing,incoming
    provider._pairs=MethodType(pairs,provider)
    candidate=provider.candidate(secrets.token_bytes(32))
    expected=[[sum(ahat.to_rows()[row][k]*b.to_rows()[k][col] for k in range(m))+
               sum(ahat.to_rows()[row][k]*peer_b[p][k][col]+peer_a[p][row][k]*b.to_rows()[k][col] for p in (1,2,3) for k in range(m))
               for col in range(m)] for row in range(4)]
    assert candidate.c.to_rows()==expected[:2]
    assert candidate.sacrifices[0][1].to_rows()==expected[2:]
    assert candidate.a.to_rows()==ahat.to_rows()[:2]
    assert candidate.sacrifices[0][0].to_rows()==ahat.to_rows()[2:]
