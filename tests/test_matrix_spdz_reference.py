"""Independent matrix algebra and malicious-fault tests; trusted fixtures only."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from rep4gpu.matrix_spdz_reference import (
    PRIME,Matrix,AuthMatrix,BackendEvidence,ColumnVoleBatch,MatrixParty,
    TrustedTestFixture,TrustedTestColumnVole,add,transpose,matmul,
)
from rep4gpu.spdz_distributed import ProtocolError
from rep4gpu.transport import IdealNetwork


def _change(matrix):
    return Matrix(matrix.rows,matrix.cols,((matrix.data[0]+1)%PRIME,*matrix.data[1:]))


def _run(action,*,width=2,columns=2,bits=128,budget=1<<32,mutate=None,pair_override=None,backend_factory=None):
    fixture=TrustedTestFixture(width,columns)
    candidates=list(fixture.candidates())
    if mutate:candidates[0]=mutate(candidates[0])
    network=IdealNetwork(timeout=5)
    backend=backend_factory() if backend_factory else TrustedTestColumnVole(pair_override=pair_override,timeout=5)
    parties=[MatrixParty(p,network.endpoint(p),fixture.keys[p],session=fixture.epoch,
              statistical_security_bits=bits,statistical_check_budget=budget,_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return action(parties[p],candidates[p],backend,fixture)
        except BaseException as error:
            backend.abort(error);network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    return results,parties,fixture


@pytest.mark.parametrize('width',[1,2,3])
def test_full_authentication_sacrifice_transposes_private_inputs_and_online(width):
    x=Matrix.from_rows([[i*width+j+1 for j in range(width)] for i in range(width)])
    y=Matrix.from_rows([[(i+2)*(j+3)+int(i==j) for j in range(width)] for i in range(width)])
    expected=Matrix.from_rows([[sum(x.to_rows()[i][k]*y.to_rows()[k][j] for k in range(width)) for j in range(width)] for i in range(width)])
    # Create owner-only masks once, outside the parallel party operations.
    masks={}
    lock=threading.Lock()
    def action(p,candidate,backend,fixture):
        with lock:
            if not masks:masks.update(x=fixture.input_masks(0),y=fixture.input_masks(1))
        sx=p.input(masks['x'][p.party],x if p.party==0 else None)
        sy=p.input(masks['y'][p.party],y if p.party==1 else None)
        sextuple=p.prepare_sextuple(candidate,backend)
        assert sextuple.test_only and not sextuple.used
        output=p.open(p.multiply(sx,sy,sextuple))
        assert sextuple.used
        assert p.statistical_accounting()['bound']<1/(1<<128)
        return output
    results,parties,_=_run(action,width=width)
    assert results==[expected]*4
    assert all(not p.poisoned for p in parties)
    if width>1:assert matmul(x,y)!=matmul(y,x)


@pytest.mark.parametrize('fault',['c','at','rt','sacrifice_c'])
def test_mac_consistent_wrong_candidate_relations_rejected(fault):
    def mutate(candidate):
        if fault=='sacrifice_c':
            pairs=list(candidate.sacrifices);pairs[0]=(pairs[0][0],_change(pairs[0][1]));return replace(candidate,sacrifices=tuple(pairs))
        return replace(candidate,**{fault:_change(getattr(candidate,fault))})
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),mutate=mutate)
    assert all(isinstance(result,BaseException) for result in results)
    messages=' '.join(str(result) for result in results)
    assert ('transpose relation' if fault in ('at','rt') else 'sacrifice failed') in messages
    assert all(p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['chosen_vector','chosen_key'])
def test_pairwise_valid_but_cross_peer_inconsistent_authentication_rejected(fault):
    def override(sender,receiver,index,h,k,vector,delta):
        if (sender,receiver,index,h,k)==(0,1,0,0,0):
            if fault=='chosen_vector':vector=((vector[0]+1)%PRIME,*vector[1:])
            else:delta=(delta+1)%PRIME
        return vector,delta
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),pair_override=override)
    assert all(isinstance(result,BaseException) for result in results)
    assert any('MAC check failed' in str(result) for result in results)
    assert all(p.poisoned for p in parties)


@pytest.mark.parametrize('fault',['value','tag'])
def test_bad_online_opening_never_releases_output(fault):
    shares={};lock=threading.Lock()
    def action(p,c,b,f):
        sextuple=p.prepare_sextuple(c,b)
        with lock:
            if not shares:
                shares['x']=list(f.authenticated(Matrix.from_rows([[1,2],[3,4]])))
                shares['y']=f.authenticated(Matrix.from_rows([[5,6],[7,8]]))
                s=shares['x'][0];shares['x'][0]=replace(s,**{fault:_change(getattr(s,fault))})
        return p.open(p.multiply(shares['x'][p.party],shares['y'][p.party],sextuple))
    results,parties,_=_run(action)
    assert all(isinstance(result,BaseException) for result in results)
    assert any('MAC check failed' in str(result) for result in results)
    assert all(p.poisoned for p in parties)


def test_one_use_sextuple_guard_after_successful_output():
    barrier=threading.Barrier(4)
    def action(p,c,b,f):
        sextuple=p.prepare_sextuple(c,b);x=p.public(Matrix.from_rows([[1,2],[3,4]]))
        p.open(p.multiply(x,x,sextuple));barrier.wait(timeout=5)
        return p.multiply(x,x,sextuple)
    results,parties,_=_run(action)
    assert all(isinstance(result,ProtocolError) and 'single-use' in str(result) for result in results)
    assert all(p.poisoned for p in parties)


def test_unverified_lower_layer_is_rejected_before_authentication_openings():
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),backend_factory=lambda:TrustedTestColumnVole(verified=False))
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.round==0 for p in parties)


@pytest.mark.parametrize('provider',['current direct COPE','current candidate spVOPE'])
def test_non_test_candidate_privacy_gate_rejects_unreviewed_backends(provider):
    evidence=BackendEvidence(provider,True,False,False)
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),mutate=lambda c:replace(c,evidence=evidence))
    assert any('privacy gate' in str(result) for result in results)
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.poisoned for p in parties)


def test_statistical_configuration_counts_whole_check_budget_and_binds_epoch():
    fixture=TrustedTestFixture(1,1);network=IdealNetwork()
    party=MatrixParty(0,network.endpoint(0),fixture.keys[0],session=fixture.epoch,
       statistical_security_bits=40,statistical_check_budget=1<<32,_allow_insecure_test_transport=True)
    assert party.columns==1
    with pytest.raises(ValueError,match='insufficient'):
        MatrixParty(1,network.endpoint(1),fixture.keys[1],session=fixture.epoch,
          statistical_security_bits=128,_allow_insecure_test_transport=True)
    with pytest.raises(ProtocolError,match='budget'):
        party._record_checks((1<<32)+1)


def test_field_canonicality_and_nonlocal_linear_operations():
    with pytest.raises(ValueError):Matrix(1,1,(PRIME,))
    x=Matrix.from_rows([[1,2],[3,4]]);v=Matrix.from_rows([[5],[7]])
    assert transpose(matmul(x,v)).rows==1
    assert matmul(transpose(x),v).rows==2


def test_standard_40_bit_configuration_with_one_matrix_mac_column():
    def action(p,c,b,f):
        sextuple=p.prepare_sextuple(c,b)
        x=p.public(Matrix.from_rows([[1,2],[3,4]]))
        return p.open(p.multiply(x,x,sextuple))
    results,parties,_=_run(action,columns=1,bits=40)
    assert results==[Matrix.from_rows([[7,10],[15,22]])]*4
    assert all(p.statistical_accounting()['bound']<1/(1<<40) for p in parties)


def test_statistical_configuration_is_bound_into_wire_session():
    fixture=TrustedTestFixture(1,2);network=IdealNetwork(timeout=5)
    parties=[MatrixParty(p,network.endpoint(p),fixture.keys[p],session=fixture.epoch,
        statistical_security_bits=40 if p==0 else 128,_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return p.open(p.public(Matrix.from_rows([[7]])))
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,parties))
    assert all(isinstance(result,BaseException) for result in results)
    assert all(p.poisoned for p in parties)


def test_authentication_batch_identifier_is_single_use():
    barrier=threading.Barrier(4)
    def action(p,c,b,f):
        p.authenticate((c.a,),b,batch_id=c.identifier)
        barrier.wait(timeout=5)
        return p.authenticate((c.b,),TrustedTestColumnVole(),batch_id=c.identifier)
    results,parties,_=_run(action)
    assert all(isinstance(result,ProtocolError) and 'single-use' in str(result) for result in results)
    assert all(p.poisoned for p in parties)


def test_ragged_public_matrix_rejected():
    with pytest.raises(ValueError):Matrix.from_rows([[1,2],[3],[4,5,6]])


def test_residual_reveal_cannot_change_after_commitment():
    class FaultingParty(MatrixParty):
        def broadcast(self,payload,*,label):
            if self.party==0 and label.endswith(b'/residual/reveal'):
                payload=payload[:32]+bytes([payload[32]^1])+payload[33:]
            return super().broadcast(payload,label=label)
    fixture=TrustedTestFixture(1,2);network=IdealNetwork(timeout=5)
    shares=fixture.authenticated(Matrix.from_rows([[9]]))
    parties=[FaultingParty(p,network.endpoint(p),fixture.keys[p],session=fixture.epoch,_allow_insecure_test_transport=True) for p in range(4)]
    def role(p):
        try:return p.open(shares[p.party])
        except BaseException as error:network.abort(error);return error
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,parties))
    assert all(isinstance(result,BaseException) for result in results)
    assert any('commitment opening mismatch' in str(result) for result in results)
    assert all(p.poisoned for p in parties)


def test_four_role_mutually_authenticated_tls_matrix_pipeline(tmp_path):
    import ssl
    from benchmarks.socket_inference import certificates,reserve_addresses
    from rep4gpu.transport import SocketTransport
    ca,identities=certificates(tmp_path);addresses=reserve_addresses()
    fixture=TrustedTestFixture(1,1);candidates=fixture.candidates();backend=TrustedTestColumnVole(timeout=10)
    def role(index):
        identity=identities[index]
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(identity['certificate'],identity['key']);server.load_verify_locations(ca)
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=ca)
        client.load_cert_chain(identity['certificate'],identity['key'])
        transport=SocketTransport(index,[tuple(address) for address in addresses],timeout=10,server_context=server,client_context=client,
            peer_hostnames={peer:f'party{peer}' for peer in range(4)})
        party=MatrixParty(index,transport,fixture.keys[index],session=fixture.epoch,statistical_security_bits=40)
        try:
            sextuple=party.prepare_sextuple(candidates[index],backend)
            result=party.open(party.multiply(party.public(Matrix.from_rows([[7]])),party.public(Matrix.from_rows([[11]])),sextuple))
            assert transport.authenticated
            return result
        except BaseException as error:party.abort(error);backend.abort(error);raise
        finally:transport.close()
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(role,range(4)))
    assert results==[Matrix.from_rows([[77]])]*4


def test_review_flag_alone_cannot_enable_an_unimplemented_private_provider():
    evidence=BackendEvidence('unimplemented private provider',True,False,True)
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),mutate=lambda c:replace(c,evidence=evidence))
    assert all(isinstance(result,BaseException) for result in results)
    assert any('no reviewed private' in str(result) for result in results)


def test_exhausted_open_budget_aborts_before_any_output_frame():
    def action(p,c,b,f):return p.open(p.public(Matrix.from_rows([[7]])))
    results,parties,_=_run(action,width=1,columns=1,bits=40,budget=1)
    assert all(isinstance(result,ProtocolError) and 'budget' in str(result) for result in results)
    assert all(p.round==0 and p.poisoned for p in parties)


def test_insufficient_compound_preprocessing_budget_aborts_before_any_frame():
    # One-column preparation needs auth2 + sacrifice4 + transpose2 + relation2.
    results,parties,_=_run(lambda p,c,b,f:p.prepare_sextuple(c,b),width=1,columns=1,bits=40,budget=9)
    assert all(isinstance(result,ProtocolError) and 'budget' in str(result) for result in results)
    assert all(p.round==0 and p.poisoned for p in parties)


def test_mac_epoch_and_key_allocation_cannot_be_reset_by_new_runtime():
    fixture=TrustedTestFixture(1,1)
    MatrixParty(0,IdealNetwork().endpoint(0),fixture.keys[0],session=fixture.epoch,
                statistical_security_bits=40,_allow_insecure_test_transport=True)
    with pytest.raises(ProtocolError,match='already allocated'):
        MatrixParty(0,IdealNetwork().endpoint(0),fixture.keys[0],session=fixture.epoch,
                    statistical_security_bits=40,_allow_insecure_test_transport=True)
    with pytest.raises(ProtocolError,match='already allocated'):
        MatrixParty(0,IdealNetwork().endpoint(0),fixture.keys[0],session=b'new-session-same-key',
                    statistical_security_bits=40,_allow_insecure_test_transport=True)
