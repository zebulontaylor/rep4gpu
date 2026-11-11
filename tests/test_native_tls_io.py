"""Real-process mutual TLS checks for the pinned EMP IOChannel adapter."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time

import pytest
from benchmarks.socket_inference import certificates

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def binary():
    path=ROOT/'native/matrix_vope/build_tls_io.py'
    spec=importlib.util.spec_from_file_location('build_matrix_tls_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build()


@pytest.fixture(scope='module')
def identities(tmp_path_factory):
    directory=tmp_path_factory.mktemp('native-mtls')
    ca,roles=certificates(directory)
    rogue=directory/'rogue';rogue.mkdir();rogue_ca,rogue_roles=certificates(rogue)
    return ca,roles,rogue_ca,rogue_roles


def _pair(binary,identities,*,server_mode='transfer',client_mode='transfer',server_peer=None,server_role=0,client_role=1,
          client_session='23'*32,client_channel='47'*32,client_cert=None,timeout_ms=2500):
    ca,roles,rogue_ca,rogue_roles=identities
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    def command(server):
        role=server_role if server else client_role
        cert=roles[role] if server or client_cert is None else client_cert
        return [str(binary),'server' if server else 'client',str(port),str(role),str((client_role if server_peer is None else server_peer) if server else server_role),
                str(ca),str(cert['certificate']),str(cert['key']),'23'*32 if server else client_session,
                server_mode if server else client_mode,str(timeout_ms),'47'*32 if server else client_channel]
    processes=[subprocess.Popen(command(server),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for server in (True,False)]
    started=time.monotonic();results=[]
    try:
        for process in processes:
            out,err=process.communicate(timeout=10)
            results.append(dict(code=process.returncode,out=out,err=err))
    finally:
        for process in processes:
            if process.poll()is None:process.kill();process.wait()
    return results,time.monotonic()-started


def test_tls_two_role_buffered_transfer_and_emp_counters(binary,identities):
    results,_=_pair(binary,identities)
    assert [result['code'] for result in results]==[0,0],results
    for result in results:
        report=json.loads(result['out'])
        assert report['passed'] and report['sent']==report['received']==200004
        assert report['tls_write_calls']<20  # 28,572 small sends coalesce.
        assert report['flushes']<20


@pytest.mark.parametrize('fault',['peer-role','session','channel','untrusted-cert','wrong-local-role'])
def test_tls_identity_and_binding_rejection(binary,identities,fault):
    options={}
    if fault=='peer-role':options['server_peer']=2
    if fault=='session':options['client_session']='24'*32
    if fault=='channel':options['client_channel']='48'*32
    if fault=='untrusted-cert':options['client_cert']=identities[3][1]
    if fault=='wrong-local-role':options['client_cert']=identities[1][2]
    results,_=_pair(binary,identities,timeout_ms=1000,**options)
    assert all(result['code']!=0 for result in results),results
    assert all(result['err'] for result in results)


@pytest.mark.parametrize('mode',['truncated-send','abrupt'])
def test_tls_eof_or_truncation_never_returns_partial_success(binary,identities,mode):
    results,_=_pair(binary,identities,server_mode=mode,client_mode='truncated-receive')
    assert results[0]['code']==0 and results[1]['code']!=0,results
    assert 'EOF' in results[1]['err'] or 'I/O failed' in results[1]['err']


def test_tls_total_receive_deadline(binary,identities):
    results,elapsed=_pair(binary,identities,server_mode='stall',client_mode='wait',timeout_ms=400)
    assert results[0]['code']==0 and results[1]['code']!=0,results
    assert 'timeout' in results[1]['err'] and elapsed<3


def test_tls_connection_deadline_without_a_listener(binary,identities):
    ca,roles,_,_=identities
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=[str(binary),'client',str(port),'1','0',str(ca),str(roles[1]['certificate']),str(roles[1]['key']),
             '23'*32,'transfer','150','47'*32]
    started=time.monotonic();result=subprocess.run(command,capture_output=True,text=True,timeout=3)
    assert result.returncode!=0 and 'timeout' in result.stderr
    assert time.monotonic()-started<2



@pytest.mark.parametrize('roles',[(2,3),(0,2),(1,3)])
def test_all_four_role_identities_are_accepted_only_at_their_assigned_endpoints(binary,identities,roles):
    results,_=_pair(binary,identities,server_role=roles[0],client_role=roles[1])
    assert [result['code'] for result in results]==[0,0],results


def test_tls_receive_deadline_does_not_reset_after_partial_progress(binary,identities):
    results,elapsed=_pair(binary,identities,server_mode='drip',client_mode='truncated-receive',timeout_ms=300)
    assert results[1]['code']!=0 and 'timeout' in results[1]['err'],results
    assert elapsed<3


def test_tls_handshake_deadline_against_a_raw_tcp_peer(binary,identities):
    ca,roles,_,_=identities
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));listener.listen(1);port=listener.getsockname()[1]
        command=[str(binary),'client',str(port),'1','0',str(ca),str(roles[1]['certificate']),str(roles[1]['key']),
                 '23'*32,'transfer','150','47'*32]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        connection,_=listener.accept()
        try:
            out,err=process.communicate(timeout=3)
            assert process.returncode!=0 and 'timeout' in err
        finally:
            connection.close()
            if process.poll()is None:process.kill();process.wait()



def test_tls_ignores_stale_openssl_errors_from_unrelated_emp_arithmetic(binary,identities):
    results,_=_pair(binary,identities,server_mode='error-queue',client_mode='error-queue')
    assert [result['code'] for result in results]==[0,0],results



def test_tls_total_write_deadline_survives_stale_openssl_errors(binary,identities):
    results,elapsed=_pair(binary,identities,server_mode='stall',client_mode='flood',timeout_ms=300)
    assert results[0]['code']==0 and results[1]['code']!=0,results
    assert 'timeout' in results[1]['err'] and elapsed<3
