"""Exercise native OLE over identity- and session-bound mutual TLS."""
from pathlib import Path
import json
import secrets
import socket
import subprocess
import sys

from direct_ole_io import PRIME,ScalarOleRequest,VectorOleRequest,LocalOleBatch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.socket_inference import certificates

OUT=ROOT/'work/direct_ole_tls'
BIN=OUT/'direct_ole_tls'


def run(name,ca,identities,*,fault='',session_mismatch=False,peer_name_mismatch=False):
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    length,calls=8,3
    scalars=tuple(secrets.randbelow(PRIME) for _ in range(calls))
    vectors=tuple(tuple(secrets.randbelow(PRIME) for _ in range(length)) for _ in range(calls))
    ScalarOleRequest(length,scalars).write(folder/'role1.bin')
    VectorOleRequest(vectors).write(folder/'role2.bin')
    session=secrets.token_hex(32)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[];logs=[]
    try:
        for role in (1,2):
            actual=role-1;peer=1-actual
            identity=identities[actual]
            sid=secrets.token_hex(32) if session_mismatch and role==2 else session
            peer_name='party3' if peer_name_mismatch and role==2 else f'party{peer}'
            log=(folder/f'role{role}.log').open('w');logs.append(log)
            command=[str(BIN),str(role),str(port),str(folder/f'role{role}.bin'),str(folder/f'role{role}.json'),str(actual),str(peer),ca,identity['certificate'],identity['key'],peer_name,sid,fault]
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        for process in processes:process.wait(timeout=45)
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait()
        for log in logs:log.close()
    codes=[process.returncode for process in processes]
    if fault or session_mismatch or peer_name_mismatch:
        assert all(code!=0 for code in codes),(name,codes)
        return {'name':name,'rejected':True,'returncodes':codes}
    assert codes==[0,0],(name,codes)
    batches=[LocalOleBatch.read(folder/f'role{role}.json',role=role,length=length,calls=calls,authenticated=True,self_role=role-1,peer_role=2-role,session=session) for role in (1,2)]
    assert all(batch.transport_authenticated for batch in batches)
    for call in range(calls):
        for row in range(length):assert (batches[0].shares[call][row]+batches[1].shares[call][row])%PRIME==scalars[call]*vectors[call][row]%PRIME
    # The typed reader never accepts authenticated output through its plain
    # default, and never accepts a different public session or peer identity.
    for options in ({},{'authenticated':True,'self_role':0,'peer_role':1,'session':'00'*32}):
        try:LocalOleBatch.read(folder/'role1.json',role=1,length=length,calls=calls,**options)
        except ValueError:pass
        else:raise AssertionError('TLS output accepted through wrong contract')
    return {'name':name,'checked_products':length*calls,'tls_identity_session_bound':True,'returncodes':codes}


def main():
    certs=OUT/'certs';certs.mkdir(parents=True,exist_ok=True)
    ca,identities=certificates(certs)
    checks=[run('honest',ca,identities),run('honest_repeated',ca,identities),run('base_fault',ca,identities,fault='response-z'),
            run('session_mismatch',ca,identities,session_mismatch=True),run('peer_name_mismatch',ca,identities,peer_name_mismatch=True)]
    (OUT/'results.json').write_text(json.dumps({'security_reviewed':False,'checks':checks},indent=2)+'\n')
    print(json.dumps({'passed':len(checks),'public_report':str(OUT/'results.json')}))


if __name__=='__main__':main()
