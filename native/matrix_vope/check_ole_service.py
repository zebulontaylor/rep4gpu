"""Check persistent role-private OLE streams, fresh correlations, and retirement."""
from pathlib import Path
import json
import os
import secrets
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.socket_inference import certificates
from native.matrix_vope.direct_ole_io import PRIME,ScalarOleRequest,VectorOleRequest
from native.matrix_vope.ole_service_io import ReplyReader,send_frame,check_ready,check_complete,LocalServiceOutput

OUT=ROOT/'work/ole_service';BIN=OUT/'ole_service'


class Peers:
    def __init__(self,name,ca,identities,fault=''):
        self.directory=OUT/name;self.directory.mkdir(parents=True,exist_ok=True)
        self.session=secrets.token_hex(32);self.sequence=0;self.rot=0;self.calls=0
        self.processes=[];self.logs=[];self.readers=[];self.metrics=[]
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        try:
            for ole_role in (1,2):
                role=ole_role-1;peer=1-role;identity=identities[role]
                log=(self.directory/f'role{ole_role}.log').open('w');self.logs.append(log)
                command=[str(BIN),str(ole_role),str(port),str(role),str(peer),ca,identity['certificate'],identity['key'],f'party{peer}',self.session,fault]
                process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
                self.processes.append(process);self.readers.append(ReplyReader(process.stdout))
            self.ready=[check_ready(reader.read(45),ole_role=index+1,self_role=index,peer_role=1-index,session=self.session) for index,reader in enumerate(self.readers)]
        except BaseException:self.cleanup();raise
    def inputs(self,length,calls):
        scalars=tuple(secrets.randbelow(PRIME) for _ in range(calls))
        vector=tuple(secrets.randbelow(PRIME) for _ in range(length))
        vectors=tuple(vector for _ in range(calls))
        return scalars,vectors
    def send(self,length,calls,*,sequences=None,identifiers=None):
        scalars,vectors=self.inputs(length,calls)
        identifier=secrets.token_bytes(32)
        identifiers=identifiers or [identifier,identifier]
        sequences=sequences or [self.sequence,self.sequence]
        paths=[]
        for index,process in enumerate(self.processes):
            request=self.directory/f'request{self.sequence}-role{index+1}.bin'
            output=self.directory/f'output{self.sequence}-role{index+1}.json'
            if index==0:ScalarOleRequest(length,scalars).write(request)
            else:VectorOleRequest(vectors).write(request)
            send_frame(process.stdin,sequence=sequences[index],request_id=identifiers[index],length=length,calls=calls,input_path=request,output_path=output)
            paths.append(output)
        return identifiers,paths,scalars,vectors
    def request(self,length,calls):
        identifiers,paths,scalars,vectors=self.send(length,calls)
        replies=[check_complete(reader.read(45),sequence=self.sequence,request_id=identifiers[index],length=length,calls=calls,start_rot_choices=self.rot) for index,reader in enumerate(self.readers)]
        outputs=[LocalServiceOutput.read(paths[index],ole_role=index+1,self_role=index,peer_role=1-index,session=self.session,sequence=self.sequence,request_id=identifiers[index],length=length,calls=calls,start_rot_choices=self.rot) for index in range(2)]
        for call in range(calls):
            for row in range(length):assert (outputs[0].batch.shares[call][row]+outputs[1].batch.shares[call][row])%PRIME==scalars[call]*vectors[call][row]%PRIME
        assert all(len(set(output.batch.shares))==calls for output in outputs)
        self.rot+=508*calls;self.calls+=calls;self.sequence+=1;self.metrics.append(replies)
        return identifiers[0]
    def close(self):
        for process in self.processes:send_frame(process.stdin,sequence=self.sequence,close=True)
        for reader in self.readers:
            reply=reader.read(45)
            assert reply=={'status':'closed','sequence':self.sequence,'total_calls':self.calls,'total_rot_choices':self.rot,'poisoned':True}
        for process in self.processes:assert process.wait(timeout=5)==0
    def rejected(self):
        for reader in self.readers:
            reply=reader.read(45);assert reply['status']=='aborted' and reply['poisoned']is True
        for process in self.processes:
            assert process.wait(timeout=5)==1
            try:process.stdin.write(b'no restart');process.stdin.flush()
            except (BrokenPipeError,OSError):pass
            else:raise AssertionError('retired service accepted another write')
    def cleanup(self):
        for process in self.processes:
            if process.poll()is None:process.kill()
            process.wait()
            if process.stdin:
                try:process.stdin.close()
                except (BrokenPipeError,OSError):pass
            if process.stdout:process.stdout.close()
        for log in self.logs:log.close()


def main():
    certs=OUT/'certs';certs.mkdir(parents=True,exist_ok=True)
    ca,identities=certificates(certs);checks=[]
    peers=Peers('changing_shapes',ca,identities)
    try:
        for length,calls in ((1,2),(8,3),(128,2)):peers.request(length,calls)
        peers.close()
        checks.append({'name':'changing_shapes','checked_products':2+24+256,'requests':3,'fresh_calls':7,'total_rot_choices':3556,'context_setup_seconds':[r['context_setup_seconds'] for r in peers.ready],'request_seconds':[[r['request_seconds'] for r in phase] for phase in peers.metrics],'fresh_correlations':True,'closed_and_retired':True})
    finally:peers.cleanup()
    for mode in ('sequence_replay','request_id_replay','peer_id_mismatch','base_fault','local_EOF','close_request_mismatch'):
        peers=Peers(mode,ca,identities,fault='response-z' if mode=='base_fault' else '')
        try:
            if mode=='sequence_replay':
                peers.request(8,1);peers.send(8,1,sequences=[0,0])
            elif mode=='request_id_replay':
                identifier=peers.request(8,1);peers.send(8,1,identifiers=[identifier,identifier])
            elif mode=='peer_id_mismatch':peers.send(8,1,identifiers=[secrets.token_bytes(32),secrets.token_bytes(32)])
            elif mode=='local_EOF':
                for process in peers.processes:process.stdin.close()
            elif mode=='close_request_mismatch':
                # Bind incompatible operations at the same stream position.
                length,calls=8,1;identifier=secrets.token_bytes(32)
                request=peers.directory/'one.bin';output=peers.directory/'one.json'
                ScalarOleRequest(length,(1,)).write(request)
                send_frame(peers.processes[0].stdin,sequence=0,request_id=identifier,length=length,calls=calls,input_path=request,output_path=output)
                send_frame(peers.processes[1].stdin,sequence=0,close=True)
            else:peers.send(8,1)
            if mode=='local_EOF':
                for reader in peers.readers:assert reader.read(45)['status']=='aborted'
                for process in peers.processes:assert process.wait(timeout=5)==1
            else:peers.rejected()
            checks.append({'name':mode,'rejected':True,'processes_retired':True,'restart_forbidden':True})
        finally:peers.cleanup()
    blocked=subprocess.run([str(BIN)],env=dict(os.environ,EMP_TEST_MODE='1'),capture_output=True,text=True)
    assert blocked.returncode==1 and 'deterministic test mode' in blocked.stderr
    checks.append({'name':'EMP_TEST_MODE','rejected_before_TLS':True})
    report={'security_reviewed':False,'transport_authenticated':True,'checks':checks}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':len(checks),'public_report':str(OUT/'results.json')}))


if __name__=='__main__':main()
