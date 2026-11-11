"""Experimental matrix OLE adapter with persistent role-private TLS/OT contexts.

The matrix protocol is inherited unchanged from the separate-process baseline.
Only endpoint lifetime changes. Every request binds its public purpose and
shape, advances ROT counters, and consumes fresh raw correlations. Any failure
retires all local native contexts and the matrix MAC epoch. No restart exists.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
import weakref
import threading

from native.matrix_vope.direct_ole_io import ScalarOleRequest,VectorOleRequest
from native.matrix_vope.ole_service_io import send_frame,ReplyReader,check_ready,check_complete,LocalServiceOutput
from .matrix_native_ole import ExperimentalNativeOle
from .matrix_spdz_reference import BackendEvidence
from .spdz_distributed import ProtocolError,_hash


_allocations=weakref.WeakKeyDictionary()
_allocation_lock=threading.Lock()


@dataclass
class _Endpoint:
    process:object
    reader:object
    log:object
    ole_role:int
    peer:int
    session:str
    sequence:int=0
    total_calls:int=0
    rot_choices:int=0
    entropy_substreams:int=0


class ExperimentalNativeOleService(ExperimentalNativeOle):
    """Test-only backend; one persistent endpoint per ordered peer direction."""
    def __init__(self,party,config):
        super().__init__(party,config)
        with party.operation():
            with _allocation_lock:
                if party in _allocations:raise ProtocolError('persistent OLE epoch already allocated; no restart')
                _allocations[party]=True
            if any((self.directory/name).exists() for name in ('RETIRED','CLOSED')):
                raise ProtocolError('native service epoch was already retired')
        self._services={};self._closed=False;self.setup_metrics=[];self.close_metrics=[]
        self.evidence=BackendEvidence('experimental role-private persistent Wolverine-r1 OLE over mTLS; TEST ONLY',True,True,False)

    def _retire_marker(self,name):
        try:
            (self.directory/name).write_text(json.dumps(dict(epoch=self.party.session.hex(),role=self.party.party,
                native_contexts_retired=True,security_target_met=False))+'\n')
        except OSError:pass

    def abort(self,error):
        self._closed=True;self._retire_marker('RETIRED')
        super().abort(error)
        self.party.abort(error)

    def _start(self):
        if self._closed:raise ProtocolError('persistent native OLE epoch is retired')
        if self._services:return
        started=time.perf_counter()
        metadata=json.dumps(dict(protocol='matrix-ole-service-v1',epoch=self.party.session.hex(),prime=(1<<127)-1,
            width=self.party.m,columns=self.party.columns,statistical_security_bits=self.party.statistical_security_bits,
            statistical_check_budget=self.party.statistical_check_budget),sort_keys=True,separators=(',',':')).encode()
        directory=self.directory/'contexts';directory.mkdir(mode=0o700)
        try:
            for sender in range(4):
                for receiver in range(4):
                    if sender==receiver or self.party.party not in (sender,receiver):continue
                    ole_role=2 if self.party.party==sender else 1
                    peer=receiver if ole_role==2 else sender
                    session=hashlib.sha256(metadata+sender.to_bytes(4,'little')+receiver.to_bytes(4,'little')).hexdigest()
                    log=(directory/f'{sender}-to-{receiver}.stderr').open('w')
                    command=[str(Path(self.config.binary).resolve()),str(ole_role),str(self.ports[sender,receiver]),
                        str(self.party.party),str(peer),str(self.config.ca_file),str(self.config.certificate),str(self.config.private_key),f'party{peer}',session]
                    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0)
                    endpoint=_Endpoint(process,ReplyReader(process.stdout),log,ole_role,peer,session)
                    self._services[sender,receiver]=endpoint
                    with self._lock:self._children.append(process)
            deadline=time.monotonic()+self.config.timeout
            for pair,endpoint in self._services.items():
                reply=endpoint.reader.read(max(.001,deadline-time.monotonic()))
                check_ready(reply,ole_role=endpoint.ole_role,self_role=self.party.party,peer_role=endpoint.peer,session=endpoint.session)
                entropy=reply['entropy_substreams']
                if type(entropy)is not int or entropy<0:raise ProtocolError('invalid service entropy counter')
                endpoint.entropy_substreams=entropy
                self.setup_metrics.append(dict(ordered_sender=pair[0],ordered_receiver=pair[1],**reply))
            self._barrier(b'service-ready/'+_hash(metadata),metadata)
            self.service_setup_wall_ms=(time.perf_counter()-started)*1000
        except BaseException as error:
            self.abort(error);self._cleanup();raise

    def _pairs(self,**arguments):
        try:return self._request_pairs(**arguments)
        except BaseException as error:
            self.abort(error);self._cleanup();raise

    def _request_pairs(self,*,purpose,batch_id,length,calls,scalars,vectors,shapes,senders=None):
        if self._closed:raise ProtocolError('persistent native OLE epoch is retired')
        if not isinstance(batch_id,bytes) or len(batch_id)!=32:raise ValueError('32-byte OLE batch ID required')
        senders=tuple(range(4)) if senders is None else tuple(senders)
        if not senders or len(set(senders))!=len(senders) or any(type(s)is not int or s not in range(4) for s in senders):raise ValueError('invalid OLE sender set')
        phase=purpose,batch_id
        if phase in self._used:raise ProtocolError('persistent OLE batch is single-use')
        self._used.add(phase)
        metadata=json.dumps(dict(protocol='matrix-ole-service-request-v1',epoch=self.party.session.hex(),purpose=purpose,
            batch_id=batch_id.hex(),length=length,calls=calls,shapes=shapes,senders=senders,
            width=self.party.m,columns=self.party.columns,statistical_security_bits=self.party.statistical_security_bits,
            statistical_check_budget=self.party.statistical_check_budget),sort_keys=True,separators=(',',':')).encode()
        phase_id=_hash(metadata);started=time.perf_counter()
        self._barrier(b'service-before/'+phase_id,metadata)
        self._start()
        directory=self.directory/(purpose+'-'+batch_id.hex());directory.mkdir(mode=0o700)
        jobs=[]
        try:
            for pair,endpoint in self._services.items():
                sender,receiver=pair
                if sender not in senders:continue
                request=directory/f'{sender}-to-{receiver}.request'
                output=directory/f'{sender}-to-{receiver}.output.json'
                if endpoint.ole_role==2:VectorOleRequest(vectors).write(request)
                else:ScalarOleRequest(length,scalars).write(request)
                request.chmod(0o600)
                request_id=hashlib.sha256(metadata+sender.to_bytes(4,'little')+receiver.to_bytes(4,'little')).digest()
                send_frame(endpoint.process.stdin,sequence=endpoint.sequence,request_id=request_id,length=length,calls=calls,
                           input_path=request,output_path=output)
                jobs.append((pair,endpoint,request_id,output))
            # Every local active endpoint receives its request before any wait.
            outgoing={};incoming={};public_metrics=[];deadline=time.monotonic()+self.config.timeout
            for pair,endpoint,request_id,output in jobs:
                reply=endpoint.reader.read(max(.001,deadline-time.monotonic()))
                check_complete(reply,sequence=endpoint.sequence,request_id=request_id,length=length,calls=calls,start_rot_choices=endpoint.rot_choices)
                entropy=reply['entropy_substreams']
                if type(entropy)is not int or entropy<=endpoint.entropy_substreams:
                    raise ProtocolError('persistent OLE entropy stream did not advance across requests')
                output.chmod(0o600)
                result=LocalServiceOutput.read(output,ole_role=endpoint.ole_role,self_role=self.party.party,peer_role=endpoint.peer,
                    session=endpoint.session,sequence=endpoint.sequence,request_id=request_id,length=length,calls=calls,start_rot_choices=endpoint.rot_choices)
                if endpoint.ole_role==2:outgoing[pair[1]]=result.batch.outgoing_u()
                else:incoming[pair[0]]=result.batch.incoming_w()
                public_metrics.append(dict(ordered_sender=pair[0],ordered_receiver=pair[1],**reply))
                endpoint.sequence+=1;endpoint.total_calls+=calls;endpoint.rot_choices=result.final_rot_choices;endpoint.entropy_substreams=entropy
            self._barrier(b'service-after/'+phase_id,metadata)
            self.metrics.append(dict(purpose=purpose,batch_id=batch_id.hex(),length=length,calls=calls,
                endpoint_count=len(jobs),wall_ms=(time.perf_counter()-started)*1000,lower_layer_checks_verified=True,
                transport_authenticated=True,privacy_reviewed=False,security_target_met=False,endpoint_metrics=public_metrics,
                persistent_contexts=True))
            return outgoing,incoming
        except BaseException as error:
            self.abort(error);self._cleanup();raise

    def close(self):
        if self._closed:return
        try:
            metadata=self.party.session+b'matrix-ole-service-close'
            self._barrier(b'service-close',metadata)
            for endpoint in self._services.values():send_frame(endpoint.process.stdin,sequence=endpoint.sequence,close=True)
            deadline=time.monotonic()+self.config.timeout
            for pair,endpoint in self._services.items():
                reply=endpoint.reader.read(max(.001,deadline-time.monotonic()))
                if (reply['status']!='closed' or reply['sequence']!=endpoint.sequence or reply['total_calls']!=endpoint.total_calls
                    or reply['total_rot_choices']!=endpoint.rot_choices or reply['poisoned']is not True):
                    raise ProtocolError('persistent OLE close or retirement acknowledgement mismatch')
                endpoint.process.wait(timeout=max(.001,deadline-time.monotonic()))
                if endpoint.process.returncode!=0:raise ProtocolError('persistent OLE service failed at close')
                self.close_metrics.append(dict(ordered_sender=pair[0],ordered_receiver=pair[1],**reply))
            self._closed=True;self._retire_marker('CLOSED')
        except BaseException as error:self.abort(error);raise
        finally:self._cleanup()

    def _cleanup(self):
        for endpoint in self._services.values():
            if endpoint.process.poll()is None:endpoint.process.kill()
            try:endpoint.process.wait(timeout=5)
            except subprocess.TimeoutExpired:endpoint.process.kill();endpoint.process.wait()
            for pipe in (endpoint.process.stdin,endpoint.process.stdout):
                try:pipe.close()
                except OSError:pass
            endpoint.log.close()
        with self._lock:self._children=[]
