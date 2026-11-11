"""Experimental role-private native OLE adapter for the F127 matrix reference.

No trusted fixture oracle is used. Every endpoint reads only its own scalar or
vector file, and Python reads only its local endpoint outputs. The backend is
explicitly TEST-ONLY and privacy-unreviewed; it does not register a production
provider or establish a security target.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
import subprocess
import threading
import time

from native.matrix_vope.direct_ole_io import ScalarOleRequest,VectorOleRequest,LocalOleBatch
from .matrix_spdz_reference import (
    PRIME,Matrix,AuthMatrix,InputMask,BackendEvidence,ColumnVoleBatch,SextupleCandidate,
    zeros,random_matrix,transpose,matmul,add,sub,scale,_pack,_unpack,
)
from .spdz_distributed import ProtocolError,_hash


@dataclass(frozen=True)
class NativeOleConfig:
    binary:Path
    private_directory:Path
    ports:tuple[tuple[int,int,int],...]
    ca_file:Path
    certificate:Path
    private_key:Path
    timeout:float=60


class ExperimentalNativeOle:
    """Actual native chosen-input OLE, with public phase barriers and local IO."""
    def __init__(self,party,config):
        if not getattr(party.transport,'authenticated',False):raise ValueError('native matrix integration requires authenticated control transport')
        self.party=party;self.config=config
        self.directory=Path(config.private_directory).resolve();self.directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.directory.chmod(0o700)
        self.ports={(sender,receiver):port for sender,receiver,port in config.ports}
        if set(self.ports)!={(a,b) for a in range(4) for b in range(4) if a!=b}:raise ValueError('complete ordered-pair port map required')
        if not Path(config.binary).is_file():raise FileNotFoundError(config.binary)
        self._children=[];self._lock=threading.Lock();self._used=set();self.metrics=[]
        self.evidence=BackendEvidence('experimental role-private native Wolverine-r1 chosen-input OLE over mTLS; TEST ONLY',True,True,False)

    def abort(self,error):
        with self._lock:children=list(self._children)
        for process in children:
            if process.poll()is None:process.kill()

    def _barrier(self,phase,metadata):
        payload=_hash(metadata)
        values=self.party.broadcast(payload,label=b'matrix-native-ole/'+phase)
        if any(value!=payload for value in values.values()):raise ProtocolError('native OLE public phase mismatch')

    def _pairs(self,*,purpose,batch_id,length,calls,scalars,vectors,shapes,senders=None):
        if not isinstance(batch_id,bytes) or len(batch_id)!=32:raise ValueError('32-byte native batch ID required')
        senders=tuple(range(4)) if senders is None else tuple(senders)
        if not senders or len(set(senders))!=len(senders) or any(type(s)is not int or s not in range(4) for s in senders):raise ValueError('invalid OLE sender set')
        phase=(purpose,batch_id)
        if phase in self._used:raise ProtocolError('native OLE batch is single-use')
        self._used.add(phase)
        metadata=json.dumps(dict(protocol='matrix-native-ole-v1',prime=PRIME,epoch=self.party.session.hex(),
            batch_id=batch_id.hex(),purpose=purpose,length=length,calls=calls,shapes=shapes,senders=senders,
            columns=self.party.columns,statistical_security_bits=self.party.statistical_security_bits,
            statistical_check_budget=self.party.statistical_check_budget),sort_keys=True,separators=(',',':')).encode()
        phase_id=_hash(metadata)
        self._barrier(b'before/'+phase_id,metadata)
        directory=self.directory/(purpose+'-'+batch_id.hex());directory.mkdir(mode=0o700)
        children=[];handles=[];jobs=[];started=time.perf_counter()
        try:
            # Each role starts all six endpoints before waiting: three outgoing
            # vector roles and three incoming scalar roles on disjoint ports.
            for sender in range(4):
                for receiver in range(4):
                    if sender not in senders or sender==receiver or self.party.party not in (sender,receiver):continue
                    vector_side=self.party.party==sender
                    ole_role=2 if vector_side else 1;peer=receiver if vector_side else sender
                    session=hashlib.sha256(metadata+struct.pack('<II',sender,receiver)).hexdigest()
                    name=f'{sender}-to-{receiver}';request=directory/(name+'.request');output=directory/(name+'.output.json')
                    if vector_side:VectorOleRequest(vectors).write(request)
                    else:ScalarOleRequest(length,scalars).write(request)
                    request.chmod(0o600)
                    log=(directory/(name+'.log')).open('w');handles.append(log)
                    command=[str(Path(self.config.binary).resolve()),str(ole_role),str(self.ports[sender,receiver]),
                        str(request),str(output),str(self.party.party),str(peer),str(self.config.ca_file),
                        str(self.config.certificate),str(self.config.private_key),f'party{peer}',session]
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    children.append(process);jobs.append((sender,receiver,ole_role,peer,session,output))
                    with self._lock:self._children.append(process)
            deadline=time.monotonic()+self.config.timeout
            for process in children:process.wait(timeout=max(.001,deadline-time.monotonic()))
            codes=[process.returncode for process in children]
            if any(code!=0 for code in codes):raise ProtocolError(f'native OLE endpoint failed: {codes}')
            outgoing={};incoming={};endpoint_metrics=[]
            for sender,receiver,role,peer,session,output in jobs:
                output.chmod(0o600)
                result=LocalOleBatch.read(output,role=role,length=length,calls=calls,authenticated=True,
                                          self_role=self.party.party,peer_role=peer,session=session)
                if not result.lower_layer_checks_verified or not result.transport_authenticated:
                    raise ProtocolError('native OLE checks or TLS contract incomplete')
                public=json.loads(output.read_text())
                endpoint_metrics.append(dict(ordered_sender=sender,ordered_receiver=receiver,ole_role=role,
                    context_setup_seconds=public['context_setup_seconds'],context_sent_bytes=public['context_sent_bytes'],
                    context_received_bytes=public['context_received_bytes'],
                    calls=[{k:call[k] for k in ('index','seconds','sent_bytes','received_bytes','fresh_rot_choices','phase_seconds')} for call in public['sessions']]))
                if role==2:outgoing[receiver]=result.outgoing_u()
                else:incoming[sender]=result.incoming_w()
            elapsed=(time.perf_counter()-started)*1000
            self.metrics.append(dict(purpose=purpose,batch_id=batch_id.hex(),length=length,calls=calls,
                                     endpoint_count=len(jobs),endpoint_return_codes=codes,wall_ms=elapsed,
                                     lower_layer_checks_verified=True,transport_authenticated=True,
                                     privacy_reviewed=False,security_target_met=False,endpoint_metrics=endpoint_metrics))
            self._barrier(b'after/'+phase_id,metadata)
            return outgoing,incoming
        except BaseException as error:
            self.abort(error);self.party.abort(error);raise
        finally:
            for process in children:
                if process.poll()is None:process.kill()
                process.wait()
            for handle in handles:handle.close()
            with self._lock:self._children=[p for p in self._children if p not in children]

    def authenticate_columns(self,role,values,key,*,epoch,batch_id):
        if role!=self.party.party or epoch!=self.party.session or key!=self.party.key:
            raise ProtocolError('native authentication role, epoch, or key mismatch')
        values=tuple(values);m=self.party.m;r=self.party.columns
        for value in values:self.party._validate_value(value)
        # One vector contains every matrix AND every fresh check mask for this
        # fixed scalar. Never split data/masks into independently keyed calls.
        vectors=tuple(tuple(value.data[row*m+k] for value in values for row in range(m)) for k in range(m) for h in range(r))
        scalars=tuple(key.data[k*r+h] for k in range(m) for h in range(r))
        outgoing,incoming=self._pairs(purpose='authenticate',batch_id=batch_id,length=m*len(values),calls=m*r,
            scalars=scalars,vectors=vectors,shapes=[len(values),m,r])
        out=[[0]*(m*r) for _ in values];inc=[[0]*(m*r) for _ in values]
        for peer,outputs in outgoing.items():
            for call,vector in enumerate(outputs):
                h=call%r
                for index in range(len(values)):
                    for row in range(m):out[index][row*r+h]=(out[index][row*r+h]+vector[index*m+row])%PRIME
        for peer,outputs in incoming.items():
            for call,vector in enumerate(outputs):
                h=call%r
                for index in range(len(values)):
                    for row in range(m):inc[index][row*r+h]=(inc[index][row*r+h]+vector[index*m+row])%PRIME
        return ColumnVoleBatch(batch_id,epoch,role,tuple(Matrix(m,r,tuple(x)) for x in out),
                               tuple(Matrix(m,r,tuple(x)) for x in inc),self.evidence)

    def candidate(self,identifier):
        """Generate local shares with genuine cross-party OLE products, no dealer."""
        with self.party.operation():
            m=self.party.m;r=self.party.columns;rows=(r+1)*m
            ahat=random_matrix(rows,m);b=random_matrix(m,m);mask=random_matrix(m,m)
            vectors=tuple(tuple(ahat.data[row*m+k] for row in range(rows)) for k in range(m) for col in range(m))
            scalars=tuple(b.data[k*m+col] for k in range(m) for col in range(m))
            outgoing,incoming=self._pairs(purpose='candidate',batch_id=identifier,length=rows,calls=m*m,
                scalars=scalars,vectors=vectors,shapes=[rows,m,m])
            values=list(matmul(ahat,b).data)
            for outputs,sign in ((outgoing,1),(incoming,-1)):
                for peer,calls in outputs.items():
                    for call,vector in enumerate(calls):
                        col=call%m
                        for row in range(rows):values[row*m+col]=(values[row*m+col]+sign*vector[row])%PRIME
            chat=Matrix(rows,m,tuple(values))
            def block(matrix,index):return Matrix(m,m,matrix.data[index*m*m:(index+1)*m*m])
            a=block(ahat,0)
            return SextupleCandidate(identifier,self.party.session,self.party.party,a,transpose(a),b,block(chat,0),mask,transpose(mask),
                tuple((block(ahat,j+1),block(chat,j+1)) for j in range(r)),self.evidence)

    def input_mask(self,owner,identifier):
        """Protocol 10: only the input owner knows the complete random mask.

        Tags authenticate the owner's FULL masks, not the recipients' claimed
        shares. Masked consistency therefore checks distribution correctness
        before the mask can alter a real input. No mask is reconstructed by a
        coordinator or disclosed to a nonowner.
        """
        with self.party.operation():
            p=self.party;m=p.m;r=p.columns
            if type(owner)is not int or owner not in range(4):raise ValueError('invalid input-mask owner')
            p._burn(_hash(identifier+b'/input-mask-preparation'))
            p._preflight_checks(1+r*(r+1))
            count=r+1
            label=b'matrix-native-input-distribution/'+identifier+struct.pack('<III',owner,m,r)
            round_id=p.round;p.round+=1
            if p.party==owner:
                full=tuple(random_matrix(m,m) for _ in range(count))
                splits=[]
                for matrix in full:
                    shares=[random_matrix(m,m) for _ in range(3)]
                    total=zeros(m,m)
                    for share in shares:total=add(total,share)
                    shares.append(sub(matrix,total));splits.append(tuple(shares))
                local=tuple(shares[owner] for shares in splits)
                outgoing={peer:p._frame(b'P',owner,round_id,label+struct.pack('<I',peer),_pack(tuple(shares[peer] for shares in splits))) for peer in range(4) if peer!=owner}
            else:full=None;outgoing={}
            bodies=p.transport.exchange(outgoing)
            expected={owner} if p.party!=owner else set()
            if set(bodies)!=expected:raise ProtocolError('input-mask distribution peer mismatch')
            if p.party!=owner:
                if any(body for peer,body in bodies.items() if peer!=owner):raise ProtocolError('nonowner mask distributor')
                body=p._body(bodies[owner],b'P',owner,round_id,label+struct.pack('<I',p.party))
                local=_unpack(body,[(m,m)]*count)
            elif any(bodies.values()):raise ProtocolError('unexpected mask-distribution response')
            vectors=tuple(tuple(value.data[row*m+k] for value in full for row in range(m)) if full is not None else (0,)*(count*m)
                          for k in range(m) for h in range(r))
            scalars=tuple(p.key.data[k*r+h] for k in range(m) for h in range(r))
            u,w=self._pairs(purpose=f'input-owner{owner}',batch_id=identifier,length=m*count,calls=m*r,
                scalars=scalars,vectors=vectors,shapes=[owner,count,m,r],senders=(owner,))
            tags=[[0]*(m*r) for _ in range(count)]
            if p.party==owner:
                for index,value in enumerate(full):tags[index]=list(matmul(value,p.key).data)
                for peer,calls in u.items():
                    for call,vector in enumerate(calls):
                        h=call%r
                        for index in range(count):
                            for row in range(m):tags[index][row*r+h]=(tags[index][row*r+h]+vector[index*m+row])%PRIME
            else:
                calls=w[owner]
                for call,vector in enumerate(calls):
                    h=call%r
                    for index in range(count):
                        for row in range(m):tags[index][row*r+h]=(tags[index][row*r+h]-vector[index*m+row])%PRIME
            authenticated=tuple(AuthMatrix(local[index],Matrix(m,r,tuple(tags[index])),p.session,p.party) for index in range(count))
            p._record_checks(1)
            coefficients=p._coins(r,label=b'matrix-native-input-check/'+identifier+b'/post-freeze')
            for j,coefficient in enumerate(coefficients):
                combined=p.add(authenticated[j+1],p.scalar(coefficient,authenticated[0]))
                p.open_many((combined,),label=b'matrix-native-input-check/'+identifier+struct.pack('<I',j))
            return InputMask(authenticated[0],owner,full[0] if full is not None else None,identifier)
