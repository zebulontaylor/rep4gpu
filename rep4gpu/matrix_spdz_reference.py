"""Exact CPU reference for matrix-vector MACs over F_(2**127-1).

This implements authentication checks, shared-right-factor sacrifice,
transpose checks, and checked online matrix multiplication. The included
candidate and chosen-input VOLE generators are central trusted TEST fixtures.
They do not implement private preprocessing or establish a security target.
See docs/matrix-spdz-reference.md for the backend boundary and paper equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import secrets
import struct
import threading
from typing import Protocol
import weakref

from .spdz_distributed import DistributedParty,ProtocolError,_hash

PRIME=(1<<127)-1
_epoch_lock=threading.Lock()
_allocated_sessions=set()
_allocated_keys=set()


@dataclass(frozen=True)
class Matrix:
    rows:int
    cols:int
    data:tuple[int,...]

    def __post_init__(self):
        if (type(self.rows)is not int or type(self.cols)is not int or self.rows<1 or self.cols<1
                or type(self.data)is not tuple or len(self.data)!=self.rows*self.cols
                or any(type(x)is not int or not 0<=x<PRIME for x in self.data)):
            raise ValueError('expected a canonical immutable F127 matrix')

    @classmethod
    def from_rows(cls,rows):
        rows=tuple(tuple(row) for row in rows)
        if not rows or not rows[0] or any(len(row)!=len(rows[0]) for row in rows):raise ValueError('matrix must have rectangular positive dimensions')
        return cls(len(rows),len(rows[0]),tuple(x%PRIME for row in rows for x in row))

    def to_rows(self):return [list(self.data[i*self.cols:(i+1)*self.cols]) for i in range(self.rows)]


def zeros(rows,cols):return Matrix(rows,cols,(0,)*(rows*cols))
def random_matrix(rows,cols):return Matrix(rows,cols,tuple(secrets.randbelow(PRIME) for _ in range(rows*cols)))
def transpose(a):return Matrix(a.cols,a.rows,tuple(a.data[i*a.cols+j] for j in range(a.cols) for i in range(a.rows)))


def add(a,b):
    if (a.rows,a.cols)!=(b.rows,b.cols):raise ValueError('matrix dimensions mismatch')
    return Matrix(a.rows,a.cols,tuple((x+y)%PRIME for x,y in zip(a.data,b.data)))


def scale(c,a):
    if type(c)is not int:raise ValueError('scalar must be an integer')
    return Matrix(a.rows,a.cols,tuple(c*x%PRIME for x in a.data))


def sub(a,b):return add(a,scale(-1,b))


def matmul(a,b):
    if a.cols!=b.rows:raise ValueError('matrix product dimensions mismatch')
    return Matrix(a.rows,b.cols,tuple(sum(a.data[i*a.cols+k]*b.data[k*b.cols+j] for k in range(a.cols))%PRIME
                                    for i in range(a.rows) for j in range(b.cols)))


def _sum(items,rows,cols):
    result=zeros(rows,cols)
    for item in items:result=add(result,item)
    return result


def _pack(items):return b''.join(x.to_bytes(16,'little') for item in items for x in item.data)


def _unpack(data,shapes):
    if len(data)!=16*sum(a*b for a,b in shapes):raise ProtocolError('matrix field payload length mismatch')
    values=tuple(int.from_bytes(data[i:i+16],'little') for i in range(0,len(data),16))
    if any(x>=PRIME for x in values):raise ProtocolError('noncanonical F127 wire value')
    result=[];offset=0
    for rows,cols in shapes:
        count=rows*cols;result.append(Matrix(rows,cols,values[offset:offset+count]));offset+=count
    return tuple(result)


@dataclass(frozen=True)
class AuthMatrix:
    value:Matrix
    tag:Matrix
    epoch:bytes
    role:int


@dataclass(frozen=True)
class BackendEvidence:
    provider:str
    lower_layer_checks_verified:bool
    test_only:bool
    privacy_reviewed:bool=False

    def __post_init__(self):
        if not isinstance(self.provider,str) or not self.provider or any(type(x)is not bool for x in (self.lower_layer_checks_verified,self.test_only,self.privacy_reviewed)):
            raise ValueError('backend evidence requires a provider name and explicit boolean claims')


@dataclass(frozen=True)
class ColumnVoleBatch:
    """Local pairwise sums: outgoing u and incoming w, with u-w=delta*x."""
    batch_id:bytes
    epoch:bytes
    role:int
    outgoing:tuple[Matrix,...]
    incoming:tuple[Matrix,...]
    evidence:BackendEvidence


class ChosenColumnVoleBackend(Protocol):
    """Role-private backend; callers never supply another role's key or values.

    For each ordered peer pair, column k, and key column h, the sender provides
    X[:,k] and the receiver its epoch-fixed V[k,h]. Every call consumes fresh
    chosen-input VOLE material. Verify lower-layer malicious checks before
    returning. Cross-peer consistency is checked separately by MatrixParty.
    Candidate spVOPE with unreviewed privacy cannot be advertised as production.
    """
    def authenticate_columns(self,role:int,values:tuple[Matrix,...],key:Matrix,*,epoch:bytes,batch_id:bytes)->ColumnVoleBatch:...
    def abort(self,error:BaseException)->None:...


@dataclass(frozen=True)
class SextupleCandidate:
    """Untrusted additive shares; matching C tags do not prove C=A@B.

    All sacrificed pairs share this same B and have independently generated
    A-prime row blocks. Candidate generation must establish privacy separately.
    """
    identifier:bytes
    epoch:bytes
    role:int
    a:Matrix
    at:Matrix
    b:Matrix
    c:Matrix
    r:Matrix
    rt:Matrix
    sacrifices:tuple[tuple[Matrix,Matrix],...]
    evidence:BackendEvidence


@dataclass
class InputMask:
    share:AuthMatrix
    owner:int
    clear:Matrix|None
    identifier:bytes
    used:bool=False


@dataclass(frozen=True)
class _SextupleState:
    owner:object
    epoch:bytes
    role:int
    key:Matrix
    identifier:bytes
    shares:tuple[AuthMatrix,...]
    evidence:tuple[BackendEvidence,...]
    used:bool=False


_sextuples=weakref.WeakKeyDictionary()


class MatrixSextuple:
    """Opaque one-use result of authentication, sacrifice and transpose checks."""
    __slots__=('__weakref__',)
    def __init__(self):raise TypeError('use MatrixParty.prepare_sextuple')
    def __reduce__(self):raise TypeError('matrix sextuples cannot be serialized')
    @property
    def used(self):return _sextuples[self].used
    @property
    def test_only(self):return any(e.test_only for e in _sextuples[self].evidence)


class MatrixParty(DistributedParty):
    """One ordered CPU stream; every returned opening has passed all checks."""
    def __init__(self,party,transport,key,*,session,statistical_security_bits=128,statistical_check_budget=1<<32,parties=4,_allow_insecure_test_transport=False):
        self._operation_lock=threading.RLock()
        self.party,self.transport,self.parties=party,transport,parties
        self.poisoned=False;self._failure=None;self.round=0;self._spent=set();self._events=0
        with self.operation():
            if parties!=4 or party not in range(4):raise ValueError('matrix reference requires four roles')
            if not getattr(transport,'authenticated',False) and not _allow_insecure_test_transport:
                raise ValueError('authenticated confidential transport required')
            if not isinstance(session,bytes) or not session:raise ValueError('nonempty MAC epoch required')
            if (type(key)is not Matrix or type(statistical_security_bits)is not int or statistical_security_bits<1
                or type(statistical_check_budget)is not int or statistical_check_budget<1):
                raise ValueError('canonical key matrix and positive statistical parameter required')
            required=1
            while PRIME**required < (1<<statistical_security_bits)*2*statistical_check_budget:required+=1
            if key.cols<required:raise ValueError('insufficient independent key columns for statistical parameter')
            self.key=key;self.m=key.rows;self.columns=key.cols;self.statistical_security_bits=statistical_security_bits;self.statistical_check_budget=statistical_check_budget;self.session=session
            self._session_hash=_hash(b'MatrixMAC-F127-v1'+struct.pack('<Q',len(session))+session+json.dumps([PRIME,self.m,self.columns,statistical_security_bits,statistical_check_budget,self.columns],separators=(',',':')).encode())
            fingerprint=_hash(b'matrix-epoch-key/'+struct.pack('<III',party,key.rows,key.cols)+_pack((key,)))
            with _epoch_lock:
                if (party,session) in _allocated_sessions or fingerprint in _allocated_keys:
                    raise ProtocolError('MAC epoch/key already allocated; do not reconstruct or reset its runtime')
                _allocated_sessions.add((party,bytes(session)));_allocated_keys.add(fingerprint)

    @contextmanager
    def operation(self):
        with self._operation_lock:
            with super().operation():yield

    def _frame(self,stage,sender,round_id,label,body):
        return b'MRF127v1'+self._session_hash+struct.pack('<QI',round_id,sender)+stage+_hash(label)+struct.pack('<Q',len(body))+body

    def _validate_value(self,value):
        if type(value)is not Matrix or (value.rows,value.cols)!=(self.m,self.m):raise ValueError('expected a square matrix of the configured width')

    def _validate(self,share):
        if type(share)is not AuthMatrix or share.epoch!=self.session or share.role!=self.party:
            raise ProtocolError('matrix sharing belongs to another role or MAC epoch')
        self._validate_value(share.value)
        if type(share.tag)is not Matrix or (share.tag.rows,share.tag.cols)!=(self.m,self.columns):
            raise ValueError('matrix-vector MAC dimensions mismatch')

    def _preflight_checks(self,count):
        if self._events+count>self.statistical_check_budget:raise ProtocolError('statistical check budget exhausted; retire the MAC epoch')

    def _record_checks(self,count):
        with self._operation_lock:
            self._preflight_checks(count)
            self._events+=count

    def _check_evidence(self,evidence):
        if type(evidence)is not BackendEvidence or not evidence.lower_layer_checks_verified:
            raise ProtocolError('backend lower-layer checks incomplete')
        if not evidence.test_only:
            if not evidence.privacy_reviewed:raise ProtocolError('backend privacy gate is not approved')
            raise ProtocolError('no reviewed private matrix preprocessing backend is registered')

    def _burn(self,identifier):
        if not isinstance(identifier,bytes) or len(identifier)!=32:raise ValueError('32-byte correlation identifier required')
        if identifier in self._spent:raise ProtocolError('matrix correlation is single-use')
        self._spent.add(identifier)

    def _commit_open(self,payload,*,label):
        salt=secrets.token_bytes(32)
        context=self._session_hash+label+struct.pack('<Q',self.round)
        commit=_hash(context+struct.pack('<I',self.party)+salt+payload)
        commits=self.broadcast(commit,label=label+b'/commit')
        reveals=self.broadcast(salt+payload,label=label+b'/reveal')
        for role,value in reveals.items():
            if (len(commits[role])!=32 or len(value)<32 or
                not secrets.compare_digest(commits[role],_hash(context+struct.pack('<I',role)+value))):
                raise ProtocolError('matrix commitment opening mismatch')
        return {role:value[32:] for role,value in reveals.items()}

    def _coins(self,count,*,label):
        # Every honest contribution is sampled after preceding values are fixed.
        contributions=self._commit_open(secrets.token_bytes(32),label=label+b'/coin')
        if any(len(value)!=32 for value in contributions.values()):raise ProtocolError('invalid coin contribution')
        seed=bytes(a^b^c^d for a,b,c,d in zip(*(contributions[p] for p in range(4))))
        result=[];counter=0
        while len(result)<count:
            word=int.from_bytes(hashlib.shake_256(self._session_hash+label+seed+struct.pack('<Q',counter)).digest(16),'little')&PRIME
            counter+=1
            if word<PRIME:result.append(word)
        return tuple(result)

    def _partial_open(self,shares,*,label):
        for share in shares:self._validate(share)
        bodies=self.broadcast(_pack([s.value for s in shares]),label=label+b'/values')
        decoded=[_unpack(bodies[p],[(self.m,self.m)]*len(shares)) for p in range(4)]
        return tuple(_sum((decoded[p][j] for p in range(4)),self.m,self.m) for j in range(len(shares)))

    def _check_exact(self,shares,opened,*,label):
        residuals=[sub(s.tag,matmul(value,self.key)) for s,value in zip(shares,opened)]
        reveals=self._commit_open(_pack(residuals),label=label+b'/residual')
        decoded=[_unpack(reveals[p],[(self.m,self.columns)]*len(shares)) for p in range(4)]
        for j in range(len(shares)):
            if any(_sum((decoded[p][j] for p in range(4)),self.m,self.columns).data):
                raise ProtocolError('matrix MAC check failed')

    def open_many(self,shares,*,label=b'output'):
        with self.operation():
            shares=tuple(shares)
            if not shares:raise ValueError('opening requires at least one authenticated matrix')
            for share in shares:self._validate(share)
            layout=json.dumps([self.m,self.columns,len(shares)],separators=(',',':')).encode()
            self._record_checks(self.columns+1)
            context=b'matrix-open/'+_hash(layout)+label
            opened=self._partial_open(shares,label=context)
            coefficients=self._coins(self.columns*len(shares),label=context+b'/post-freeze')
            combined=[];public=[]
            for repetition in range(self.columns):
                a=self.public(zeros(self.m,self.m));b=zeros(self.m,self.m)
                for j,(share,value) in enumerate(zip(shares,opened)):
                    coefficient=coefficients[repetition*len(shares)+j]
                    a=self.add(a,self.scalar(coefficient,share));b=add(b,scale(coefficient,value))
                combined.append(a);public.append(b)
            self._check_exact(combined,public,label=context+b'/batch')
            return opened

    def open(self,share):return self.open_many((share,))[0]

    def public(self,value):
        with self.operation():
            self._validate_value(value)
            return AuthMatrix(value if self.party==0 else zeros(self.m,self.m),matmul(value,self.key),self.session,self.party)

    def add(self,x,y):
        with self.operation():
            self._validate(x);self._validate(y)
            return AuthMatrix(add(x.value,y.value),add(x.tag,y.tag),self.session,self.party)

    def subtract(self,x,y):return self.add(x,self.scalar(-1,y))

    def scalar(self,c,x):
        with self.operation():
            self._validate(x)
            return AuthMatrix(scale(c,x.value),scale(c,x.tag),self.session,self.party)

    def left(self,public,x):
        with self.operation():
            self._validate_value(public);self._validate(x)
            return AuthMatrix(matmul(public,x.value),matmul(public,x.tag),self.session,self.party)

    def input(self,mask,value=None):
        with self.operation():
            self._burn(mask.identifier)
            if mask.used:raise ProtocolError('input mask is single-use')
            mask.used=True;self._validate(mask.share)
            if type(mask.owner)is not int or mask.owner not in range(4):raise ValueError('invalid input owner')
            if self.party==mask.owner:
                self._validate_value(mask.clear);self._validate_value(value)
                payload=_pack((sub(value,mask.clear),))
            else:
                if mask.clear is not None or value is not None:raise ProtocolError('nonowner supplied private input or mask')
                payload=b''
            bodies=self.broadcast(payload,label=b'matrix-input/'+mask.identifier+struct.pack('<I',mask.owner))
            if any(bodies[p] for p in range(4) if p!=mask.owner):raise ProtocolError('nonowner input correction')
            correction=_unpack(bodies[mask.owner],[(self.m,self.m)])[0]
            return self.add(mask.share,self.public(correction))

    def authenticate(self,values,backend,*,batch_id):
        """Authenticate fixed local matrices and verify masked batch consistency."""
        with self.operation():
            self._burn(batch_id)
            values=tuple(values)
            if not values:raise ValueError('authentication requires matrices')
            for value in values:self._validate_value(value)
            self._record_checks(self.columns+1)
            masks=tuple(random_matrix(self.m,self.m) for _ in range(self.columns))
            try:result=backend.authenticate_columns(self.party,values+masks,self.key,epoch=self.session,batch_id=batch_id)
            except BaseException as error:backend.abort(error);raise
            if (type(result)is not ColumnVoleBatch or result.role!=self.party or result.epoch!=self.session
                or result.batch_id!=batch_id or len(result.outgoing)!=len(values)+self.columns
                or len(result.incoming)!=len(values)+self.columns or not result.evidence.lower_layer_checks_verified):
                raise ProtocolError('unverified or mismatched chosen-column VOLE batch')
            self._check_evidence(result.evidence)
            authenticated=[]
            for value,u,w in zip(values+masks,result.outgoing,result.incoming):
                if (u.rows,u.cols)!=(self.m,self.columns) or (w.rows,w.cols)!=(self.m,self.columns):
                    raise ValueError('chosen-column VOLE output dimensions mismatch')
                authenticated.append(AuthMatrix(value,add(matmul(value,self.key),sub(u,w)),self.session,self.party))
            coefficients=self._coins(len(values)*self.columns,label=b'matrix-auth/'+batch_id+b'/post-freeze')
            for repetition in range(self.columns):
                combined=authenticated[len(values)+repetition]
                for j in range(len(values)):
                    combined=self.add(combined,self.scalar(coefficients[repetition*len(values)+j],authenticated[j]))
                opened=self._partial_open((combined,),label=b'matrix-auth/'+batch_id+struct.pack('<I',repetition))
                self._check_exact((combined,),opened,label=b'matrix-auth/'+batch_id+struct.pack('<I',repetition)+b'/check')
            return tuple(authenticated[:len(values)]),result.evidence

    def prepare_sextuple(self,candidate,backend):
        """Authenticate, sacrifice and verify transposes before issuing a handle."""
        with self.operation():
            if type(candidate)is not SextupleCandidate or candidate.epoch!=self.session or candidate.role!=self.party:
                raise ProtocolError('candidate belongs to another role or epoch')
            self._burn(candidate.identifier)
            self._check_evidence(candidate.evidence)
            if len(candidate.sacrifices)!=self.columns:raise ValueError('one independent shared-B sacrifice per key column required')
            self._preflight_checks((self.columns+1)*(1+3*self.columns)+2)
            values=[candidate.a,candidate.at,candidate.b,candidate.c,candidate.r,candidate.rt]
            for pair in candidate.sacrifices:values.extend(pair)
            masks=tuple(random_matrix(self.m,self.m) for _ in range(self.columns))
            for mask in masks:values.extend((mask,transpose(mask)))
            authenticated,evidence=self.authenticate(values,backend,batch_id=_hash(candidate.identifier+b'/authenticate'))
            a,at,b,c,r,rt=authenticated[:6]
            self._record_checks(1)
            challenges=self._coins(self.columns,label=b'matrix-sacrifice/'+candidate.identifier+b'/post-freeze')
            for j,challenge in enumerate(challenges):
                ap,cp=authenticated[6+2*j:8+2*j]
                d=self.subtract(self.scalar(challenge,a),ap)
                opened_d=self.open_many((d,),label=b'matrix-sacrifice/D'+candidate.identifier+struct.pack('<I',j))[0]
                error=self.subtract(self.subtract(self.scalar(challenge,c),cp),self.left(opened_d,b))
                opened_e=self.open_many((error,),label=b'matrix-sacrifice/E'+candidate.identifier+struct.pack('<I',j))[0]
                if any(opened_e.data):raise ProtocolError('matrix product sacrifice failed')
            self._record_checks(1)
            coefficients=self._coins(2*self.columns,label=b'matrix-transpose/'+candidate.identifier+b'/post-freeze')
            mask_offset=6+2*self.columns
            for j in range(self.columns):
                mask,mask_t=authenticated[mask_offset+2*j:mask_offset+2*j+2]
                left=self.add(mask,self.add(self.scalar(coefficients[2*j],a),self.scalar(coefficients[2*j+1],r)))
                right=self.add(mask_t,self.add(self.scalar(coefficients[2*j],at),self.scalar(coefficients[2*j+1],rt)))
                opened_left,opened_right=self.open_many((left,right),label=b'matrix-transpose/check'+candidate.identifier+struct.pack('<I',j))
                if opened_left!=transpose(opened_right):raise ProtocolError('authenticated transpose relation failed')
            handle=object.__new__(MatrixSextuple)
            _sextuples[handle]=_SextupleState(weakref.ref(self),self.session,self.party,self.key,candidate.identifier,(a,at,b,c,r,rt),(candidate.evidence,evidence))
            return handle

    def multiply(self,x,y,sextuple):
        with self.operation():
            if type(sextuple)is not MatrixSextuple or sextuple not in _sextuples:raise ProtocolError('unknown matrix sextuple')
            state=_sextuples[sextuple]
            if state.used:raise ProtocolError('matrix sextuple is single-use')
            from dataclasses import replace
            _sextuples[sextuple]=replace(state,used=True)
            if state.owner()is not self or state.epoch!=self.session or state.role!=self.party or state.key!=self.key:
                raise ProtocolError('matrix sextuple role, epoch, or key mismatch')
            self._validate(x);self._validate(y)
            self._preflight_checks(2*(self.columns+1))
            a,at,b,c,r,rt=state.shares
            d,e=self.open_many((self.subtract(x,a),self.subtract(y,b)),label=b'matrix-online/DE'+state.identifier)
            ft=self.open_many((self.subtract(self.left(transpose(e),at),rt),),label=b'matrix-online/FT'+state.identifier)[0]
            f=transpose(ft);de=matmul(d,e)
            value=add(add(c.value,matmul(d,b.value)),add(matmul(a.value,e),de if self.party==0 else zeros(self.m,self.m)))
            tag=add(add(c.tag,matmul(d,b.tag)),add(add(matmul(f,self.key),r.tag),matmul(de,self.key)))
            return AuthMatrix(value,tag,self.session,self.party)

    def statistical_accounting(self):
        """Conservative union-bound bookkeeping, not a complete security claim."""
        return dict(statistical_security_bits=self.statistical_security_bits,statistical_check_budget=self.statistical_check_budget,
                    key_columns=self.columns,linear_check_repetitions=self.columns,sacrifice_repetitions=self.columns,
                    check_events=self._events,bound=Fraction(2*self._events,PRIME**self.columns),
                    cryptographic_and_backend_assumptions_excluded=True,security_target_met=False)


class TrustedTestColumnVole:
    """Central ideal chosen-input VOLE oracle. TEST ONLY; sees all local keys.

    ``pair_override`` can supply inconsistent chosen vectors or key scalars to
    one peer, while preserving that pair's VOLE relation, for adversarial tests.
    """
    def __init__(self,parties=4,pair_override=None,verified=True,timeout=10):
        if parties!=4:raise ValueError('test oracle requires four roles')
        self.condition=threading.Condition();self.calls={};self.failure=None
        self.pair_override=pair_override;self.verified=verified;self.timeout=timeout

    def abort(self,error):
        with self.condition:self.failure=error;self.condition.notify_all()

    def authenticate_columns(self,role,values,key,*,epoch,batch_id):
        with self.condition:
            if self.failure is not None:raise ProtocolError('test VOLE oracle aborted') from self.failure
            call=self.calls.setdefault((epoch,batch_id),dict(inputs={},outputs=None,readers=0))
            if role in call['inputs']:raise ProtocolError('duplicate test VOLE role invocation')
            call['inputs'][role]=(values,key)
            if len(call['inputs'])==4:
                try:call['outputs']=self._generate(call['inputs'],epoch,batch_id)
                except BaseException as error:self.failure=error
                self.condition.notify_all()
            ready=self.condition.wait_for(lambda:call['outputs']is not None or self.failure is not None,timeout=self.timeout)
            if not ready:self.failure=TimeoutError('test VOLE oracle rendezvous timed out');self.condition.notify_all()
            if self.failure is not None:raise ProtocolError('test VOLE oracle failed') from self.failure
            result=call['outputs'][role];call['readers']+=1
            # Keep identifiers to reject a second invocation, without retaining inputs.
            if call['readers']==4:call['inputs']={p:None for p in range(4)};call['outputs']=None
            return result

    def _generate(self,inputs,epoch,batch_id):
        values,key=inputs[0];m=key.rows;columns=key.cols;count=len(values)
        if any(len(v)!=count or (k.rows,k.cols)!=(m,columns) for v,k in inputs.values()):raise ValueError('test VOLE public layout mismatch')
        outgoing=[[zeros(m,columns) for _ in range(count)] for _ in range(4)]
        incoming=[[zeros(m,columns) for _ in range(count)] for _ in range(4)]
        for sender in range(4):
            for receiver in range(4):
                if sender==receiver:continue
                for index in range(count):
                    for h in range(columns):
                        for k in range(m):
                            vector=tuple(inputs[sender][0][index].data[row*m+k] for row in range(m))
                            delta=inputs[receiver][1].data[k*columns+h]
                            if self.pair_override is not None:vector,delta=self.pair_override(sender,receiver,index,h,k,vector,delta)
                            if len(vector)!=m or any(type(x)is not int or not 0<=x<PRIME for x in (*vector,delta)):
                                raise ValueError('invalid test VOLE chosen input')
                            w=tuple(secrets.randbelow(PRIME) for _ in range(m));u=tuple((delta*x+y)%PRIME for x,y in zip(vector,w))
                            out=list(outgoing[sender][index].data);inc=list(incoming[receiver][index].data)
                            for row in range(m):out[row*columns+h]=(out[row*columns+h]+u[row])%PRIME;inc[row*columns+h]=(inc[row*columns+h]+w[row])%PRIME
                            outgoing[sender][index]=Matrix(m,columns,tuple(out));incoming[receiver][index]=Matrix(m,columns,tuple(inc))
        evidence=BackendEvidence('central trusted ideal chosen-input VOLE TEST fixture',self.verified,True,False)
        return {p:ColumnVoleBatch(batch_id,epoch,p,tuple(outgoing[p]),tuple(incoming[p]),evidence) for p in range(4)}


class TrustedTestFixture:
    """Central matrix candidate/input factory. TEST ONLY, not private preprocessing."""
    def __init__(self,width,columns=2):
        self.m=width;self.columns=columns;self.epoch=secrets.token_bytes(32)
        self.keys=tuple(random_matrix(width,columns) for _ in range(4))
        self.evidence=BackendEvidence('central trusted matrix-candidate TEST fixture',True,True,False)

    def _split(self,value):
        parts=[random_matrix(value.rows,value.cols) for _ in range(3)]
        parts.append(sub(value,_sum(parts,value.rows,value.cols)))
        return tuple(parts)

    def authenticated(self,value):
        if (value.rows,value.cols)!=(self.m,self.m):raise ValueError('fixture width mismatch')
        values=self._split(value);global_key=_sum(self.keys,self.m,self.columns)
        tags=self._split(matmul(value,global_key))
        return tuple(AuthMatrix(values[p],tags[p],self.epoch,p) for p in range(4))

    def input_masks(self,owner):
        if owner not in range(4):raise ValueError('invalid fixture owner')
        clear=random_matrix(self.m,self.m);shares=self.authenticated(clear);identifier=secrets.token_bytes(32)
        return tuple(InputMask(shares[p],owner,clear if p==owner else None,identifier) for p in range(4))

    def candidates(self):
        a,b,r=(random_matrix(self.m,self.m) for _ in range(3))
        aa,bb,cc,rr=(self._split(value) for value in (a,b,matmul(a,b),r))
        sacrifices=[]
        for _ in range(self.columns):
            ap=random_matrix(self.m,self.m);sacrifices.append((self._split(ap),self._split(matmul(ap,b))))
        identifier=secrets.token_bytes(32)
        return tuple(SextupleCandidate(identifier,self.epoch,p,aa[p],transpose(aa[p]),bb[p],cc[p],rr[p],transpose(rr[p]),
                     tuple((ap[p],cp[p]) for ap,cp in sacrifices),self.evidence) for p in range(4))
