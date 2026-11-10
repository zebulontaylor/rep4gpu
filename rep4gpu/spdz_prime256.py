"""Role-local bigint online reference over explicitly supported prime fields.

One field MAC coordinate authenticates each value. Correct fresh private
preprocessing and authenticated confidential links are required. This CPU
reference makes no GPU performance or complete security proof claim.
"""
from dataclasses import dataclass
import secrets
import struct
import json

from .spdz_distributed import DistributedParty,ProtocolError,_hash

PRIME=2**256-189
LOWGEAR_PRIME=2**255+62*2**20+1
GOLDILOCKS_PRIME=2**64-2**32+1
SUPPORTED_PRIMES=frozenset((PRIME,LOWGEAR_PRIME,GOLDILOCKS_PRIME))


@dataclass
class QShare:
    values: list
    macs: list
    shape: tuple


def _canonical(values,prime=PRIME):
    return all(type(value) is int and 0<=value<prime for value in values)


def _pack(values,prime=PRIME):
    if not _canonical(values,prime):raise ValueError('noncanonical prime field values')
    width=(prime.bit_length()+7)//8
    return b''.join(value.to_bytes(width,'little') for value in values)


def _unpack(data,count,prime=PRIME):
    width=(prime.bit_length()+7)//8
    if len(data)!=width*count:raise ProtocolError('prime field payload length mismatch')
    values=[int.from_bytes(data[i:i+width],'little') for i in range(0,len(data),width)]
    if not _canonical(values,prime):raise ProtocolError('noncanonical prime field encoding')
    return values


def clear_matmul(a,b,shape_a,shape_b,*,prime=PRIME):
    m,k=shape_a;k2,n=shape_b
    if k!=k2:raise ValueError('matrix dimensions mismatch')
    return [sum(a[i*k+t]*b[t*n+j] for t in range(k))%prime for i in range(m) for j in range(n)]


class Prime256Party(DistributedParty):
    """One ordered authenticated stream with failure-wide key retirement."""
    def __init__(self,party,transport,alpha_share,*,session,parties=4,field_prime=PRIME,_allow_insecure_test_transport=False):
        self.party,self.transport,self.parties=party,transport,parties
        self.poisoned=False;self._failure=None;self.round=0;self.spent_triples=set();self.spent_masks=set()
        with self.operation():
            if type(field_prime) is not int or field_prime not in SUPPORTED_PRIMES:
                raise ValueError('unsupported prime field')
            self.prime=field_prime
            self.field_bytes=(field_prime.bit_length()+7)//8
            if parties!=4 or party not in range(4):raise ValueError('q256 reference requires four roles')
            if not isinstance(session,bytes) or not session:raise ValueError('nonempty session required')
            if not getattr(transport,'authenticated',False) and not _allow_insecure_test_transport:
                raise ValueError('authenticated confidential transport required')
            if not self._canonical([alpha_share]):raise ValueError('canonical field MAC key required')
            self.session=session;self.alpha_share=alpha_share;self.mac_keys=1
            self._session_hash=_hash(b'SPDZuint256-session-v2'+self.prime.to_bytes(32,'little')+struct.pack('<Q',len(session))+session)

    def _canonical(self,values):return _canonical(values,self.prime)

    def _pack(self,values):return _pack(values,self.prime)

    def _unpack(self,data,count):return _unpack(data,count,self.prime)

    def _matmul(self,a,b,shape_a,shape_b):
        return clear_matmul(a,b,shape_a,shape_b,prime=self.prime)

    def _frame(self,stage,sender,round_id,label,body):
        return (b'SPDZQ256'+self._session_hash+struct.pack('<QI',round_id,sender)
                +stage+_hash(label)+struct.pack('<Q',len(body))+body)

    def _validate(self,x):
        if (not isinstance(x.shape,tuple) or len(x.shape)!=2 or any(type(n) is not int or n<0 for n in x.shape)
                or len(x.values)!=x.shape[0]*x.shape[1] or len(x.macs)!=len(x.values)
                or not self._canonical(x.values) or not self._canonical(x.macs)):
            raise ValueError('invalid canonical q256 authenticated matrix')

    def open_many(self,shares,*,label=b''):
        with self.operation():
            shares=tuple(shares)
            if not shares:raise ValueError('opening requires at least one share')
            for share in shares:self._validate(share)
            layout=json.dumps({'field':self.prime,'encoding':f'uint{8*self.field_bytes}-le','keys':1,'shapes':[x.shape for x in shares]},
                              sort_keys=True,separators=(',',':')).encode()
            context=b'SPDZq256-open-v1'+self._session_hash+struct.pack('<Q',self.round)+_hash(layout)+_hash(label)
            values=[v for share in shares for v in share.values]
            macs=[v for share in shares for v in share.macs]
            count=len(values)
            incoming=self.broadcast(self._pack(values),label=context+b'values')
            decoded=[self._unpack(incoming[p],count) for p in range(4)]
            opened=[sum(row[i] for row in decoded)%self.prime for i in range(count)]
            residual=[(mac-self.alpha_share*value)%self.prime for mac,value in zip(macs,opened)]
            salt=self._commit_salt()
            if not isinstance(salt,bytes) or len(salt)!=32:raise ProtocolError('invalid commitment salt')
            reveal=salt+self._pack(residual)
            commits=self.broadcast(_hash(context+struct.pack('<I',self.party)+reveal),label=context+b'commit')
            if any(len(commit)!=32 for commit in commits.values()):raise ProtocolError('invalid commitment size')
            reveals=self.broadcast(reveal,label=context+b'reveal')
            checked=[]
            for p in range(4):
                if len(reveals[p])!=32+self.field_bytes*count or not secrets.compare_digest(_hash(context+struct.pack('<I',p)+reveals[p]),commits[p]):
                    raise ProtocolError('q256 residual commitment mismatch')
                checked.append(self._unpack(reveals[p][32:],count))
            if any(sum(row[i] for row in checked)%self.prime for i in range(count)):
                raise ProtocolError('q256 MAC check failed')
            output=[];offset=0
            for share in shares:
                n=len(share.values);output.append(opened[offset:offset+n]);offset+=n
            return output

    def open(self,x):return self.open_many((x,))[0]

    def add(self,x,y):
        with self.operation():
            self._validate(x);self._validate(y)
            if x.shape!=y.shape:raise ValueError('share dimensions mismatch')
            return QShare([(a+b)%self.prime for a,b in zip(x.values,y.values)],
                          [(a+b)%self.prime for a,b in zip(x.macs,y.macs)],x.shape)

    def subtract(self,x,y):
        with self.operation():
            self._validate(x);self._validate(y)
            if x.shape!=y.shape:raise ValueError('share dimensions mismatch')
            return QShare([(a-b)%self.prime for a,b in zip(x.values,y.values)],[(a-b)%self.prime for a,b in zip(x.macs,y.macs)],x.shape)

    def add_public(self,x,value):
        with self.operation():
            self._validate(x)
            values=[value]*len(x.values) if type(value) is int else list(value)
            if len(values)!=len(x.values) or not self._canonical(values):raise ValueError('invalid public matrix')
            return QShare([(a+(b if self.party==0 else 0))%self.prime for a,b in zip(x.values,values)],
                          [(a+self.alpha_share*b)%self.prime for a,b in zip(x.macs,values)],x.shape)

    def _burn_q(self,correlation):
        identifier=correlation.identifier
        old=correlation.used;correlation.used=True
        if old or identifier in self.spent_triples:raise ValueError('q256 correlations are single-use')
        self.spent_triples.add(identifier)
        if not isinstance(identifier,bytes) or len(identifier)!=32:raise ValueError('invalid correlation identifier')
        return identifier

    def matmul(self,x,y,triple):
        with self.operation():
            identifier=self._burn_q(triple)
            for share in (x,y,triple.a,triple.b,triple.c):self._validate(share)
            m,k=x.shape;k2,n=y.shape
            if k!=k2 or triple.a.shape!=x.shape or triple.b.shape!=y.shape or triple.c.shape!=(m,n):
                raise ValueError('q256 matrix triple dimensions mismatch')
            d,e=self.open_many((self.subtract(x,triple.a),self.subtract(y,triple.b)),label=b'matmul'+identifier)
            db=self._matmul(d,triple.b.values,x.shape,y.shape)
            ae=self._matmul(triple.a.values,e,x.shape,y.shape)
            de=self._matmul(d,e,x.shape,y.shape)
            dm=self._matmul(d,triple.b.macs,x.shape,y.shape)
            me=self._matmul(triple.a.macs,e,x.shape,y.shape)
            return QShare([(c+a+b+(v if self.party==0 else 0))%self.prime for c,a,b,v in zip(triple.c.values,db,ae,de)],
                          [(c+a+b+self.alpha_share*v)%self.prime for c,a,b,v in zip(triple.c.macs,dm,me,de)],(m,n))

    def square(self,x,mask):
        with self.operation():
            identifier=self._burn_q(mask)
            for share in (x,mask.a,mask.a_squared):self._validate(share)
            if x.shape!=mask.a.shape or x.shape!=mask.a_squared.shape:raise ValueError('square mask dimensions mismatch')
            d=self.open_many((self.subtract(x,mask.a),),label=b'square'+identifier)[0]
            return QShare([(c+2*v*a+(v*v if self.party==0 else 0))%self.prime for c,v,a in zip(mask.a_squared.values,d,mask.a.values)],
                          [(c+2*v*a+self.alpha_share*v*v)%self.prime for c,v,a in zip(mask.a_squared.macs,d,mask.a.macs)],x.shape)

    def input(self,mask,value=None):
        with self.operation():
            identifier=self._burn_q(mask)
            self._validate(mask.share)
            if type(mask.owner) is not int or mask.owner not in range(4):raise ValueError('invalid input owner')
            if self.party==mask.owner:
                if mask.clear is None or len(mask.clear)!=len(mask.share.values) or not self._canonical(mask.clear):raise ValueError('owner clear mask missing')
                values=list(value) if value is not None else []
                if len(values)!=len(mask.clear) or not self._canonical(values):raise ValueError('invalid private input')
                correction=self._pack([(x-r)%self.prime for x,r in zip(values,mask.clear)])
            else:
                if value is not None or mask.clear is not None:raise ValueError('nonowner supplied clear input or mask')
                correction=b''
            shape=struct.pack('<QQ',*mask.share.shape)
            received=self.broadcast(correction,label=b'q256-input'+identifier+struct.pack('<I',mask.owner)+shape)
            if any(received[p] for p in range(4) if p!=mask.owner):raise ProtocolError('nonowner input payload')
            epsilon=self._unpack(received[mask.owner],len(mask.share.values))
            return self.add_public(mask.share,epsilon)
