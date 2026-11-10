"""Per-party authenticated prime-field Beaver online protocol.

Preprocessing is trusted: this module does not generate or verify distributed
triples. Every failed session retires its MAC keys and all preprocessing. Never
recreate a party using the keys of an aborted session.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import secrets
import struct

import numpy as np
import torch

from .spdz import PRIME, AuthenticationError, field_matmul


class ProtocolError(AuthenticationError):
    """A frame, consistency check, or session contract failed."""


@dataclass
class AuthShare:
    values: torch.Tensor
    macs: torch.Tensor


@dataclass
class MatrixTriple:
    a: AuthShare
    b: AuthShare
    c: AuthShare
    used: bool = False
    identifier: bytes | None = None


@dataclass
class ElementwiseTriple(MatrixTriple):
    pass


def split_share(central):
    """Clone a trusted dealer's central share into independent local bundles."""
    return [AuthShare(central.values[p].clone(),central.macs[:,p].clone())
            for p in range(central.values.shape[0])]


def split_triple(central):
    """Consume one dealer triple and return single-use local matrix triples."""
    if central.used:
        raise ValueError('central triple was already consumed or split')
    central.used=True
    a,b,c=split_share(central.a),split_share(central.b),split_share(central.c)
    identifier=secrets.token_bytes(32)
    return [MatrixTriple(x,y,z,identifier=identifier) for x,y,z in zip(a,b,c)]


def split_elementwise_triple(central):
    """Split trusted elementwise correlations; the dealer must supply C=A*B."""
    return [ElementwiseTriple(t.a,t.b,t.c,identifier=t.identifier) for t in split_triple(central)]


def _hash(data):
    return hashlib.blake2b(data,digest_size=32).digest()


def _pack(tensor):
    return tensor.detach().contiguous().cpu().numpy().astype('<u4').tobytes()


class DistributedParty:
    """Authenticated online evaluation with trusted preprocessing and abort.

    ``alpha_share`` is this party's vector of independent field MAC-key shares.
    Five keys are the default dealer configuration (approximately 155 key bits).
    The transport must provide authenticated confidential point-to-point links.
    Public shapes, protocol operations, and preprocessing must agree at all roles.
    """
    def __init__(self,party,transport,alpha_share,*,parties=4,session,
                 _allow_insecure_test_transport=False,stack_macs=False):
        self.party,self.transport,self.parties=party,transport,parties
        self.poisoned=False
        self.round=0
        self.spent_triples=set()
        self._failure=None
        self.stack_macs=bool(stack_macs)
        with self.operation():
            if not isinstance(session,bytes) or not session:
                raise ValueError('session must be nonempty bytes')
            if not isinstance(parties,int) or not 2<=parties<=65535 or party not in range(parties):
                raise ValueError('invalid party count or party identity')
            if not getattr(transport,'authenticated',False) and not _allow_insecure_test_transport:
                raise ValueError('authenticated confidential transport is required')
            if alpha_share.dtype!=torch.int64 or alpha_share.ndim!=1 or alpha_share.numel()<5:
                raise ValueError('alpha_share must contain at least five int64 MAC-key shares')
            if torch.any((alpha_share<0)|(alpha_share>=PRIME)).item():
                raise ValueError('MAC-key shares must be canonical field elements')
            self.session=session
            self._session_hash=_hash(b'SPDZ31-session-v1'+struct.pack('<Q',len(session))+session)
            self.alpha_share=alpha_share.clone()
            self.device=alpha_share.device
            self.mac_keys=alpha_share.numel()

    def abort(self,error):
        """Poison this session and interrupt peers; callers must retire its keys."""
        self.poisoned=True
        self._failure=error
        current=self.transport
        seen=set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            try:
                method=getattr(current,'abort',None)
                if method is not None:
                    method(error)
                    return
                network=getattr(current,'network',None)
                if network is not None and hasattr(network,'abort'):
                    network.abort(error)
                    return
                if hasattr(current,'close'):
                    current.close()
                    return
            except BaseException:
                return
            current=getattr(current,'underlying',None)

    @contextmanager
    def operation(self):
        if self.poisoned:
            raise ProtocolError('session is poisoned; retire its MAC keys and preprocessing') from self._failure
        try:
            yield
        except BaseException as error:
            self.abort(error)
            raise

    def _frame(self,stage,sender,round_id,label,body):
        header=(b'SPDZ31v1'+self._session_hash+struct.pack('<QI',round_id,sender)
                +stage+_hash(label)+struct.pack('<Q',len(body)))
        return header+body

    def _body(self,frame,stage,sender,round_id,label):
        if not isinstance(frame,bytes) or len(frame)<93:
            raise ProtocolError('invalid broadcast frame')
        size=struct.unpack('<Q',frame[85:93])[0]
        if len(frame)!=93+size or frame[:93]!=self._frame(stage,sender,round_id,label,b'')[:85]+struct.pack('<Q',size):
            raise ProtocolError('broadcast session, round, type, label, or length mismatch')
        return frame[93:]

    def broadcast(self,payload,*,label):
        """Broadcast bytes with an ordered all-to-all digest echo, including self.

        This provides consistency with abort, rather than reliable broadcast.
        ``label`` must bind the operation's public types, shapes, and identifiers.
        """
        with self.operation():
            if not isinstance(payload,bytes) or not isinstance(label,bytes):
                raise TypeError('broadcast payload and label must be bytes')
            if not 0<=self.round<2**64:
                raise ProtocolError('broadcast round counter exhausted')
            round_id=self.round
            self.round+=1
            own=self._frame(b'V',self.party,round_id,label,payload)
            peers=set(range(self.parties))-{self.party}
            incoming=self.transport.exchange({p:own for p in peers})
            if set(incoming)!=peers:
                raise ProtocolError('broadcast peer set mismatch')
            frames={self.party:own,**incoming}
            bodies={p:self._body(frames[p],b'V',p,round_id,label) for p in range(self.parties)}
            view=b''.join(_hash(struct.pack('<I',p)+frames[p]) for p in range(self.parties))
            echo=self._frame(b'E',self.party,round_id,label,view)
            checks=self.transport.exchange({p:echo for p in peers})
            if set(checks)!=peers:
                raise ProtocolError('broadcast echo peer set mismatch')
            for p in peers:
                if not secrets.compare_digest(self._body(checks[p],b'E',p,round_id,label),view):
                    raise ProtocolError('broadcast equivocation or inconsistent echo')
            return bodies

    def _validate(self,x):
        if not isinstance(x,AuthShare) or x.values.dtype!=torch.int64 or x.values.ndim!=2:
            raise ValueError('expected a local authenticated int64 matrix')
        if x.macs.dtype!=torch.int64 or x.macs.shape!=(self.mac_keys,*x.values.shape):
            raise ValueError('authenticated matrix MAC dimensions mismatch')
        if x.values.device!=self.device or x.macs.device!=self.device:
            raise ValueError('authenticated matrices must use the party device')
        for t in (x.values,x.macs):
            if torch.any((t<0)|(t>=PRIME)).item():
                raise ValueError('shares must be canonical field elements')

    def _unpack(self,data,count):
        if len(data)!=count*4:
            raise ProtocolError('field payload length mismatch')
        array=np.frombuffer(data,dtype='<u4')
        if np.any(array>=PRIME):
            raise ProtocolError('noncanonical field encoding')
        return torch.from_numpy(array.astype(np.int64)).to(self.device)

    def _commit_salt(self):
        """Return fresh private commitment entropy for one live opening."""
        return secrets.token_bytes(32)

    def open_many(self,shares,*,label=b''):
        with self.operation():
            shares=tuple(shares)
            if not shares:
                raise ValueError('open_many requires at least one matrix')
            for x in shares:
                self._validate(x)
            layout=json.dumps({'field':PRIME,'dtype':'uint32-le','keys':self.mac_keys,
                               'shapes':[list(x.values.shape) for x in shares]},sort_keys=True,separators=(',',':')).encode()
            context=(b'SPDZ31-open-v1'+self._session_hash+struct.pack('<Q',self.round)
                     +_hash(layout)+_hash(label))
            count=sum(x.values.numel() for x in shares)
            values=torch.cat([x.values.reshape(-1) for x in shares])
            opened=torch.zeros_like(values)
            received=self.broadcast(_pack(values),label=context+b'values')
            for p in range(self.parties):
                opened=(opened+self._unpack(received[p],count)).remainder(PRIME)
            macs=torch.cat([x.macs.reshape(self.mac_keys,-1) for x in shares],dim=1)
            residual=(macs-self.alpha_share[:,None]*opened).remainder(PRIME)
            raw=_pack(residual)
            salt=self._commit_salt()
            binding=context+struct.pack('<I',self.party)
            commitment=_hash(binding+salt+raw)
            commitments=self.broadcast(commitment,label=context+b'commit')
            if any(len(value)!=32 for value in commitments.values()):
                raise ProtocolError('invalid residual commitment length')
            reveals=self.broadcast(salt+raw,label=context+b'reveal')
            total=torch.zeros_like(residual)
            for p in range(self.parties):
                reveal=reveals[p]
                if len(reveal)!=32+self.mac_keys*count*4 or not secrets.compare_digest(
                        _hash(context+struct.pack('<I',p)+reveal),commitments[p]):
                    raise ProtocolError('residual commitment opening failed')
                total=(total+self._unpack(reveal[32:],self.mac_keys*count).reshape(self.mac_keys,count)).remainder(PRIME)
            if torch.any(total!=0).item():
                raise AuthenticationError('distributed field MAC check failed')
            result=[]
            offset=0
            for x in shares:
                size=x.values.numel()
                result.append(opened[offset:offset+size].reshape(x.values.shape))
                offset+=size
            return result

    def open(self,x):
        return self.open_many((x,))[0]

    def add(self,x,y):
        with self.operation():
            self._validate(x);self._validate(y)
            if x.values.shape!=y.values.shape:
                raise ValueError('authenticated addition dimensions mismatch')
            return AuthShare((x.values+y.values).remainder(PRIME),(x.macs+y.macs).remainder(PRIME))

    def subtract(self,x,y):
        with self.operation():
            self._validate(x);self._validate(y)
            if x.values.shape!=y.values.shape:
                raise ValueError('authenticated subtraction dimensions mismatch')
            return AuthShare((x.values-y.values).remainder(PRIME),(x.macs-y.macs).remainder(PRIME))

    def add_public(self,x,value):
        with self.operation():
            self._validate(x)
            value=torch.as_tensor(value,dtype=torch.int64,device=self.device).remainder(PRIME)
            value=torch.broadcast_to(value,x.values.shape)
            return AuthShare((x.values+value if self.party==0 else x.values.clone()).remainder(PRIME),
                             (x.macs+self.alpha_share[:,None,None]*value).remainder(PRIME))

    def _burn(self,triple):
        old=triple.used
        triple.used=True
        identifier=triple.identifier
        if not isinstance(identifier,bytes) or len(identifier)!=32:
            raise ValueError('triples require a trusted 32-byte preprocessing identifier')
        if old or identifier in self.spent_triples:
            raise ValueError('triples are single-use')
        self.spent_triples.add(identifier)
        return identifier

    def matmul(self,x,y,triple):
        with self.operation():
            identifier=self._burn(triple)
            if type(triple) is not MatrixTriple:
                raise ValueError('matmul requires a matrix correlation triple')
            for value in (x,y,triple.a,triple.b,triple.c):self._validate(value)
            if x.values.shape!=triple.a.values.shape or y.values.shape!=triple.b.values.shape or x.values.shape[1]!=y.values.shape[0] or triple.c.values.shape!=(x.values.shape[0],y.values.shape[1]):
                raise ValueError('matrix triple dimensions mismatch')
            d,e=self.open_many((self.subtract(x,triple.a),self.subtract(y,triple.b)),label=b'matmul'+identifier)
            return self._matmul_local(triple,d,e)

    def _matmul_local(self,triple,d,e):
        public=field_matmul(d,e)
        if self.stack_macs:
            m,n=public.shape
            right=field_matmul(torch.cat((triple.a.values,*triple.a.macs.unbind(0)),dim=0),e)
            right=right.reshape(self.mac_keys+1,m,n)
            left=field_matmul(d,torch.cat((triple.b.values,*triple.b.macs.unbind(0)),dim=1))
            value=(triple.c.values+left[:,:n]+right[0]).remainder(PRIME)
            left_macs=left[:,n:].reshape(m,self.mac_keys,n).permute(1,0,2)
            macs=(triple.c.macs+left_macs+right[1:]).remainder(PRIME)
        else:
            value=(triple.c.values+field_matmul(d,triple.b.values)+field_matmul(triple.a.values,e)).remainder(PRIME)
            macs=torch.stack([(triple.c.macs[key]+field_matmul(d,triple.b.macs[key])+
                               field_matmul(triple.a.macs[key],e)).remainder(PRIME)
                              for key in range(self.mac_keys)])
        return self.add_public(AuthShare(value,macs),public)

    def mul(self,x,y,triple):
        with self.operation():
            identifier=self._burn(triple)
            if type(triple) is not ElementwiseTriple:
                raise ValueError('mul requires an elementwise correlation triple')
            for value in (x,y,triple.a,triple.b,triple.c):self._validate(value)
            if any(value.values.shape!=x.values.shape for value in (y,triple.a,triple.b,triple.c)):
                raise ValueError('elementwise triple dimensions mismatch')
            d,e=self.open_many((self.subtract(x,triple.a),self.subtract(y,triple.b)),label=b'elementwise'+identifier)
            value=(triple.c.values+d*triple.b.values+triple.a.values*e).remainder(PRIME)
            macs=(triple.c.macs+d*triple.b.macs+triple.a.macs*e).remainder(PRIME)
            return self.add_public(AuthShare(value,macs),(d*e).remainder(PRIME))
