"""Canonical role-private request and local output types for the native OLE driver."""
from dataclasses import dataclass
from pathlib import Path
import json
import struct

PRIME=(1<<127)-1
MAGIC=b'SIVOLE01'


def _fields(values):
    if any(type(value)is not int or not 0<=value<PRIME for value in values):
        raise ValueError('expected canonical F127 inputs')
    return b''.join(value.to_bytes(16,'little') for value in values)


def _shape(length,calls):
    if type(length)is not int or type(calls)is not int or not 1<=length<=8192 or not 1<=calls<=4096 or length*calls>(1<<24):
        raise ValueError('unsupported native OLE dimensions')
    return MAGIC+struct.pack('<QQ',length,calls)


@dataclass(frozen=True)
class ScalarOleRequest:
    length:int
    scalars:tuple[int,...]

    def write(self,path):
        if type(self.scalars)is not tuple:
            raise ValueError('scalar requests require an immutable tuple')
        Path(path).write_bytes(_shape(self.length,len(self.scalars))+_fields(self.scalars))


@dataclass(frozen=True)
class VectorOleRequest:
    vectors:tuple[tuple[int,...],...]

    @property
    def length(self):
        return len(self.vectors[0]) if self.vectors else 0

    def write(self,path):
        if type(self.vectors)is not tuple or any(type(vector)is not tuple or len(vector)!=self.length for vector in self.vectors):
            raise ValueError('vector requests require equal immutable tuples')
        Path(path).write_bytes(_shape(self.length,len(self.vectors))+_fields(tuple(value for vector in self.vectors for value in vector)))


@dataclass(frozen=True)
class LocalOleBatch:
    role:int
    shares:tuple[tuple[int,...],...]
    lower_layer_checks_verified:bool
    privacy_reviewed:bool=False
    transport_authenticated:bool=False

    @classmethod
    def read(cls,path,*,role,length,calls,authenticated=False,self_role=None,peer_role=None,session=None):
        result=json.loads(Path(path).read_text())
        if (result['protocol']!='wolverine-r1-chosen-input-ole' or result['role']!=role or
                result['length']!=length or result['calls']!=calls or result['prime_bits']!=127 or
                len(result['sessions'])!=calls):
            raise ValueError('native OLE output contract mismatch')
        shares=[]
        for index,call in enumerate(result['sessions']):
            if call['index']!=index or call['fresh_rot_choices']!=508 or len(call['share'])!=length:
                raise ValueError('incomplete or reordered native OLE batch')
            _fields(call['share'])
            shares.append(tuple(call['share']))
        if result['security_reviewed'] is not False or result['transport_authenticated'] is not authenticated:
            raise ValueError('unexpected native OLE security metadata')
        if authenticated:
            if (self_role is None or peer_role is None or session is None or
                    result.get('self_role')!=self_role or result.get('peer_role')!=peer_role or
                    result.get('tls_session')!=session):
                raise ValueError('TLS OLE output identity/session mismatch')
        return cls(role,tuple(shares),True,False,authenticated)

    def outgoing_u(self):
        if self.role!=2:
            raise ValueError('only vector-side shares are outgoing u')
        return self.shares

    def incoming_w(self):
        if self.role!=1:
            raise ValueError('only scalar-side shares are incoming w')
        return tuple(tuple((-value)%PRIME for value in row) for row in self.shares)
