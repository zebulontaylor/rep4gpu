"""One-use fixed-model q256 row correlations supplied by native preprocessing.

The provider authenticates W, A, and F=A@W before import. Online evaluation
checks D=X-A and computes F+D@W. This module does not prove the provider's
relation; a correctly authenticated wrong F still gives a wrong result.
"""
from dataclasses import dataclass
import json
import struct
import weakref

from .spdz_distributed import ProtocolError,_hash
from .spdz_prime256 import PRIME,QShare,clear_matmul


@dataclass(frozen=True)
class _FrozenShare:
    values: tuple
    macs: tuple
    shape: tuple


@dataclass
class _Rows:
    owner: object
    session: bytes
    alpha: int
    role: int
    model_id: bytes
    identifier: bytes
    weights: _FrozenShare
    a: _FrozenShare
    factor: _FrozenShare
    used: bool=False


_states=weakref.WeakKeyDictionary()


class PreparedQRows:
    """Opaque role/session-bound capability for independent query rows."""
    __slots__=('__weakref__',)
    def __init__(self):raise TypeError('use import_prepared_rows')
    def __reduce__(self):raise TypeError('prepared correlations cannot be copied or serialized')
    @property
    def used(self):return _states[self].used
    @property
    def input_shape(self):return _states[self].a.shape
    @property
    def output_shape(self):
        state=_states[self]
        return state.a.shape[0],state.weights.shape[1]


def _freeze(share):return _FrozenShare(tuple(share.values),tuple(share.macs),tuple(share.shape))


def import_prepared_rows(party,batch,row_counts,*,model_id):
    """Consume a native batch into complete disjoint contiguous query handles.

    The caller supplies only this role's W/A/F shares. Positive row counts must
    cover every A/F row. Model shares are copied once into immutable tuples and
    shared by all children; A/F rows are separate immutable copies. Import does
    no communication. Each query's checked-opening label binds the complete
    partition, provider identifier, model ID, and exact row range.
    """
    with party.operation():
        if getattr(party,'prime',PRIME)!=PRIME:
            raise ValueError('prepared q256 rows require the q=2**256-189 field')
        identifier=party._burn_q(batch)
        if not isinstance(model_id,bytes) or not model_id or model_id!=batch.model_id:
            raise ValueError('prepared model identifier mismatch')
        for share in (batch.weights,batch.a,batch.factor):party._validate(share)
        rows,k=batch.a.shape
        if batch.weights.shape[0]!=k or batch.factor.shape!=(rows,batch.weights.shape[1]):
            raise ValueError('prepared provider dimensions mismatch')
        counts=tuple(row_counts)
        if not counts or any(type(count) is not int or count<=0 for count in counts) or sum(counts)!=rows:
            raise ValueError('row counts must positively cover the complete batch')
        weights=_freeze(batch.weights)
        context=b'SPDZq256-fixed-right-v1'+identifier+_hash(model_id)+_hash(json.dumps(counts).encode())
        result=[];offset=0;n=weights.shape[1]
        for count in counts:
            end=offset+count
            a=_FrozenShare(tuple(batch.a.values[offset*k:end*k]),tuple(batch.a.macs[offset*k:end*k]),(count,k))
            factor=_FrozenShare(tuple(batch.factor.values[offset*n:end*n]),tuple(batch.factor.macs[offset*n:end*n]),(count,n))
            state=_Rows(weakref.ref(party),party.session,party.alpha_share,party.party,bytes(model_id),
                        _hash(context+struct.pack('<QQ',offset,end)),weights,a,factor)
            handle=object.__new__(PreparedQRows);_states[handle]=state;result.append(handle);offset=end
        return tuple(result)


def matmul_prepared_rows(party,x,prepared,*,model_id):
    """Consume fresh A/F rows, check X-A, and return authenticated F+D@W."""
    with party.operation():
        if getattr(party,'prime',PRIME)!=PRIME:
            raise ValueError('prepared q256 rows require the q=2**256-189 field')
        if type(prepared) is not PreparedQRows or prepared not in _states:raise ValueError('unknown prepared q256 rows')
        state=_states[prepared]
        if state.used:raise ValueError('prepared q256 rows are single-use')
        state.used=True
        if (state.owner() is not party or state.session!=party.session or state.alpha!=party.alpha_share
                or state.role!=party.party or state.model_id!=model_id):
            raise ProtocolError('prepared q256 party, session, key, or model mismatch')
        party._validate(x)
        if x.shape!=state.a.shape:raise ValueError('prepared query dimensions mismatch')
        d=party.open_many((party.subtract(x,state.a),),label=state.identifier+b'q256-fixed-query')[0]
        product=clear_matmul(d,state.weights.values,x.shape,state.weights.shape)
        mac_product=clear_matmul(d,state.weights.macs,x.shape,state.weights.shape)
        return QShare([(a+b)%PRIME for a,b in zip(state.factor.values,product)],
                      [(a+b)%PRIME for a,b in zip(state.factor.macs,mac_product)],prepared.output_shape)
