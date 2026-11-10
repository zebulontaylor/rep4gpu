"""Single-use model-dependent preparation for authenticated Beaver matmul.

Prepare one fresh correlation for each future query before seeing that query.
Preparation checks E=W-B and computes F=C+A@E. Online evaluation opens only
D=X-A through the runtime's masked-opening policy, then returns F+D@W with the
corresponding authenticated MACs. Deferred runtimes authenticate the pending
fixed circuit before their final ordinary output opening. Preparation
includes model-dependent communication and matrix work; report it separately
from query latency. ``model_id`` is application metadata, not a proof of which
model supplied the authenticated weights.
"""
from dataclasses import dataclass,replace
import struct
import json
import weakref

import torch

from .spdz import PRIME,field_matmul
from .spdz_distributed import AuthShare,MatrixTriple,ProtocolError,_hash


@dataclass
class _State:
    party: object
    session: bytes
    session_hash: bytes
    alpha: object
    alpha_version: object
    role: int
    model_id: bytes
    identifier: bytes
    a: AuthShare
    factor: AuthShare
    weights: AuthShare
    versions: tuple = ()
    used: bool = False


_states = weakref.WeakKeyDictionary()


class PreparedRightMatmul:
    """Opaque party-bound handle created by ``prepare_right_matmul``.

    Handles cannot be serialized, copied into another role, or reset after use.
    The stored A, F, and model snapshot are private owned tensors.
    """
    __slots__ = ('__weakref__',)

    def __init__(self):
        raise TypeError('use prepare_right_matmul to create a prepared handle')

    def __reduce__(self):
        raise TypeError('prepared handles cannot be serialized or copied')

    @property
    def used(self):
        return _states[self].used

    @property
    def model_id(self):
        return _states[self].model_id

    @property
    def triple_id(self):
        return _states[self].identifier

    @property
    def input_shape(self):
        return tuple(_states[self].a.values.shape)

    @property
    def output_shape(self):
        state = _states[self]
        return (state.a.values.shape[0],state.weights.values.shape[1])


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _clone(share):
    with torch.inference_mode(False):
        return AuthShare(share.values.clone(),share.macs.clone())


def _versions(state):
    versions=tuple(_version(tensor) for share in (state.a,state.factor,state.weights)
                   for tensor in (share.values,share.macs))
    if any(version is None for version in versions):
        raise ProtocolError('prepared storage requires tensor mutation counters')
    return versions


def _consume(party,prepared,model_id):
    if type(prepared) is not PreparedRightMatmul or prepared not in _states:
        raise ValueError('unknown prepared-right capability')
    state=_states[prepared]
    if state.used:
        raise ValueError('prepared-right correlations are single-use')
    state.used=True
    if (state.party() is not party or state.session!=party.session or state.session_hash!=party._session_hash
            or state.alpha() is not party.alpha_share or state.alpha_version!=_version(party.alpha_share)
            or state.role!=party.party):
        raise ProtocolError('prepared-right capability belongs to another party, session, or key')
    if not isinstance(model_id,bytes) or model_id!=state.model_id:
        raise ValueError('prepared-right model_id mismatch')
    if state.versions!=_versions(state):
        raise ProtocolError('prepared-right private storage was mutated')
    return state


def _factor(a,c,e):
    if a.values.is_cuda:
        try:
            from .field_prepared import prepared_right_factor
        except ImportError:
            pass
        else:
            values,macs = prepared_right_factor(a.values,a.macs,c.values,c.macs,e)
            return AuthShare(values,macs)
    h,m = a.macs.shape[:2]
    stacked = torch.cat((a.values.long(),*a.macs.long().unbind(0)),dim=0)
    product = field_matmul(stacked,e.long()).reshape(h+1,m,e.shape[1])
    return AuthShare((c.values.long()+product[0]).remainder(PRIME),
                     (c.macs.long()+product[1:]).remainder(PRIME))


def _online(d,weights,factor):
    if weights.values.is_cuda:
        if d.shape[0] <= 4 and d.shape[1] < 2**31:
            from .field_prepared_canonical import prepared_right_online_canonical
            values,macs = prepared_right_online_canonical(d,weights.values,weights.macs,factor.values,factor.macs)
            return AuthShare(values,macs)
        try:
            from .field_prepared import prepared_right_online
        except ImportError:
            pass
        else:
            values,macs = prepared_right_online(d,weights.values,weights.macs,factor.values,factor.macs)
            return AuthShare(values,macs)
    h = weights.macs.shape[0]
    m,n = d.shape[0],weights.values.shape[1]
    stacked = torch.cat((weights.values.long(),*weights.macs.long().unbind(0)),dim=1)
    product = field_matmul(d.long(),stacked)
    return AuthShare((factor.values.long()+product[:,:n]).remainder(PRIME),
                     (factor.macs.long()+product[:,n:].reshape(m,h,n).permute(1,0,2)).remainder(PRIME))


def prepare_right_matmul(party,weights,triple,*,model_id):
    """Burn a fresh triple and authenticate its model-dependent preparation.

    Every role supplies the same nonempty public ``model_id``. No query input
    enters this function. All pending deferred openings must already be checked.
    The caller's tensors are cloned so later mutations cannot alter the handle.
    """
    with party.operation():
        identifier = party._burn(triple)
        if type(triple) is not MatrixTriple:
            raise ValueError('right preparation requires a matrix triple')
        if not isinstance(model_id,bytes) or not model_id:
            raise ValueError('model_id must be nonempty bytes')
        if getattr(party,'pending_values',0):
            raise ProtocolError('check pending openings before model-dependent preparation')
        for share in (weights,triple.a,triple.b,triple.c):
            party._validate(share)
        ensure = getattr(party,'_ensure_canonical',None)
        if ensure is not None:
            ensure((weights,triple.a,triple.b,triple.c))
        m,k = triple.a.values.shape
        if weights.values.shape != triple.b.values.shape or weights.values.shape[0] != k or triple.c.values.shape != (m,weights.values.shape[1]):
            raise ValueError('prepared model and matrix triple dimensions mismatch')
        a,b,c,right = (_clone(share) for share in (triple.a,triple.b,triple.c,weights))
        mark = getattr(party,'_mark',lambda share:share)
        for share in (a,b,c,right):
            mark(share)
        layout = json.dumps(dict(a=list(a.values.shape),w=list(right.values.shape),
                                 keys=party.mac_keys,model_id=model_id.hex()),
                            sort_keys=True,separators=(',',':')).encode()
        label = b'SPDZ31-prepare-right-v1'+identifier+_hash(layout)
        e = party.open_many((party.subtract(right,b),),label=label)[0]
        with torch.inference_mode(False):
            factor = mark(_factor(a,c,e))
        # Ordinary open_many above must finish authentication even on runtimes
        # that defer their private masked-opening hook.
        if getattr(party,'pending_values',0):
            raise ProtocolError('model preparation returned with unchecked openings')
        handle = object.__new__(PreparedRightMatmul)
        state = _State(weakref.ref(party),party.session,party._session_hash,
                       weakref.ref(party.alpha_share),_version(party.alpha_share),
                       party.party,bytes(model_id),identifier,a,factor,right)
        state.versions=_versions(state)
        _states[handle]=state
        return handle


def matmul_prepared_right(party,x,prepared,*,model_id):
    """Consume prepared A/F once, open D=X-A, and evaluate F+D@W.

    The runtime's masked-opening policy controls immediate or deferred MAC
    authentication. Deferred results stay inside its fixed circuit until an
    ordinary output opening flushes all checks. A failed attempt cannot restore
    the handle. ``model_id`` must match the
    preparation metadata, and the exact party/session/key object must own it.
    """
    with party.operation():
        state = _consume(party,prepared,model_id)
        party._validate(x)
        if x.values.shape != state.a.values.shape:
            raise ValueError('prepared-right query dimensions mismatch')
        label = b'SPDZ31-query-prepared-right-v1'+state.identifier+_hash(model_id)
        masked_open = getattr(party,'_masked_open_many',party.open_many)
        d = masked_open((party.subtract(x,state.a),),label=label)[0]
        result = _online(d,state.weights,state.factor)
        return getattr(party,'_mark',lambda share:share)(result)



def partition_prepared_right(party,prepared,row_counts,*,model_id):
    """Consume a prepared batch and return single-use contiguous row handles.

    Positive integer counts must cover every parent row exactly, in order.
    Partitioning communicates nothing: each child opening binds the complete
    partition and its row range, so different role partitions abort before an
    opened candidate is used. Children own their A/F rows and share the owned,
    mutation-checked model snapshot. Even an invalid partition attempt consumes
    the parent and poisons the party; it cannot be retried under the same keys.
    """
    with party.operation():
        state=_consume(party,prepared,model_id)
        counts=tuple(row_counts)
        if (not counts or any(type(count) is not int or count<=0 for count in counts)
                or sum(counts)!=state.a.values.shape[0]):
            raise ValueError('row counts must be positive integers covering all prepared rows')
        layout=json.dumps(counts,separators=(',',':')).encode()
        prefix=b'SPDZ31-partition-right-v1'+state.identifier+_hash(layout)
        children=[]
        offset=0
        mark=getattr(party,'_mark',lambda share:share)
        for index,count in enumerate(counts):
            end=offset+count
            a=mark(_clone(AuthShare(state.a.values[offset:end],state.a.macs[:,offset:end])))
            factor=mark(_clone(AuthShare(state.factor.values[offset:end],state.factor.macs[:,offset:end])))
            identifier=_hash(prefix+struct.pack('<QQQ',index,offset,end))
            child_state=replace(state,identifier=identifier,a=a,factor=factor,used=False)
            child_state.versions=_versions(child_state)
            child=object.__new__(PreparedRightMatmul)
            _states[child]=child_state
            children.append(child)
            offset=end
        return tuple(children)
