"""One-use Goldilocks row masks prepared from checked generic matrix triples."""
from dataclasses import dataclass
import hashlib
import json
import struct
import weakref

from .spdz_distributed import ProtocolError
from .spdz_prime256 import GOLDILOCKS_PRIME as PRIME, QShare


@dataclass(frozen=True)
class _Share:
    values: tuple
    macs: tuple
    shape: tuple


@dataclass
class _State:
    owner: object
    session: bytes
    key: int
    role: int
    model: bytes
    identifier: bytes
    a: _Share
    factor: _Share
    weights: _Share | None = None
    used: bool = False


_rows = weakref.WeakKeyDictionary()
_squares = weakref.WeakKeyDictionary()


class FixedWeightRows:
    """Opaque one-use query rows bound to their party, key, epoch, and model."""
    __slots__ = ('__weakref__',)
    def __init__(self): raise TypeError('use prepare_fixed_weights')
    def __reduce__(self): raise TypeError('row masks cannot be copied or serialized')
    @property
    def used(self): return _rows[self].used
    @property
    def input_shape(self): return _rows[self].a.shape


class SquareRows:
    """Opaque one-use partition of an authenticated square-mask batch."""
    __slots__ = ('__weakref__',)
    def __init__(self): raise TypeError('use partition_square_rows')
    def __reduce__(self): raise TypeError('square masks cannot be copied or serialized')
    @property
    def used(self): return _squares[self].used


def _freeze(share): return _Share(tuple(share.values), tuple(share.macs), tuple(share.shape))


def _field(party):
    if getattr(party, 'prime', None) != PRIME:
        raise ValueError('Goldilocks row preparation requires p=2**64-2**32+1')


def _counts(counts, rows):
    counts = tuple(counts)
    if not counts or any(type(n) is not int or n < 1 for n in counts) or sum(counts) != rows:
        raise ValueError('positive row counts must cover the complete batch')
    return counts


def _slice(share, start, end):
    n = share.shape[1]
    return _Share(share.values[start*n:end*n], share.macs[start*n:end*n], (end-start, n))


def _state(party, handle, registry, model=None):
    if handle not in registry:
        raise ProtocolError('unknown Goldilocks row handle')
    state = registry[handle]
    if state.used:
        raise ProtocolError('Goldilocks row masks are single-use')
    state.used = True
    if (state.owner() is not party or state.session != party.session or state.key != party.alpha_share
            or state.role != party.party or (model is not None and state.model != model)):
        raise ProtocolError('Goldilocks row party, epoch, key, or model mismatch')
    return state


def _validate_masked_capability(party, handle, kind, label):
    """Validate a burned row capability for the protected deferred-opening hook."""
    _field(party)
    registry = _rows if kind == 'matmul' else _squares if kind == 'square' else None
    if registry is None or handle not in registry:
        raise ProtocolError('unknown masked-difference capability')
    state = registry[handle]
    expected = state.identifier+b'/D' if kind == 'matmul' else b'square'+state.identifier
    if (not state.used or state.owner() is not party or state.session != party.session or
            state.key != party.alpha_share or state.role != party.party or label != expected):
        raise ProtocolError('masked-difference capability or label mismatch')
    return state


def _open_difference(party, difference, handle, kind, label):
    hook = getattr(party, '_open_goldilocks_difference', None)
    if hook is not None:
        return hook(difference, handle, kind=kind, label=label)
    return party.open_many((difference,), label=label)[0]


def prepare_fixed_weights(party, weights, triple, row_counts, *, model_id):
    """Burn a triple, check W-B, and partition A and F=C+A@(W-B).

    F's MAC is MAC(C)+MAC(A)@(W-B). Every child owns disjoint A/F rows;
    their fixed weight snapshot is shared and immutable. All source snapshots
    are taken before the checked opening.
    """
    with party.operation():
        _field(party)
        flush = getattr(party, 'flush_pending', None)
        if flush is not None:
            flush()
        identifier = party._burn_q(triple)
        if not isinstance(model_id, bytes) or not model_id:
            raise ValueError('nonempty public model identifier required')
        for share in (weights, triple.a, triple.b, triple.c): party._validate(share)
        rows, k = triple.a.shape
        n = weights.shape[1]
        if weights.shape != (k, n) or triple.b.shape != weights.shape or triple.c.shape != (rows, n):
            raise ValueError('fixed weight triple dimensions mismatch')
        counts = _counts(row_counts, rows)
        w, a, b, c = map(_freeze, (weights, triple.a, triple.b, triple.c))
        context = hashlib.sha256(b'Goldilocks-fixed-weights-v1\0' + identifier +
                                 hashlib.sha256(model_id).digest() +
                                 json.dumps([counts, a.shape, w.shape], separators=(',', ':')).encode()).digest()
        e = party.open_many((party.subtract(w, b),), label=context+b'/E')[0]
        ae = party._matmul(a.values, e, a.shape, w.shape)
        me = party._matmul(a.macs, e, a.shape, w.shape)
        f = _Share(tuple((x+y)%PRIME for x,y in zip(c.values, ae)),
                   tuple((x+y)%PRIME for x,y in zip(c.macs, me)), c.shape)
        handles = []
        start = 0
        for count in counts:
            end = start + count
            child = hashlib.sha256(context+struct.pack('<QQ', start, end)).digest()
            state = _State(weakref.ref(party), party.session, party.alpha_share, party.party,
                           bytes(model_id), child, _slice(a, start, end), _slice(f, start, end), w)
            handle = object.__new__(FixedWeightRows)
            _rows[handle] = state
            handles.append(handle)
            start = end
        return tuple(handles)


def matmul_rows(party, x, handle, *, model_id):
    """Check D=X-A before using a fresh row handle to compute F+D@W."""
    with party.operation():
        _field(party)
        if not isinstance(model_id, bytes) or not model_id:
            raise ValueError('nonempty public model identifier required')
        state = _state(party, handle, _rows, model_id)
        party._validate(x)
        if x.shape != state.a.shape:
            raise ValueError('query row dimensions mismatch')
        d = _open_difference(party, party.subtract(x, state.a), handle, 'matmul', state.identifier+b'/D')
        values = party._matmul(d, state.weights.values, x.shape, state.weights.shape)
        macs = party._matmul(d, state.weights.macs, x.shape, state.weights.shape)
        return QShare([(a+b)%PRIME for a,b in zip(state.factor.values, values)],
                      [(a+b)%PRIME for a,b in zip(state.factor.macs, macs)], state.factor.shape)


def partition_square_rows(party, mask, row_counts):
    """Burn one native square batch into complete disjoint row capabilities."""
    with party.operation():
        _field(party)
        identifier = party._burn_q(mask)
        party._validate(mask.a); party._validate(mask.a_squared)
        if mask.a.shape != mask.a_squared.shape:
            raise ValueError('square mask dimensions mismatch')
        counts = _counts(row_counts, mask.a.shape[0])
        a, c = map(_freeze, (mask.a, mask.a_squared))
        context = hashlib.sha256(b'Goldilocks-square-rows-v1\0'+identifier+
                                 json.dumps(counts, separators=(',', ':')).encode()).digest()
        handles = []
        start = 0
        for count in counts:
            end = start + count
            handle = object.__new__(SquareRows)
            _squares[handle] = _State(weakref.ref(party), party.session, party.alpha_share, party.party,
                                     b'', hashlib.sha256(context+struct.pack('<QQ',start,end)).digest(),
                                     _slice(a,start,end), _slice(c,start,end))
            handles.append(handle)
            start = end
        return tuple(handles)


def square_rows(party, x, handle):
    with party.operation():
        _field(party)
        state = _state(party, handle, _squares)
        party._validate(x)
        if x.shape != state.a.shape:
            raise ValueError('square query row dimensions mismatch')
        # Preserve the immediate path's single-use identifier ledger and label.
        if state.identifier in party.spent_triples:
            raise ProtocolError('square row correlation identifier is single-use')
        party.spent_triples.add(state.identifier)
        d = _open_difference(party, party.subtract(x, state.a), handle, 'square', b'square'+state.identifier)
        return QShare([(c+2*v*a+(v*v if party.party==0 else 0))%PRIME
                       for c,v,a in zip(state.factor.values,d,state.a.values)],
                      [(c+2*v*a+party.alpha_share*v*v)%PRIME
                       for c,v,a in zip(state.factor.macs,d,state.a.macs)],x.shape)
