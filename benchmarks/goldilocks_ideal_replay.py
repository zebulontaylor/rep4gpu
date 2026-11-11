"""Benchmark-only role-local replay of a fresh checked Goldilocks stream.

Fixtures contain private preprocessing snapshots and must remain in ignored
storage or worker memory. Replay uses fixed transcripts, not live connections.
It models an ideal confidential channel and omits local TLS cryptography.
"""
from dataclasses import dataclass,field
from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
import statistics
import time
import weakref

from rep4gpu.spdz_distributed import ProtocolError
from rep4gpu.spdz_goldilocks_deferred import DeferredGoldilocksParty
from rep4gpu.spdz_goldilocks_stream import (
    PRIME,FixedWeightRows,SquareRows,_rows,_squares,matmul_rows,square_rows,
)
from rep4gpu.spdz_prime256 import QShare
from rep4gpu.transport import RecordingTransport,ReplayTransport


@dataclass(frozen=True)
class Share:
    values: tuple
    macs: tuple
    shape: tuple


def freeze(share):return Share(tuple(share.values),tuple(share.macs),tuple(share.shape))


@dataclass(frozen=True)
class Handle:
    kind: str
    session: bytes=field(repr=False)
    key: int=field(repr=False)
    role: int
    model: bytes=field(repr=False)
    identifier: bytes=field(repr=False)
    a: object=field(repr=False)
    factor: object=field(repr=False)
    weights: object=field(repr=False)


@dataclass(frozen=True)
class State:
    source: object=field(repr=False)
    party: int
    session: bytes=field(repr=False)
    alpha: int=field(repr=False)
    round: int
    spent: frozenset=field(repr=False)
    authorized: frozenset=field(repr=False)
    flush_index: int
    cap: int
    initial: object=field(repr=False)
    b1: object=field(repr=False)
    b2: object=field(repr=False)
    first: tuple=field(repr=False)
    second: tuple=field(repr=False)
    squares: tuple=field(repr=False)


@dataclass(frozen=True)
class Frame:
    sent: tuple=field(repr=False)
    received: tuple=field(repr=False)


@dataclass(frozen=True)
class Fixture:
    state: State=field(repr=False)
    frames: tuple=field(repr=False)
    salts: tuple=field(repr=False)
    outputs: tuple=field(repr=False)
    tokens: tuple=field(repr=False)
    recorded_live_ms: float
    def __reduce__(self):raise TypeError('private replay fixtures cannot be serialized by this helper')


class GoldReplayTransport(ReplayTransport):
    """Exact fixed byte fixtures; no socket, native provider, or live forwarding."""
    def __init__(self,frames,check_outgoing=True):
        super().__init__([(dict(frame.sent),dict(frame.received)) for frame in frames],check_outgoing=check_outgoing)


class _ReplayParty(DeferredGoldilocksParty):
    def _commit_salt(self):
        # Preserve the live OS entropy cost while using this fixed fixture's
        # previously revealed salt solely to reproduce its byte transcript.
        super()._commit_salt()
        if self._salt_index>=len(self._salts):raise ValueError('replay entropy fixture exhausted')
        value=self._salts[self._salt_index];self._salt_index+=1
        return value


def _snapshot_handle(party,handle,registry,kind):
    if handle not in registry:raise ValueError('unknown stream capability')
    state=registry[handle]
    if (state.used or state.owner() is not party or state.session!=party.session or state.key!=party.alpha_share
            or state.role!=party.party):raise ValueError('snapshot requires unused owned capabilities')
    return Handle(kind,state.session,state.key,state.role,state.model,state.identifier,state.a,state.factor,state.weights)


def snapshot(party,x,model,first,second,squares):
    """Snapshot before a fresh live execution, never after consuming its rows."""
    if type(party) is not DeferredGoldilocksParty or party.poisoned or party.pending_values or party._pending:
        raise TypeError('snapshot requires a live checked-boundary DeferredGoldilocksParty')
    if not first or len(first)!=len(second) or len(first)!=len(squares):raise ValueError('stream capability counts differ')
    for share in (x,model['b1'],model['b2']):party._validate(share)
    return State(weakref.ref(party),party.party,party.session,party.alpha_share,party.round,
                 frozenset(party.spent_triples),frozenset(party._authorized),party._flush_index,
                 party.max_pending_values,freeze(x),freeze(model['b1']),freeze(model['b2']),
                 tuple(_snapshot_handle(party,h,_rows,'row') for h in first),
                 tuple(_snapshot_handle(party,h,_rows,'row') for h in second),
                 tuple(_snapshot_handle(party,h,_squares,'square') for h in squares))


def _execute(party,x,b1,b2,first,second,squares):
    outputs=[];tokens=[]
    for h1,h2,square in zip(first,second,squares):
        model1=_rows[h1].model;model2=_rows[h2].model
        hidden=matmul_rows(party,x,h1,model_id=model1)
        hidden=square_rows(party,party.add(hidden,b1),square)
        output=party.add(matmul_rows(party,hidden,h2,model_id=model2),b2)
        opened=party.open(output)
        signed=[value-PRIME if value>PRIME//2 else value for value in opened]
        if len(signed)!=x.shape[1]:raise ValueError('dependent stream needs output classes equal to input width')
        token=max(range(len(signed)),key=lambda i:signed[i])
        x=party.add_public(QShare([0]*len(signed),[0]*len(signed),(1,len(signed))),
                           [int(i==token) for i in range(len(signed))])
        outputs.append(tuple(opened));tokens.append(token)
    if party.pending_values or party._pending:raise AssertionError('stream finished with unchecked residuals')
    return tuple(outputs),tuple(tokens)


def record_stream(party,x,model,first,second,squares):
    """Execute fresh live rows once, retaining a private role-local fixture.

    Snapshot/setup and later replay are separate from recorded_live_ms. That
    diagnostic live time includes RecordingTransport's fixture-copy overhead.
    The caller's live counters, rows, and transport are never reset.
    """
    state=snapshot(party,x,model,first,second,squares)
    original_transport=party.transport;recorder=RecordingTransport(original_transport)
    original_salt=party._commit_salt;had_override='_commit_salt' in party.__dict__
    previous=party.__dict__.get('_commit_salt');salts=[]
    def record_salt():
        value=original_salt();salts.append(value);return value
    party.transport=recorder;party._commit_salt=record_salt
    try:
        started=time.perf_counter()
        outputs,tokens=_execute(party,x,model['b1'],model['b2'],first,second,squares)
        elapsed=(time.perf_counter()-started)*1000
    finally:
        party.transport=original_transport
        if had_override:party._commit_salt=previous
        else:del party._commit_salt
    frames=tuple(Frame(tuple(sorted(sent.items())),tuple(sorted(received.items()))) for sent,received in recorder.transcript)
    return Fixture(state,frames,tuple(salts),outputs,tokens,elapsed)


def _clone_handle(party,transport,handle):
    if type(transport) is not GoldReplayTransport or party.transport is not transport:
        raise TypeError('capability restoration is restricted to exact fixed replay transport')
    # Native production APIs never use this benchmark-only minting operation.
    from rep4gpu.spdz_goldilocks_stream import _State
    registry,cls=(_rows,FixedWeightRows) if handle.kind=='row' else (_squares,SquareRows)
    local=object.__new__(cls)
    registry[local]=_State(weakref.ref(party),handle.session,handle.key,handle.role,handle.model,
                           handle.identifier,handle.a,handle.factor,handle.weights,False)
    return local


def restore(fixture,transport):
    """Create isolated mutable counters only on the exact non-network class."""
    if type(transport) is not GoldReplayTransport:raise TypeError('restoration requires exact fixed replay transport')
    state=fixture.state;source=state.source()
    if (source is None or source.poisoned or source.session!=state.session or source.alpha_share!=state.alpha
            or source.party!=state.party):raise ValueError('original fixture session was changed or retired')
    local=_ReplayParty(state.party,transport,state.alpha,session=state.session,max_pending_values=state.cap,
                       _allow_insecure_test_transport=True)
    local.round=state.round;local.spent_triples=set(state.spent);local._authorized=set(state.authorized)
    local._flush_index=state.flush_index;local._salts=fixture.salts;local._salt_index=0
    first=tuple(_clone_handle(local,transport,h) for h in state.first)
    second=tuple(_clone_handle(local,transport,h) for h in state.second)
    squares=tuple(_clone_handle(local,transport,h) for h in state.squares)
    return local,first,second,squares


def _one_replay(fixture,*,check_outgoing,timed):
    transport=GoldReplayTransport(fixture.frames,check_outgoing=check_outgoing)
    party,first,second,squares=restore(fixture,transport)
    state=fixture.state
    if timed:transport.start()
    outputs,tokens=_execute(party,state.initial,state.b1,state.b2,first,second,squares)
    phases=transport.finish() if timed else []
    transport.assert_consumed()
    if outputs!=fixture.outputs or tokens!=fixture.tokens:raise AssertionError('replay checked outputs differ from live execution')
    if party._salt_index!=len(fixture.salts):raise AssertionError('replay entropy sequence was not fully consumed')
    if not all(h.used for h in (*first,*second,*squares)):raise AssertionError('replay left unused correlations')
    return phases


def run_local_replay(fixture,*,repeats=7,warmups=2):
    """Return public phase timings; outgoing-byte verification is untimed."""
    if type(repeats)is not int or repeats<1 or type(warmups)is not int or warmups<0:
        raise ValueError('positive repeats and nonnegative warmups required')
    _one_replay(fixture,check_outgoing=True,timed=False)
    for _ in range(warmups):_one_replay(fixture,check_outgoing=False,timed=True)
    samples=[_one_replay(fixture,check_outgoing=False,timed=True) for _ in range(repeats)]
    count=len(samples[0])
    if any(len(sample)!=count for sample in samples):raise AssertionError('phase counts differ')
    totals=[1000*sum(sample) for sample in samples]
    return dict(party=fixture.state.party,steps=len(fixture.outputs),rounds=len(fixture.frames),
                output_equality=True,outgoing_bytes_verified=True,repeats=repeats,warmups=warmups,
                phase_samples_seconds=samples,phase_medians_seconds=[statistics.median(row[i] for row in samples) for i in range(count)],
                samples_ms=totals,median_ms=statistics.median(totals),min_ms=min(totals),max_ms=max(totals),
                recorded_live_ms_with_fixture_copy_overhead=fixture.recorded_live_ms,
                commitment_salts_per_stream=len(fixture.salts),
                outgoing_bytes_per_round=[sum(len(data) for peer,data in frame.sent) for frame in fixture.frames],
                incoming_bytes_per_round=[sum(len(data) for peer,data in frame.received) for frame in fixture.frames],
                entropy_cost='Each replay commitment charges a fresh OS salt draw, then uses the fixed recorded salt only for fixture compatibility.',
                scope='Ideal confidential-channel replay. Includes local bigint arithmetic, encoding/decoding, hashes and MAC checks; excludes transport exchange machinery, network wait, and TLS encryption/decryption. Not physical perfect-network latency or live preprocessing reuse.')


def phase_estimate(role_reports):
    """Sum per-phase maxima across role-local medians, with no wait subtraction."""
    if len(role_reports)!=4 or {row['party'] for row in role_reports}!={0,1,2,3}:raise ValueError('one report per role required')
    if any(row.get('output_equality') is not True or row.get('outgoing_bytes_verified') is not True for row in role_reports):
        raise ValueError('phase aggregation requires byte-verified checked replay outputs')
    arrays=[row['phase_medians_seconds'] for row in role_reports]
    if any(not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 for array in arrays for value in array):
        raise ValueError('phase times must be finite and nonnegative')
    if not arrays[0] or any(len(array)!=len(arrays[0]) for array in arrays):raise ValueError('role phase schedules differ')
    if len({row['steps'] for row in role_reports})!=1 or len({row['rounds'] for row in role_reports})!=1:
        raise ValueError('role execution dimensions differ')
    phases=[max(array[i] for array in arrays) for i in range(len(arrays[0]))]
    total=1000*sum(phases);steps=role_reports[0]['steps']
    return dict(ideal_confidential_channel_barrier_ms=total,per_token_ms=total/steps,
                phase_critical_seconds=phases,phase_count=len(phases),steps=steps,
                aggregation='Sum of per-phase maxima of isolated role-local median times; no socket-time subtraction.',
                physical_four_node_measurement=False,tls_crypto_included=False,
                preprocessing_accounting='Unchanged: charge the actual fresh preprocessing pool cost per distinct one-use row pack, plus separately reported model/input/setup stages. Replay creates no new query capacity.')



@contextmanager
def exclusive_local_replay(path):
    """Keep other role workers idle while one role's CPU replay is timed."""
    path=Path(path)
    descriptor=path.open('a+')
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor,fcntl.LOCK_UN)
        descriptor.close()
