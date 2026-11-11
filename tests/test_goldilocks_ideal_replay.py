"""Private replay fixtures reproduce bytes, outputs, and one-use control flow."""
from dataclasses import replace

import pytest

from test_spdz_goldilocks_deferred import run
from rep4gpu.spdz_prime256 import QShare
from rep4gpu.transport import IdealNetwork
import benchmarks.goldilocks_ideal_replay as bench


@pytest.fixture(scope='module')
def recorded():
    def operation(p,x,h1,h2,sq):
        model=dict(b1=QShare([0]*3,[0]*3,(1,3)),b2=QShare([0]*2,[0]*2,(1,2)))
        return bench.record_stream(p,x,model,h1,h2,sq)
    fixtures,parties,_=run(operation)
    assert all(type(fixture)is bench.Fixture for fixture in fixtures)
    return fixtures,parties


def test_exact_bytes_checked_outputs_and_phase_estimate(recorded):
    fixtures,parties=recorded
    reports=[bench.run_local_replay(fixture,repeats=2,warmups=1) for fixture in fixtures]
    assert all(row['output_equality'] and row['outgoing_bytes_verified'] for row in reports)
    assert all(row['rounds']==32 and len(row['phase_medians_seconds'])==33 for row in reports)
    summary=bench.phase_estimate(reports)
    expected=sum(max(row['phase_medians_seconds'][i] for row in reports) for i in range(33))*1000
    assert summary['ideal_confidential_channel_barrier_ms']==expected
    assert summary['per_token_ms']==expected/2 and summary['tls_crypto_included'] is False
    assert all(not party.poisoned for party in parties)


def test_outgoing_byte_mismatch_is_rejected(recorded):
    fixture=recorded[0][0];frame=fixture.frames[0]
    peer,data=frame.sent[0]
    changed=replace(frame,sent=((peer,data[:-1]+bytes([data[-1]^1])),)+frame.sent[1:])
    corrupt=replace(fixture,frames=(changed,)+fixture.frames[1:])
    with pytest.raises(ValueError,match='outgoing mismatch'):bench.run_local_replay(corrupt,repeats=1,warmups=0)


def test_incoming_frame_reordering_fails_even_without_outgoing_checks(recorded):
    fixture=recorded[0][0]
    a,b=fixture.frames[:2]
    frames=(replace(a,received=b.received),replace(b,received=a.received))+fixture.frames[2:]
    with pytest.raises(RuntimeError):bench._one_replay(replace(fixture,frames=frames),check_outgoing=False,timed=True)


def test_output_equality_is_mandatory(recorded):
    fixture=recorded[0][0]
    first=fixture.outputs[0]
    bad=replace(fixture,outputs=((first[0]+1,)+first[1:],)+fixture.outputs[1:])
    with pytest.raises(AssertionError,match='outputs differ'):bench.run_local_replay(bad,repeats=1,warmups=0)


def test_salt_order_is_checked_and_replay_still_draws_fresh_entropy(recorded,monkeypatch):
    fixture=recorded[0][0];draws=[]
    from rep4gpu.spdz_prime256 import Prime256Party
    original=Prime256Party._commit_salt
    def charged(party):draws.append(1);return original(party)
    monkeypatch.setattr(Prime256Party,'_commit_salt',charged)
    bench._one_replay(fixture,check_outgoing=True,timed=False)
    assert len(draws)==len(fixture.salts)
    bad=replace(fixture,salts=(fixture.salts[1],fixture.salts[0])+fixture.salts[2:])
    with pytest.raises(RuntimeError):bench._one_replay(bad,check_outgoing=False,timed=False)


def test_private_capabilities_cannot_restore_to_live_or_subclassed_transport(recorded):
    fixture=recorded[0][0]
    with pytest.raises(TypeError,match='exact fixed'):bench.restore(fixture,IdealNetwork().endpoint(0))
    class ForwardingSubclass(bench.GoldReplayTransport):pass
    with pytest.raises(TypeError,match='exact fixed'):bench.restore(fixture,ForwardingSubclass(fixture.frames))


def test_replay_does_not_reset_or_retire_live_state(recorded):
    fixtures,parties=recorded;party=parties[0];fixture=fixtures[0]
    before=(party.round,party._flush_index,frozenset(party.spent_triples),frozenset(party._authorized),party.transport)
    bench.run_local_replay(fixture,repeats=1,warmups=0)
    after=(party.round,party._flush_index,frozenset(party.spent_triples),frozenset(party._authorized),party.transport)
    assert before==after and not party.poisoned


def test_phase_estimate_rejects_missing_roles_or_mismatched_schedule(recorded):
    reports=[bench.run_local_replay(fixture,repeats=1,warmups=0) for fixture in recorded[0]]
    with pytest.raises(ValueError):bench.phase_estimate(reports[:3])
    reports[0]=dict(reports[0],phase_medians_seconds=reports[0]['phase_medians_seconds'][:-1])
    with pytest.raises(ValueError):bench.phase_estimate(reports)


def test_snapshots_are_immutable_and_public_reports_contain_no_private_fixture(recorded):
    fixture=recorded[0][0]
    with pytest.raises(TypeError):fixture.state.initial.values[0]=7
    with pytest.raises(TypeError):fixture.state.first[0].a.values[0]=7
    import pickle,json
    with pytest.raises(TypeError):pickle.dumps(fixture)
    report=bench.run_local_replay(fixture,repeats=1,warmups=0)
    encoded=json.dumps(report)
    assert fixture.salts[0].hex() not in encoded
    assert str(fixture.state.alpha) not in encoded
    assert not {'state','frames','salts','initial','alpha'} & set(report)


def test_aggregation_rejects_unverified_or_invalid_phase_measurements(recorded):
    reports=[bench.run_local_replay(fixture,repeats=1,warmups=0) for fixture in recorded[0]]
    unchecked=[dict(row) for row in reports];unchecked[0]['outgoing_bytes_verified']=False
    with pytest.raises(ValueError,match='verified'):bench.phase_estimate(unchecked)
    invalid=[dict(row) for row in reports]
    invalid[0]['phase_medians_seconds']=[float('nan')]+invalid[0]['phase_medians_seconds'][1:]
    with pytest.raises(ValueError,match='finite'):bench.phase_estimate(invalid)
