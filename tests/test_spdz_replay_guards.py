"""Guard the benchmark's separation of live entropy and frozen fixtures."""
import pytest

from benchmarks.spdz_inference import SaltFixture


def test_live_commitment_salts_record_fresh_os_entropy(monkeypatch):
    draws = iter([b'a'*32,b'b'*32])
    calls = []
    def entropy(count):
        calls.append(count)
        return next(draws)
    monkeypatch.setattr('benchmarks.spdz_inference.secrets.token_bytes',entropy)
    fixture = SaltFixture()
    fixture._benchmark_salt_log = []
    assert fixture._commit_salt() == b'a'*32
    assert fixture._commit_salt() == b'b'*32
    assert fixture._benchmark_salt_log == [b'a'*32,b'b'*32]
    assert calls == [32,32]


def test_fixed_replay_charges_entropy_and_consumes_every_recorded_salt(monkeypatch):
    calls = []
    def entropy(count):
        calls.append(count)
        return b'fresh-discarded-entropy'.ljust(count,b'!')
    monkeypatch.setattr('benchmarks.spdz_inference.secrets.token_bytes',entropy)
    fixture = SaltFixture()
    fixture._benchmark_replay_salts = [b'fixture-a'.ljust(32,b'a'),b'fixture-b'.ljust(32,b'b')]
    fixture._benchmark_salt_index = 0
    fixture._benchmark_salt_log = []
    assert fixture._commit_salt() == fixture._benchmark_replay_salts[0]
    assert fixture._commit_salt() == fixture._benchmark_replay_salts[1]
    assert calls == [32,32]
    assert fixture._benchmark_salt_index == 2
    assert fixture._benchmark_salt_log == []
    with pytest.raises(AssertionError,match='exhausted'):
        fixture._commit_salt()
    assert calls == [32,32,32]
