"""CPU checks for primitive benchmark accounting and ring references."""
import pytest
import torch
from benchmarks.primitives import fixture, run_case


@pytest.mark.parametrize('bits,binary_bits', [(64,64),(32,32)])
@pytest.mark.parametrize('name', ['matmul','trunc','matmul_relu'])
def test_checked_primitive_replay(bits,binary_bits,name):
    torch.set_num_threads(1)
    row=run_case(1,4,3,repeats=1,bits=bits,binary_bits=binary_bits,name=name)
    assert row['exact_integer_match']
    assert len(row['verification_opening_samples_ms'])==2
    assert len(row['replay_phase_samples_seconds'])==4
    assert len(row['replay_phase_samples_seconds'][0][0])==row['rounds']+1
    assert row['rounds'] >= 1
    assert row['payload_bytes'] > 0
    assert row['ideal_network_phase_max_ms'] > 0
    assert set(row['ideal_slowdowns'])==set(row['baselines'])


def test_fixture_rejects_empty_matrices():
    with pytest.raises(ValueError,match='positive'):
        fixture(0,4,3,32)


def test_fixture_reports_pre_wrapping_range(monkeypatch):
    import numpy as np
    import benchmarks.primitives as benchmark
    class ConstantGenerator:
        def normal(self, mean, deviation, shape):
            return np.full(shape, 2.0)
    monkeypatch.setattr(benchmark.np.random, 'default_rng', lambda seed: ConstantGenerator())
    _, _, wrapped, metadata = fixture(1,64,1,32)
    assert metadata['maximum_accumulator_before_ring_wrap']==2**32
    assert not metadata['accumulator_fits_signed_ring']
    assert wrapped.item()==0


def test_worker_failure_releases_waiting_peers(monkeypatch):
    import time
    from rep4gpu.runtime import Party
    original=Party.matmul
    def fail_one_party(self,left,right):
        if self.party==0:
            raise RuntimeError('injected primitive failure')
        return original(self,left,right)
    monkeypatch.setattr(Party,'matmul',fail_one_party)
    start=time.monotonic()
    with pytest.raises(RuntimeError,match='injected primitive failure'):
        run_case(1,4,3,repeats=1)
    assert time.monotonic()-start < 2
