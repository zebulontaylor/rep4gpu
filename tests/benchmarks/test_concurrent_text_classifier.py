import pytest

from experiments.benchmarks.inference.concurrent_text_classifier import overlapping_throughput


def report(offset):
    rows = [dict(start_monotonic=i+offset, end_monotonic=i+offset+0.8) for i in range(5)]
    return dict(passed=True, parties=[dict(measurements=rows) for _ in range(4)])


def test_common_window_excludes_partial_batches():
    result = overlapping_throughput([report(0), report(.5)], 100)
    assert result['window_seconds'] == 4.3
    assert result['complete_batches_per_session'] == [4, 4]
    assert result['texts_per_second'] == pytest.approx(800/4.3)


def test_reject_nonoverlap_and_failed_sessions():
    with pytest.raises(ValueError, match='overlapping'):
        overlapping_throughput([report(0), report(9)], 100)
    failed = report(0)
    failed['passed'] = False
    with pytest.raises(ValueError, match='output checks'):
        overlapping_throughput([failed], 100)
