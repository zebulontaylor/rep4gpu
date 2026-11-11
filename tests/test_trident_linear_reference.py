"""Check role-local linear algebra independently of trusted fixture products."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rep4gpu.trident_linear_reference import Party, TrustedFixture, run_matmul
from rep4gpu.transport import IdealNetwork


@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 3, 4), (3, 7, 2), (1, 128, 16)])
def test_ring_matmul_and_exact_payload(shape):
    m, k, n = shape
    rng = np.random.default_rng(91)
    x = rng.integers(0, 1 << 32, (m, k), dtype=np.uint32)
    y = rng.integers(0, 1 << 32, (k, n), dtype=np.uint32)
    expected = np.array([[sum(int(x[i, t])*int(y[t, j]) for t in range(k)) % (1 << 32)
                          for j in range(n)] for i in range(m)], dtype=np.uint32)
    result = run_matmul(x, y)
    np.testing.assert_array_equal(result['output'], expected)
    assert result['online_payload_bytes'] == 12*m*n + 96
    assert result['opening_payload_bytes'] == 16*m*n + 128
    assert result['online_bytes_per_party'][0] == 0
    assert result['preprocessing_payload_bytes'] is None


@pytest.mark.parametrize('target', ['data', 'hash', 'length'])
def test_online_fault_aborts(target):
    def mutate(p, round, outgoing):
        if p == 1 and round == 0:
            peer = 2 if target == 'hash' else 3
            data = outgoing[peer]
            outgoing[peer] = data[:-1] if target == 'length' else bytes([data[0] ^ 1]) + data[1:]
        return outgoing
    with pytest.raises((ValueError, ConnectionError)):
        run_matmul([[3, 7]], [[4], [9]], mutate=mutate)


@pytest.mark.parametrize('sender', [1, 2])
def test_output_data_or_hash_fault_never_returns_result(sender):
    def mutate(p, round, outgoing):
        if p == sender and round == 2:
            data = outgoing[0]
            outgoing[0] = bytes([data[0] ^ 1]) + data[1:]
        return outgoing
    with pytest.raises((ValueError, ConnectionError)):
        run_matmul([[3]], [[7]], mutate=mutate)


def test_signed_inputs_use_ring_arithmetic():
    result = run_matmul([[-1, -(1 << 31)]], [[-3], [2]])
    np.testing.assert_array_equal(result['output'], [[3]])


def test_correlations_are_bound_and_single_use():
    f = TrustedFixture()
    xs, ys = f.share([[3]]), f.share([[7]])
    prepared = f.prepare(xs, ys)
    network = IdealNetwork(timeout=2)
    parties = [Party(p, network.endpoint(p)) for p in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda p: p.matmul(xs[p.party], ys[p.party], prepared[p.party]), parties))
    assert len(outputs) == 4 and all(item.used for item in prepared)
    with pytest.raises(ValueError, match='single-use'):
        parties[0].matmul(xs[0], ys[0], prepared[0])
    fresh = f.prepare(xs, ys)
    wrong = f.share([[9]])
    with pytest.raises(ValueError, match='different input wires'):
        Party(0, IdealNetwork().endpoint(0)).matmul(wrong[0], ys[0], fresh[0])


def test_each_role_only_has_its_share():
    shares = TrustedFixture().share([[123]])
    assert shares[0].masked is None and set(shares[0].masks) == {1, 2, 3}
    for p in (1, 2, 3):
        assert set(shares[p].masks) == {1, 2, 3} - {p}


def test_fresh_fixture_masks_change_transcript_values_not_result():
    records = []
    for seed in (5, 6):
        observed = []
        def observe(p, round, outgoing):
            if round == 0 and p == 1:
                observed.append(outgoing[3])
            return outgoing
        result = run_matmul([[9, 11]], [[7], [5]], seed=seed, mutate=observe)
        np.testing.assert_array_equal(result['output'], [[118]])
        records.append(observed[0])
    assert records[0] != records[1]
