from concurrent.futures import ThreadPoolExecutor
import hashlib
import struct

import numpy as np
import pytest
import torch

from rep4gpu import spdz_compression as compression
from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import DistributedParty, split_share
from rep4gpu.transport import IdealNetwork


def _run(*, mutation=None, payload_fault=None, labels=None, keys=10):
    dealer = TrustedDealer(parties=4, mac_keys=keys)
    clear = torch.tensor([[7, 19, -3], [41, 0, 73]]).remainder(PRIME)
    shares = split_share(dealer.share(clear))
    other = split_share(dealer.share(clear.T.contiguous()))
    network = IdealNetwork(timeout=5)
    parties, events = [], [[] for _ in range(4)]
    for p in range(4):
        party = compression.CompressedParty(p, network.endpoint(p), dealer.alpha_shares[:, p],
                    session=b'compression-test', _allow_insecure_test_transport=True)
        broadcast = party.broadcast
        def record(payload, *, label, p=p, broadcast=broadcast):
            events[p].append(label)
            if payload_fault is not None:
                payload = payload_fault(p, label, payload, network)
            return broadcast(payload, label=label)
        party.broadcast = record
        parties.append(party)
        if mutation is not None:
            mutation(p, shares[p])
    def work(p):
        try:
            return parties[p].open_many((shares[p], other[p]), label=b'open-test' if labels is None else labels[p])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        result = list(pool.map(work, range(4)))
    return result, parties, events, clear


def test_compressed_open_matches_full_field_values_and_order():
    result, parties, events, clear = _run()
    assert all(torch.equal(value[0], clear) and torch.equal(value[1], clear.T) for value in result)
    for order in events:
        assert len(order) == 5
        assert order[0].endswith(b'/values')
        assert b'/coin/' in order[1] and order[1].endswith(b'/commit')
        assert b'/coin/' in order[2] and order[2].endswith(b'/reveal')
        assert b'/residual/' in order[3] and order[3].endswith(b'/commit')
        assert b'/residual/' in order[4] and order[4].endswith(b'/reveal')
    assert all(not party.poisoned for party in parties)


@pytest.mark.parametrize('kind', ['value', 'mac'])
def test_mutated_local_authentication_aborts_all_roles(kind):
    def mutation(p, share):
        if p == 0:
            target = share.values if kind == 'value' else share.macs
            target.reshape(-1)[0] = (target.reshape(-1)[0] + 1) % PRIME
    result, parties, _, _ = _run(mutation=mutation)
    assert all(isinstance(value, BaseException) for value in result)
    assert all(party.poisoned for party in parties)


@pytest.mark.parametrize('stage', ['coin-commit', 'coin-reveal', 'residual-commit', 'residual-reveal'])
def test_committed_seed_and_residual_cannot_be_replaced(stage):
    family, action = stage.split('-')
    def fault(p, label, payload, network):
        if p == 0 and b'/' + family.encode() + b'/' in label and label.endswith(b'/' + action.encode()):
            # Alter every recipient's copy consistently. Broadcast consistency
            # alone therefore cannot detect the forged commitment opening.
            return payload[:-1] + bytes([payload[-1] ^ 1])
        return payload
    result, parties, _, _ = _run(payload_fault=fault)
    assert all(isinstance(value, BaseException) for value in result)
    assert all(party.poisoned for party in parties)


def test_public_label_disagreement_aborts_before_coin():
    result, parties, events, _ = _run(labels=[b'a', b'b', b'a', b'a'])
    assert all(isinstance(value, BaseException) for value in result)
    assert all(party.poisoned for party in parties)
    assert all(len(order) == 1 for order in events)


@pytest.mark.parametrize('keys', [5, 6])
def test_fewer_than_seven_compressed_mac_keys_are_rejected(keys):
    result, parties, events, _ = _run(keys=keys)
    assert all(isinstance(value, BaseException) for value in result)
    assert all(party.poisoned for party in parties)
    assert all(not order for order in events)


@pytest.mark.parametrize('keys', [7, 10, 12])
def test_all_selected_mac_coordinates_are_retained(keys):
    result, parties, _, clear = _run(keys=keys)
    assert all(party.mac_keys == keys for party in parties)
    assert all(torch.equal(value[0], clear) for value in result)


def test_coefficient_stream_matches_independent_reference_and_domains():
    seed, context, count = bytes(range(32)), b'typed-operation', 47
    key = 3
    prefix = (compression.DOMAIN + b'/coefficients' + struct.pack('<Q', len(context)) + context
              + seed + struct.pack('<IQ', key, count))
    raw = hashlib.shake_256(prefix + struct.pack('<Q', 0)).digest(4 * (count + 16))
    values = [(word & PRIME) for (word,) in struct.iter_unpack('<I', raw)]
    expected = [value for value in values if value < PRIME][:count]
    actual = compression.coefficient_row(seed, context, key, count)
    assert actual.tolist() == expected
    assert actual.min() >= 0 and actual.max() < PRIME
    assert not torch.equal(actual, compression.coefficient_row(seed, context, key + 1, count))
    assert not torch.equal(actual, compression.coefficient_row(seed, context + b'x', key, count))
    assert compression.coefficient_row(seed, context, key, 0).shape == (0,)


def test_rejection_sampler_discards_prime_and_retries(monkeypatch):
    class Stream:
        def __init__(self, block):
            self.block = block
        def digest(self, count):
            words = [PRIME] * (count // 4) if self.block == 0 else [PRIME, 0, 1, PRIME - 1] * (count // 16 + 1)
            return np.array(words[:count // 4], dtype='<u4').tobytes()
    monkeypatch.setattr(compression.hashlib, 'shake_256', lambda data: Stream(struct.unpack('<Q', data[-8:])[0]))
    assert compression.coefficient_row(bytes(32), b'ctx', 0, 3).tolist() == [0, 1, PRIME - 1]


def test_compression_matches_python_big_integer_reduction():
    seed, context = bytes(range(32)), b'exact-reduction'
    residual = torch.tensor([[PRIME - 1, PRIME - 2, 19, 0]] * 10)
    actual = compression.compress_residual(residual, seed, context)
    expected = [sum(int(a) * int(b) for a, b in zip(row, compression.coefficient_row(seed, context, key, 4))) % PRIME
                for key, row in enumerate(residual)]
    assert actual.tolist() == expected


def test_toy_field_adaptive_guess_bound_and_common_projection_counterexample():
    # Error delta=(1,1). The attacker chooses its check after seeing r but
    # before learning the honest alpha. Enumerate its best possible guess.
    p = 3
    best_successes = 0
    projection_cancellation = 0
    for r0 in range(p):
        for r1 in range(p):
            t = (r0 + r1) % p
            projection_cancellation += t == 0
            successes = max(sum((alpha * t - guess) % p == 0 for alpha in range(p)) for guess in range(p))
            best_successes += successes
    assert best_successes == 2 * p * p - p
    assert best_successes / p**3 == pytest.approx(2 / p - 1 / p**2)
    # Sharing r across keys leaves a common cancellation event with mass1/p,
    # regardless of how many independent MAC keys protect the opened values.
    assert projection_cancellation == p
    assert projection_cancellation / p**2 == pytest.approx(1 / p)


def test_rushing_adversary_cannot_replace_committed_check_after_honest_reveals():
    def mutation(p, share):
        if p == 0:
            share.values[0, 0] = (share.values[0, 0] + 1) % PRIME
    def fault(p, label, payload, network):
        if p != 0 or b'/residual/' not in label or not label.endswith(b'/reveal'):
            return payload
        # Let the corrupt role inspect all honest residual reveals before it
        # sends its own. Its new cancelling vector cannot open its old commit.
        with network.condition:
            assert network.condition.wait_for(
                lambda: 8 in network.rounds and len(network.rounds[8][0]) == 3,
                timeout=3)
            frames = [network.rounds[8][0][sender][0] for sender in (1, 2, 3)]
        honest = [np.frombuffer(frame[93 + 32:], dtype='<u4').astype(np.int64) for frame in frames]
        cancel = (-sum(honest)) % PRIME
        return payload[:32] + cancel.astype('<u4').tobytes()
    result, parties, _, _ = _run(mutation=mutation, payload_fault=fault)
    assert all(isinstance(value, BaseException) for value in result)
    assert any('compressed residual commitment opening failed' in str(value) for value in result)
    assert all(party.poisoned for party in parties)


def test_chacha_cpu_oracle_matches_independent_raw_stream():
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    seed, context, key, count = bytes(range(32)), b'cpu-chacha-context', 4, 67
    derived = hashlib.sha256(compression.DOMAIN + b'/' + compression.CHACHA_BACKEND
                            + struct.pack('<Q', len(context)) + context + seed
                            + struct.pack('<IQ', key, count)).digest()
    encryptor = Cipher(algorithms.ChaCha20(derived, bytes(16)), mode=None).encryptor()
    raw = encryptor.update(bytes(4 * count)) + encryptor.finalize()
    expected = np.frombuffer(raw, dtype='<u4') & PRIME
    assert np.all(expected < PRIME)
    assert compression.coefficient_row_chacha(seed, context, key, count).tolist() == expected.tolist()
    assert not torch.equal(compression.coefficient_row_chacha(seed, context, key, count),
                           compression.coefficient_row_chacha(seed, context, key + 1, count))


def test_chacha_retries_only_rejected_positions_and_fails_closed(monkeypatch):
    import cryptography.hazmat.primitives.ciphers as ciphers
    class Encryptor:
        def __init__(self, attempt, exhausted):
            self.attempt, self.exhausted = attempt, exhausted
        def update(self, data):
            values = [PRIME] * 4 if self.exhausted else ([PRIME, 7, PRIME, 9] if self.attempt == 0 else [11, 12, 13, 14])
            return np.array(values, dtype='<u4').tobytes()
        def finalize(self):
            return b''
    exhausted = False
    class Cipher:
        def __init__(self, algorithm, mode):
            self.attempt = struct.unpack('<QQ', algorithm.nonce)[1]
        def encryptor(self):
            return Encryptor(self.attempt, exhausted)
    monkeypatch.setattr(ciphers, 'Cipher', Cipher)
    assert compression.coefficient_row_chacha(bytes(32), b'ctx', 0, 4).tolist() == [11, 7, 13, 9]
    exhausted = True
    with pytest.raises(compression.ProtocolError, match='attempts exhausted'):
        compression.coefficient_row_chacha(bytes(32), b'ctx', 0, 4)


@pytest.mark.parametrize('count', [1, 17, 2049, 65539])
def test_gpu_chacha_coefficients_match_cpu_oracle(count):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    seed, context, keys = bytes(range(32)), b'gpu-chacha-context/' + compression.CHACHA_BACKEND, 10
    actual = compression.coefficient_rows_gpu(seed, context, keys, count, 'cuda')
    assert actual.dtype == torch.int32
    expected = torch.stack([compression.coefficient_row_chacha(seed, context, key, count) for key in range(keys)])
    assert torch.equal(actual.cpu().to(torch.int64), expected)


def test_gpu_coefficient_exhaustion_returns_failure_flag_and_default_raises(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    monkeypatch.setattr(compression, 'MAX_CHACHA_ATTEMPTS', 0)
    coefficients, failure = compression.coefficient_rows_gpu(bytes(32), b'exhaustion', 10, 17, 'cuda', return_status=True)
    assert failure.item() == 1
    assert torch.count_nonzero(coefficients).item() == 0
    with pytest.raises(compression.ProtocolError, match='attempts exhausted'):
        compression.coefficient_rows_gpu(bytes(32), b'exhaustion', 10, 17, 'cuda')


def test_rushing_adversary_cannot_replace_seed_after_honest_coin_reveals():
    def fault(p, label, payload, network):
        if p != 0 or b'/coin/' not in label or not label.endswith(b'/reveal'):
            return payload
        with network.condition:
            assert network.condition.wait_for(
                lambda: 4 in network.rounds and len(network.rounds[4][0]) == 3,
                timeout=3)
            frames = [network.rounds[4][0][sender][0] for sender in (1, 2, 3)]
        # A rushing role could force the XOR coin to zero if it were allowed
        # to choose its seed contribution after reading honest contributions.
        contribution = bytes(a ^ b ^ c for a, b, c in zip(*(frame[93 + 32:] for frame in frames)))
        return payload[:32] + contribution
    result, parties, events, _ = _run(payload_fault=fault)
    assert all(isinstance(value, BaseException) for value in result)
    assert any('coin commitment opening failed' in str(value) for value in result)
    assert all(party.poisoned for party in parties)
    assert all(not any(b'/residual/' in label for label in order) for order in events)


def test_compressed_beaver_product_and_repeated_checks_match_field_reference():
    from rep4gpu.spdz import field_matmul
    from rep4gpu.spdz_distributed import split_triple
    dealer = TrustedDealer(parties=4, mac_keys=10)
    a = torch.tensor([[3, -7, 9], [11, 27, -99]]).remainder(PRIME)
    b = torch.tensor([[71, 0], [89, -3], [22, 42]]).remainder(PRIME)
    xs, ys = split_share(dealer.share(a)), split_share(dealer.share(b))
    triples = split_triple(dealer.triple(2, 3, 2))
    network = IdealNetwork(timeout=5)
    parties = [compression.CompressedParty(p, network.endpoint(p), dealer.alpha_shares[:, p],
               session=b'compressed-beaver', _allow_insecure_test_transport=True) for p in range(4)]
    def work(p):
        try:
            product = parties[p].matmul(xs[p], ys[p], triples[p])
            first = parties[p].open(product)
            second = parties[p].open(product)
            assert torch.equal(first, second)
            return first
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        result = list(pool.map(work, range(4)))
    assert all(torch.equal(value, field_matmul(a, b)) for value in result)
    assert all(party.round == 15 for party in parties)
    assert all(not party.poisoned for party in parties)


def test_gpu_rejection_retries_preserve_previously_accepted_positions(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import triton
    import triton.language as tl
    from rep4gpu import gpu_prg_batch

    @triton.jit
    def fake_words(KEY, nonce, block, BLOCK: tl.constexpr):
        return (
            tl.where(nonce == 0, 0x7fffffff, block*16 + 0 + nonce*100).to(tl.uint32),
            (block*16 + 1 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 2 + nonce*100).to(tl.uint32),
            (block*16 + 3 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 4 + nonce*100).to(tl.uint32),
            (block*16 + 5 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 6 + nonce*100).to(tl.uint32),
            (block*16 + 7 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 8 + nonce*100).to(tl.uint32),
            (block*16 + 9 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 10 + nonce*100).to(tl.uint32),
            (block*16 + 11 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 12 + nonce*100).to(tl.uint32),
            (block*16 + 13 + nonce*100).to(tl.uint32),
            tl.where(nonce == 0, 0x7fffffff, block*16 + 14 + nonce*100).to(tl.uint32),
            (block*16 + 15 + nonce*100).to(tl.uint32))
    monkeypatch.setattr(gpu_prg_batch, '_block_words', fake_words)
    monkeypatch.setattr(compression, '_GPU_COEFFICIENT_KERNEL', None)
    count = 2051
    actual = compression.coefficient_rows_gpu(bytes(32), b'forced-rejection', 10, count, 'cuda')
    positions = torch.arange(count)
    expected = positions + (positions.remainder(2) == 0).to(torch.int64)*100
    assert torch.equal(actual.cpu().to(torch.int64), expected.expand(10, -1))


def test_toy_field_sacrifice_needs_independent_check_challenges():
    # Enumerate every pair of fixed check errors for a bad retained triple.
    # Reusing t gives a common root; independent draws give two constraints.
    p = 5
    worst_common = 0
    for delta0 in range(1, p):
        for delta1 in range(p):
            for delta2 in range(p):
                common = sum((t*delta0 - delta1) % p == 0 and (t*delta0 - delta2) % p == 0
                             for t in range(p))
                independent = sum((t0*delta0 - delta1) % p == 0 and (t1*delta0 - delta2) % p == 0
                                  for t0 in range(p) for t1 in range(p))
                worst_common = max(worst_common, common)
                assert independent == 1
    assert worst_common / p == pytest.approx(1 / p)
    assert independent / p**2 == pytest.approx(1 / p**2)



def test_gpu_key_metadata_upload_retains_more_than_128_mac_keys():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    seed, context, keys, count = bytes(range(32)), b'large-key-metadata', 129, 1
    actual = compression.coefficient_rows_gpu(seed, context, keys, count, 'cuda')
    expected = torch.stack([compression.coefficient_row_chacha(seed, context, key, count) for key in range(keys)])
    assert torch.equal(actual.cpu().to(torch.int64), expected)



def test_seven_key_online_union_bound_exceeds_128_bit_statistical_target():
    from fractions import Fraction
    q = Fraction(2*PRIME - 1, PRIME**2)
    checks = 2**64 - 1
    assert checks*q**7 < Fraction(1, 2**145)
    assert checks*q**7 < Fraction(1, 2**128)
    assert checks*q**6 > Fraction(1, 2**128)


def test_six_key_coefficient_request_fails_before_gpu_setup():
    with pytest.raises(ValueError, match='seven'):
        compression.coefficient_rows_gpu(bytes(32), b'reject-six', 6, 1, 'cuda')


def test_seventh_cpu_mac_coordinate_is_checked():
    def mutation(p, share):
        if p == 0:
            share.macs[6, 0, 0] = (share.macs[6, 0, 0] + 1) % PRIME
    result, parties, _, _ = _run(keys=7, mutation=mutation)
    assert all(isinstance(value, BaseException) for value in result)
    assert all(party.poisoned for party in parties)
