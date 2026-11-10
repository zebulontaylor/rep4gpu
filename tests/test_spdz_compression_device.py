"""CPU-crypto wire oracles for independent-key GPU MAC compression."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer, AuthenticationError, field_matmul
from rep4gpu.spdz_distributed import AuthShare, ProtocolError, split_share, split_triple
from rep4gpu.spdz_compression import (DeviceCompressedParty, DOMAIN, CHACHA_BACKEND,
                                     coefficient_row_chacha)
from rep4gpu.device_transport import DeviceFrame, DeviceReplay, DeviceIdealEndpoint, DeviceRecorder
from rep4gpu.transport import IdealNetwork


def _hash(data):
    return hashlib.blake2b(data, digest_size=32).digest()


def _pack(tensor):
    return tensor.numpy().astype('<u4').tobytes()


def _fixture(shapes, keys=10):
    dealer = TrustedDealer(mac_keys=keys)
    clear = [torch.arange(shape[0]*shape[1]).reshape(shape).remainder(PRIME) for shape in shapes]
    by_matrix = [split_share(dealer.share(value)) for value in clear]
    shares = [[matrix[p] for matrix in by_matrix] for p in range(4)]
    # Fixed entropy is confined to this wire fixture; production constructors
    # have no seed argument and their entropy hooks draw from the OS.
    coins = [bytes([p + 1])*32 for p in range(4)]
    coin_salts = [bytes([p + 7])*32 for p in range(4)]
    tau_salts = [bytes([p + 13])*32 for p in range(4)]
    return dealer, clear, shares, coins, coin_salts, tau_salts


def _oracle(shares, alpha, session, coins, coin_salts, tau_salts, *, fault=None):
    from blake3 import blake3
    keys = alpha.shape[0]
    layout = json.dumps({'field': PRIME, 'dtype': 'uint32-le', 'keys': keys,
                         'shapes': [list(x.values.shape) for x in shares[0]]},
                        sort_keys=True, separators=(',', ':')).encode()
    session_hash = _hash(b'SPDZ31-session-v1' + struct.pack('<Q', len(session)) + session)
    context = (b'SPDZ31G-independent-MAC-v1' + session_hash + struct.pack('<Q', 0)
               + _hash(layout) + _hash(b'') + _hash(CHACHA_BACKEND))
    transcripts = [[] for _ in range(4)]

    def broadcast(payloads, round_id, label, kind):
        hash_fn = (lambda data: blake3(data).digest()) if kind == 'device' else _hash
        magic = b'SPDZ31G1' if kind == 'device' else b'SPDZ31v1'
        def frame(stage, sender, body):
            return (magic + session_hash + struct.pack('<QI', round_id, sender) + stage
                    + _hash(label) + struct.pack('<Q', len(body)) + body)
        frames = [frame(b'V', p, payloads[p]) for p in range(4)]
        view = b''.join(hash_fn(struct.pack('<I', p) + frames[p]) for p in range(4))
        echoes = [frame(b'E', p, view) for p in range(4)]
        for p in range(4):
            peers = set(range(4)) - {p}
            transcripts[p].append(DeviceFrame(kind, {q: frames[p] for q in peers}, {q: frames[q] for q in peers}))
            transcripts[p].append(DeviceFrame(kind, {q: echoes[p] for q in peers}, {q: echoes[q] for q in peers}))

    values = [torch.cat([x.values.reshape(-1) for x in role]) for role in shares]
    opened = sum(values).remainder(PRIME)
    broadcast([_pack(value) for value in values], 0, context + b'/values', 'device')
    coin_binding = DOMAIN + b'/coin/' + context
    coin_reveals = [coin_salts[p] + coins[p] for p in range(4)]
    coin_commits = [_hash(coin_binding + struct.pack('<I', p) + coin_reveals[p]) for p in range(4)]
    broadcast(coin_commits, 1, coin_binding + b'/commit', 'bytes')
    if fault == 'coin-reveal':
        coin_reveals[0] = coin_reveals[0][:-1] + bytes([coin_reveals[0][-1] ^ 1])
    broadcast(coin_reveals, 2, coin_binding + b'/reveal', 'bytes')
    joint = bytes(a ^ b ^ c ^ d for a, b, c, d in zip(*coins))
    coefficients = torch.stack([coefficient_row_chacha(joint, context, key, opened.numel()) for key in range(keys)])
    checks = []
    for p in range(4):
        macs = torch.cat([x.macs.reshape(keys, -1) for x in shares[p]], dim=1)
        residual = (macs - alpha[:, p, None]*opened).remainder(PRIME)
        checks.append((residual*coefficients).remainder(PRIME).sum(1).remainder(PRIME))
    binding = DOMAIN + b'/residual/' + context
    tau_reveals = [tau_salts[p] + _pack(checks[p]) for p in range(4)]
    tau_commits = [blake3(binding + struct.pack('<I', p) + tau_reveals[p]).digest() for p in range(4)]
    broadcast(tau_commits, 3, binding + b'/commit', 'device')
    if fault == 'residual-reveal':
        # The transcript models a rushing corrupt role choosing a cancelling
        # check after reading honest reveals, while retaining its old commit.
        cancelling = (-sum(checks[1:])).remainder(PRIME)
        tau_reveals[0] = tau_salts[0] + _pack(cancelling)
    broadcast(tau_reveals, 4, binding + b'/reveal', 'device')
    expected, offset = [], 0
    for share in shares[0]:
        size = share.values.numel()
        expected.append(opened[offset:offset + size].reshape(share.values.shape))
        offset += size
    return transcripts, expected


def _runtime(p, transcript, alpha, coins, coin_salts, tau_salts):
    replay = DeviceReplay(transcript[p], check_outgoing=True)
    party = DeviceCompressedParty(p, replay, alpha[:, p].to('cuda'), session=b'compressed-device-oracle',
                                  _allow_insecure_test_transport=True)
    party._coin_contribution = lambda: coins[p]
    party._coin_salt = lambda: coin_salts[p]
    party._commit_salt = lambda: tau_salts[p]
    return party, replay


@pytest.mark.parametrize('keys', [7, 10])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('shapes', [[(3, 5)], [(1, 1), (5, 3)], [(0, 3)]])
def test_device_compressed_open_matches_every_cpu_crypto_wire_byte(dtype, shapes, keys):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer, _, shares, coins, coin_salts, tau_salts = _fixture(shapes, keys=keys)
    transcript, expected = _oracle(shares, dealer.alpha_shares, b'compressed-device-oracle', coins, coin_salts, tau_salts)
    for p in range(4):
        party, replay = _runtime(p, transcript, dealer.alpha_shares, coins, coin_salts, tau_salts)
        local = [AuthShare(x.values.to(device='cuda', dtype=dtype), x.macs.to(device='cuda', dtype=dtype)) for x in shares[p]]
        actual = party.open_many(local)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(actual, expected))
        assert not party.poisoned
        assert party.mac_keys == keys and party.round == 5
        replay.assert_consumed()


@pytest.mark.parametrize('kind', ['value', 'mac'])
def test_device_compression_rejects_consistent_but_unauthenticated_opening(kind):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer, _, shares, coins, coin_salts, tau_salts = _fixture([(3, 5)])
    target = shares[0][0].values if kind == 'value' else shares[0][0].macs
    target.reshape(-1)[0] = (target.reshape(-1)[0] + 1) % PRIME
    transcript, _ = _oracle(shares, dealer.alpha_shares, b'compressed-device-oracle', coins, coin_salts, tau_salts)
    for p in range(4):
        party, replay = _runtime(p, transcript, dealer.alpha_shares, coins, coin_salts, tau_salts)
        local = [AuthShare(x.values.to('cuda'), x.macs.to('cuda')) for x in shares[p]]
        with pytest.raises(AuthenticationError, match='compressed field MAC check failed'):
            party.open_many(local)
        assert party.poisoned
        replay.assert_consumed()


@pytest.mark.parametrize('fault', ['coin-reveal', 'residual-reveal'])
def test_device_compression_rejects_committed_coin_or_check_replacement(fault):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer, _, shares, coins, coin_salts, tau_salts = _fixture([(3, 5)])
    if fault == 'residual-reveal':
        shares[0][0].values[0, 0] = (shares[0][0].values[0, 0] + 1) % PRIME
    transcript, _ = _oracle(shares, dealer.alpha_shares, b'compressed-device-oracle', coins, coin_salts, tau_salts, fault=fault)
    for p in (1, 2, 3):
        party, _ = _runtime(p, transcript, dealer.alpha_shares, coins, coin_salts, tau_salts)
        local = [AuthShare(x.values.to('cuda'), x.macs.to('cuda')) for x in shares[p]]
        with pytest.raises(ProtocolError, match='commitment opening failed'):
            party.open_many(local)
        assert party.poisoned



def test_fresh_device_compressed_beaver_records_and_replays_real_entropy():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=10)
    x = torch.tensor([[3, -7, 9], [11, 27, -99]]).remainder(PRIME)
    y = torch.tensor([[71, 0], [89, -3], [22, 42]]).remainder(PRIME)
    xs, ys = split_share(dealer.share(x)), split_share(dealer.share(y))
    triples = split_triple(dealer.triple(2, 3, 2))
    network = IdealNetwork(timeout=60)
    device = torch.device('cuda', torch.cuda.current_device())
    entropy = [{} for _ in range(4)]
    records = [None]*4
    inputs = [None]*4
    def move(share):
        return AuthShare(share.values.to(device).int(), share.macs.to(device).int())
    def role(p):
        try:
            transport = DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p), device=device))
            records[p] = transport
            party = DeviceCompressedParty(p, transport, dealer.alpha_shares[:, p].to(device),
                        session=b'fresh-compressed-beaver', _allow_insecure_test_transport=True)
            for name in ('_coin_contribution', '_coin_salt', '_commit_salt'):
                original, draws = getattr(party, name), []
                entropy[p][name] = draws
                def draw(original=original, draws=draws):
                    value = original()
                    draws.append(value)
                    return value
                setattr(party, name, draw)
            triple = triples[p]
            triple.a, triple.b, triple.c = move(triple.a), move(triple.b), move(triple.c)
            local_x, local_y = move(xs[p]), move(ys[p])
            inputs[p] = local_x, local_y
            product = party.matmul(local_x, local_y, triple)
            return party.open(product).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        result = list(pool.map(role, range(4)))
    assert all(torch.equal(value, field_matmul(x, y)) for value in result)
    for p in range(4):
        from rep4gpu.spdz_distributed import MatrixTriple
        replay = DeviceReplay(records[p].transcript, check_outgoing=True)
        party = DeviceCompressedParty(p, replay, dealer.alpha_shares[:, p].to(device),
                    session=b'fresh-compressed-beaver', _allow_insecure_test_transport=True)
        for name, draws in entropy[p].items():
            assert len(draws) == 2 and all(len(draw) == 32 for draw in draws)
            queue = iter(draws)
            setattr(party, name, lambda queue=queue: next(queue))
        old = triples[p]
        triple = MatrixTriple(old.a, old.b, old.c, identifier=old.identifier)
        product = party.matmul(*inputs[p], triple)
        assert torch.equal(party.open(product).cpu(), result[p])
        replay.assert_consumed()



def test_seventh_device_mac_coordinate_is_checked_by_cpu_wire_oracle():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer, _, shares, coins, coin_salts, tau_salts = _fixture([(3, 5)], keys=7)
    shares[0][0].macs[6, 0, 0] = (shares[0][0].macs[6, 0, 0] + 1) % PRIME
    transcript, _ = _oracle(shares, dealer.alpha_shares, b'compressed-device-oracle', coins, coin_salts, tau_salts)
    for p in range(4):
        party, replay = _runtime(p, transcript, dealer.alpha_shares, coins, coin_salts, tau_salts)
        local = [AuthShare(x.values.to('cuda'), x.macs.to('cuda')) for x in shares[p]]
        with pytest.raises(AuthenticationError, match='compressed field MAC check failed'):
            party.open_many(local)
        assert party.poisoned
        replay.assert_consumed()


@pytest.mark.parametrize('party_type', ['immediate', 'deferred'])
def test_six_key_device_compression_is_rejected_before_online_frames(party_type):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
    class NoOnline:
        def exchange_device(self, *args, **kwargs):
            raise AssertionError('invalid MAC key count must not communicate')
    cls = DeviceCompressedParty if party_type == 'immediate' else DeviceDeferredCompressedParty
    with pytest.raises(ValueError, match='seven'):
        cls(0, NoOnline(), torch.ones(6, dtype=torch.int64, device='cuda'),
            session=b'reject-six', _allow_insecure_test_transport=True)
