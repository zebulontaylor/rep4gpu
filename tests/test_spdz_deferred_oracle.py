"""Independent CPU-crypto transcripts for segmented deferred MAC batches."""
import hashlib
import json
import struct

import pytest
import torch

from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import AuthShare, split_share
from rep4gpu.spdz_compression import DOMAIN, CHACHA_BACKEND, coefficient_row_chacha
from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
from rep4gpu.device_transport import DeviceFrame, DeviceReplay


def _h(data):
    return hashlib.blake2b(data, digest_size=32).digest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _pack(value):
    return value.numpy().astype('<u4').tobytes()


def _oracle(masked, outputs, alpha, session, cap, threshold, coins, coin_salts, tau_salts):
    from blake3 import blake3
    keys = alpha.shape[0]
    session_hash = _h(b'SPDZ31-session-v1' + struct.pack('<Q', len(session)) + session)
    transcripts, schemas, residuals = [[] for _ in range(4)], [], []
    round_id = 0
    def broadcast(payloads, label, kind):
        nonlocal round_id
        hash_fn = (lambda data: blake3(data).digest()) if kind == 'device' else _h
        magic = b'SPDZ31G1' if kind == 'device' else b'SPDZ31v1'
        def frame(stage, p, body):
            return (magic + session_hash + struct.pack('<QI', round_id, p) + stage
                    + _h(label) + struct.pack('<Q', len(body)) + body)
        values = [frame(b'V', p, payloads[p]) for p in range(4)]
        view = b''.join(hash_fn(struct.pack('<I', p) + values[p]) for p in range(4))
        echoes = [frame(b'E', p, view) for p in range(4)]
        for p in range(4):
            peers = set(range(4)) - {p}
            transcripts[p].append(DeviceFrame(kind, {q: values[p] for q in peers}, {q: values[q] for q in peers}))
            transcripts[p].append(DeviceFrame(kind, {q: echoes[p] for q in peers}, {q: echoes[q] for q in peers}))
        round_id += 1
    for index, shares in enumerate(masked):
        count = shares[0].values.numel()
        label = b'fresh-mask-' + bytes([index])
        schema = {'index': index, 'round': round_id, 'count': count,
                  'shapes': [list(shares[0].values.shape)], 'label': label.hex(),
                  'max_pending_values': cap, 'output_check_threshold': threshold}
        context = (b'SPDZ31G-deferred-masked-v1' + session_hash + struct.pack('<Q', round_id)
                   + _h(_json(schema)) + _h(CHACHA_BACKEND))
        broadcast([_pack(share.values) for share in shares], context + b'/values', 'device')
        opened = sum(share.values for share in shares).remainder(PRIME).reshape(-1)
        residuals.append([(share.macs.reshape(keys, -1) - alpha[:, p, None]*opened).remainder(PRIME)
                          for p, share in enumerate(shares)])
        schema['context'] = context.hex()
        schemas.append(_json(schema).hex())
    layout = {'field': PRIME, 'keys': keys, 'backend': CHACHA_BACKEND.hex(),
              'max_pending_values': cap, 'output_check_threshold': threshold, 'segments': schemas}
    context = (b'SPDZ31G-deferred-check-v1' + session_hash + struct.pack('<QQ', round_id, 0)
               + _h(_json(layout)))
    coin_binding = DOMAIN + b'/coin/' + context
    reveals = [coin_salts[p] + coins[p] for p in range(4)]
    commits = [_h(coin_binding + struct.pack('<I', p) + reveals[p]) for p in range(4)]
    broadcast(commits, coin_binding + b'/commit', 'bytes')
    broadcast(reveals, coin_binding + b'/reveal', 'bytes')
    joint = bytes(a ^ b ^ c ^ d for a, b, c, d in zip(*coins))
    checks = [torch.zeros(keys, dtype=torch.int64) for _ in range(4)]
    for index, segment in enumerate(residuals):
        segment_context = context + b'/segment/' + struct.pack('<Q', index)
        chi = torch.stack([coefficient_row_chacha(joint, segment_context, key, segment[0].shape[1]) for key in range(keys)])
        for p in range(4):
            part = (segment[p]*chi).remainder(PRIME).sum(1).remainder(PRIME)
            checks[p] = (checks[p] + part).remainder(PRIME)
    binding = DOMAIN + b'/residual/' + context
    reveals = [tau_salts[p][0] + _pack(checks[p]) for p in range(4)]
    commits = [blake3(binding + struct.pack('<I', p) + reveals[p]).digest() for p in range(4)]
    broadcast(commits, binding + b'/commit', 'device')
    broadcast(reveals, binding + b'/reveal', 'device')
    label = b'deferred-output/' + struct.pack('<QQ', cap, threshold) + b'full-vector\0checked-output'
    layout = {'field': PRIME, 'dtype': 'uint32-le', 'keys': keys, 'shapes': [list(outputs[0].values.shape)]}
    context = (b'SPDZ31G-open-v1' + session_hash + struct.pack('<Q', round_id)
               + _h(_json(layout)) + _h(label))
    broadcast([_pack(share.values) for share in outputs], context + b'values', 'device')
    expected = sum(share.values for share in outputs).remainder(PRIME)
    residual = [(share.macs - alpha[:, p, None, None]*expected).remainder(PRIME) for p, share in enumerate(outputs)]
    reveals = [tau_salts[p][1] + _pack(residual[p]) for p in range(4)]
    commits = [blake3(context + struct.pack('<I', p) + reveals[p]).digest() for p in range(4)]
    broadcast(commits, context + b'commit', 'device')
    broadcast(reveals, context + b'reveal', 'device')
    return transcripts, expected


@pytest.mark.parametrize('keys', [7, 10])
@pytest.mark.parametrize('segments', [1, 2])
def test_deferred_segment_checks_match_every_cpu_crypto_wire_byte(segments, keys):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    dealer = TrustedDealer(mac_keys=keys)
    masked = [split_share(dealer.share(torch.arange(3 - index).reshape(1, 3 - index))) for index in range(segments)]
    output = split_share(dealer.share(torch.tensor([[42]])))
    coins = [bytes([p + 1])*32 for p in range(4)]
    coin_salts = [bytes([p + 7])*32 for p in range(4)]
    tau_salts = [[bytes([p + 13])*32, bytes([p + 19])*32] for p in range(4)]
    session, cap, threshold = b'deferred-segment-wire-oracle', 123, 4
    transcript, expected = _oracle(masked, output, dealer.alpha_shares, session, cap, threshold, coins, coin_salts, tau_salts)
    for p in range(4):
        replay = DeviceReplay(transcript[p], check_outgoing=True)
        party = DeviceDeferredCompressedParty(p, replay, dealer.alpha_shares[:, p].to('cuda'), session=session,
                    max_pending_values=cap, output_check_threshold=threshold, _allow_insecure_test_transport=True)
        party._coin_contribution = lambda p=p: coins[p]
        party._coin_salt = lambda p=p: coin_salts[p]
        salts = iter(tau_salts[p])
        party._commit_salt = lambda salts=salts: next(salts)
        def move(share):
            return AuthShare(share.values.to('cuda').int(), share.macs.to('cuda').int())
        for index, shares in enumerate(masked):
            party._masked_open_many((move(shares[p]),), label=b'fresh-mask-' + bytes([index]))
        actual = party.open_many((move(output[p]),), label=b'checked-output')[0]
        assert torch.equal(actual.cpu(), expected)
        assert not party.poisoned and not party._pending and party._flush_index == 1
        replay.assert_consumed()
