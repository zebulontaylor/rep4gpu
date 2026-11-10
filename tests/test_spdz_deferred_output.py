"""Public threshold selection after the deferred checked-output boundary."""
import struct

import pytest
import torch

from rep4gpu.spdz_compression import DeviceCompressedParty
from rep4gpu.spdz_device import DeviceDistributedParty
from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
from rep4gpu.spdz_distributed import AuthShare


@pytest.mark.parametrize('count', [0, 1, 4, 5])
@pytest.mark.parametrize('threshold', [0, 4])
def test_output_policy_flushes_before_selecting_public_count_only(monkeypatch, count, threshold):
    events = []
    def full(self, shares, *, label):
        events.append(('full-vector', label))
        return 'full-vector'
    def compressed(self, shares, *, label):
        events.append(('compressed', label))
        return 'compressed'
    monkeypatch.setattr(DeviceDistributedParty, 'open_many', full)
    monkeypatch.setattr(DeviceCompressedParty, 'open_many', compressed)
    party = object.__new__(DeviceDeferredCompressedParty)
    party.poisoned = False
    party.max_pending_values = 123
    party.output_check_threshold = threshold
    party.flush_pending = lambda: events.append(('flush', None))
    shares = [AuthShare(torch.empty((1, count), dtype=torch.int32), torch.empty((10, 1, count), dtype=torch.int32))]
    result = party.open_many(shares, label=b'typed-output')
    mode = 'full-vector' if threshold > 0 and count <= threshold else 'compressed'
    assert result == mode and events[0] == ('flush', None)
    assert [event[0] for event in events] == ['flush', mode]
    assert struct.pack('<QQ', 123, threshold) in events[1][1]
    assert mode.encode() in events[1][1]
    assert events[1][1].endswith(b'typed-output')


@pytest.mark.parametrize('threshold', [-1, 2**31, True, 1.5, None])
def test_output_threshold_validation_precedes_gpu_setup(threshold):
    with pytest.raises(ValueError, match='output_check_threshold'):
        DeviceDeferredCompressedParty(output_check_threshold=threshold)


def _full_output_oracle(shares, alpha, session, salts, cap, threshold, user_label):
    import hashlib
    import json
    from blake3 import blake3
    from rep4gpu.device_transport import DeviceFrame
    from rep4gpu.spdz import PRIME
    def h(data):
        return hashlib.blake2b(data, digest_size=32).digest()
    def pack(value):
        return value.numpy().astype('<u4').tobytes()
    keys = alpha.shape[0]
    layout = json.dumps({'field': PRIME, 'dtype': 'uint32-le', 'keys': keys,
                         'shapes': [list(shares[0].values.shape)]}, sort_keys=True, separators=(',', ':')).encode()
    public_label = b'deferred-output/' + struct.pack('<QQ', cap, threshold) + b'full-vector\0' + user_label
    session_hash = h(b'SPDZ31-session-v1' + struct.pack('<Q', len(session)) + session)
    context = b'SPDZ31G-open-v1' + session_hash + struct.pack('<Q', 0) + h(layout) + h(public_label)
    transcript = [[] for _ in range(4)]
    def broadcast(payloads, round_id, label):
        def frame(stage, p, body):
            return (b'SPDZ31G1' + session_hash + struct.pack('<QI', round_id, p) + stage
                    + h(label) + struct.pack('<Q', len(body)) + body)
        values = [frame(b'V', p, payloads[p]) for p in range(4)]
        view = b''.join(blake3(struct.pack('<I', p) + values[p]).digest() for p in range(4))
        echoes = [frame(b'E', p, view) for p in range(4)]
        for p in range(4):
            peers = set(range(4)) - {p}
            transcript[p].append(DeviceFrame('device', {q: values[p] for q in peers}, {q: values[q] for q in peers}))
            transcript[p].append(DeviceFrame('device', {q: echoes[p] for q in peers}, {q: echoes[q] for q in peers}))
    opened = sum(share.values for share in shares).remainder(PRIME)
    residual = [(share.macs - alpha[:, p, None, None]*opened).remainder(PRIME) for p, share in enumerate(shares)]
    broadcast([pack(share.values) for share in shares], 0, context + b'values')
    reveals = [salts[p] + pack(residual[p]) for p in range(4)]
    commits = [blake3(context + struct.pack('<I', p) + reveals[p]).digest() for p in range(4)]
    broadcast(commits, 1, context + b'commit')
    broadcast(reveals, 2, context + b'reveal')
    return transcript, opened


@pytest.mark.parametrize('keys', [7, 10])
@pytest.mark.parametrize('count', [0, 4])
def test_hybrid_full_vector_output_matches_every_independent_cpu_wire_byte(count, keys):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from rep4gpu.spdz import TrustedDealer
    from rep4gpu.spdz_distributed import split_share
    from rep4gpu.device_transport import DeviceReplay
    dealer = TrustedDealer(mac_keys=keys)
    shares = split_share(dealer.share(torch.arange(count).reshape(1, count)))
    salts = [bytes([p + 11])*32 for p in range(4)]
    session, user_label, cap, threshold = b'hybrid-fullvector-byte-oracle', b'typed-output', 123, 4
    transcript, expected = _full_output_oracle(shares, dealer.alpha_shares, session, salts, cap, threshold, user_label)
    def no_coin():
        raise AssertionError('full-vector output checks do not draw a challenge coin')
    for p in range(4):
        replay = DeviceReplay(transcript[p], check_outgoing=True)
        party = DeviceDeferredCompressedParty(p, replay, dealer.alpha_shares[:, p].to('cuda'),
                    session=session, max_pending_values=cap, output_check_threshold=threshold,
                    _allow_insecure_test_transport=True)
        party._commit_salt = lambda p=p: salts[p]
        party._coin_contribution, party._coin_salt = no_coin, no_coin
        local = AuthShare(shares[p].values.to('cuda').int(), shares[p].macs.to('cuda').int())
        actual = party.open_many((local,), label=user_label)[0]
        assert torch.equal(actual.cpu(), expected)
        assert party.last_output_check_mode == 'full-vector' and party.round == 3
        replay.assert_consumed()
