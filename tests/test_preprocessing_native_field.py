"""Independent q256 persistence fixtures for role-local native preprocessing."""
import json
from pathlib import Path
import struct

import pytest

from rep4gpu import preprocessing_mascot_direct as direct


Q = 2**256 - 189
R = 189
SPEC = b'SPDZ gfp\x00' + struct.pack('<I', 32) + Q.to_bytes(32, 'big') + struct.pack('<I', 1)


def _montgomery(value):
    return (value*R % Q).to_bytes(32, 'little')


def _native_bytes(rows, key):
    header = SPEC + _montgomery(key)
    return (struct.pack('<Q', len(header)) + header
            + b''.join(_montgomery(value) + _montgomery(mac) for value, mac in rows))


def test_four_limb_montgomery_values_and_one_mac_decode_exactly(tmp_path):
    values = [0, 1, Q - 1, 2**64 - 1, 2**64, 2**127, 2**128 + 3, 2**192 + 7, Q//2]
    key = 2**255 + 2**128 + 29
    rows = [(value, (value*7919 + 104729) % Q) for value in values]
    path = tmp_path/'native.bin'
    path.write_bytes(_native_bytes(rows, key))
    actual, alpha = direct._native_file(path, len(rows))
    assert list(map(tuple, actual)) == rows
    assert alpha == key
    assert pow(2, 256, Q) == R
    assert _montgomery(1) == bytes([189]) + bytes(31)
    assert len(path.read_bytes()) == 8 + len(SPEC) + 32 + len(rows)*64


@pytest.mark.parametrize('kind', ['protocol', 'prime-endian', 'montgomery-flag', 'header-size', 'trailing', 'truncated-key', 'truncated-share'])
def test_native_header_and_exact_lengths_fail_closed(tmp_path, kind):
    raw = bytearray(_native_bytes([(1, 37)], 37))
    if kind == 'protocol':
        raw[8] ^= 1
    elif kind == 'prime-endian':
        start = 8 + len(b'SPDZ gfp\x00') + 4
        raw[start:start + 32] = Q.to_bytes(32, 'little')
    elif kind == 'montgomery-flag':
        start = 8 + len(SPEC) - 4
        raw[start:start + 4] = struct.pack('<I', 0)
    elif kind == 'header-size':
        raw[:8] = struct.pack('<Q', len(SPEC) + 64)
    elif kind == 'trailing':
        raw += b'\x00'
    elif kind == 'truncated-key':
        raw = raw[:8 + len(SPEC) + 31]
    else:
        raw = raw[:-1]
    path = tmp_path/'malformed.bin'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        direct._native_file(path, 1)


@pytest.mark.parametrize('location', ['key', 'value', 'mac'])
@pytest.mark.parametrize('word', [Q, 2**256 - 1])
def test_noncanonical_raw_montgomery_field_words_are_rejected(tmp_path, location, word):
    raw = bytearray(_native_bytes([(1, 37)], 37))
    offset = {'key': 8 + len(SPEC), 'value': 8 + len(SPEC) + 32,
              'mac': 8 + len(SPEC) + 64}[location]
    raw[offset:offset + 32] = word.to_bytes(32, 'little')
    path = tmp_path/'noncanonical.bin'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='canonical'):
        direct._native_file(path, 1)


def test_wrong_share_count_and_multiple_mac_layout_are_rejected(tmp_path):
    path = tmp_path/'wrong-count.bin'
    path.write_bytes(_native_bytes([(1, 37), (9, 19)], 37))
    with pytest.raises(ValueError):
        direct._native_file(path, 1)
    # Native SPDZ here has one MAC per value, not a seven-coordinate p31 row.
    path.write_bytes(_native_bytes([(1, 37)], 37) + bytes(32))
    with pytest.raises(ValueError):
        direct._native_file(path, 1)


def _clear_bytes(allocations):
    return b''.join(struct.pack('<Q', (value >> (64*limb)) & (2**64 - 1))
                    for values in allocations for limb in range(4) for value in values)


def test_owner_clear_masks_are_limb_major_unsigned_and_segmented():
    allocations = [[2**255 + 2**200 + 2**129 + 2**65 + 3, 2**128 + 2**64 - 1, Q - 1],
                   [2**192 + 2**127 + 19]]
    assert direct._decode_owner_clear(_clear_bytes(allocations), [3, 1]) == sum(allocations, [])


@pytest.mark.parametrize('kind', ['truncated', 'trailing', 'noncanonical'])
def test_owner_clear_mask_encoding_rejects_malformed_words(kind):
    raw = _clear_bytes([[17, 2**255 + 1]])
    if kind == 'truncated':
        raw = raw[:-1]
    elif kind == 'trailing':
        raw += b'\x00'
    else:
        raw = _clear_bytes([[17, Q]])
    with pytest.raises(ValueError):
        direct._decode_owner_clear(raw, [2])


def _role_bundle(tmp_path, monkeypatch, *, party=1, calibration_fault=None):
    import hashlib
    request = dict(matrices=[dict(name='matrix', m=1, k=2, n=1)],
                   squares=[dict(name='square', shape=[1, 2])],
                   inputs=[dict(name='owned', owner=party, shape=[1, 3]),
                           dict(name='other', owner=(party + 1) % 4, shape=[1, 2])])
    session = '12'*32
    records, total, counts = direct._layout(request, session)
    key = 2**255 + 2**129 + 7919
    rows = [((2**200 + index*2**72 + 17) % Q, (key*index + 104729) % Q) for index in range(total)]
    rows[0] = (int(party == 0), key)
    if calibration_fault == 'value':
        rows[0] = (1 - int(party == 0), key)
    elif calibration_fault == 'mac':
        rows[0] = (int(party == 0), (key + 1) % Q)
    role = tmp_path/f'party{party}'
    (role/'Persistence').mkdir(parents=True)
    (role/'Player-Data').mkdir()
    native = tmp_path/'public-native-build'
    native.mkdir()
    binary = native/'mascot-party.x'
    binary.write_bytes(b'synthetic unit-test executable identity')
    policy = dict(post_authentication_sacrifice_coin=True,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    encoded_policy = json.dumps(policy, sort_keys=True).encode()
    policy_hash = hashlib.sha256(encoded_policy).hexdigest()
    (native/'mascot256-preprocessing-policy.json').write_bytes(encoded_policy)
    (role/'native-policy.json').write_bytes(encoded_policy)
    metadata = dict(format='mascot256-native-role-v1', ready=True, return_codes=[0]*4,
                    parties=4, prime=Q, mac_keys=1, security_parameter=128, session=session,
                    request=request, records=records, share_count=total, clear_counts=counts,
                    native_policy=policy, native_policy_sha256=policy_hash,
                    native_binary=str(binary), native_binary_sha256=policy['binary_sha256'],
                    native_root=str(native),
                    export_source_sha256=hashlib.sha256(direct._program(request).encode()).hexdigest())
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    data_path = role/'Persistence'/f'Transactions-P{party}.data'
    data_path.write_bytes(_native_bytes(rows, key))
    clear = [2**255 + 2**128 + 3, 2**192 + 2**64 + 19, Q - 1]
    (role/'Player-Data'/f'Binary-Output-P{party}-0').write_bytes(_clear_bytes([clear]))
    # Binary-format tests supply a synthetic reviewed-policy boundary. Actual
    # source/build verification is exercised separately by the native builder.
    def reviewed_policy(root, executable):
        assert Path(root).resolve() == native.resolve()
        assert Path(executable).resolve() == binary.resolve()
        return policy, policy_hash
    monkeypatch.setattr(direct, '_verify_policy', reviewed_policy)
    return role, metadata, rows, key, clear


def test_role_loader_calibration_shapes_owner_privacy_and_single_use(tmp_path, monkeypatch):
    role, metadata, rows, key, clear = _role_bundle(tmp_path, monkeypatch)
    original = direct._native_file
    calls = []
    def assigned_only(path, count):
        assert Path(path).parent.parent == role
        calls.append(Path(path))
        return original(path, count)
    monkeypatch.setattr(direct, '_native_file', assigned_only)
    loaded = direct.load_role(role, 1)
    assert loaded['alpha_share'] == key
    triple = loaded['matrix_triples']['matrix']
    assert triple.a.shape == (1, 2) and triple.b.shape == (2, 1) and triple.c.shape == (1, 1)
    assert loaded['square_masks']['square'].a.shape == (1, 2)
    assert loaded['input_masks']['owned'].clear == clear
    assert loaded['input_masks']['other'].clear is None
    assert loaded['input_masks']['owned'].owner == 1
    assert loaded['input_masks']['other'].owner == 2
    assert len(loaded['session']) == 32 and loaded['session'] != bytes.fromhex(metadata['session'])
    for record in metadata['records'][1:]:
        collection = {'matrices': loaded['matrix_triples'], 'squares': loaded['square_masks'],
                      'inputs': loaded['input_masks']}[record['kind']]
        item = collection[record['name']]
        share = item.share if record['kind'] == 'inputs' else getattr(item, record['component'])
        expected = rows[record['offset']:record['offset'] + record['count']]
        assert share.values == [value for value, _ in expected]
        assert share.macs == [mac for _, mac in expected]
    with pytest.raises(FileExistsError):
        direct.load_role(role, 1)
    assert len(calls) == 1 and (role/'consumed').is_file()
    assert not (tmp_path/'party0').exists() and not (tmp_path/'party2').exists() and not (tmp_path/'party3').exists()


@pytest.mark.parametrize('fault', ['value', 'mac'])
def test_bad_constant_one_calibration_burns_role_bundle(tmp_path, monkeypatch, fault):
    role, _, _, _, _ = _role_bundle(tmp_path, monkeypatch, calibration_fault=fault)
    with pytest.raises(ValueError, match='calibration'):
        direct.load_role(role, 1)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):
        direct.load_role(role, 1)


def test_wrong_role_cannot_read_or_consume_private_payload(tmp_path, monkeypatch):
    role, _, _, _, _ = _role_bundle(tmp_path, monkeypatch)
    def forbidden(*args):
        raise AssertionError('wrong role must not read authenticated shares')
    monkeypatch.setattr(direct, '_native_file', forbidden)
    with pytest.raises(ValueError, match='role'):
        direct.load_role(role, 2)
    assert not (role/'consumed').exists()


def test_self_consistent_role_policy_cannot_replace_reviewed_native_build(tmp_path, monkeypatch):
    import hashlib
    role, metadata, _, _, _ = _role_bundle(tmp_path, monkeypatch)
    forged = dict(metadata['native_policy'], unauthorized_change=True)
    encoded = json.dumps(forged, sort_keys=True).encode()
    (role/'native-policy.json').write_bytes(encoded)
    metadata['native_policy'] = forged
    metadata['native_policy_sha256'] = hashlib.sha256(encoded).hexdigest()
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='provenance'):
        direct.load_role(role, 1)
    assert not (role/'consumed').exists()


def test_role_loader_reads_only_assigned_private_directory(tmp_path, monkeypatch):
    role, _, _, _, _ = _role_bundle(tmp_path, monkeypatch)
    original_bytes, original_text = Path.read_bytes, Path.read_text
    def check(path):
        assert not any(f'party{p}' in path.parts for p in (0, 2, 3))
    def read_bytes(path):
        check(path)
        return original_bytes(path)
    def read_text(path, *args, **kwargs):
        check(path)
        return original_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    monkeypatch.setattr(Path, 'read_text', read_text)
    assert direct.load_role(role, 1)['input_masks']['other'].clear is None


def _native_role_probe(root, party, addresses):
    """Consume only this disposable role's correlations and report booleans."""
    import ssl
    from rep4gpu.spdz_prime256 import Prime256Party
    from rep4gpu.transport import SocketTransport
    role = Path(root)/f'party{party}'
    player = role/'Player-Data'
    ca = role/'all-public-certs.pem'
    ca.write_bytes(b''.join((player/f'P{p}.pem').read_bytes() for p in range(4)))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.verify_mode = ssl.CERT_REQUIRED
    server.load_cert_chain(player/f'P{party}.pem', player/f'P{party}.key')
    server.load_verify_locations(cafile=ca)
    client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
    client.load_cert_chain(player/f'P{party}.pem', player/f'P{party}.key')
    material = direct.load_role(role, party)
    transport = SocketTransport(party, [tuple(address) for address in addresses], timeout=30,
                               server_context=server, client_context=client,
                               peer_hostnames={p: f'P{p}' for p in range(4)})
    runtime = Prime256Party(party, transport, material['alpha_share'], session=material['session'])
    try:
        triple = material['matrix_triples']['probe']
        triple.used = True
        a, b, c = runtime.open_many((triple.a, triple.b, triple.c), label=b'disposable-q256-matrix')
        m, k = triple.a.shape
        _, n = triple.b.shape
        expected = [sum(a[i*k+t]*b[t*n+j] for t in range(k)) % Q for i in range(m) for j in range(n)]
        assert c == expected
        square = material['square_masks']['sq']
        square.used = True
        a, aa = runtime.open_many((square.a, square.a_squared), label=b'disposable-q256-square')
        assert aa == [value*value % Q for value in a]
        owner_only = True
        for name in sorted(material['input_masks']):
            mask = material['input_masks'][name]
            mask.used = True
            opened = runtime.open_many((mask.share,), label=b'disposable-q256-mask/' + name.encode())[0]
            if mask.owner == party:
                assert mask.clear == opened
            else:
                owner_only &= mask.clear is None
        assert owner_only
        report = dict(party=party, passed=True, matrix_bigint_relation=True,
                      square_bigint_relation=True, unsigned_owner_masks=True,
                      owner_only_clear=True, one_native_mac=True,
                      disposable=True, role_local_private_reads=True)
        (role/'format-probe.json').write_text(json.dumps(report, indent=2) + '\n')
    except BaseException as error:
        runtime.abort(error)
        raise
    finally:
        transport.close()



def test_owner_clear_export_uses_unsigned_conversion_for_every_limb():
    request = dict(inputs=[dict(name='owned', owner=1, shape=[1, 3])],
                   squares=[dict(name='sq', shape=[1, 2])])
    source = direct._program(request)
    for shift in (0, 64, 128, 192):
        assert f'(ic0>>{shift}).to_regint(n_bits=0,sync=False).binary_output(1)' in source
    assert 'sint.write_to_file(sq0)' in source and 'sint.write_to_file(sqc0)' in source
    # CONVMODP with n_bits64 first centers x in the field. For x=q-1 that
    # changes its low limb by189; unsigned conversion must preserve q-1.
    value = Q - 1
    canonical_low = value % 2**64
    centered_low = (value - Q) % 2**64
    assert (centered_low - canonical_low) % 2**64 == 189


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5 or sys.argv[1] != '--native-role':
        raise SystemExit('usage: test_preprocessing_native_field.py --native-role ROOT PARTY ADDRESSES_JSON')
    configuration = json.loads(sys.argv[4])
    _native_role_probe(sys.argv[2], int(sys.argv[3]), configuration)
