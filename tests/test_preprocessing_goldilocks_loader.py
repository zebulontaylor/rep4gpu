"""Independent one-limb fixtures for the role-local Goldilocks loader."""
import hashlib
import json
from pathlib import Path
import random
import struct

import pytest

from rep4gpu import preprocessing_lowgear_goldilocks_loader as native

Q = 2**64 - 2**32 + 1
R = 2**32 - 1
SPEC = b'SPDZ gfp\0' + struct.pack('<I', 8) + Q.to_bytes(8, 'big') + struct.pack('<I', 1)


def mont(value):
    return (value * R % Q).to_bytes(8, 'little')


def encoded(rows, key):
    header = SPEC + mont(key)
    return struct.pack('<Q', len(header)) + header + b''.join(mont(v) + mont(m) for v, m in rows)


def test_full_unsigned_limb_montgomery_oracle(tmp_path):
    rng = random.Random(918237)
    values = [0, 1, 2**32 - 1, 2**32, 2**63 - 1, 2**63, Q//2, Q - 2, Q - 1]
    values += [rng.randrange(Q) for _ in range(300)]
    rows = [(v, rng.randrange(Q)) for v in values]
    key = Q - 17
    path = tmp_path/'rows'
    path.write_bytes(encoded(rows, key))
    assert native._native_file(path, len(rows)) == (rows, key)
    assert pow(2, 64, Q) == R
    assert mont(1) == bytes.fromhex('ffffffff00000000')
    assert len(path.read_bytes()) == 8 + len(SPEC) + 8 + 16 * len(rows)


@pytest.mark.parametrize('fault', ['signature', 'prime-endian', 'montgomery', 'header-length',
                                   'short-key', 'short-row', 'trailing', 'extra-mac'])
def test_native_spec_and_exact_lengths(tmp_path, fault):
    raw = bytearray(encoded([(1, 17)], 17))
    if fault == 'signature': raw[8] ^= 1
    elif fault == 'prime-endian': raw[21:29] = Q.to_bytes(8, 'little')
    elif fault == 'montgomery': raw[29:33] = struct.pack('<I', 0)
    elif fault == 'header-length': raw[:8] = struct.pack('<Q', len(SPEC) + 16)
    elif fault == 'short-key': raw = raw[:8 + len(SPEC) + 7]
    elif fault == 'short-row': raw = raw[:-1]
    elif fault == 'extra-mac': raw += bytes(8)
    else: raw += b'\0'
    path = tmp_path/'bad'
    path.write_bytes(raw)
    with pytest.raises(ValueError): native._native_file(path, 1)


@pytest.mark.parametrize('where', ['key', 'value', 'mac'])
@pytest.mark.parametrize('bad', [Q, 2**64 - 1])
def test_raw_noncanonical_words(tmp_path, where, bad):
    raw = bytearray(encoded([(1, 17)], 17))
    at = 8 + len(SPEC) + {'key': 0, 'value': 8, 'mac': 16}[where]
    raw[at:at + 8] = bad.to_bytes(8, 'little')
    path = tmp_path/'bad'
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='canonical'): native._native_file(path, 1)


def test_owner_masks_are_unsigned_one_word_per_record():
    values = [0, 2**63, Q - 1, 17]
    raw = b''.join(v.to_bytes(8, 'little') for v in values)
    assert native._decode_owner_clear(raw, [3, 1]) == values
    for bad in [raw[:-1], raw + b'\0', raw[:8] + Q.to_bytes(8, 'little') + raw[16:]]:
        with pytest.raises(ValueError): native._decode_owner_clear(bad, [3, 1])


def bundle(tmp_path, monkeypatch, party=1):
    request = dict(matrices=[dict(name='mat', m=1, k=2, n=1)],
                   squares=[dict(name='sq', shape=[1, 2])],
                   inputs=[dict(name='owned', shape=[1, 3], owner=party),
                           dict(name='other', shape=[1, 2], owner=(party + 1) % 4)])
    session = 'a7' * 32
    records, total, counts = native._layout(request, session)
    key = Q - 7919
    rows = [((2**63 + i*9991) % Q, (key*i + 104729) % Q) for i in range(total)]
    rows[0] = (int(party == 0), key)
    role = tmp_path/f'party{party}'
    (role/'Persistence').mkdir(parents=True)
    (role/'Player-Data').mkdir()
    root = tmp_path/'public-build'
    root.mkdir()
    binary = root/'lowgear-party.x'
    binary.write_bytes(b'synthetic executable identity')
    policy = dict(protocol='LowGear', covert=False, prime=Q, statistical_security=40,
                  computational_security=128, finite_bit_probe_default=False)
    text = json.dumps(policy, sort_keys=True).encode()
    digest = hashlib.sha256(text).hexdigest()
    (root/'lowgear-goldilocks-s40-policy.json').write_bytes(text)
    (role/'native-policy.json').write_bytes(text)
    metadata = dict(format='lowgear-goldilocks-s40-native-field-v1', ready=True, passed=True, parties=4,
                    return_codes=[0]*4, prime=Q, mac_keys=1, security_parameter=40,
                    statistical_security=40, computational_security=128, finite_bit_probe=False,
                    field_conversion=False, experimental=True, security_target_met=False,
                    native_root=str(root), native_binary=str(binary),
                    native_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    native_policy=policy, native_policy_sha256=digest, session=session,
                    request=request, records=records, share_count=total, clear_counts=counts,
                    export_source_sha256=hashlib.sha256(native._program(request).encode()).hexdigest())
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    (tmp_path/'public-manifest.json').write_text(json.dumps(metadata))
    (role/'Persistence'/f'Transactions-P{party}.data').write_bytes(encoded(rows, key))
    clear = [2**63, Q - 1, Q//2 + 1]
    (role/'Player-Data'/f'Binary-Output-P{party}-0').write_bytes(b''.join(v.to_bytes(8, 'little') for v in clear))
    def gate(actual_root, actual_binary):
        assert Path(actual_root) == root and Path(actual_binary) == binary
        return policy, digest
    monkeypatch.setattr(native, '_verify_policy', gate)
    return role, metadata, rows, key, clear


def test_role_local_shapes_calibration_owner_privacy_and_one_use(tmp_path, monkeypatch):
    role, metadata, rows, key, clear = bundle(tmp_path, monkeypatch)
    reader = native._read_local
    reads = []
    def assigned_only(path, directory, filename, **kwargs):
        assert path == role and filename in ['Transactions-P1.data', 'Binary-Output-P1-0']
        reads.append(filename)
        return reader(path, directory, filename, **kwargs)
    monkeypatch.setattr(native, '_read_local', assigned_only)
    loaded = native.load_role(role, 1)
    assert loaded['alpha_share'] == key
    assert loaded['matrix_triples']['mat'].a.shape == (1, 2)
    assert loaded['matrix_triples']['mat'].b.shape == (2, 1)
    assert loaded['matrix_triples']['mat'].c.shape == (1, 1)
    assert loaded['square_masks']['sq'].a.shape == (1, 2)
    assert loaded['input_masks']['owned'].clear == clear
    assert loaded['input_masks']['other'].clear is None
    assert len(loaded['session']) == 32 and loaded['session'] != bytes.fromhex(metadata['session'])
    for record in metadata['records'][1:]:
        item = loaded[{'matrices':'matrix_triples', 'squares':'square_masks', 'inputs':'input_masks'}[record['kind']]][record['name']]
        share = item.share if record['kind'] == 'inputs' else getattr(item, record['component'])
        expected = rows[record['offset']:record['offset'] + record['count']]
        assert share.values == [v for v, _ in expected] and share.macs == [m for _, m in expected]
    assert sorted(reads) == ['Binary-Output-P1-0', 'Transactions-P1.data']
    with pytest.raises(FileExistsError): native.load_role(role, 1)
    assert not any((tmp_path/f'party{p}').exists() for p in [0, 2, 3])


@pytest.mark.parametrize('fault', ['calibration-value', 'calibration-mac', 'bad-payload', 'missing-clear', 'symlink-file', 'symlink-directory'])
def test_private_payload_failures_burn_whole_bundle(tmp_path, monkeypatch, fault):
    role, metadata, rows, key, _ = bundle(tmp_path, monkeypatch)
    data = role/'Persistence/Transactions-P1.data'
    if fault.startswith('calibration'):
        rows[0] = (1, key) if fault.endswith('value') else (0, (key + 1) % Q)
        data.write_bytes(encoded(rows, key))
    elif fault == 'bad-payload': data.write_bytes(data.read_bytes()[:-1])
    elif fault == 'missing-clear': (role/'Player-Data/Binary-Output-P1-0').unlink()
    elif fault == 'symlink-file':
        outside = tmp_path/'outside-private'; outside.write_bytes(data.read_bytes());data.unlink();data.symlink_to(outside)
    else:
        original = role/'Persistence'; renamed = tmp_path/'outside-persistence';original.rename(renamed);original.symlink_to(renamed)
    with pytest.raises((ValueError, OSError)): native.load_role(role, 1)
    assert (role/'consumed').is_file()
    with pytest.raises(FileExistsError): native.load_role(role, 1)


@pytest.mark.parametrize('fault', ['wrong-role', 'prime', 'stat', 'comp', 'probe', 'program', 'policy', 'binary', 'layout', 'session'])
def test_public_provenance_failures_do_not_read_private_data(tmp_path, monkeypatch, fault):
    role, metadata, *_ = bundle(tmp_path, monkeypatch)
    if fault == 'prime': metadata['prime'] += 2
    elif fault == 'stat': metadata['statistical_security'] = 128
    elif fault == 'comp': metadata['computational_security'] = 80
    elif fault == 'probe': metadata['finite_bit_probe'] = True
    elif fault == 'program': metadata['export_source_sha256'] = '00'*32
    elif fault == 'policy': metadata['native_policy'] = dict(metadata['native_policy'], unauthorized=True)
    elif fault == 'binary': metadata['native_binary_sha256'] = '00'*32
    elif fault == 'layout': metadata['records'][1]['count'] += 1
    elif fault == 'session': metadata['session'] = '00'
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    def forbidden(*args, **kwargs): raise AssertionError('public gate must precede private reads')
    monkeypatch.setattr(native, '_read_local', forbidden)
    with pytest.raises(ValueError): native.load_role(role, 2 if fault == 'wrong-role' else 1)
    assert not (role/'consumed').exists()


def test_export_program_uses_one_unsigned_word_per_owner_mask():
    request = dict(inputs=[dict(name='x', owner=1, shape=[1, 2])])
    source = native._program(request)
    assert 'program.security=40' in source
    assert source.count('.to_regint(n_bits=0,sync=False).binary_output(1)') == 1
    assert all('>>'+str(shift)+')' not in source for shift in [64, 128, 192])


def test_concurrent_load_claims_allow_exactly_one_consumer(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    role, *_ = bundle(tmp_path, monkeypatch)
    def claim():
        try:
            return native.load_role(role, 1)
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sum(value is not None for value in results) == 1


@pytest.mark.parametrize('key,value', [('return_codes', [False]*4), ('parties', True),
                                      ('share_count', True), ('clear_counts', [False]*4)])
def test_metadata_numeric_booleans_fail_closed(tmp_path, monkeypatch, key, value):
    role, metadata, *_ = bundle(tmp_path, monkeypatch)
    metadata[key] = value
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError): native.load_role(role, 1)
    assert not (role/'consumed').exists()


def warm_bundle(tmp_path, monkeypatch):
    parent_dir = tmp_path/'parent'; parent_dir.mkdir()
    parent_role, parent, *_ = bundle(parent_dir, monkeypatch)
    (parent_role/'consumed').write_text(parent['session'])
    report = dict(passed=True, return_codes=[0]*4, source_stable_during_run=True,
                  exact_integer_match=True, public_preprocessing_manifest=parent)
    report_path = tmp_path/'parent-success.json'; report_path.write_text(json.dumps(report))
    child_dir = tmp_path/'child'; child_dir.mkdir()
    role, metadata, rows, key, clear = bundle(child_dir, monkeypatch)
    # Both pools use the same pinned build; only the correlation session changes.
    metadata.update(session='b9'*32, native_root=parent['native_root'],
                    native_binary=parent['native_binary'], native_policy=parent['native_policy'],
                    native_policy_sha256=parent['native_policy_sha256'],
                    native_binary_sha256=parent['native_binary_sha256'])
    records, total, counts = native._layout(metadata['request'], metadata['session'])
    metadata.update(records=records, share_count=total, clear_counts=counts,
                    native_key_cache_reused=True, key_epoch_root=str(parent_dir),
                    key_epoch_session=parent['session'], parent_session=parent['session'],
                    parent_native_manifest=str(parent_dir/'public-manifest.json'),
                    parent_native_manifest_sha256=native._sha(parent_dir/'public-manifest.json'),
                    parent_online_report=str(report_path), parent_online_report_sha256=native._sha(report_path))
    (role/'native-policy.json').write_bytes((parent_role/'native-policy.json').read_bytes())
    (parent_role/'key-cache-advanced').write_text(json.dumps(dict(successor_session=metadata['session'],destination=str(role))))
    for path in (role/'public-manifest.json', child_dir/'public-manifest.json'):
        path.write_text(json.dumps(metadata))
    monkeypatch.setattr(native, '_verify_policy', lambda root,binary:(parent['native_policy'],parent['native_policy_sha256']))
    return role, metadata, parent_role, report_path


def test_warm_lineage_reads_only_own_handoff_and_returns_new_correlations(tmp_path, monkeypatch):
    role, metadata, parent_role, _ = warm_bundle(tmp_path, monkeypatch)
    original = native._read_local
    reads=[]
    def own_only(path, directory, filename, **kwargs):
        assert path in (role, parent_role) and path.name == 'party1'
        reads.append(filename)
        return original(path,directory,filename,**kwargs)
    monkeypatch.setattr(native,'_read_local',own_only)
    loaded=native.load_role(role,1)
    assert loaded['provenance']['native_key_cache_reused'] is True
    assert reads[:2] == ['consumed','key-cache-advanced']
    assert (role/'consumed').read_text() == metadata['session']


@pytest.mark.parametrize('fault', ['retired', 'advanced', 'epoch-session', 'epoch-policy',
                                   'root-manifest', 'parent-hash', 'report-hash', 'handoff-session',
                                   'handoff-destination', 'missing-consumed', 'cold-root', 'parent-cycle'])
def test_epoch_guards_fail_before_burn_or_payload_read(tmp_path, monkeypatch, fault):
    role,m,parent,report_path=warm_bundle(tmp_path,monkeypatch)
    epoch=parent.parent
    if fault=='retired': (epoch/'key-epoch-retired').write_text('failed check')
    elif fault=='advanced': (role/'key-cache-advanced').write_text('already advanced')
    elif fault=='epoch-session': m['key_epoch_session']='cc'*32
    elif fault=='epoch-policy':
        first=json.loads((epoch/'public-manifest.json').read_text());first['native_policy_sha256']='00'*32
        (epoch/'public-manifest.json').write_text(json.dumps(first))
    elif fault=='root-manifest':
        changed=dict(m, session='cc'*32);(role.parent/'public-manifest.json').write_text(json.dumps(changed))
    elif fault=='parent-hash': m['parent_native_manifest_sha256']='00'*32
    elif fault=='report-hash': m['parent_online_report_sha256']='00'*32
    elif fault.startswith('handoff'):
        handoff=json.loads((parent/'key-cache-advanced').read_text())
        handoff['successor_session' if fault.endswith('session') else 'destination']='wrong'
        (parent/'key-cache-advanced').write_text(json.dumps(handoff))
    elif fault=='missing-consumed': (parent/'consumed').unlink()
    elif fault=='cold-root': m['native_key_cache_reused']=False
    else:
        m['parent_native_manifest']=str(role.parent/'public-manifest.json')
    (role/'public-manifest.json').write_text(json.dumps(m))
    if fault!='root-manifest': (role.parent/'public-manifest.json').write_text(json.dumps(m))
    def forbidden(path,directory,filename,**kwargs):
        if directory!='.': raise AssertionError('guard must precede private payload reads')
        return original(path,directory,filename,**kwargs)
    original=native._read_local;monkeypatch.setattr(native,'_read_local',forbidden)
    with pytest.raises((ValueError,OSError)): native.load_role(role,1)
    assert not (role/'consumed').exists()
