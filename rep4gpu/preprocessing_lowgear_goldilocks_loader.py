"""Load one role's single-use Goldilocks LowGear diagnostic correlations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from .preprocessing_mama import _layout, _sha
from .preprocessing_mascot_direct import QShare, QTriple, QSquare, QInputMask
from .preprocessing_lowgear_goldilocks import _program, _verify_policy

PRIME = 2**64 - 2**32 + 1
MAC_KEYS = 1
_FORMAT = 'lowgear-goldilocks-s40-native-field-v1'
_SPEC = b'SPDZ gfp\x00' + struct.pack('<I', 8) + PRIME.to_bytes(8, 'big') + struct.pack('<I', 1)
_R = 2**32 - 1
_INV_R = pow(_R, -1, PRIME)


def _decode_native(raw: bytes, count: int):
    if type(count) is not int or count < 1:
        raise ValueError('native share count must be positive')
    if len(raw) < 8:
        raise ValueError('truncated Goldilocks header')
    size = struct.unpack('<Q', raw[:8])[0]
    if size != len(_SPEC) + 8:
        raise ValueError('Goldilocks header length mismatch')
    if raw[8:8 + len(_SPEC)] != _SPEC:
        raise ValueError('Goldilocks native specification mismatch')
    if len(raw) != 8 + size + count * 16:
        raise ValueError('Goldilocks native file length mismatch')

    def decode(word):
        value = int.from_bytes(word, 'little')
        if value >= PRIME:
            raise ValueError('noncanonical Goldilocks Montgomery value')
        return value * _INV_R % PRIME

    key = decode(raw[8 + len(_SPEC):8 + size])
    body = raw[8 + size:]
    rows = [(decode(body[i:i + 8]), decode(body[i + 8:i + 16]))
            for i in range(0, len(body), 16)]
    return rows, key


def _native_file(path, count):
    return _decode_native(Path(path).read_bytes(), count)


def _decode_owner_clear(raw: bytes, counts):
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError('invalid owner mask count')
    if len(raw) != 8 * sum(counts):
        raise ValueError('Goldilocks owner mask length mismatch')
    values = [int.from_bytes(raw[i:i + 8], 'little') for i in range(0, len(raw), 8)]
    if any(value >= PRIME for value in values):
        raise ValueError('noncanonical Goldilocks owner mask')
    return values


def _read_local(role: Path, directory: str, filename: str, *, optional=False):
    """Open private data relative to the assigned role without following links."""
    role_fd = os.open(role, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=role_fd)
        try:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            except FileNotFoundError:
                if optional:
                    return b''
                raise
            with os.fdopen(fd, 'rb') as payload:
                if not stat.S_ISREG(os.fstat(payload.fileno()).st_mode):
                    raise ValueError('private native payload must be a regular file')
                return payload.read()
        finally:
            os.close(directory_fd)
    finally:
        os.close(role_fd)


def load_role(bundle_path, party, device='cpu'):
    """Verify public provenance, claim the whole role bundle, then read its data.

    Failed payload parsing or calibration leaves the consume marker in place.
    Only the assigned role's persistence file and clear-mask file are opened.
    """
    if str(device) != 'cpu':
        raise ValueError('Goldilocks native shares support CPU big integers only')
    role = Path(bundle_path).resolve()
    if type(party) is not int or party not in range(4) or role.name != f'party{party}':
        raise ValueError('assigned Goldilocks role mismatch')
    m = json.loads((role / 'public-manifest.json').read_text())
    codes = m.get('return_codes')
    if (m.get('format') != _FORMAT or m.get('ready') is not True or
            type(codes) is not list or len(codes) != 4 or
            any(type(code) is not int or code != 0 for code in codes) or
            type(m.get('parties')) is not int or m['parties'] != 4):
        raise ValueError('Goldilocks export incomplete')
    expected = dict(prime=PRIME, mac_keys=1, security_parameter=40,
                    statistical_security=40, computational_security=128,
                    finite_bit_probe=False, field_conversion=False,
                    experimental=True, security_target_met=False)
    if any(type(m.get(key)) is not type(value) or m[key] != value
           for key, value in expected.items()):
        raise ValueError('Goldilocks diagnostic parameters mismatch')
    try:
        session = bytes.fromhex(m['session'])
    except (ValueError, TypeError) as error:
        raise ValueError('invalid Goldilocks session') from error
    if len(session) != 32:
        raise ValueError('invalid Goldilocks session length')
    policy_path = role / 'native-policy.json'
    policy = json.loads(policy_path.read_text())
    if policy != m['native_policy'] or _sha(policy_path) != m['native_policy_sha256']:
        raise ValueError('Goldilocks role policy mismatch')
    native_root = Path(m['native_root']).resolve()
    binary = Path(m['native_binary']).resolve()
    if binary != native_root / 'lowgear-party.x':
        raise ValueError('Goldilocks executable path mismatch')
    verified, digest = _verify_policy(native_root, binary)
    if (verified != policy or digest != m['native_policy_sha256'] or
            _sha(binary) != m['native_binary_sha256']):
        raise ValueError('Goldilocks build provenance mismatch')
    if hashlib.sha256(_program(m['request']).encode()).hexdigest() != m['export_source_sha256']:
        raise ValueError('Goldilocks export program mismatch')
    records, total, counts = _layout(m['request'], m['session'])
    if (records != m['records'] or type(m['share_count']) is not int or total != m['share_count'] or
            type(m['clear_counts']) is not list or
            any(type(count) is not int for count in m['clear_counts']) or counts != m['clear_counts']):
        raise ValueError('Goldilocks native layout mismatch')

    fd = os.open(role / 'consumed', os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as claim:
        claim.write(m['session'])
        claim.flush()
        os.fsync(claim.fileno())
    fd = os.open(role, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    raw = _read_local(role, 'Persistence', f'Transactions-P{party}.data')
    rows, key = _decode_native(raw, total)
    if rows[0] != (int(party == 0), key):
        raise ValueError('Goldilocks constant-one calibration failed')
    owner_counts = [r['count'] for r in records if r['kind'] == 'inputs' and r['owner'] == party]
    clear = _decode_owner_clear(_read_local(role, 'Player-Data', f'Binary-Output-P{party}-0',
                                          optional=True), owner_counts)
    matrices, squares, inputs = {}, {}, {}
    for record in records[1:]:
        local = rows[record['offset']:record['offset'] + record['count']]
        share = QShare([value for value, _ in local], [mac for _, mac in local], tuple(record['shape']))
        target = matrices if record['kind'] == 'matrices' else squares if record['kind'] == 'squares' else inputs
        target.setdefault(record['name'], {})[record['component']] = share
    for kind, target in [('matrices', matrices), ('squares', squares)]:
        for name, parts in list(target.items()):
            record = next(r for r in records if r['kind'] == kind and r['name'] == name)
            identifier = bytes.fromhex(record['identifier'])
            target[name] = (QTriple(parts['a'], parts['b'], parts['c'], identifier)
                            if kind == 'matrices' else QSquare(parts['a'], parts['a_squared'], identifier))
    for record in records:
        if record['kind'] == 'inputs':
            plain = (clear[record['clear_offset']:record['clear_offset'] + record['count']]
                     if record['owner'] == party else None)
            inputs[record['name']] = QInputMask(inputs[record['name']]['share'], plain,
                                                record['owner'], bytes.fromhex(record['identifier']))
    return dict(alpha_share=key, matrix_triples=matrices, square_masks=squares,
                input_masks=inputs,
                session=hashlib.sha256(b'LowGear-Goldilocks-S40-native-v1\0' + session).digest(),
                provenance=m)
