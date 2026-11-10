"""Safety and private dependency ownership in the native build helper."""
import hashlib
import json

import pytest

from scripts import build_mama_secure as build


def test_existing_build_is_not_modified(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    (source/'one.h').write_text('source')
    patch = tmp_path/'reviewed.patch'
    patch.write_text('patch')
    lock = tmp_path/'lock.json'
    lock.write_text(json.dumps(dict(source_files_sha256={'one.h': build.digest(source/'one.h')},
                                    patch_sha256=build.digest(patch))))
    monkeypatch.setattr(build, 'LOCK_PATH', lock)
    monkeypatch.setattr(build, 'PATCH_PATH', patch)
    existing = tmp_path/'already-built'
    existing.mkdir()
    sentinel = existing/'do-not-replace'
    sentinel.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='already exists'):
        build.main(['--source', str(source), '--sodium-prefix', str(tmp_path/'unused'),
                    '--build-dir', str(existing)])
    assert sentinel.read_bytes() == b'preserve'
    assert list(existing.iterdir()) == [sentinel]


def test_sodium_symlinks_become_private_files_with_versioned_soname(tmp_path):
    prefix = tmp_path/'prefix'
    (prefix/'include/sodium').mkdir(parents=True)
    (prefix/'lib').mkdir()
    (prefix/'include/sodium.h').write_text('header')
    (prefix/'include/sodium/version.h').write_text('#define SODIUM_VERSION_STRING "1.0.20"')
    library = prefix/'lib/libsodium.so.26'
    library.write_bytes(b'library')
    (prefix/'lib/libsodium.so').symlink_to(library)
    target = tmp_path/'private'
    hashes = build.copy_sodium(prefix, target, {'sodium_version':'1.0.20'})
    library.unlink()
    for name in ('libsodium.so', 'libsodium.so.26'):
        copied = target/'lib'/name
        assert copied.read_bytes() == b'library'
        assert not copied.is_symlink()
        assert hashes[name] == hashlib.sha256(b'library').hexdigest()


def test_failed_native_smoke_cannot_create_functional_evidence(tmp_path):
    directory = tmp_path/'build'
    directory.mkdir()
    (directory/'mama-party.x').write_bytes(b'binary')
    policy = directory/'native-preprocessing-policy.json'
    policy.write_bytes(b'policy')
    report = tmp_path/'public-report.json'
    report.write_text(json.dumps(dict(ready=True, passed=True, return_codes=[0, 1, 0, 0],
                                     prime=2147483647, mac_keys=10, security_parameter=128,
                                     native_binary_sha256=build.digest(directory/'mama-party.x'),
                                     native_policy_sha256=build.digest(policy))))
    with pytest.raises(ValueError, match='roles or security parameters'):
        build.record_functional_smoke(directory, report)
    assert not (directory/'build-functional-smoke.json').exists()
    assert policy.read_bytes() == b'policy'


def test_ambient_make_flags_cannot_enable_insecure_build(monkeypatch):
    for name in ('SECURE', 'MY_CFLAGS', 'CFLAGS', 'MAKEFLAGS', 'MAKEOVERRIDES', 'MAKEFILES'):
        monkeypatch.setenv(name, '-DINSECURE')
    monkeypatch.setenv('MAMA_TEST_PRESERVE', 'keep')
    environment = build.clean_build_environment()
    assert not any(name in environment for name in
                   ('SECURE', 'MY_CFLAGS', 'CFLAGS', 'MAKEFLAGS', 'MAKEOVERRIDES', 'MAKEFILES'))
    assert environment['MAMA_TEST_PRESERVE'] == 'keep'
