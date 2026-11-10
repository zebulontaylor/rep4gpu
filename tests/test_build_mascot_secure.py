"""Fail-closed provenance checks for the reviewed native backend."""
import json

import pytest

from scripts import build_mascot_secure as build


@pytest.fixture
def pinned_build(tmp_path, monkeypatch):
    root = tmp_path/'native'
    root.mkdir()
    source = root/'OT/selected.hpp'
    source.parent.mkdir()
    source.write_text('fresh joint coin after authentication')
    (root/'CONFIG.mine').write_text('USE_KOS = 0\n')
    for name in ('mascot-party.x', 'libSPDZ.so', 'local/lib/libsodium.so',
                 'local/lib/liblibOTe.a'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    patch = tmp_path/'amplify.patch'
    patch.write_text('amplification four')
    fresh = tmp_path/'fresh.patch'
    fresh.write_text('fresh challenge')
    lock = tmp_path/'lock.json'
    lock.write_text(json.dumps(dict(
        source_files_sha256={'OT/selected.hpp': build.digest(source)},
        config=(root/'CONFIG.mine').read_text(), internal_prime=str(2**256-189),
        dependency_archives_sha256={'local/lib/liblibOTe.a': build.digest(root/'local/lib/liblibOTe.a')},
        patch_sha256=build.digest(patch), fresh_sacrifice_patch_sha256=build.digest(fresh))))
    monkeypatch.setattr(build, 'LOCK_PATH', lock)
    monkeypatch.setattr(build, 'PATCH_PATH', patch)
    monkeypatch.setattr(build, 'FRESH_PATCH_PATH', fresh)
    build.write_policy(root)
    return root


def test_policy_requires_post_authentication_coin(pinned_build):
    path = pinned_build/'mascot256-preprocessing-policy.json'
    policy = json.loads(path.read_text())
    del policy['post_authentication_sacrifice_coin']
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='parameters differ'):
        build.verify_policy(pinned_build)


@pytest.mark.parametrize('name', ['mascot-party.x', 'libSPDZ.so',
                                 'local/lib/liblibOTe.a', 'OT/selected.hpp'])
def test_changed_native_artifact_rejected(pinned_build, name):
    with (pinned_build/name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='differs'):
        build.verify_policy(pinned_build)


def test_pinned_artifacts_verify_without_rewriting_policy(pinned_build):
    path = pinned_build/'mascot256-preprocessing-policy.json'
    before = path.read_bytes()
    assert build.verify_policy(pinned_build)['post_authentication_sacrifice_coin']
    assert path.read_bytes() == before
