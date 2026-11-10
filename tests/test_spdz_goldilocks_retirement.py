"""A failed online run retires the original key epoch before continuation."""
import json
import pytest
from scripts import run_spdz_prime256 as runner


def lineage(tmp_path):
    root=tmp_path/'epoch';pool=tmp_path/'next-pool'
    root.mkdir();pool.mkdir()
    manifest={'key_epoch_root':str(root)}
    (pool/'public-manifest.json').write_text(json.dumps(manifest))
    role=pool/'party0';role.mkdir()
    (role/'public-manifest.json').write_text(json.dumps(manifest))
    return root,pool


@pytest.mark.parametrize('failure',['exception','report'])
def test_coordinator_retires_ancestor_on_any_failed_result(tmp_path,monkeypatch,failure):
    root,pool=lineage(tmp_path)
    def fail(*args):
        if failure=='exception':raise RuntimeError('injected coordinator failure')
        return {'passed':False}
    monkeypatch.setattr(runner,'_run',fail)
    if failure=='exception':
        with pytest.raises(RuntimeError,match='injected'):
            runner.run(pool,tmp_path/'result.json',preprocessing_backend='lowgear-goldilocks-s40')
    else:
        assert not runner.run(pool,tmp_path/'result.json',preprocessing_backend='lowgear-goldilocks-s40')['passed']
    assert (root/'key-epoch-retired').is_file()


def test_success_does_not_retire_epoch(tmp_path,monkeypatch):
    root,pool=lineage(tmp_path)
    monkeypatch.setattr(runner,'_run',lambda *args:{'passed':True})
    assert runner.run(pool,tmp_path/'result.json',preprocessing_backend='lowgear-goldilocks-s40')['passed']
    assert not (root/'key-epoch-retired').exists()


def test_worker_loader_failure_retires_epoch(tmp_path,monkeypatch):
    root,pool=lineage(tmp_path)
    def bad_loader(*args):raise ValueError('injected native loader failure')
    monkeypatch.setattr(runner,'native_backend',lambda name:(bad_loader,2**64-2**32+1))
    config=tmp_path/'worker.json'
    config.write_text(json.dumps({'party':0,'bundle':str(pool),'preprocessing_backend':'lowgear-goldilocks-s40'}))
    with pytest.raises(ValueError,match='injected'):
        runner.worker(config,tmp_path/'result.json')
    assert (root/'key-epoch-retired').is_file()
    assert json.loads((tmp_path/'result.json').read_text())['passed'] is False
