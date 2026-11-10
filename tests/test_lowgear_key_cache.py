import json,os
from pathlib import Path
import pytest
from rep4gpu import lowgear_key_cache as cache
from scripts import build_lowgear_goldilocks as build

@pytest.fixture
def state(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();native=tmp_path/'native';native.mkdir();policy={'test_only':True};pp=native/'lowgear-goldilocks-s40-policy.json';pp.write_text(json.dumps(policy));monkeypatch.setattr(build,'verify_policy',lambda root:policy)
    m=dict(format=cache.FORMAT,ready=True,passed=True,return_codes=[0]*4,prime=cache.PRIME,finite_bit_probe=False,session='11'*32,native_root=str(native),native_policy=policy,native_policy_sha256=cache._sha(pp))
    (old/'public-manifest.json').write_text(json.dumps(m))
    for p in range(4):
        role=old/f'party{p}';(role/'Player-Data').mkdir(parents=True);(role/'public-manifest.json').write_text(json.dumps(m));(role/'consumed').write_text(m['session'])
        for name in [f'PairwiseParams-gfp-64-40-40-128-1-{cache.PRIME}-1-P{p}-4',f'LowGear-Secrets-40-fixture-P{p}-4']:(role/'Player-Data'/name).write_bytes(f'only role{p}'.encode())
        (role/'Player-Data'/f'P{p}.key').write_text('TLS must not copy')
        (role/'Persistence').mkdir();(role/'Persistence'/f'Transactions-P{p}.data').write_text('correlations must not copy')
    report=tmp_path/'success.json';report.write_text(json.dumps(dict(passed=True,return_codes=[0]*4,source_stable_during_run=True,exact_integer_match=True,public_preprocessing_manifest=m)))
    dest=tmp_path/'new'/'party0';(dest/'Player-Data').mkdir(parents=True)
    return old,dest,report

def test_own_role_only_and_single_advance(state):
    old,dest,report=state;r=cache.carry_role(old,dest,0,'22'*32,report)
    files=list((dest/'Player-Data').iterdir());assert r['cache_files']==2 and len(files)==2
    assert all(p.read_bytes()==b'only role0' and p.stat().st_mode&0o777==0o600 for p in files)
    assert not (dest/'Persistence').exists()
    with pytest.raises(FileExistsError):cache.carry_role(old,dest,0,'33'*32,report)
    assert (old/'key-epoch-retired').exists()

@pytest.mark.parametrize('field,value',[('passed',False),('return_codes',[0,0,0,1]),('source_stable_during_run',False),('exact_integer_match',False)])
def test_failed_online_parent_rejected(state,field,value):
    old,dest,report=state;r=json.loads(report.read_text());r[field]=value;report.write_text(json.dumps(r))
    with pytest.raises(ValueError):cache.validate_parent(old,report)
    assert not list((dest/'Player-Data').iterdir())

def test_retirement_blocks_future_pool(state):
    old,dest,report=state;cache.retire_epoch(old/'party0','test abort')
    with pytest.raises(ValueError,match='retired'):cache.validate_parent(old,report)

def test_symlink_cache_rejected_and_epoch_retired(state,tmp_path):
    old,dest,report=state;p=next((old/'party0/Player-Data').glob('LowGear-Secrets*'));p.unlink();target=tmp_path/'external';target.write_text('outside');p.symlink_to(target)
    with pytest.raises(OSError):cache.carry_role(old,dest,0,'22'*32,report)
    assert (old/'key-epoch-retired').exists()

def test_missing_cache_retires_epoch(state):
    old,dest,report=state;next((old/'party0/Player-Data').glob('LowGear-Secrets*')).unlink()
    with pytest.raises(ValueError):cache.carry_role(old,dest,0,'22'*32,report)
    assert (old/'key-epoch-retired').exists()

def test_unconsumed_parent_rejected(state):
    old,dest,report=state;(old/'party2/consumed').unlink()
    with pytest.raises(ValueError,match='consumed'):cache.validate_parent(old,report)

def test_wrong_consumed_session_rejected(state):
    old,dest,report=state;(old/'party0/consumed').write_text('33'*32)
    with pytest.raises(ValueError,match='consumed'):cache.validate_parent(old,report)

def test_standalone_helper_avoids_torch_import():
    import subprocess,sys
    script=Path(cache.__file__).resolve()
    code='import runpy,sys;runpy.run_path('+repr(str(script))+',run_name="stdlib_probe");assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)
