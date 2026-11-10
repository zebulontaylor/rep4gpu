import json
from pathlib import Path
import pytest
from scripts import build_lowgear_goldilocks as build

@pytest.fixture
def tree(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();patch=tmp_path/'patch';patch.write_text('fixture patch\n')
    for name,data in {'CONFIG.mine':'MOD = -DMAX_MOD_SZ=6\n','field.h':'field source','lowgear-party.x':'native binary','libSPDZ.so':'core','libFHE.so':'HE','ot.a':'OT'}.items():(root/name).write_text(data)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(dict(source_sha256={n:build.digest(root/n)for n in ('CONFIG.mine','field.h')},config=(root/'CONFIG.mine').read_text(),patch_sha256=build.digest(patch),dependency_archives_sha256={'ot.a':build.digest(root/'ot.a')})))
    monkeypatch.setattr(build,'LOCK',lock);monkeypatch.setattr(build,'PATCH',patch)
    build.write_policy(root);return root

def test_exact_policy(tree):
    p=build.verify_policy(tree)
    assert p['statistical_security']==40 and p['computational_security']==128
    assert p['max_mod_sz']==6 and p['finite_bit_probe_default'] is False

@pytest.mark.parametrize('name',['CONFIG.mine','field.h','lowgear-party.x','libSPDZ.so','libFHE.so','ot.a'])
def test_changed_component_rejected(tree,name):
    (tree/name).write_text('changed')
    with pytest.raises(ValueError):build.verify_policy(tree)

@pytest.mark.parametrize('field,value',[('statistical_security',128),('max_mod_sz',16),('finite_bit_probe_default',True),('finite_bit_probe_default',0),('covert',True),('covert',0),('field_bits',256),('prime',2**256-189)])
def test_changed_parameters_rejected(tree,field,value):
    path=tree/'lowgear-goldilocks-s40-policy.json';p=json.loads(path.read_text());p[field]=value;path.write_text(json.dumps(p))
    with pytest.raises(ValueError):build.verify_policy(tree)
