"""Independent serialization checks for the experimental LowGear field."""
import hashlib,json,struct
from pathlib import Path
import pytest
from rep4gpu import preprocessing_lowgear as module

P=57896044618658097711785492504343953926634992332820282019728792003956629831681
R=pow(2,256,P)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_fixture(tmp_path,monkeypatch,party=0):
    role=tmp_path/f'party{party}';(role/'Persistence').mkdir(parents=True);(role/'Player-Data').mkdir()
    request=dict(matrices=[dict(name='first',m=1,k=2,n=1)],squares=[dict(name='sq',shape=[1,2])],inputs=[dict(name='x',owner=0,shape=[1,3])])
    session='26'*32;records,count,clear_counts=module._layout(request,session);key=123456789
    spec=b'SPDZ gfp'+b'\x00'+struct.pack('<I',32)+P.to_bytes(32,'big')+struct.pack('<I',1)
    word=lambda value:((value*R)%P).to_bytes(32,'little')
    header=spec+word(key);rows=[(int(party==0),key)]
    rows += [((P-1-i)%P,(2**190+31*i)%P)for i in range(count-1)]
    raw=struct.pack('<Q',len(header))+header+b''.join(word(v)+word(g)for v,g in rows)
    (role/'Persistence'/f'Transactions-P{party}.data').write_bytes(raw)
    clear=[P-1,2**254+2**63+19,2**130+7]if party==0 else []
    body=b''.join(((value>>(64*limb))&((1<<64)-1)).to_bytes(8,'little')for limb in range(4)for value in clear)
    (role/'Player-Data'/f'Binary-Output-P{party}-0').write_bytes(body)
    binary=tmp_path/'lowgear-party.x';binary.write_bytes(b'isolated-test-binary')
    policy=dict(covert=False,binary_sha256=digest(binary));policy_file=role/'native-policy.json';policy_file.write_text(json.dumps(policy))
    m=dict(format=module._FORMAT,ready=True,return_codes=[0]*4,prime=P,mac_keys=1,security_parameter=128,request=request,session=session,
           records=records,share_count=count,clear_counts=clear_counts,native_policy=policy,native_policy_sha256=digest(policy_file),
           native_binary=str(binary),native_binary_sha256=digest(binary),export_source_sha256=hashlib.sha256(module._program(request).encode()).hexdigest())
    (role/'public-manifest.json').write_text(json.dumps(m))
    monkeypatch.setattr(module,'_verify_policy',lambda *args:(policy,digest(policy_file)))
    return role,m,rows,key,clear


@pytest.mark.parametrize('party',range(4))
def test_native_field_and_owner_clear_are_exact_and_single_use(tmp_path,monkeypatch,party):
    role,m,rows,key,clear=role_fixture(tmp_path,monkeypatch,party)
    parsed,alpha=module._native_file(role/'Persistence'/f'Transactions-P{party}.data',len(rows))
    assert parsed==rows and alpha==key
    material=module.load_role(role,party)
    assert material['alpha_share']==key
    first=material['matrix_triples']['first']
    assert first.a.values==[v for v,g in rows[1:3]]
    assert first.a.macs==[g for v,g in rows[1:3]]
    assert material['input_masks']['x'].clear==(clear if party==0 else None)
    assert len(material['session'])==32
    with pytest.raises(FileExistsError):module.load_role(role,party)


def test_native_field_failure_still_retires_bundle(tmp_path,monkeypatch):
    role,m,rows,key,clear=role_fixture(tmp_path,monkeypatch)
    path=role/'Persistence/Transactions-P0.data';raw=bytearray(path.read_bytes());raw[-32:]=P.to_bytes(32,'little');path.write_bytes(raw)
    with pytest.raises(ValueError,match='noncanonical'):module.load_role(role,0)
    assert(role/'consumed').exists()


@pytest.mark.parametrize('field,value',[('security_parameter',40),('format','lowgear-s40-native-field-experiment-v1'),('prime',2**256-189),('mac_keys',2)])
def test_s128_loader_rejects_other_security_and_field_formats(tmp_path,monkeypatch,field,value):
    role,m,*_=role_fixture(tmp_path,monkeypatch);m[field]=value;(role/'public-manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):module.load_role(role,0)
    assert not(role/'consumed').exists()


def test_clear_limb_export_is_unsigned_and_square_blocks_are_separate():
    source=module._program(dict(squares=[dict(name='sq',shape=[1,3])],inputs=[dict(name='x',owner=0,shape=[1,3])]))
    assert source.count('to_regint(n_bits=0,sync=False)')==4
    assert 'sint.write_to_file(sq0)\nsint.write_to_file(sqc0)'in source
