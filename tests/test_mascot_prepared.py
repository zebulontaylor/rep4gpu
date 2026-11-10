"""Native fixed-right source, private provisioning, and role-format oracles."""
import hashlib
import json
import struct
import pytest
from rep4gpu.preprocessing_mascot_prepared import program,write_owner_inputs,load_role,FORMAT
from rep4gpu.preprocessing_mascot_direct import PRIME,_SPEC

SESSION='17'*32
REQUEST=dict(models=[dict(name='first',owner=1,k=2,n=3,rows=4,model_id='42'*32)],
             static_inputs=[dict(name='bias',owner=1,shape=[1,3])],
             squares=[dict(name='hidden',shape=[4,3])],inputs=[dict(name='query',owner=0,shape=[4,2])])


def test_native_source_computes_fixed_product_and_never_exports_model_clear():
    source,records,total,plan,clear=program(REQUEST,SESSION)
    assert 'w0=sint.get_input_from(1,size=6)' in source
    assert 'am0.assign_vector(sint.get_random(size=8))' in source
    assert 'fm0=am0.dot(wm0)' in source
    assert 'to_regint(n_bits=0,sync=False)' in source and 'n_bits=64' not in source
    assert [item['component'] for item in records[:3]]==['weights','a','factor']
    assert [item['count'] for item in records[:3]]==[6,8,12]
    assert [item['name'] for item in plan[1]]==['first','bias']
    assert clear==[8,0,0,0] and total==1+6+8+12+3+24+8
    assert all('model_id' in item for item in records[:3])


def test_owner_provisioning_only_accepts_assigned_canonical_model(tmp_path):
    role=tmp_path/'party1';(role/'Player-Data').mkdir(parents=True)
    values=dict(first=[PRIME-1]*6,bias=[0,1,PRIME-1])
    write_owner_inputs(role,1,REQUEST,SESSION,values)
    data=(role/'Player-Data/Input-P1-0').read_text().splitlines()
    assert len(data)==2 and list(map(int,data[0].split()))==values['first']
    with pytest.raises(FileExistsError):write_owner_inputs(role,1,REQUEST,SESSION,values)
    with pytest.raises(ValueError,match='ownership'):write_owner_inputs(role,1,REQUEST,SESSION,dict(query=[0]*8))


def _bundle(tmp_path,monkeypatch,party=1):
    from rep4gpu import preprocessing_mascot_prepared as module
    role=tmp_path/f'party{party}';(role/'Persistence').mkdir(parents=True);(role/'Player-Data').mkdir()
    source,records,total,plan,clear_counts=program(REQUEST,SESSION)
    binary=tmp_path/'mascot-party.x';binary.write_bytes(b'fake-test-executable')
    policy=dict(post_authentication_sacrifice_coin=True,test_only=True)
    policyfile=role/'native-policy.json';policyfile.write_text(json.dumps(policy))
    digest=hashlib.sha256(policyfile.read_bytes()).hexdigest()
    monkeypatch.setattr(module,'_verify_policy',lambda root,binary:(policy,digest))
    manifest=dict(format=FORMAT,ready=True,return_codes=[0]*4,prime=PRIME,mac_keys=1,security_parameter=128,
                  request=REQUEST,session=SESSION,records=records,share_count=total,owner_input_plan=plan,clear_counts=clear_counts,
                  export_source_sha256=hashlib.sha256(source.encode()).hexdigest(),native_binary=str(binary),
                  native_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),native_policy=policy,native_policy_sha256=digest)
    (role/'public-manifest.json').write_text(json.dumps(manifest))
    key=PRIME-19
    def encoded(value):return (value*189%PRIME).to_bytes(32,'little')
    header=_SPEC+encoded(key)
    body=encoded(int(party==0))+encoded(key)
    for i in range(1,total):body+=encoded(i)+encoded((i+7)%PRIME)
    (role/'Persistence'/f'Transactions-P{party}.data').write_bytes(struct.pack('<Q',len(header))+header+body)
    if party==0:
        # Independent little-endian, limb-major public test masks.
        values=[PRIME-1-i for i in range(8)]
        raw=b''.join(((value>>(64*limb))&((1<<64)-1)).to_bytes(8,'little') for limb in range(4) for value in values)
        (role/'Player-Data/Binary-Output-P0-0').write_bytes(raw)
    return role,manifest


@pytest.mark.parametrize('party',[0,1])
def test_role_import_uses_only_local_native_shares_and_claims_once(tmp_path,monkeypatch,party):
    role,manifest=_bundle(tmp_path,monkeypatch,party)
    loaded=load_role(role,party)
    assert loaded['alpha_share']==PRIME-19
    batch=loaded['prepared_batches']['first']
    assert batch.weights.shape==(2,3) and batch.a.shape==(4,2) and batch.factor.shape==(4,3)
    assert batch.weights.values==list(range(1,7)) and batch.model_id==bytes.fromhex('42'*32)
    mask=loaded['input_masks']['query']
    assert mask.clear==([PRIME-1-i for i in range(8)] if party==0 else None)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):load_role(role,party)


def test_invalid_private_encoding_stays_consumed(tmp_path,monkeypatch):
    role,manifest=_bundle(tmp_path,monkeypatch)
    path=role/'Persistence/Transactions-P1.data';data=bytearray(path.read_bytes());data[-32:]=PRIME.to_bytes(32,'little');path.write_bytes(data)
    with pytest.raises(ValueError,match='noncanonical'):load_role(role,1)
    assert (role/'consumed').exists()


def test_changed_public_layout_fails_before_secret_claim(tmp_path,monkeypatch):
    role,manifest=_bundle(tmp_path,monkeypatch);manifest['share_count']+=1
    (role/'public-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='layout'):load_role(role,1)
    assert not (role/'consumed').exists()
