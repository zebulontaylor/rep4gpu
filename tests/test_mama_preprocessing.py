"""Synthetic native-format checks and an opt-in real MAMA/TLS integration."""
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from rep4gpu.preprocessing_mama import _layout,_native_file,load_role,run_native

PRIME=2147483647
REPOSITORY=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_role(root):
    """Encode disposable synthetic words independently from the decoder."""
    role=root/'party0'
    (role/'Persistence').mkdir(parents=True)
    (role/'Player-Data').mkdir()
    request={'matrices':[{'name':'first','m':2,'k':3,'n':2}],
             'squares':[{'name':'hidden','shape':[2,2]}],
             'inputs':[{'name':'x','owner':0,'shape':[2,3]}]}
    session='13'*32
    records,count,clear_counts=_layout(request,session)
    keys=np.arange(1,11,dtype=np.int64)
    values=np.arange(count*11,dtype=np.int64).reshape(count,11)%PRIME
    values[0,0]=1
    values[0,1:]=keys
    values[-1]=PRIME-1
    # File metadata uses GMP's big-endian bigint encoding; stored field
    # elements and the outer stream length are little-endian native words.
    header=(b'Mama10 gfp'+b'\x00'+struct.pack('<I',4)+PRIME.to_bytes(4,'big')
            +struct.pack('<I',1)+((keys*4)%PRIME).astype('<u8').tobytes())
    raw=(struct.pack('<Q',len(header))+header+((values*4)%PRIME).astype('<u8').tobytes())
    (role/'Persistence/Transactions-P0.data').write_bytes(raw)
    clear=np.arange(6,dtype=np.int64)
    (role/'Player-Data/Binary-Output-P0-0').write_bytes(clear.astype('<i8').tobytes())
    policy=dict(format='mama-prime31-reviewed-build-v1',prime=PRIME,mac_keys=10,
                security_parameter=128,ot='SoftSpokenMalOt',binary_sha256='synthetic-unit-fixture',
                source_lock_sha256=sha(REPOSITORY/'native/mama-source-lock.json'),
                reviewed_patch_sha256=sha(REPOSITORY/'native/patches/mama-independent-sacrifice.patch'),
                build_properties=dict(LONG_BIT_MAC_KEY=True,USE_KOS=0,USE_LIBOTE2=0,INSECURE=False))
    (role/'native-policy.json').write_text(json.dumps(policy))
    manifest=dict(format='mama-prime31-role-v1',ready=True,return_codes=[0]*4,parties=4,
                  prime=PRIME,mac_keys=10,security_parameter=128,request=request,session=session,
                  records=records,share_count=count,clear_counts=clear_counts,native_policy=policy,
                  native_policy_sha256=sha(role/'native-policy.json'),
                  native_binary_sha256=policy['binary_sha256'],native_source_sha256={},
                  export_source_sha256='synthetic-unit-fixture',preprocessing_ms=0,native_preprocessing_ms=0)
    (role/'public-manifest.json').write_text(json.dumps(manifest))
    return role,values,keys,clear


def test_native_montgomery_and_key_order(tmp_path):
    role,expected,keys,_=fixture_role(tmp_path)
    values,key=_native_file(role/'Persistence/Transactions-P0.data',len(expected))
    np.testing.assert_array_equal(values,expected)
    np.testing.assert_array_equal(key,keys)


def test_int32_storage_preserves_every_canonical_bit(tmp_path):
    role64,_,_,_=fixture_role(tmp_path/'words64')
    role32,_,_,_=fixture_role(tmp_path/'words32')
    a=load_role(role64,0)
    b=load_role(role32,0,storage_dtype=torch.int32)
    for name in ('a','b','c'):
        left=getattr(a['matrix_triples']['first'],name)
        right=getattr(b['matrix_triples']['first'],name)
        assert right.values.dtype==right.macs.dtype==torch.int32
        assert torch.equal(left.values,right.values.long())
        assert torch.equal(left.macs,right.macs.long())
    assert b['alpha_share'].dtype==b['input_masks']['x'].clear.dtype==torch.int64
    assert b['provenance']['privacy_audit_status']=='failed_small_field_selective_failure'
    assert b['provenance']['security_target_met'] is False and b['provenance']['functional_only'] is True
    assert torch.equal(a['input_masks']['x'].share.macs,b['input_masks']['x'].share.macs.long())
    with pytest.raises(FileExistsError):load_role(role32,0)


def test_canonical_parse_failure_still_retires_bundle(tmp_path):
    role,_,_,_=fixture_role(tmp_path)
    path=role/'Persistence/Transactions-P0.data'
    raw=bytearray(path.read_bytes())
    raw[-8:]=struct.pack('<Q',PRIME)
    path.write_bytes(raw)
    with pytest.raises(ValueError,match='noncanonical'):load_role(role,0)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):load_role(role,0)


def test_policy_change_rejected_without_importing_secrets(tmp_path):
    role,_,_,_=fixture_role(tmp_path)
    (role/'native-policy.json').write_text('{}')
    with pytest.raises(ValueError,match='policy'):load_role(role,0)
    assert not (role/'consumed').exists()


def test_ready_and_assigned_role_are_required(tmp_path):
    role,_,_,_=fixture_role(tmp_path)
    with pytest.raises(ValueError,match='assigned'):load_role(role,1)
    path=role/'public-manifest.json';manifest=json.loads(path.read_text());manifest['ready']=False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='complete'):load_role(role,0)


def test_native_requires_fresh_output_and_finite_timeout(tmp_path):
    with pytest.raises(ValueError,match='timeout'):run_native({},tmp_path/'new',timeout=float('nan'))
    with pytest.raises(FileExistsError):run_native({},tmp_path)


@pytest.mark.parametrize('compile_status',[0,1])
def test_legacy_producer_marks_even_successful_bundles_functional_only(tmp_path,monkeypatch,compile_status):
    from types import SimpleNamespace
    import rep4gpu.preprocessing_mama as module
    native=tmp_path/'native';native.mkdir()
    (native/'mama-party.x').write_bytes(b'unit-test executable placeholder')
    (native/'native-preprocessing-policy.json').write_text('{}')
    monkeypatch.setattr(module,'_verify_policy',lambda *args:({},'unit-test-policy'))
    monkeypatch.setattr(module,'_native_environment',lambda *args:({},{}))
    def run(command,**kwargs):
        if command[0]=='openssl' and command[1]=='req':
            for option in ('-keyout','-out'):
                Path(command[command.index(option)+1]).write_bytes(b'unit-test TLS placeholder')
        return SimpleNamespace(returncode=compile_status if 'compile.py' in ' '.join(command) else 0)
    class Process:
        returncode=0
        def __init__(self,*args,**kwargs):pass
        def poll(self):return self.returncode
        def wait(self,**kwargs):return self.returncode
        def kill(self):self.returncode=-9
    monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    report=run_native({},tmp_path/'output',native_root=native)
    assert report['passed'] is (compile_status==0)
    assert report['privacy_audit_status']=='failed_small_field_selective_failure'
    assert report['security_target_met'] is False and report['functional_only'] is True
    persisted=json.loads((tmp_path/'output/public-manifest.json').read_text())
    assert persisted['privacy_audit_status']==report['privacy_audit_status']
    assert persisted['security_target_met'] is False


def test_real_native_preprocessing_and_four_tls_roles(tmp_path):
    native=os.environ.get('MAMA_NATIVE_ROOT')
    if not native:pytest.skip('set MAMA_NATIVE_ROOT to a reviewed native build for this integration')
    from scripts.check_mama_preprocessing import run_fresh
    report=run_fresh(native,tmp_path/'fresh-native-check')
    assert report['passed'] and report['native_return_codes']==report['checked_role_return_codes']==[0]*4
    assert report['mac_keys']==10 and not report['trusted_dealer']
    assert report['disposable_correlations']


def test_native_loader_environment_is_pinned(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import rep4gpu.preprocessing_mama as module
    (tmp_path/'local/lib').mkdir(parents=True)
    spdz=tmp_path/'libSPDZ.so';spdz.write_bytes(b'synthetic shared library')
    sodium=tmp_path/'local/lib/libsodium.so';sodium.write_bytes(b'synthetic sodium')
    policy={'dependency_sha256':{'libSPDZ.so':sha(spdz),'local/lib/libsodium.so':sha(sodium)}}
    monkeypatch.setenv('LD_PRELOAD','untrusted-preload.so')
    monkeypatch.setenv('LD_AUDIT','untrusted-audit.so')
    monkeypatch.setenv('LD_LIBRARY_PATH','/untrusted/libraries')
    def inspect(command,**kwargs):
        env=kwargs['env']
        assert 'LD_PRELOAD' not in env and 'LD_AUDIT' not in env
        assert env['LD_LIBRARY_PATH']==f'{tmp_path}/local/lib:{tmp_path}'
        return SimpleNamespace(returncode=0,stdout=f'libSPDZ.so => {spdz} (0x1)\nlibsodium.so.26 => {sodium} (0x2)\n')
    monkeypatch.setattr(module.subprocess,'run',inspect)
    _,resolved=module._native_environment(tmp_path,tmp_path/'mama-party.x',policy)
    assert resolved['libSPDZ.so']['sha256']==sha(spdz)


@pytest.mark.parametrize('storage_dtype',[torch.int32,torch.int64])
def test_fixed_online_subset_returns_only_owned_selected_coordinates(tmp_path,storage_dtype):
    role10,_,_,_=fixture_role(tmp_path/'native10')
    role7,_,_,_=fixture_role(tmp_path/'native7')
    full=load_role(role10,0,storage_dtype=storage_dtype)
    selected=load_role(role7,0,storage_dtype=storage_dtype,online_mac_keys=7)
    assert selected['alpha_share'].shape==(7,)
    assert selected['alpha_share'].untyped_storage().nbytes()==7*8
    assert torch.equal(selected['alpha_share'],full['alpha_share'][:7])
    pairs=[(getattr(full['matrix_triples']['first'],name),getattr(selected['matrix_triples']['first'],name)) for name in ('a','b','c')]
    pairs += [(getattr(full['square_masks']['hidden'],name),getattr(selected['square_masks']['hidden'],name)) for name in ('a','a_squared')]
    pairs += [(full['input_masks']['x'].share,selected['input_masks']['x'].share)]
    for a,b in pairs:
        assert torch.equal(a.values,b.values)
        assert torch.equal(a.macs[:7],b.macs)
        assert b.macs.shape[0]==7
        assert b.macs.untyped_storage().nbytes()==7*b.values.numel()*b.macs.element_size()
    assert torch.equal(full['input_masks']['x'].clear,selected['input_masks']['x'].clear)
    assert selected['session']!=full['session'] and selected['session']!=bytes.fromhex('13'*32)
    binding=(b'SPDZ31-MAMA-online-selection-v1\x00'+struct.pack('<Q',32)+bytes.fromhex('13'*32)
             +struct.pack('<II',10,7)+b''.join(struct.pack('<I',i) for i in range(7)))
    assert selected['session']==hashlib.sha256(binding).digest()
    assert selected['provenance']['native_mac_keys']==10
    assert selected['provenance']['online_mac_keys']==7
    assert selected['provenance']['online_mac_indices']==list(range(7))
    assert json.loads((role7/'consumed').read_text())['online_mac_keys']==7
    with pytest.raises(FileExistsError):load_role(role7,0,online_mac_keys=10)


@pytest.mark.parametrize('corruption',['dropped_noncanonical','dropped_calibration','dropped_key'])
def test_unused_native_coordinates_are_checked_before_projection(tmp_path,corruption):
    role,values,_,_=fixture_role(tmp_path)
    path=role/'Persistence/Transactions-P0.data';raw=bytearray(path.read_bytes())
    header_size=struct.unpack('<Q',raw[:8])[0]
    if corruption=='dropped_noncanonical':
        offset=8+header_size+(len(values)-1)*88+10*8
        replacement=PRIME
    elif corruption=='dropped_calibration':
        offset=8+header_size+10*8
        replacement=44
    else:
        offset=8+header_size-8
        replacement=44
    raw[offset:offset+8]=struct.pack('<Q',replacement)
    path.write_bytes(raw)
    with pytest.raises(ValueError,match='noncanonical|calibration'):
        load_role(role,0,online_mac_keys=7)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):load_role(role,0,online_mac_keys=10)


def test_mixed_subset_claim_race_consumes_whole_native_bundle_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    role,_,_,_=fixture_role(tmp_path)
    ready=Barrier(2)
    def claim(keys):
        ready.wait(timeout=2)
        try:return load_role(role,0,online_mac_keys=keys)
        except FileExistsError:return None
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(claim,(7,10)))
    winners=[result for result in results if result is not None]
    assert len(winners)==1
    marker=json.loads((role/'consumed').read_text())
    assert marker['native_mac_keys']==10
    assert marker['online_mac_keys']==winners[0]['alpha_share'].numel()
    for keys in (7,10):
        with pytest.raises(FileExistsError):load_role(role,0,online_mac_keys=keys)


@pytest.mark.parametrize('keys',[0,5,6,8,9,11,7.0,True])
def test_online_projection_choice_is_fixed_and_explicit(tmp_path,keys):
    role,_,_,_=fixture_role(tmp_path)
    with pytest.raises(ValueError,match='seven or ten'):
        load_role(role,0,online_mac_keys=keys)
    assert not (role/'consumed').exists()


def test_different_online_selections_abort_before_a_broadcast_is_accepted(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.spdz_distributed import DistributedParty
    from rep4gpu.transport import IdealNetwork
    role7,_,_,_=fixture_role(tmp_path/'selection7')
    role10,_,_,_=fixture_role(tmp_path/'selection10')
    subset=load_role(role7,0,online_mac_keys=7)
    full=load_role(role10,0,online_mac_keys=10)
    network=IdealNetwork(timeout=2)
    parties=[DistributedParty(p,network.endpoint(p),torch.ones(7,dtype=torch.int64),
                              session=full['session'] if p==1 else subset['session'],
                              _allow_insecure_test_transport=True) for p in range(4)]
    def work(party):
        try:return party.broadcast(b'public-probe',label=b'projection-session-test')
        except BaseException as error:return error
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(work,parties))
    assert all(isinstance(result,BaseException) for result in results)
    assert all(party.poisoned for party in parties)
