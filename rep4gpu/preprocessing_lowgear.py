"""Experimental native-field LowGear export, separate from MASCOT providers.

LowGear selects the malicious HE key-generation path. This module records the
actual build and runtime parameters; it does not certify the HE parameter set.
"""
from pathlib import Path
import hashlib,json,math,os,secrets,shutil,socket,struct,subprocess,sys,time
from .preprocessing_mama import _layout,_sha,_port_base,_native_environment as _base_environment
from .preprocessing_mascot_direct import _program

PRIME=57896044618658097711785492504343953926634992332820282019728792003956629831681
MAC_KEYS=1
_FORMAT='lowgear-native-field-experiment-v1'


def _binary(root):
    binary=Path(root)/'lowgear-party.x'
    if not binary.is_file():raise FileNotFoundError(binary)
    return binary


def _verify_policy(root,binary):
    from scripts.build_lowgear_secure import verify_policy
    verify_policy(root)
    root=Path(root);path=root/'lowgear-preprocessing-policy.json';policy=json.loads(path.read_text())
    expected=dict(format='lowgear-experimental-build-v1',protocol='LowGear',covert=False,
                  prime=PRIME,field_bits=256,statistical_security=128,computational_security=128)
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('LowGear experiment parameters mismatch')
    if _sha(binary)!=policy.get('binary_sha256'):raise ValueError('LowGear binary mismatch')
    for category in ('source_sha256','dependency_sha256'):
        for name,digest in policy[category].items():
            if Path(name).is_absolute()or'..'in Path(name).parts or _sha(root/name)!=digest:raise ValueError('LowGear provenance mismatch')
    return policy,_sha(path)


def _native_environment(root,binary,policy):
    environment,libraries=_base_environment(root,binary,policy)
    selected=libraries.get('libFHE.so')
    if not selected or selected['path']!=str((root/'libFHE.so').resolve()) or selected['sha256']!=policy['dependency_sha256']['libFHE.so']:
        raise ValueError('LowGear did not load the pinned HE library')
    return environment,libraries


def run_native(manifest_path,output_dir,*,native_root='work/lowgear-build',
               timeout=900,batch_size=1000):
    """Generate native q256 role bundles without any integer field conversion.

    ``output_dir`` must not exist. No fake/file preprocessing or insecure flags
    are accepted. Existing native sources and executables are read-only inputs.
    Failed output directories stay unready and their correlations cannot load.
    """
    request=json.loads(Path(manifest_path).read_text()) if not isinstance(manifest_path,dict) else manifest_path
    if not math.isfinite(timeout) or timeout<=0 or type(batch_size) is not int or batch_size<1:raise ValueError('invalid native timeout or batch size')
    session=secrets.token_hex(32)
    records,total,clear_counts=_layout(request,session)
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    native_root=Path(native_root).resolve();binary=_binary(native_root)
    policy,policy_hash=_verify_policy(native_root,binary)
    native_env,loaded_libraries=_native_environment(native_root,binary,policy)
    compiler=output/'compiler'
    for directory in ('Source','Bytecode','Schedules'):(compiler/'Programs'/directory).mkdir(parents=True,exist_ok=True)
    program='native_q_export'
    source=compiler/'Programs/Source'/f'{program}.mpc';source.write_text(_program(request))
    metadata=dict(format=_FORMAT,ready=False,parties=4,prime=PRIME,mac_keys=MAC_KEYS,
                  security_parameter=128,session=session,request=request,records=records,
                  privacy_audit_status='unreviewed_lowgear_parameter_experiment',
                  security_target_met=False,experimental=True,field_conversion=False,
                  share_count=total,clear_counts=clear_counts,
                  native_binary=str(binary),native_binary_sha256=_sha(binary),
                  native_policy=policy,native_policy_sha256=policy_hash,native_loaded_libraries=loaded_libraries,
                  export_source_sha256=_sha(source),native_source_sha256={})
    for name in ('CONFIG','CONFIG.mine','compile.py','Machines/lowgear-party.cpp','Protocols/LowGearKeyGen.hpp','Protocols/ShuffleSacrifice.hpp',
                 'Protocols/MAC_Check.hpp','OT/NPartyTripleGenerator.hpp','OT/OTExtensionWithMatrix.cpp',
                 'OT/OTMultiplier.hpp','OT/MascotMacKey.hpp'):
        path=native_root/name
        if path.is_file():metadata['native_source_sha256'][name]=_sha(path)
    started=time.perf_counter()
    compile_log=compiler/'compile.log'
    with compile_log.open('w') as log:
        compiled=subprocess.run([sys.executable,'-B',str(native_root/'compile.py'),'-F','64',program],
                                cwd=compiler,stdout=log,stderr=subprocess.STDOUT,
                                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=timeout)
    if compiled.returncode:
        metadata.update(return_codes=[],passed=False,compile_return_code=compiled.returncode)
        (output/'public-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return metadata
    compile_ms=(time.perf_counter()-started)*1000
    certs=output/'public-certificates';certs.mkdir()
    roles=[]
    def openssl(*args):
        subprocess.run(['openssl',*map(str,args)],check=True,capture_output=True,timeout=60)
    for p in range(4):
        role=output/f'party{p}';role.mkdir(mode=0o700)
        player=role/'Player-Data';player.mkdir(mode=0o700)
        (role/'Programs').symlink_to(compiler/'Programs',target_is_directory=True)
        openssl('req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:prime256v1','-nodes',
                '-keyout',player/f'P{p}.key','-out',certs/f'P{p}.pem','-days','1','-subj',f'/CN=P{p}')
        (player/f'P{p}.key').chmod(0o600)
        roles.append(role)
    for role in roles:
        for cert in certs.glob('*.pem'):shutil.copy2(cert,role/'Player-Data'/cert.name)
        openssl('rehash',role/'Player-Data')
    base=_port_base();processes=[];logs=[]
    native_started=time.perf_counter()
    try:
        for p,role in enumerate(roles):
            log=(role/'native.log').open('w');logs.append(log)
            command=[str(binary),'-p',str(p),'-N','4','-P',str(PRIME),'-lgp','256',
                     '-S','128','-o','comp_sec=128,verbose_he','-e','-pn',str(base),'-b',str(batch_size),program]
            processes.append(subprocess.Popen(command,cwd=role,stdout=log,stderr=subprocess.STDOUT,
                                              env=native_env))
        deadline=time.monotonic()+timeout
        for process in processes:process.wait(timeout=max(.001,deadline-time.monotonic()))
    except BaseException as error:
        metadata['error']=f'{type(error).__name__}: {error}'
    finally:
        for process in processes:
            if process.poll() is None:process.kill()
        for process in processes:process.wait()
        for log in logs:log.close()
    codes=[process.returncode for process in processes]
    passed=len(codes)==4 and all(code==0 for code in codes)
    metadata.update(ready=passed,passed=passed,return_codes=codes,compile_ms=compile_ms,
                    native_preprocessing_ms=(time.perf_counter()-native_started)*1000,
                    preprocessing_ms=(time.perf_counter()-started)*1000,
                    private_role_directories=[str(role) for role in roles],
                    transport='native encrypted peer channels',trusted_dealer=False)
    encoded=json.dumps(metadata,indent=2)+'\n'
    (output/'public-manifest.json').write_text(encoded)
    for role in roles:
        (role/'public-manifest.json').write_text(encoded)
        (role/'native-policy.json').write_text((native_root/'lowgear-preprocessing-policy.json').read_text())
        # No native secret file is copied between roles or read by this coordinator.
        for private in (role/'Player-Data').glob('Mascot-Secrets-*'):private.chmod(0o600)
        for private in (role/'Persistence').glob('*') if (role/'Persistence').exists() else []:private.chmod(0o600)
        for private in (role/'Player-Data').glob('Binary-Output-*'):private.chmod(0o600)
    return metadata


from .preprocessing_mascot_direct import QShare,QTriple,QSquare,QInputMask
_SPEC=b'SPDZ gfp'+b'\x00'+struct.pack('<I',32)+PRIME.to_bytes(32,'big')+struct.pack('<I',1)
_INV_R=pow(pow(2,256,PRIME),-1,PRIME)


def _native_file(path,count):
    raw=Path(path).read_bytes()
    if len(raw)<8:raise ValueError('truncated native header')
    size=struct.unpack('<Q',raw[:8])[0]
    if size!=len(_SPEC)+32:raise ValueError('native q256 header length mismatch')
    header=raw[8:8+size]
    if header[:len(_SPEC)]!=_SPEC:raise ValueError('native q256 specification mismatch')
    if len(raw)!=8+size+count*64:raise ValueError('native q256 file length mismatch')
    def decode(word):
        value=int.from_bytes(word,'little')
        if value>=PRIME:raise ValueError('noncanonical native Montgomery value')
        return value*_INV_R%PRIME
    key=decode(header[len(_SPEC):]);body=raw[8+size:]
    rows=[(decode(body[i:i+32]),decode(body[i+32:i+64])) for i in range(0,len(body),64)]
    return rows,key


def _decode_owner_clear(raw,counts):
    if len(raw)!=32*sum(counts):raise ValueError('owner q256 mask length mismatch')
    result=[];offset=0
    for count in counts:
        words=[int.from_bytes(raw[offset+i:offset+i+8],'little') for i in range(0,count*32,8)]
        values=[sum(words[limb*count+j]<<(64*limb) for limb in range(4)) for j in range(count)]
        if any(value>=PRIME for value in values):raise ValueError('noncanonical owner q256 mask')
        result.extend(values);offset+=count*32
    return result


def load_role(bundle_path,party,device='cpu'):
    if str(device)!='cpu':raise ValueError('native q256 shares currently support CPU big integers only')
    role=Path(bundle_path).resolve();m=json.loads((role/'public-manifest.json').read_text())
    if m.get('format')!=_FORMAT or not m.get('ready') or m.get('return_codes')!=[0]*4:raise ValueError('native q256 export incomplete')
    if m.get('prime')!=PRIME or m.get('mac_keys')!=1 or m.get('security_parameter')!=128:raise ValueError('native q256 parameters mismatch')
    if party not in range(4) or role.name!=f'party{party}':raise ValueError('assigned native role mismatch')
    policy=json.loads((role/'native-policy.json').read_text())
    if policy!=m['native_policy'] or _sha(role/'native-policy.json')!=m['native_policy_sha256'] or policy.get('covert')is not False:raise ValueError('native q256 policy mismatch')
    binary=Path(m['native_binary']);verified,digest=_verify_policy(binary.parent,binary)
    if verified!=policy or digest!=m['native_policy_sha256'] or _sha(binary)!=m['native_binary_sha256']:
        raise ValueError('native q256 build provenance mismatch')
    if hashlib.sha256(_program(m['request']).encode()).hexdigest()!=m['export_source_sha256']:raise ValueError('native q256 program mismatch')
    fd=os.open(role/'consumed',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w')as f:f.write(m['session']);f.flush();os.fsync(f.fileno())
    fd=os.open(role,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    records,total,counts=_layout(m['request'],m['session'])
    if records!=m['records'] or total!=m['share_count']:raise ValueError('native q256 layout mismatch')
    rows,key=_native_file(role/'Persistence'/f'Transactions-P{party}.data',total)
    if rows[0]!=(int(party==0),key):raise ValueError('native q256 constant calibration failed')
    path=role/'Player-Data'/f'Binary-Output-P{party}-0'
    clear=_decode_owner_clear(path.read_bytes()if path.exists()else b'',[r['count']for r in records if r['kind']=='inputs'and r['owner']==party])
    matrices={};squares={};inputs={}
    for r in records[1:]:
        local=rows[r['offset']:r['offset']+r['count']]
        share=QShare([v for v,g in local],[g for v,g in local],tuple(r['shape']))
        target=matrices if r['kind']=='matrices'else squares if r['kind']=='squares'else inputs
        target.setdefault(r['name'],{})[r['component']]=share
    for kind,target in [('matrices',matrices),('squares',squares)]:
        for name,parts in list(target.items()):
            r=next(r for r in records if r['kind']==kind and r['name']==name);identifier=bytes.fromhex(r['identifier'])
            target[name]=QTriple(parts['a'],parts['b'],parts['c'],identifier)if kind=='matrices'else QSquare(parts['a'],parts['a_squared'],identifier)
    for r in records:
        if r['kind']=='inputs':
            plain=clear[r['clear_offset']:r['clear_offset']+r['count']]if r['owner']==party else None
            inputs[r['name']]=QInputMask(inputs[r['name']]['share'],plain,r['owner'],bytes.fromhex(r['identifier']))
    return dict(alpha_share=key,matrix_triples=matrices,square_masks=squares,input_masks=inputs,
                session=hashlib.sha256(b'LOWGEAR-native-field-v1\0'+PRIME.to_bytes(32,'little')+bytes.fromhex(m['session'])).digest(),provenance=m)
