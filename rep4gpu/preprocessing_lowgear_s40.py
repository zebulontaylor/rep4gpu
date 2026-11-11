"""Explicit S40/128-computational LowGear diagnostic; separate from S128."""
from pathlib import Path
import hashlib,json,math,os,secrets,shutil,socket,struct,subprocess,sys,time
from .preprocessing_mama import _layout,_sha,_port_base
from .preprocessing_lowgear import PRIME,_native_environment,_binary,_program as _strong_program
_FORMAT='lowgear-s40-native-field-experiment-v1'
MAC_KEYS=1

def _program(request):
    return _strong_program(request).replace('program.security=128','program.security=40')

def _verify_policy(root,binary):
    from scripts.build_lowgear_secure import verify_policy
    strong=verify_policy(root)
    path=Path(root)/'lowgear-s40-preprocessing-policy.json';policy=json.loads(path.read_text())
    if policy.get('format')!='lowgear-s40-experimental-build-v1' or policy.get('statistical_security')!=40 or policy.get('computational_security')!=128 or policy.get('covert')is not False or policy.get('prime')!=PRIME:
        raise ValueError('S40 LowGear diagnostic parameters mismatch')
    if policy['binary_build_policy_sha256']!=_sha(Path(root)/'lowgear-preprocessing-policy.json'):
        raise ValueError('S40 diagnostic does not match the verified public backend')
    for name in ('binary_sha256','source_sha256','dependency_sha256'):
        if policy[name]!=strong[name]:raise ValueError('S40 diagnostic backend mismatch')
    return policy,_sha(path)


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
                  security_parameter=40,statistical_security=40,computational_security=128,session=session,request=request,records=records,
                  privacy_audit_status='explicit_s40_comp128_lowgear_diagnostic',
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
                     '-S','40','-o','comp_sec=128,verbose_he','-e','-pn',str(base),'-b',str(batch_size),program]
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
        (role/'native-policy.json').write_text((native_root/'lowgear-s40-preprocessing-policy.json').read_text())
        # No native secret file is copied between roles or read by this coordinator.
        for private in (role/'Player-Data').glob('Mascot-Secrets-*'):private.chmod(0o600)
        for private in (role/'Persistence').glob('*') if (role/'Persistence').exists() else []:private.chmod(0o600)
        for private in (role/'Player-Data').glob('Binary-Output-*'):private.chmod(0o600)
    return metadata
