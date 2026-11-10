"""Run the pinned native q256 fixed-model producer with private role inputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

from rep4gpu.preprocessing_mascot_prepared import program,write_owner_inputs,FORMAT
from rep4gpu.preprocessing_mascot_direct import _binary,_verify_policy,PRIME
from rep4gpu.preprocessing_mama import _sha,_port_base,_native_environment


def native_worker(config_path):
    config=json.loads(Path(config_path).read_text());role=Path(config['role']);party=config['party']
    metadata=json.loads((role/'public-manifest.json').read_text())
    source,records,total,plan,clear=program(metadata['request'],metadata['session'])
    if hashlib.sha256(source.encode()).hexdigest()!=metadata['export_source_sha256']:
        raise ValueError('native prepared source changed')
    own=json.loads(Path(config['owner_inputs']).read_text()) if config['owner_inputs'] else {}
    write_owner_inputs(role,party,metadata['request'],metadata['session'],own)
    binary=Path(config['command'][0]);policy,digest=_verify_policy(binary.parent,binary)
    if policy!=metadata['native_policy'] or digest!=metadata['native_policy_sha256'] or _sha(binary)!=metadata['native_binary_sha256']:
        raise ValueError('native policy changed before role launch')
    environment,_=_native_environment(binary.parent,binary,policy)
    os.chdir(role)
    os.execve(binary,config['command'],environment)


def run_native(request,output_dir,*,owner_input_paths,native_root='work/mascot256-fresh-build',timeout=120,batch_size=1000):
    """Generate one batch; only each native worker reads its owner-input file.

    ``owner_input_paths`` maps roles to independently provisioned JSON files.
    The coordinator copies public programs/certificates and passes file paths;
    it does not open these model files or any native secret output.
    """
    if not math.isfinite(timeout) or timeout<=0 or type(batch_size)is not int or batch_size<1:
        raise ValueError('invalid timeout or batch size')
    if not isinstance(request,dict):request=json.loads(Path(request).read_text())
    session=secrets.token_hex(32);source,records,total,plan,clear_counts=program(request,session)
    required={p for p in range(4) if plan[p]}
    if set(owner_input_paths)!=required:raise ValueError('exact private model-owner input paths required')
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    root=Path(native_root).resolve();binary=_binary(root);policy,policy_hash=_verify_policy(root,binary)
    environment,libraries=_native_environment(root,binary,policy)
    compiler=output/'compiler'
    for directory in ('Source','Bytecode','Schedules'):(compiler/'Programs'/directory).mkdir(parents=True,exist_ok=True)
    name='native_q_prepared_export';source_path=compiler/'Programs/Source'/f'{name}.mpc';source_path.write_text(source)
    metadata=dict(format=FORMAT,ready=False,return_codes=[],prime=PRIME,mac_keys=1,security_parameter=128,
                  request=request,session=session,records=records,share_count=total,owner_input_plan=plan,clear_counts=clear_counts,
                  native_binary=str(binary),native_binary_sha256=_sha(binary),native_policy=policy,native_policy_sha256=policy_hash,
                  native_loaded_libraries=libraries,export_source_sha256=_sha(source_path),trusted_dealer=False,
                  security_target_met=False,privacy_audit_status='conditional_native_fixed_model_experiment')
    started=time.perf_counter()
    with (compiler/'compile.log').open('w') as log:
        compiled=subprocess.run([sys.executable,'-B',str(root/'compile.py'),'-F','64',name],cwd=compiler,
                                stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
                                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    metadata['compile_ms']=(time.perf_counter()-started)*1000
    if compiled.returncode:
        metadata.update(passed=False,compile_return_code=compiled.returncode)
        (output/'public-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata
    certs=output/'public-certificates';certs.mkdir()
    def openssl(*args):subprocess.run(['openssl',*map(str,args)],check=True,capture_output=True,timeout=60)
    roles=[]
    for p in range(4):
        role=output/f'party{p}';role.mkdir(mode=0o700);player=role/'Player-Data';player.mkdir(mode=0o700)
        (role/'Programs').symlink_to(compiler/'Programs',target_is_directory=True)
        openssl('req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:prime256v1','-nodes','-keyout',player/f'P{p}.key',
                '-out',certs/f'P{p}.pem','-days','1','-subj',f'/CN=P{p}')
        (player/f'P{p}.key').chmod(0o600);roles.append(role)
    base=_port_base();processes=[];logs=[]
    for p,role in enumerate(roles):
        for cert in certs.glob('*.pem'):shutil.copy2(cert,role/'Player-Data'/cert.name)
        openssl('rehash',role/'Player-Data')
        (role/'public-manifest.json').write_text(json.dumps(metadata))
        (role/'native-policy.json').write_text((root/'mascot256-preprocessing-policy.json').read_text())
        command=[str(binary),'-p',str(p),'-N','4','-P',str(PRIME),'-lgp','256','-S','128','-e','-pn',str(base),'-b',str(batch_size),name]
        config=dict(role=str(role),party=p,owner_inputs=str(Path(owner_input_paths[p]).resolve()) if p in required else None,command=command)
        path=role/'launch.json';path.write_text(json.dumps(config));path.chmod(0o600)
    native_started=time.perf_counter()
    try:
        for role in roles:
            log=(role/'native.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-m','scripts.preprocess_mascot_prepared','--native-worker',str(role/'launch.json')],
                                               stdout=log,stderr=subprocess.STDOUT,env=environment))
        deadline=time.monotonic()+timeout
        for process in processes:process.wait(timeout=max(.001,deadline-time.monotonic()))
    except BaseException as error:metadata['error']=f'{type(error).__name__}: {error}'
    finally:
        for process in processes:
            if process.poll()is None:process.kill()
        for process in processes:process.wait()
        for log in logs:log.close()
    codes=[process.returncode for process in processes];passed=codes==[0]*4
    metadata.update(ready=passed,passed=passed,return_codes=codes,native_preprocessing_ms=(time.perf_counter()-native_started)*1000,
                    preprocessing_including_compile_ms=(time.perf_counter()-started)*1000,
                    measurement='Fresh native generation including isolated owner provisioning and worker startup; shared-host contended functional run.')
    encoded=json.dumps(metadata,indent=2)+'\n';(output/'public-manifest.json').write_text(encoded)
    for role in roles:
        (role/'public-manifest.json').write_text(encoded)
        for directory in ('Persistence','Player-Data'):
            for private in (role/directory).glob('*'):
                if private.is_file() and not private.name.endswith('.pem'):private.chmod(0o600)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native-worker',type=Path)
    args=parser.parse_args()
    if args.native_worker:native_worker(args.native_worker)
    else:parser.error('use run_native from the prepared toy orchestrator')


if __name__=='__main__':main()
