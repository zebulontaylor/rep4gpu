"""Build and verify the pinned large-prime, amplification-four MASCOT backend.

Requires a verified dependency cache containing the locked SoftSpoken/base-OT
archives and libsodium SDK. The destination must be new. Party data and old
MP-SPDZ object files are never copied. This records local build provenance;
it is not remote attestation or a review of the separate p31 bridge program.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.build_mama_secure import clean_build_environment

ROOT=Path(__file__).resolve().parents[1]
LOCK_PATH=ROOT/'native/mascot256-source-lock.json'
PATCH_PATH=ROOT/'native/patches/mascot-amplify-four.patch'
FRESH_PATCH_PATH=ROOT/'native/patches/mascot-fresh-sacrifice.patch'
FORMAT='mascot256-tau4-fresh-sacrifice-build-v2'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_policy(native_root):
    root=Path(native_root);lock=json.loads(LOCK_PATH.read_text())
    sources={name:digest(root/name) for name in lock['source_files_sha256']}
    sources['CONFIG.mine']=digest(root/'CONFIG.mine')
    dependencies={name:digest(root/name) for name in lock['dependency_archives_sha256']}
    dependencies.update({name:digest(root/name) for name in ('libSPDZ.so','local/lib/libsodium.so')})
    policy=dict(format=FORMAT,internal_prime=lock['internal_prime'],field_bits=256,
                runtime_security=128,compiler_security=176,compiler_bit_length=78,amplification=4,check=True,
                post_authentication_sacrifice_coin=True,
                native_mac_keys=1,application_prime=2147483647,application_mac_keys=7,
                ot='SoftSpokenMalOt',binary_sha256=digest(root/'mascot-party.x'),
                source_sha256=sources,dependency_sha256=dependencies,
                source_lock_sha256=digest(LOCK_PATH),reviewed_patch_sha256=digest(PATCH_PATH),
                fresh_sacrifice_patch_sha256=digest(FRESH_PATCH_PATH),
                build_properties=dict(LONG_BIT_MAC_KEY=True,USE_KOS=0,USE_LIBOTE2=0,INSECURE=False))
    path=root/'mascot256-preprocessing-policy.json';path.write_text(json.dumps(policy,indent=2)+'\n')
    verify_policy(root)
    return path


def verify_policy(native_root):
    root=Path(native_root).resolve();lock=json.loads(LOCK_PATH.read_text())
    policy=json.loads((root/'mascot256-preprocessing-policy.json').read_text())
    expected=dict(format=FORMAT,internal_prime=str(2**256-189),field_bits=256,
                  runtime_security=128,compiler_security=176,compiler_bit_length=78,amplification=4,check=True,
                  post_authentication_sacrifice_coin=True,
                native_mac_keys=1,application_prime=2147483647,application_mac_keys=7,ot='SoftSpokenMalOt')
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('MASCOT policy parameters differ')
    if policy.get('source_lock_sha256')!=digest(LOCK_PATH) or policy.get('reviewed_patch_sha256')!=digest(PATCH_PATH) or lock['patch_sha256']!=digest(PATCH_PATH):
        raise ValueError('MASCOT source or amplification patch lock differs')
    if policy.get('fresh_sacrifice_patch_sha256')!=digest(FRESH_PATCH_PATH) or lock.get('fresh_sacrifice_patch_sha256')!=digest(FRESH_PATCH_PATH):
        raise ValueError('fresh post-authentication sacrifice patch differs')
    if (root/'CONFIG.mine').read_text()!=lock['config']:raise ValueError('MASCOT configuration differs')
    properties=dict(LONG_BIT_MAC_KEY=True,USE_KOS=0,USE_LIBOTE2=0,INSECURE=False)
    if policy.get('build_properties')!=properties:raise ValueError('MASCOT build properties differ')
    if policy.get('binary_sha256')!=digest(root/'mascot-party.x'):raise ValueError('MASCOT binary differs')
    for name,original in lock['source_files_sha256'].items():
        expected_hash=lock.get('patched_source_sha256',{}).get(name,original)
        if digest(root/name)!=expected_hash or policy.get('source_sha256',{}).get(name)!=expected_hash:
            raise ValueError(f'MASCOT source differs: {name}')
    if policy.get('source_sha256',{}).get('CONFIG.mine')!=digest(root/'CONFIG.mine'):raise ValueError('MASCOT config digest differs')
    required=set(lock['dependency_archives_sha256'])|{'libSPDZ.so','local/lib/libsodium.so'}
    if set(policy.get('dependency_sha256',{}))!=required:raise ValueError('MASCOT dependency list differs')
    for name in required:
        actual=digest(root/name)
        if policy['dependency_sha256'][name]!=actual or (name in lock['dependency_archives_sha256'] and actual!=lock['dependency_archives_sha256'][name]):
            raise ValueError(f'MASCOT dependency differs: {name}')
    return policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--dependency-root',type=Path)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();root=args.build_dir.resolve()
    if args.verify_only:
        verify_policy(root);print('MASCOT build policy verified.');return 0
    if args.source is None or args.dependency_root is None:parser.error('--source and --dependency-root are required for a build')
    source=args.source.resolve();cache=args.dependency_root.resolve();lock=json.loads(LOCK_PATH.read_text())
    if root.exists():raise ValueError('destination already exists')
    if source==root or source in root.parents:raise ValueError('destination must be outside source tree')
    for name,expected in lock['source_files_sha256'].items():
        if digest(source/name)!=expected:raise ValueError(f'unpinned source: {name}')
    for name,expected in lock['dependency_archives_sha256'].items():
        if digest(cache/name)!=expected:raise ValueError(f'unpinned dependency: {name}')
    def ignore(directory,names):
        return [n for n in names if n in ('Player-Data','Programs','deps','local','bin','.git','__pycache__') or n.endswith(('.o','.d','.a','.so','.x','.map'))]
    shutil.copytree(source,root,ignore=ignore)
    shutil.copytree(cache/'local',root/'local',symlinks=True)
    shutil.copytree(cache/'deps',root/'deps',symlinks=True)
    (root/'Player-Data').mkdir(mode=0o700)
    (root/'CONFIG.mine').write_text(lock['config'])
    # External archives do not depend on field/template configuration. Their
    # bytes were verified above; avoid Make's conservative CONFIG prerequisite.
    os.utime(root/'local/lib/liblibOTe.a',None)
    commands=[]
    def run(command,name):
        commands.append(list(map(str,command)))
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(list(map(str,command)),cwd=root,stdout=log,stderr=subprocess.STDOUT,
                           env=clean_build_environment(),check=True)
    run(['patch','-p1','--batch','--forward','-i',PATCH_PATH],'apply-amplification-four')
    run(['patch','-p1','--batch','--forward','-i',FRESH_PATCH_PATH],'apply-fresh-sacrifice')
    run(['make','-j'+str(args.jobs),'mascot-party.x'],'mascot256-build')
    policy=write_policy(root)
    (root/'build-report.json').write_text(json.dumps(dict(passed=True,commands=commands,
        binary_sha256=digest(root/'mascot-party.x'),library_sha256=digest(root/'libSPDZ.so'),
        policy_sha256=digest(policy),source_lock_sha256=digest(LOCK_PATH),
        compiler=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0]),indent=2)+'\n')
    print(root/'mascot-party.x');return 0


if __name__=='__main__':raise SystemExit(main())
