"""Build the pinned experimental LowGear backend in a new private workspace.

This records source and binary provenance. It does not certify the selected HE
parameters or provide remote attestation. No player data or old native objects
are copied from the source tree.
"""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
from scripts.build_mama_secure import clean_build_environment

ROOT=Path(__file__).resolve().parents[1]
LOCK_PATH=ROOT/'native/lowgear-source-lock.json'
PATCH_PATH=ROOT/'native/patches/lowgear-experimental-build.patch'
FORMAT='lowgear-experimental-build-v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_policy(native_root):
    root=Path(native_root).resolve();lock=json.loads(LOCK_PATH.read_text())
    policy=json.loads((root/'lowgear-preprocessing-policy.json').read_text())
    expected=dict(format=FORMAT,protocol='LowGear',covert=False,prime=lock['prime'],field_bits=256,
                  statistical_security=128,computational_security=128)
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('LowGear parameters differ')
    if lock['patch_sha256']!=digest(PATCH_PATH):raise ValueError('LowGear patch lock differs')
    if policy.get('source_lock_sha256',digest(LOCK_PATH))!=digest(LOCK_PATH):raise ValueError('LowGear source lock differs')
    if policy['binary_sha256']!=digest(root/'lowgear-party.x'):raise ValueError('LowGear binary differs')
    if (root/'CONFIG.mine').read_text()!=lock['config']:raise ValueError('LowGear configuration differs')
    for name,value in lock['patched_source_sha256'].items():
        if digest(root/name)!=value or policy['source_sha256'].get(name)!=value:raise ValueError(f'LowGear source differs: {name}')
    required={'libSPDZ.so','libFHE.so'}|set(lock['dependency_archives_sha256'])
    if set(policy['dependency_sha256'])!=required:raise ValueError('LowGear dependency list differs')
    for name in required:
        actual=digest(root/name)
        if policy['dependency_sha256'][name]!=actual:raise ValueError(f'LowGear dependency differs: {name}')
        if name in lock['dependency_archives_sha256']and actual!=lock['dependency_archives_sha256'][name]:raise ValueError('LowGear external archive differs')
    return policy


def write_policy(root):
    root=Path(root);lock=json.loads(LOCK_PATH.read_text())
    dependencies={'libSPDZ.so','libFHE.so'}|set(lock['dependency_archives_sha256'])
    policy=dict(format=FORMAT,protocol='LowGear',covert=False,prime=lock['prime'],field_bits=256,
                statistical_security=128,computational_security=128,max_mod_sz=16,
                review_status='parameter_and_functional_evaluation_only',
                binary_sha256=digest(root/'lowgear-party.x'),
                source_sha256={name:digest(root/name)for name in lock['patched_source_sha256']},
                dependency_sha256={name:digest(root/name)for name in dependencies},
                source_lock_sha256=digest(LOCK_PATH),patch_sha256=digest(PATCH_PATH),
                binary_fallback_protocol='MASCOT Share over GF(2^128), unused by prime export',
                auxiliary_setup_transport='CryptoPlayer with role-local identities',public_cost_instrumentation=True)
    path=root/'lowgear-preprocessing-policy.json';path.write_text(json.dumps(policy,indent=2)+'\n')
    verify_policy(root);return path


def build(source,cache,destination,jobs):
    source=Path(source).resolve();cache=Path(cache).resolve();root=Path(destination).resolve()
    lock=json.loads(LOCK_PATH.read_text())
    if root.exists():raise ValueError('build destination already exists')
    for name,value in lock['source_files_sha256'].items():
        path=source/name
        if value is None:
            if path.exists():raise ValueError(f'expected absent source configuration: {name}')
        elif digest(path)!=value:raise ValueError(f'unpinned source: {name}')
    for name,value in lock['dependency_archives_sha256'].items():
        if digest(cache/name)!=value:raise ValueError(f'unpinned dependency: {name}')
    def ignore(directory,names):
        return [n for n in names if n in ('Player-Data','Programs','deps','local','.git','__pycache__','bin')or n.endswith(('.o','.d','.x','.a','.so','.map'))]
    shutil.copytree(source,root,ignore=ignore)
    shutil.copytree(cache/'local',root/'local',symlinks=True)
    shutil.copytree(cache/'deps',root/'deps',symlinks=True)
    (root/'Player-Data').mkdir(mode=0o700)
    with (root/'apply-lowgear-patch.log').open('w')as log:
        subprocess.run(['patch','-p1','--batch','--forward','-i',str(PATCH_PATH)],cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
    # External OT/sodium archives do not depend on MAX_MOD_SZ. Their bytes were
    # checked above; only native field-layout-dependent objects are rebuilt.
    os.utime(root/'local/lib/liblibOTe.a',None)
    with (root/'lowgear-build.log').open('w')as log:
        subprocess.run(['make','-j'+str(jobs),'lowgear-party.x'],cwd=root,env=clean_build_environment(),check=True,stdout=log,stderr=subprocess.STDOUT)
    return write_policy(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--dependency-root',type=Path)
    parser.add_argument('--build-dir',type=Path,required=True);parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:verify_policy(args.build_dir);print('Experimental LowGear provenance verified.')
    else:
        if args.source is None or args.dependency_root is None:parser.error('--source and --dependency-root are required')
        print(build(args.source,args.dependency_root,args.build_dir,args.jobs))
