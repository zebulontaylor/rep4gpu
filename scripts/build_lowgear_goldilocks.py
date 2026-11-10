"""Pin the separate Goldilocks/S40 LowGear experiment and finite-bit diagnostic.

The public bit diagnostic is disabled in normal preprocessing. This records
build provenance; the HE parameter selection remains an experiment.
"""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
from scripts.build_mama_secure import clean_build_environment
ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'native/lowgear-goldilocks-source-lock.json'
PATCH=ROOT/'native/patches/lowgear-goldilocks-six-limbs.patch'
FORMAT='lowgear-goldilocks-s40-experimental-build-v2'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_policy(root):
    root=Path(root).resolve();lock=json.loads(LOCK.read_text());p=json.loads((root/'lowgear-goldilocks-s40-policy.json').read_text())
    expected=dict(format=FORMAT,protocol='LowGear',covert=False,prime=2**64-2**32+1,field_bits=64,statistical_security=40,computational_security=128,max_mod_sz=6,finite_bit_probe_default=False)
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('covert') is not False or p.get('finite_bit_probe_default') is not False:raise ValueError('Goldilocks build parameters differ')
    if p.get('source_lock_sha256')!=digest(LOCK) or lock['patch_sha256']!=digest(PATCH):raise ValueError('Goldilocks source lock differs')
    if p['source_sha256']!=lock['source_sha256'] or (root/'CONFIG.mine').read_text()!=lock['config']:raise ValueError('Goldilocks source configuration differs')
    for name,value in lock['source_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Goldilocks source differs: '+name)
    required={'libSPDZ.so','libFHE.so'}|set(lock['dependency_archives_sha256'])
    if set(p['dependency_sha256'])!=required:raise ValueError('Goldilocks dependencies differ')
    for name in required:
        actual=digest(root/name)
        if p['dependency_sha256'][name]!=actual:raise ValueError('Goldilocks dependency differs: '+name)
        if name in lock['dependency_archives_sha256'] and actual!=lock['dependency_archives_sha256'][name]:raise ValueError('Goldilocks external dependency differs')
    if p['binary_sha256']!=digest(root/'lowgear-party.x'):raise ValueError('Goldilocks binary differs')
    return p

def write_policy(root):
    root=Path(root).resolve();lock=json.loads(LOCK.read_text());deps={'libSPDZ.so','libFHE.so'}|set(lock['dependency_archives_sha256'])
    p=dict(format=FORMAT,protocol='LowGear',covert=False,prime=2**64-2**32+1,field_bits=64,statistical_security=40,computational_security=128,max_mod_sz=6,finite_bit_probe_default=False,review_status='parameter_and_functional_evaluation_only',binary_sha256=digest(root/'lowgear-party.x'),source_sha256=lock['source_sha256'],dependency_sha256={n:digest(root/n)for n in deps},source_lock_sha256=digest(LOCK),patch_sha256=digest(PATCH))
    path=root/'lowgear-goldilocks-s40-policy.json';path.write_text(json.dumps(p,indent=2)+'\n');verify_policy(root);return path

def build(base,destination,jobs=2):
    from scripts.build_lowgear_secure import verify_policy as verify_base
    base=Path(base).resolve();root=Path(destination).resolve();verify_base(base)
    if root.exists() or jobs not in (1,2):raise ValueError('fresh build directory and one or two jobs required')
    def ignore(directory,names):return[n for n in names if n in ('Player-Data','Programs','.git','__pycache__','bin')or (n.endswith(('.o','.d','.x','.a','.so','.map'))and not any(x in Path(directory).relative_to(base).parts for x in ('local','deps')))]
    shutil.copytree(base,root,ignore=ignore,symlinks=True)
    subprocess.run(['patch','-p1','--batch','--forward','-i',str(PATCH)],cwd=root,check=True)
    with (root/'goldilocks-build.log').open('w')as log:subprocess.run(['make','-j'+str(jobs),'lowgear-party.x'],cwd=root,env=clean_build_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    return write_policy(root)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-build',type=Path)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        verify_policy(args.build_dir);print('Goldilocks/S40 experimental build provenance verified.')
    else:
        if args.base_build is None:parser.error('--base-build is required for a new build')
        print(build(args.base_build,args.build_dir,args.jobs))
