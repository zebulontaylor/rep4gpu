"""Carry one role's validated HE key cache into a fresh correlation pool.

The coordinator handles public metadata. Each copying subprocess opens only its
own role's cache. An epoch is retired after any failed continuation or online
run because its MAC key must never be used after a failed authenticated opening.
"""
import argparse,hashlib,json,os,shutil,stat
from pathlib import Path
PRIME=2**64-2**32+1
FORMAT='lowgear-goldilocks-s40-native-field-v1'
def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def epoch_root(metadata,root):return Path(metadata.get('key_epoch_root',root)).resolve()
def retire_epoch(role_or_root,reason='failed authenticated session'):
    path=Path(role_or_root).resolve();m=_read(path/'public-manifest.json');root=epoch_root(m,path.parent if path.name in {f'party{p}' for p in range(4)} and (path.parent/'public-manifest.json').is_file() else path)
    marker=root/'key-epoch-retired'
    try:fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    except FileExistsError:return
    with os.fdopen(fd,'w')as f:f.write(str(reason)[:512]+'\n');f.flush();os.fsync(f.fileno())
    _sync(root)
def validate_parent(previous_root,success_report):
    root=Path(previous_root).resolve();manifest=root/'public-manifest.json';m=_read(manifest);r=_read(success_report)
    if (m.get('format')!=FORMAT or m.get('ready')is not True or m.get('passed')is not True or m.get('return_codes')!=[0]*4 or m.get('prime')!=PRIME or m.get('finite_bit_probe')is not False):raise ValueError('parent native pool was not successful')
    if (r.get('passed')is not True or r.get('return_codes')!=[0]*4 or r.get('source_stable_during_run')is not True or r.get('exact_integer_match')is not True or r.get('public_preprocessing_manifest')!=m):raise ValueError('parent online run was not successful')
    epoch=epoch_root(m,root)
    if (epoch/'key-epoch-retired').exists():raise ValueError('LowGear key epoch is retired')
    first=_read(epoch/'public-manifest.json')
    if first.get('session')!=m.get('key_epoch_session',m['session']) or first.get('native_policy_sha256')!=m.get('native_policy_sha256'):raise ValueError('LowGear epoch lineage differs')
    for party in range(4):
        marker=root/f'party{party}/consumed'
        if marker.is_symlink()or not marker.is_file()or marker.stat().st_size!=64 or marker.read_text()!=m['session']:raise ValueError('parent online bundles were not consumed with this session')
    return dict(key_epoch_root=str(epoch),key_epoch_session=first['session'],parent_native_manifest=str(manifest),parent_native_manifest_sha256=_sha(manifest),parent_online_report=str(Path(success_report).resolve()),parent_online_report_sha256=_sha(success_report),parent_session=m['session'])
def _carry_role(previous_root,destination_role,party,successor_session,success_report):
    if type(party)is not int or party not in range(4):raise ValueError('invalid role')
    root=Path(previous_root).resolve();dest=Path(destination_role).resolve();info=validate_parent(root,success_report)
    if dest.name!=f'party{party}' or type(successor_session)is not str or len(successor_session)!=64 or bytes.fromhex(successor_session).hex()!=successor_session or successor_session==info['parent_session'] or dest.parent==root:raise ValueError('invalid successor identity')
    old=root/f'party{party}';m=_read(old/'public-manifest.json')
    if m!=_read(root/'public-manifest.json')or old.is_symlink()or (old/'Player-Data').is_symlink()or (dest/'Player-Data').is_symlink():raise ValueError('role cache boundary differs')
    from scripts.build_lowgear_goldilocks import verify_policy
    policy=verify_policy(m['native_root'])
    if policy!=m['native_policy']or _sha(Path(m['native_root'])/'lowgear-goldilocks-s40-policy.json')!=m['native_policy_sha256']:raise ValueError('parent native policy differs')
    claim=old/'key-cache-advanced';fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w')as f:json.dump(dict(successor_session=successor_session,destination=str(dest)),f);f.flush();os.fsync(f.fileno())
    _sync(old)
    names=[]
    for prefix in ('PairwiseParams-gfp-64-40-40-128-1-'+str(PRIME)+'-1-P'+str(party)+'-4','LowGear-Secrets-40-*'+f'-P{party}-4'):
        found=list((old/'Player-Data').glob(prefix))
        if len(found)!=1:raise ValueError('missing or ambiguous own-role HE cache')
        names.append(found[0].name)
    total=0
    try:
        for name in names:
            source=old/'Player-Data'/name;target=dest/'Player-Data'/name
            fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd)
            if not stat.S_ISREG(s.st_mode)or not 0<s.st_size<=32*1024*1024:os.close(fd);raise ValueError('invalid own-role cache size/type')
            with os.fdopen(fd,'rb')as src:
                output=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
                with os.fdopen(output,'wb')as dst:shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
            total+=s.st_size
        _sync(dest/'Player-Data')
    except BaseException:
        retire_epoch(root,'failed private key-cache continuation');raise
    return dict(party=party,cache_files=len(names),cache_bytes=total,key_epoch_session=info['key_epoch_session'])
def carry_role(previous_root,destination_role,party,successor_session,success_report):
    try:return _carry_role(previous_root,destination_role,party,successor_session,success_report)
    except BaseException:
        if (Path(previous_root)/'public-manifest.json').is_file():
            retire_epoch(previous_root,'failed private key-cache continuation')
        raise
if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-root',required=True);p.add_argument('--destination-role',required=True);p.add_argument('--party',type=int,required=True);p.add_argument('--session',required=True);p.add_argument('--success-report',required=True);a=p.parse_args()
    print(json.dumps(carry_role(a.previous_root,a.destination_role,a.party,a.session,a.success_report)))
