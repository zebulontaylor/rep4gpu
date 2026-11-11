"""Build the mTLS direct chosen-input OLE adapter without changing the backend."""
from pathlib import Path
import subprocess
import json
import hashlib

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
BACKEND=REPO/'work/vole-labs'
OUT=REPO/'work/direct_ole_tls'
OUT.mkdir(parents=True,exist_ok=True)
lock=json.loads((HERE/'source-lock.json').read_text())
for dependency in lock['dependencies']:
    directory=BACKEND/dependency['path']
    head=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if head!=dependency['commit']:
        raise SystemExit('Dependency commit mismatch: '+str(directory))
    dirty=subprocess.check_output(['git','-C',str(directory),'diff','HEAD','--name-only'],text=True)
    if dirty.strip():
        raise SystemExit('Dependency has modified tracked files: '+str(directory))
for name,expected in lock['source_files_sha256'].items():
    if hashlib.sha256((BACKEND/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Dependency source hash mismatch: '+name)

command=['c++','-std=c++20','-O2','-Wall','-Wno-ignored-attributes',
         '-DSIV_OLE_TLS','-march=native','-maes','-mrdseed','-DENABLE_RDSEED','-pthread',
         '-I'+str(BACKEND),'-I'+str(BACKEND/'thirdparty/emp-ot'),
         '-I'+str(BACKEND/'thirdparty/emp-tool'),'-I'+str(BACKEND/'build/emp-tool'),
         str(HERE/'direct_ole.cpp'),str(HERE/'tls_io.cpp'),
         str(BACKEND/'build/emp-ot/libemp-ot.a'),str(BACKEND/'build/emp-tool/libemp-tool.a'),
         '-lssl','-lcrypto','-o',str(OUT/'direct_ole_tls')]
with (OUT/'build.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
manifest={'dependencies':lock['dependencies'],'source_sha256':{}}
for name in ('tls_io.cpp','tls_io.h','direct_ole.cpp','direct_ole.hpp','wolverine_base.hpp','seeded_vole.hpp','fp127.h','fp127ext4.h','direct_ole_io.py','check_direct_ole_tls.py','build_direct_ole_tls.py'):
    manifest['source_sha256'][name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
(OUT/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'direct_ole_tls')
