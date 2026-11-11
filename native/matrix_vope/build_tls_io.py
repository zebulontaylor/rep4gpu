"""Build the TLS IOChannel regression driver against the pinned EMP headers."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
BACKEND=REPO/'work/vole-labs'
OUT=REPO/'work/matrix_vope_tls'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=json.loads((HERE/'source-lock.json').read_text())
    for dependency in lock['dependencies']:
        root=BACKEND/dependency['path']
        if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=dependency['commit']:
            raise ValueError('EMP dependency commit mismatch')
        if subprocess.check_output(['git','-C',str(root),'diff','HEAD','--name-only'],text=True).strip():
            raise ValueError('EMP dependency tracked sources changed')
    for name,expected in lock['source_files_sha256'].items():
        if hashlib.sha256((BACKEND/name).read_bytes()).hexdigest()!=expected:raise ValueError('EMP pinned source differs: '+name)
    binary=OUT/'test_tls_io'
    command=['c++','-std=c++20','-O2','-Wall','-Wextra','-Wno-ignored-attributes','-march=native','-maes','-pthread',
             '-I'+str(BACKEND/'thirdparty/emp-tool'),'-I'+str(BACKEND/'build/emp-tool'),
             str(HERE/'test_tls_io.cpp'),str(HERE/'tls_io.cpp'),str(BACKEND/'build/emp-tool/libemp-tool.a'),
             '-lssl','-lcrypto','-o',str(binary)]
    with (OUT/'build.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    report=dict(dependencies=lock['dependencies'],command=command,
                source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('tls_io.h','tls_io.cpp','test_tls_io.cpp','build_tls_io.py')},
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (OUT/'build-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return binary


if __name__=='__main__':print(build())
