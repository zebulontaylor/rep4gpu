"""Build F127^4 GMP-oracle checks and repeat them under undefined-behavior checks."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
BACKEND=REPO/'work/vole-labs'
OUT=REPO/'work/matrix_vope/fp127ext4'
OUT.mkdir(parents=True,exist_ok=True)
flags=['-std=c++20','-maes','-msse4.1','-mpclmul','-mrdseed',
       '-I'+str(HERE),'-I'+str(BACKEND),'-I'+str(BACKEND/'thirdparty/emp-tool'),
       '-I'+str(BACKEND/'thirdparty/emp-ot'),'-I'+str(BACKEND/'build/emp-tool')]
link=['-L'+str(BACKEND/'build/emp-tool'),'-lemp-tool','-lcrypto','-lgmpxx','-lgmp','-pthread']
results={'sources_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                         for name in ('fp127.h','fp127ext4.h','test_fp127ext4.cpp')},'checks':{}}
for name,compiler,extra in (('arithmetic','c++',['-O2']),
                            ('ubsan','clang++',['-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all'])):
    command=[compiler,*flags,*extra,str(HERE/'test_fp127ext4.cpp'),*link,'-o',str(OUT/name)]
    with (OUT/(name+'-build.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run([str(OUT/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
    results['checks'][name]={'output':(OUT/(name+'.log')).read_text().strip(),'build_command':command}
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
