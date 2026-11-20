"""Run the compiled HPMPC research fixture; public seeds and plain TCP are insecure.

Build program.hpp as HPMPC's YourFirstProgram (function 65), protocol 12,
PRE=0, CV_FIX=1, BITLENGTH=16, DATTYPE=512, FRACTIONAL=0. This benchmark
opens only predictions and verifies them against the trained ring16 model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('.local/deps/hpmpc'))
parser.add_argument('--checkpoint',type=Path,default=Path('experiments/results/examples/tiny_ring16_fixed_head/model.npz'))
parser.add_argument('--output',type=Path,default=Path('experiments/results/inference/hpmpc_fixed_head'))
args=parser.parse_args()
# Socket helper threads share the same four-core budget as party computation.
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
root=args.root.resolve(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
fixture=root/'evidence'
model=np.load(args.checkpoint)
x=np.resize(model['x_test'],(8192,512)).astype(np.int64)
w=model['w1'].astype(np.int64); b=model['b1'].astype(np.int64); bias=int(model['b2'][0])
assert w.shape==(512,4) and np.array_equal(model['signs'],[1,1,-1,-1])
expected=(((x@w+b)**2@model['signs']+bias)<0).astype(np.uint16)
x.reshape(256,32,512).transpose(0,2,1).astype('<u2').tofile(fixture/'owner0.bin')
np.concatenate([w.ravel(),b,[bias]]).astype('<u2').tofile(fixture/'owner1.bin')
native_source=Path(__file__).with_name('native.cpp')
subprocess.run(['g++','-O3','-march=native','-std=c++20',str(native_source),'-o',str(fixture/'native_fixed_head')],check=True)
native=subprocess.check_output([str(fixture/'native_fixed_head')],cwd=root,text=True)
for name in ['native_predictions.bin','native_sparse_predictions.bin']:
    assert np.array_equal(np.fromfile(fixture/name,dtype='<u2'),expected)
rows=[]
for repeat in range(7):
    ps=[]; handles=[]
    for party in range(4):
        (fixture/f'predictions{party}.bin').unlink(missing_ok=True)
        handle=open(output/f'r{repeat}_p{party}.log','w');handles.append(handle)
        ps.append(subprocess.Popen([str(root/f'executables/run-P{party}.o')],cwd=root,stdout=handle,stderr=subprocess.STDOUT))
    codes=[]
    try:
        codes=[p.wait(timeout=90) for p in ps]
    finally:
        for p in ps:
            if p.poll() is None:p.kill();p.wait()
        for h in handles:h.close()
    parties=[]
    for p in range(4):
        text=(output/f'r{repeat}_p{p}.log').read_text()
        path=fixture/f'predictions{p}.bin'
        exact=path.exists() and np.array_equal(np.fromfile(path,dtype='<u2'),expected)
        total=re.search(r'computation chrono: (\S+)s',text)
        phases=re.search(r'PHASE_MS input (\S+) matmul (\S+) sign (\S+) reveal (\S+)',text)
        parties.append(dict(party=p,exact=bool(exact),checks_pass='Checkeqs passed!' in text,
                            total_ms=float(total[1])*1000 if total else None,
                            phases_ms=list(map(float,phases.groups())) if phases else None))
    rows.append(dict(codes=codes,parties=parties))
passed=all(r['codes']==[0]*4 and all(p['exact'] and p['checks_pass'] for p in r['parties']) for r in rows)
result=dict(passed=passed,batch=8192,bits=16,hidden=4,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            program_sha256=hashlib.sha256(Path(__file__).with_name('program.hpp').read_bytes()).hexdigest(),
            native_sha256=hashlib.sha256(native_source.read_bytes()).hexdigest(),
            security_scope='Research fixture: public deterministic seeds, plain loopback TCP; not a privacy-preserving deployment.',
            timing_scope='Total MPC includes private input/model sharing, sign-only reveal and final checks. Native excludes loading; one CPU thread. Four MPC processes on shared CPU.',
            protocol='Quad protocol12 PRE=0 CV_FIX=1',native=native,repeats=rows)
if passed:
    result['median_max_total_ms']=float(np.median([max(p['total_ms'] for p in r['parties']) for r in rows]))
    result['native_best_ms']=min(float(line.split()[-1]) for line in native.splitlines())
    result['slowdown']=result['median_max_total_ms']/result['native_best_ms']
(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='repeats'},indent=2))
raise SystemExit(0 if passed else 1)
