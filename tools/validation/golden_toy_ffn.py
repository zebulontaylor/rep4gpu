#!/usr/bin/env python3
"""Compare native MP-SPDZ trained toy MLP logits with the exact Q12 reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def program(bits, batch):
    compose_hidden = 'sint.bit_compose([sint.conv(bit & keep) for bit in first_bits], signed=True)' if bits == 32 else 'sint.bit_compose([bit & keep for bit in first_bits], signed=True)'
    compose_output = 'sint.bit_compose([sint.conv(bit) for bit in second_bits], signed=True)' if bits == 32 else 'sint.bit_compose(second_bits, signed=True)'
    return f'''from Compiler.types import sint
from Compiler.library import print_ln, check_point
# Party 0 supplies contexts; party 1 supplies the trained model.
x = sint.Matrix({batch}, 16)
w1 = sint.Matrix(64, 16)
b1 = sint.Array(64)
w2 = sint.Matrix(8, 64)
b2 = sint.Array(8)
x.input_from(0)
w1.input_from(1)
b1.input_from(1)
w2.input_from(1)
b2.input_from(1)
first = x.direct_mul_trans(w1) + (sint.concat([b1.get_vector()] * {batch}) << 12)
first_bits = first.bit_decompose({bits}, maybe_mixed=True)[12:]
keep = first_bits[-1].bit_not()
hidden = sint.Matrix({batch}, 64)
hidden.assign_vector({compose_hidden})
second = hidden.direct_mul_trans(w2)
second_bits = second.bit_decompose({bits}, maybe_mixed=True)[12:]
output = {compose_output} + sint.concat([b2.get_vector()] * {batch})
revealed = output.reveal()
check_point()
for row in range({batch}):
    for column in range(8):
        print_ln('LOGIT %s %s %s', row, column, revealed[row * 8 + column])
'''


def run(model_path, mpspdz, bits, directory, context_count, timeout):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch import nn
    from experiments.examples.toy_next_token import dataset, quantized_fixture
    torch.set_num_threads(1)
    model_path, mpspdz, directory = map(lambda p: Path(p).resolve(), (model_path, mpspdz, directory))
    directory.mkdir(parents=True, exist_ok=True)
    model = nn.Sequential(nn.Linear(16,64), nn.ReLU(), nn.Linear(64,8))
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    encoded, expected_all, targets_all, proof = quantized_fixture(model, fractional_bits=12, bits=bits)
    indices = [0,7,56,63] if context_count == 4 else list(range(64))
    contexts, _, _ = dataset()
    x,w1,b1,w2,b2 = encoded
    x,expected = x[indices], expected_all[indices]
    arrays = [x,w1.T.contiguous(),b1,w2.T.contiguous(),b2]
    inputs = [[int(v) for v in x.reshape(-1)], [int(v) for array in arrays[1:] for v in array.reshape(-1)], [], []]
    fixture = dict(context_indices=indices, contexts=contexts[indices].tolist(), targets=targets_all[indices].tolist(),
                   owners=[0,1,1,1,1], shapes=[list(array.shape) for array in arrays],
                   names=['context_input','weight1_transposed','bias1','weight2_transposed','bias2'],
                   arrays=[array.tolist() for array in arrays], expected_logits=expected.tolist())
    fixture_text = json.dumps(fixture, indent=2)+'\n'
    (directory/'fixture.json').write_text(fixture_text)
    for p in range(4):
        (directory/f'Input-P{p}-0.txt').write_text(' '.join(map(str,inputs[p]))+'\n')
    source = program(bits,len(indices))
    (directory/'reference.mpc').write_text(source)
    executable = mpspdz/'rep4-ring-party.x'
    compiler_command = [sys.executable,str(mpspdz/'compile.py'),'-R',str(bits),'-Z','4','toy_ffn_reference']
    compile_code, codes, timed_out = None, [], False
    with tempfile.TemporaryDirectory(prefix='rep4-trained-ffn-') as temporary:
        work=Path(temporary)
        (work/'Programs/Source').mkdir(parents=True)
        (work/'Player-Data').mkdir()
        (work/'Programs/Source/toy_ffn_reference.mpc').write_text(source)
        for cert in (mpspdz/'Player-Data').iterdir():
            if cert.suffix in ('.pem','.key','.0'):
                shutil.copy(cert,work/'Player-Data'/cert.name)
        for p in range(4):
            shutil.copy(directory/f'Input-P{p}-0.txt',work/f'Player-Data/Input-P{p}-0')
        compiled=subprocess.run(compiler_command,cwd=work,capture_output=True,text=True)
        compile_code=compiled.returncode
        (directory/'compile.log').write_text(compiled.stdout+compiled.stderr)
        if compile_code==0:
            logs,processes=[],[]
            try:
                for p in range(4):
                    log=open(directory/f'party{p}.log','w')
                    logs.append(log)
                    processes.append(subprocess.Popen([str(executable),str(p),'toy_ffn_reference','-R',str(bits),
                        '-v','-pn',str(18900+bits)],cwd=work,stdout=log,stderr=subprocess.STDOUT))
                deadline=time.monotonic()+timeout
                for process in processes:
                    process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out=True
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                codes=[process.returncode for process in processes]
                for log in logs:
                    log.close()
    party0=(directory/'party0.log').read_text() if (directory/'party0.log').exists() else ''
    observed=[[None]*8 for _ in indices]
    records=re.findall(r'^LOGIT (\d+) (\d+) (-?\d+)$',party0,re.MULTILINE)
    duplicate=False
    for row,column,value in records:
        row,column,value=map(int,(row,column,value))
        if row>=len(indices) or column>=8 or observed[row][column] is not None:
            duplicate=True
        else:
            observed[row][column]=value
    match=not duplicate and observed==expected.tolist() and len(records)==len(indices)*8
    traffic={name:float(m.group(1)) for name,pattern in [('party0_sent_mb',r'^Data sent = ([0-9.e+-]+) MB'),
                ('global_sent_mb',r'^Global data sent = ([0-9.e+-]+) MB')] if (m:=re.search(pattern,party0,re.MULTILINE))}
    passed=compile_code==0 and codes==[0]*4 and not timed_out and match
    summary=dict(passed=passed, ring_bits=bits, fractional_bits=12, truncation='signed floor; no half-up offset',
                 network='16 -> 64 ReLU -> 8 logits', input_owner=0,model_owner=1,
                 context_count=len(indices),logit_count=len(records),context_indices=indices,
                 compile_command=compiler_command,compile_return_code=compile_code,return_codes=codes,timed_out=timed_out,
                 checkpoint=str(model_path),checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 executable=str(executable),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                 reference_example_sha256=hashlib.sha256((ROOT/'experiments/examples/toy_next_token.py').read_bytes()).hexdigest(),
                 program_sha256=hashlib.sha256(source.encode()).hexdigest(),fixture_sha256=hashlib.sha256(fixture_text.encode()).hexdigest(),
                 source_sha256={name:hashlib.sha256((mpspdz/name).read_bytes()).hexdigest() for name in
                    ['Protocols/Rep4.hpp','Protocols/Rep4Input.hpp','Protocols/Rep4MC.hpp','GC/Rep4Secret.h']},
                 binary_to_arithmetic='generic per-bit conversion' if bits==32 else 'optimized masked compose',
                 range_proof=proof,expected_logits=expected.tolist(),observed_logits=observed,
                 exact_integer_match=match,traffic=traffic,
                 evaluation='Synthetic all-context trained fixture; not held-out language-model accuracy.')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(bits=bits,passed=passed,contexts=len(indices),logits=len(records),return_codes=codes)),flush=True)
    return 0 if passed else (compile_code or next((code for code in codes if code),1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ROOT/'experiments/results/examples/toy_next_token/model.pt')
    parser.add_argument('--mpspdz32',type=Path,default=ROOT/'.local/work/mpspdz32')
    parser.add_argument('--mpspdz64',type=Path,default=Path('/home/zeb/Desktop/mp-spdz-toy/MP-SPDZ'))
    parser.add_argument('--contexts',type=int,choices=(4,64),default=64)
    parser.add_argument('--output',type=Path,default=ROOT/'tests/fixtures/mpspdz')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    if args.timeout<=0:
        parser.error('timeout must be positive')
    codes=[run(args.model,path,bits,args.output/f'toy_ffn{bits}',args.contexts,args.timeout)
           for bits,path in [(32,args.mpspdz32),(64,args.mpspdz64)]]
    return next((code for code in codes if code),0)


if __name__=='__main__':
    sys.exit(main())
