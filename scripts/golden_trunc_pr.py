#!/usr/bin/env python3
"""Validate actual MP-SPDZ Rep4 small-gap probabilistic truncation in both rings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0),
            (3, 0, 2, 1), (0, 2, 3, 1), (1, 3, 2, 0))


def cases(bits):
    limit = 1 << (bits-2)
    return [-limit+1, -limit+4096, -8193, -8192, -8191, -4097,
            -4096, -4095, -1, 0, 1, 4095, 4096, 4097, 8191,
            8192, 8193, limit-4096, limit-1]


def topology(bits, count):
    """Source-derived payload routes for the three trunc_pr phases."""
    width = bits // 8
    masked = {(0, 2): count*width, (0, 3): 32,
              (1, 3): count*width, (1, 2): 32}
    inputs = {(0, 3): 2*count*width, (1, 3): 32,
              (2, 1): 2*count*width, (3, 1): 32}
    products = {}
    for sender, backup, receiver, _ in SCHEDULE:
        products[sender, receiver] = products.get((sender, receiver), 0) + count*width
        products[backup, receiver] = products.get((backup, receiver), 0) + 32
    return [masked, inputs, products]


def runtime_trace(bits, values):
    """Compare the runtime's real payload routes/counts with the source mapping."""
    sys.path.insert(0, str(ROOT))
    import torch
    from rep4gpu.probabilistic import trunc_pr
    from rep4gpu.runtime import Party, make_seeds
    from rep4gpu.transport import IdealNetwork
    net, seeds = IdealNetwork(), make_seeds()
    tensors = torch.tensor(values, dtype=torch.int64)
    class Trace:
        def __init__(self, endpoint):
            self.endpoint, self.records = endpoint, []
        def exchange(self, outgoing):
            self.records.append({receiver: len(data) for receiver, data in outgoing.items()})
            return self.endpoint.exchange(outgoing)
    traces = [Trace(net.endpoint(p)) for p in range(4)]
    def work(p):
        party = Party(p, traces[p], {e: seed for e, seed in seeds.items() if e != p}, bits=bits)
        x = party.input(0, tensors if p == 0 else None, tensors.shape)
        traces[p].records.clear()
        result = trunc_pr(party, x, 12)
        return result
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(4)))
    actual = [{(p, receiver): size for p in range(4) for receiver, size in traces[p].records[phase].items()}
              for phase in range(3)]
    expected = topology(bits, len(values))
    assert actual == expected, (actual, expected)
    return [dict(phase=name, edges=[dict(sender=p, receiver=q, bytes=size) for (p,q),size in sorted(edges.items())],
                 global_payload_bytes=sum(edges.values()))
            for name,edges in zip(('masked partial opening', 'msb and upper joint inputs', 'msb correction multiplication'), actual)]


def run(mpspdz, bits, output, repeats, port, timeout):
    mpspdz, output = Path(mpspdz).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = cases(bits) * repeats
    source = f'''from Compiler.types import sint
from Compiler.instructions import trunc_pr
from Compiler.library import print_ln, check_point
x = sint.get_input_from(0, size={len(values)})
y = sint(size={len(values)})
# Direct opcode selects Rep4<T>::trunc_pr, including its signed small-gap bias.
trunc_pr(y, x, {bits-1}, 12)
revealed = y.reveal()
check_point()
for index in range({len(values)}):
    print_ln('TRUNC %s %s', index, revealed[index])
'''
    (output/'reference.mpc').write_text(source)
    executable = mpspdz/'rep4-ring-party.x'
    compile_code, codes, timed_out = None, [], False
    with tempfile.TemporaryDirectory(prefix='rep4-trunc-pr-') as temporary:
        work = Path(temporary)
        (work/'Programs/Source').mkdir(parents=True)
        (work/'Player-Data').mkdir()
        (work/'Programs/Source/trunc_pr_reference.mpc').write_text(source)
        for cert in (mpspdz/'Player-Data').iterdir():
            if cert.suffix in ('.pem', '.key', '.0'):
                shutil.copy(cert, work/'Player-Data'/cert.name)
        for p in range(4):
            (work/f'Player-Data/Input-P{p}-0').write_text(' '.join(map(str, values)) if p == 0 else '\n')
        command = [sys.executable, str(mpspdz/'compile.py'), '-R', str(bits), 'trunc_pr_reference']
        compiled = subprocess.run(command, cwd=work, capture_output=True, text=True)
        compile_code = compiled.returncode
        (output/'compile.log').write_text(compiled.stdout+compiled.stderr)
        if compile_code == 0:
            logs, processes = [], []
            try:
                for p in range(4):
                    log = open(output/f'party{p}.log', 'w')
                    logs.append(log)
                    processes.append(subprocess.Popen([str(executable), str(p), 'trunc_pr_reference', '-R', str(bits),
                        '-v', '--code-locations', '-pn', str(port)], cwd=work, stdout=log, stderr=subprocess.STDOUT))
                deadline = time.monotonic()+timeout
                for process in processes:
                    process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                codes = [process.returncode for process in processes]
                for log in logs:
                    log.close()
    observed = []
    log_text = (output/'party0.log').read_text() if (output/'party0.log').exists() else ''
    for index, value in re.findall(r'^TRUNC (\d+) (-?\d+)$', log_text, re.MULTILINE):
        observed.append((int(index), int(value)))
    validation = []
    for index, value in observed:
        original = values[index]
        floor = original >> 12
        ceiling = floor + int(original % 4096 != 0)
        validation.append(dict(index=index, input=original, output=value, floor=floor, ceiling=ceiling,
                               exact_multiple=original % 4096 == 0, passed=value in (floor, ceiling)))
    phases = runtime_trace(bits, values)
    protocol = (mpspdz/'Protocols/Rep4.hpp').read_text()
    # These source clauses identify the small-gap branch and its exact tuples.
    clauses = ['P.send_to(2 + P.my_num(), c_os)', 'P.send_to(3 - P.my_num(), c_os.hash())',
               'prepare_joint_input(0, 1, 3, 2, inputs, gen_results)',
               'prepare_joint_input(2, 3, 1, 0, eval_inputs, eval_results)',
               'prepare_mul(gen_results.next().res, eval_results.next().res)',
               '- T::constant(info.subtract_after(), P.my_num())']
    source_topology_match = all(clause in protocol for clause in clauses)
    reached_native = 'Rep4<T>::trunc_pr' in log_text or 'void Rep4<' in log_text and '::trunc_pr' in log_text
    traffic = {name: float(match.group(1)) for name, pattern in [('party0_sent_mb', r'^Data sent = ([0-9.e+-]+) MB'),
                ('global_sent_mb', r'^Global data sent = ([0-9.e+-]+) MB')] if (match := re.search(pattern, log_text, re.MULTILINE))}
    passed = (compile_code == 0 and codes == [0]*4 and not timed_out and len(observed) == len(values)
              and [index for index,_ in observed] == list(range(len(values)))
              and all(row['passed'] for row in validation) and source_topology_match and reached_native)
    summary = dict(passed=passed, ring_bits=bits, k=bits-1, shift=12,
                   range_contract=f'abs(signed(x)) < 2**{bits-2}', repeats=repeats,
                   compile_command=command, compile_return_code=compile_code, return_codes=codes,
                   timed_out=timed_out, executable=str(executable), executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                   program_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   source_sha256={name: hashlib.sha256((mpspdz/name).read_bytes()).hexdigest()
                        for name in ['Protocols/Rep4.hpp','Processor/TruncPrTuple.h','Tools/Hash.cpp']},
                   native_trunc_pr_reached=reached_native, source_topology_match=source_topology_match,
                   topology_audit='Source-derived MP-SPDZ phase routes compared with actual runtime outgoing edge byte counts; MP-SPDZ native opcode execution confirmed with code-location logs. No packet capture.',
                   runtime_topology_case_count=len(values), runtime_phases=phases, traffic=traffic,
                   joint_input_order='MP-SPDZ interleaves (msb,upper) per value; runtime stacks the two planes. This is a semantic batch permutation with the same tuples, routes, and byte counts.',
                   exact_multiple_count=sum(row['exact_multiple'] for row in validation),
                   exact_multiples_match=all(row['output'] == row['floor'] for row in validation if row['exact_multiple']), observed=validation)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(bits=bits, passed=passed, count=len(observed), return_codes=codes, native_trunc_pr_reached=reached_native)))
    return 0 if passed else (compile_code or next((code for code in codes if code), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpspdz64', type=Path, default=Path('/home/zeb/Desktop/mp-spdz-toy/MP-SPDZ'))
    parser.add_argument('--mpspdz32', type=Path, default=ROOT/'work/mpspdz32')
    parser.add_argument('--output', type=Path, default=ROOT/'tests/golden')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('repeats and timeout must be positive')
    codes = [run(path, bits, args.output/f'trunc_pr{bits}', args.repeats, 18700+bits, args.timeout)
             for bits,path in [(32,args.mpspdz32),(64,args.mpspdz64)]]
    return next((code for code in codes if code), 0)


if __name__ == '__main__':
    sys.exit(main())
