#!/usr/bin/env python3
"""Run the pinned local MP-SPDZ executable without changing its source tree."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

def signed(value, bits=64):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpspdz', type=Path, default=Path('/home/zeb/Desktop/mp-spdz-toy/MP-SPDZ'))
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/mpspdz/mpspdz'))
    parser.add_argument('--ring', type=int, choices=(32, 64), default=64)
    parser.add_argument('--split', type=int, choices=(2, 4), default=2)
    parser.add_argument('--port', type=int, default=18640)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    executable = args.mpspdz.resolve() / 'rep4-ring-party.x'
    ring_mask = (1 << args.ring) - 1
    a_values = [[(1 << (args.ring-1))-1, -7, 8193], [-(1 << (args.ring-1)), 4096, -4097]]
    b_values = [[3, -5, 2048], [-1, 1 << (args.ring//2), -8192]]
    x_values = [-4097, -2049, -2048, -1, 0, 2048, (1 << (args.ring-1))-1, -(1 << (args.ring-1))]
    expected = [f'MAT {i} {j} {signed(sum(a*b for a,b in zip(a_values[i], b_values[j])), args.ring)}'
                for i in range(2) for j in range(2)]
    for i, x in enumerate(x_values):
        scaled = signed(((x + 2048) & ring_mask) >> 12, args.ring-12)
        expected.append(f'Q12 {i} {scaled} {max(0, scaled)}')
    with tempfile.TemporaryDirectory(prefix='siv-mpspdz-') as temporary:
        work = Path(temporary)
        (work / 'Programs/Source').mkdir(parents=True)
        (work / 'Player-Data').mkdir()
        source = (root / 'tests/fixtures/mpspdz/rep4_reference.mpc').read_text()
        source = source.replace('bit_decompose(64,', f'bit_decompose({args.ring},')
        if args.ring == 32:
            # The optimized masked compose path in this MP-SPDZ snapshot fails
            # its 32-bit ring edge cases. Convert each bit with the generic path.
            source = source.replace('sint.bit_compose(bits, signed=True)', 'sint.bit_compose([sint.conv(bit) for bit in bits], signed=True)')
            source = source.replace('sint.bit_compose([bit & positive for bit in bits], signed=True)', 'sint.bit_compose([sint.conv(bit & positive) for bit in bits], signed=True)')
        (work / 'Programs/Source/rep4_reference.mpc').write_text(source)
        (out / 'reference.mpc').write_text(source)
        # Use the installed local test certificates in this isolated run directory.
        for certificate in (args.mpspdz / 'Player-Data').iterdir():
            if certificate.suffix in ('.pem', '.key', '.0'):
                shutil.copy(certificate, work / 'Player-Data' / certificate.name)
        inputs = [[v for row in a_values for v in row] + x_values, [v for row in b_values for v in row], [], []]
        for p, values in enumerate(inputs):
            (work / f'Player-Data/Input-P{p}-0').write_text(' '.join(map(str, values)) + '\n')
        command = [sys.executable, str(args.mpspdz.resolve() / 'compile.py'), '-R', str(args.ring), '-Z', str(args.split), 'rep4_reference']
        compile_result = subprocess.run(command, cwd=work, capture_output=True, text=True)
        (out / 'compile.log').write_text(compile_result.stdout + compile_result.stderr)
        if compile_result.returncode:
            return compile_result.returncode
        logs = [open(out / f'party{p}.log', 'w') for p in range(4)]
        processes = [subprocess.Popen([str(executable), str(p), 'rep4_reference', '-R', str(args.ring), '-v', '-pn', str(args.port)],
                                     cwd=work, stdout=logs[p], stderr=subprocess.STDOUT) for p in range(4)]
        try:
            codes = [process.wait(timeout=180) for process in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
        observed = [line for line in (out / 'party0.log').read_text().splitlines() if line.startswith(('MAT ', 'Q12 '))]
        sources = ['Protocols/Rep4.hpp', 'Protocols/Rep4Input.hpp', 'Protocols/Rep4MC.hpp', 'Protocols/Rep4Share.h', 'GC/Rep4Secret.h']
        party0 = (out / 'party0.log').read_text()
        traffic = {name: float(match.group(1)) for name, pattern in [('party0_sent_mb', r'^Data sent = ([0-9.e+-]+) MB'), ('global_sent_mb', r'^Global data sent = ([0-9.e+-]+) MB')] if (match := re.search(pattern, party0, re.MULTILINE))}
        summary = dict(ring_bits=args.ring, split=args.split,
                       binary_to_arithmetic='generic per-bit conversion' if args.ring == 32 else 'optimized masked compose',
                       program_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       inputs=dict(a=a_values, b=b_values, q12=x_values), traffic=traffic, executable=str(executable), executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                       source_sha256={name: hashlib.sha256((args.mpspdz / name).read_bytes()).hexdigest() for name in sources},
                       compile_command=command, return_codes=codes, expected=expected, observed=observed,
                       passed=all(code == 0 for code in codes) and observed == expected)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return next((code for code in codes if code), 0) or (0 if summary['passed'] else 1)


if __name__ == '__main__':
    sys.exit(main())
