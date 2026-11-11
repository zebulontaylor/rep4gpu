"""Record each live SiLU optimization stage before applying the next stage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.witness_optimization import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    row = run_case('after','cuda','silu',elements=14336)
    row['stage'] = args.stage
    row['operation_profile_note'] = (
        'Runtime-method instrumentation only: bounded floors call a standalone '
        'protocol, so their internal multiplications appear in mul. Use '
        'silu_exact_accounting.json for semantic operation counts and bytes.')
    paths = ['rep4gpu/numerical.py','rep4gpu/witness_activations.py',
             'rep4gpu/witness_prover.py','rep4gpu/exact_truncation.py',
             'rep4gpu/silu_exp.py','rep4gpu/bounded_exact_truncation_gpu.py']
    row['source_sha256'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    target = Path('results/silu_four_step.json')
    report = json.loads(target.read_text()) if target.exists() else {'results':[]}
    report['results'] = [old for old in report['results'] if old['stage'] != args.stage]+[row]
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row,indent=2))


if __name__ == '__main__':
    main()
