"""Compare shared carry prefixes and proved-range sign extraction on CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

import benchmarks.witness_optimization as harness
from rep4gpu.device_inference import DeviceInferenceParty
from rep4gpu import witness_activations, witness_prover

BASELINE = Path('results/carry_prefix_baseline_sources')


def load_baseline():
    result = {}
    for name in ('carry_tree_swar', 'numerical', 'witness_activations'):
        full_name = 'rep4gpu._prefix_before_' + name
        spec = importlib.util.spec_from_file_location(full_name, BASELINE/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        result[name] = module
    result['witness_activations'].NumericalFixedPoint = result['numerical'].NumericalFixedPoint
    return result


def compare(kind, elements=257, rows=2, width=256):
    old = load_baseline()
    original_options = DeviceInferenceParty._conversion_options
    original_implementations = harness.implementations
    results = []
    try:
        for version in ('before', 'after'):
            if version == 'before':
                def options(party):
                    selected = original_options(party)
                    if party.optimized:
                        selected['carry_impl'] = old['carry_tree_swar'].truncation_bits
                    return selected
                DeviceInferenceParty._conversion_options = options
                legacy_prover = SimpleNamespace(
                    numerical_witnesses=lambda kind, inputs: witness_prover.numerical_witnesses(
                        kind, inputs, direct_output=False),
                    rmsnorm_witnesses=witness_prover.rmsnorm_witnesses,
                    softmax_witnesses=witness_prover.softmax_witnesses)
                harness.implementations = lambda _: (old['witness_activations'], legacy_prover)
            else:
                DeviceInferenceParty._conversion_options = original_options
                harness.implementations = lambda _: (witness_activations, witness_prover)
            results.append(harness.run_case(version, 'cuda', kind, elements, rows, width))
    finally:
        DeviceInferenceParty._conversion_options = original_options
        harness.implementations = original_implementations
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', nargs='+', choices=['silu','gelu','softmax','rmsnorm'],
                        default=['silu','gelu','softmax','rmsnorm'])
    parser.add_argument('--elements', type=int, default=257)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--output', type=Path, default=Path('results/carry_prefix_optimization.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [BASELINE/(name+'.py') for name in ('carry_tree_swar','numerical','witness_activations')]
    sources += [Path('rep4gpu')/(name+'.py') for name in ('carry_tree_swar','numerical','witness_activations',
                'witness_prover','bounded_activation','device_inference','compact_conversion_gpu','exact_truncation')]
    sources += [Path('benchmarks/witness_optimization.py'),Path(__file__)]
    report = dict(methodology='Four live optimized CUDA Rep4 roles, exact floor ring64 Q24. '
        'Baseline precedes shared-carry truncation and proved-domain narrow signs. Current SiLU '
        'also includes the direct-output Remez and bounded-floor changes. Global bytes include witness ingestion and acceptance; '
        'exclude input sharing and final validation output opening. No distributed timing claim.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for kind in args.kind:
        for row in compare(kind,args.elements,args.rows,args.width):
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
