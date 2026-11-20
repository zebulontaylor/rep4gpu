"""Compare shared one-layer activations and a simulated helper at output bit level.

The helper is ordinary CUDA, not a TEE. Floating formats are output casts;
separate FP16/BF16 baselines execute native PyTorch layers on the GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import secrets
import time

import numpy as np
import torch

from experiments.benchmarks.activations.float_bit_metrics import compare_formats
from experiments.benchmarks.activations.numerical_primitives import IntegerOracle, Interval, evaluate
from experiments.benchmarks.activations.seeded_tee_simulation import clear_activation, seeded_tee_party
from experiments.benchmarks.inference.numerical_inference_common import ForwardOracle
from rep4gpu.arithmetic.numerical import NumericalFixedPoint
from rep4gpu.models.phase_graph import PhaseGraphParty
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork

KINDS = ('relu', 'silu', 'gelu', 'swiglu')


def activation(kind, gate, value=None):
    if kind == 'relu':
        return torch.relu(gate)
    if kind == 'gelu':
        return torch.nn.functional.gelu(gate, approximate='none')
    result = torch.nn.functional.silu(gate)
    return result * value if kind == 'swiglu' else result


def fixture(suite, f, batch, width, hidden, grid_elements):
    scale = 1 << f
    if suite == 'dense':
        rng = np.random.default_rng(20261111)
        arrays = [rng.normal(0, .5, (batch, width)),
                  rng.normal(0, 1/np.sqrt(width), (width, hidden)),
                  rng.normal(0, .05, hidden),
                  rng.normal(0, 1/np.sqrt(width), (width, hidden)),
                  rng.normal(0, .05, hidden)]
        floats = [torch.from_numpy(a) for a in arrays]
        integers = [torch.round(a*scale).long() for a in floats]
        rt = ForwardOracle()
        iv = [rt.public(a) for a in integers]
        raw = [rt.add(rt.matmul(iv[0], iv[i]), rt.mul_public(iv[i+1], scale)).lo for i in (1, 3)]
        original = [floats[0] @ floats[i] + floats[i+1] for i in (1, 3)]
        decoded = [a.double()/scale for a in integers]
        quantized = [decoded[0] @ decoded[i] + decoded[i+1] for i in (1, 3)]
        proof = {'maximum_accumulator_bound': rt.maximum}
    else:
        grid = torch.round(torch.linspace(-4, 4, grid_elements, dtype=torch.float64)*scale).long()
        near_zero = torch.arange(-64, 65, dtype=torch.int64)
        near_endpoints = torch.cat((torch.arange(0, 17)-4*scale, 4*scale-torch.arange(0, 17)))
        # Offset uniform-grid points to avoid testing only low-precision-friendly dyadics.
        jitter = torch.round(torch.linspace(-4, 4, grid_elements, dtype=torch.float64)*scale).long()
        jitter = (jitter + 1).clamp(-4*scale, 4*scale)
        gate = torch.cat((grid, jitter, near_zero, near_endpoints)).reshape(1, -1)
        value = torch.round(torch.linspace(4, -4, gate.numel(), dtype=torch.float64)*scale).long().reshape_as(gate)
        integers, floats = [gate*scale, value*scale], None
        raw = integers
        original = quantized = [a.double()/scale for a in (gate, value)]
        proof = {'maximum_accumulator_bound': max(int(a.abs().max()) for a in raw)}
    floored = [(a >> f).double()/scale for a in raw]
    if any(bool((a.abs() > 4).any()) for a in floored):
        raise ValueError('fixture exceeds supported activation domain [-4,4]')
    # Include ceil outcomes of probabilistic affine rescaling in the domain check.
    if any(bool((((a+scale-1) >> f).abs() > 4*scale).any()) for a in raw):
        raise ValueError('ceil-rescaled fixture exceeds supported activation domain')
    return {'integers': integers, 'floats': floats, 'raw': raw, 'original': original,
            'quantized': quantized, 'floored': floored, 'range': proof}


def integer_interval(kind, raw, f, probabilistic):
    rt = IntegerOracle(probabilistic)
    inputs = [rt.trunc(rt.public(a), f) for a in raw]
    if kind == 'relu':
        result = Interval(inputs[0].lo.clamp_min(0), inputs[0].hi.clamp_min(0))
    else:
        result = evaluate(rt, kind, inputs[0], inputs[1], f)
    return result


def run_backend(kind, suite, data, f, backend, repeats):
    """Run four actual Rep4 roles; all model tensors enter through private input."""
    network, seeds = IdealNetwork(timeout=300), make_seeds()
    cls = seeded_tee_party(secrets.token_bytes(32)) if backend == 'tee_simulation' else PhaseGraphParty
    rounding = 'probabilistic' if backend == 'private_probabilistic' else 'floor'
    roles = [cls(i, DeviceIdealEndpoint(network.endpoint(i), 'cuda:0'),
                 {j: s for j, s in seeds.items() if j != i}, device='cuda:0', bits=64,
                 binary_bits=64, optimized=True, gpu_open=True, rounding=rounding,
                 sparse_products=True) for i in range(4)]
    scale = 1 << f
    values = data['integers'] if kind == 'swiglu' else data['integers'][:3 if suite == 'dense' else 1]
    with ThreadPoolExecutor(4) as pool:
        def parallel(fn):
            def guarded(p):
                try:
                    return fn(p)
                except BaseException as exc:
                    network.abort(exc)
                    raise
            return list(pool.map(guarded, roles))
        shares = parallel(lambda p: [p.input(0 if i == 0 else 1,
            value if p.party == (0 if i == 0 else 1) else None, value.shape)
            for i, value in enumerate(values)])
        def forward(p):
            own = shares[p.party]
            raw = ([p.add(p.matmul(own[0], own[i]), p.mul_public(own[i+1], scale))
                    for i in ((1, 3) if kind == 'swiglu' else (1,))]
                   if suite == 'dense' else own)
            if backend == 'tee_simulation':
                result = p.tee_activation(kind, raw[0], raw[1] if kind == 'swiglu' else None, f)
            else:
                fixed = NumericalFixedPoint(p, f)
                inputs = [p.trunc(a, f) for a in raw]
                result = (fixed.swiglu(*inputs) if kind == 'swiglu'
                          else getattr(fixed, kind)(inputs[0]))
            return p.open(result)
        outputs = []
        for _ in range(repeats):
            results = parallel(forward)
            cpu = [v.cpu() for v in results]
            if not all(torch.equal(cpu[0], v) for v in cpu):
                raise AssertionError('Rep4 roles disagree')
            outputs.append(cpu[0])
    del roles, shares
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def real_metrics(actual, reference):
    error = actual-reference
    return {'max_abs_error': float(error.abs().max()), 'rmse': float(error.square().mean().sqrt())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/activations/layer_bit_accuracy.json'))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--grid-elements', type=int, default=4097)
    parser.add_argument('--fractional-bits', type=int, nargs='+', default=[24])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--suites', nargs='+', choices=['dense', 'stress'], default=['dense', 'stress'])
    args = parser.parse_args()
    if min(args.batch, args.width, args.hidden, args.grid_elements, args.repeats) < 1:
        parser.error('dimensions and repeats must be positive')
    if any(f < 20 or f > 24 for f in args.fractional_bits):
        parser.error('fractional bits must be in [20,24]')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'configuration': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'methodology': 'Four live colocated CUDA Rep4 roles. Dense fixture shares inputs, weights and biases. Stress fixture shares raw activation inputs only. TEE is an ordinary CUDA helper simulation, not an enclave. Float64 PyTorch references are numerical references, not proven correctly rounded transcendental oracles. Output-format comparisons cast both decoded Qf and float64 reference values; they do not simulate native FP8 inference. Native FP16/BF16 dense GPU baselines are reported separately. GELU uses approximate=none. All supported activation inputs lie in [-4,4]. Probabilistic repeats use fresh masks; fixture seed is fixed.',
              'results': []}
    raw_outputs = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for f in args.fractional_bits:
        for suite in args.suites:
            data = fixture(suite, f, args.batch, args.width, args.hidden, args.grid_elements)
            for kind in args.kinds:
                refs = {name: activation(kind, pair[0], pair[1])
                        for name, pair in ((k, data[k]) for k in ('original', 'quantized', 'floored'))}
                native = {}
                if suite == 'dense':
                    for name, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16)):
                        v = [a.to(device='cuda', dtype=dtype) for a in data['floats']]
                        pair = [v[0] @ v[i] + v[i+1] for i in (1,3)]
                        native[name] = activation(kind, pair[0], pair[1]).double().cpu()
                tee_word = None
                for backend in ('tee_simulation', 'private_floor', 'private_probabilistic'):
                    start = time.monotonic()
                    outputs = run_backend(kind, suite, data, f, backend, args.repeats)
                    if backend == 'tee_simulation':
                        expected = clear_activation(kind, data['raw'][0].cuda(),
                            data['raw'][1].cuda() if kind == 'swiglu' else None, f).cpu()
                        if not all(torch.equal(v, expected) for v in outputs):
                            raise AssertionError('helper differs from exact clear helper computation')
                        tee_word = outputs[0]
                    else:
                        interval = integer_interval(kind, data['raw'], f, backend == 'private_probabilistic')
                        if not all(bool(((v >= interval.lo) & (v <= interval.hi)).all()) for v in outputs):
                            raise AssertionError('private output outside integer oracle interval')
                    row = {'suite': suite, 'kind': kind, 'fractional_bits': f, 'backend': backend,
                           'elements_per_repeat': outputs[0].numel(), 'repeats': len(outputs),
                           'elapsed_seconds': time.monotonic()-start, 'party_agreement': True,
                           'integer_oracle_passed': True, 'range': data['range'],
                           'repeats_identical_words': all(torch.equal(outputs[0], v) for v in outputs),
                           'runs': []}
                    for r, word in enumerate(outputs):
                        actual = word.double()/(1 << f)
                        pairdiff = (word-tee_word).abs()
                        run = {'repeat': r, 'references': {name: {
                                    **real_metrics(actual, truth), 'formats': compare_formats(actual, truth)}
                                for name, truth in refs.items()},
                               'vs_tee': {'exact_q_words': int((word == tee_word).sum()),
                                          'max_q_lsb_error': int(pairdiff.max()),
                                          'formats': compare_formats(actual, tee_word.double()/(1 << f))},
                               'vs_native_gpu': {name: compare_formats(actual, truth)[name]
                                                 for name, truth in native.items()}}
                        row['runs'].append(run)
                        raw_outputs[f'{suite}_{kind}_q{f}_{backend}_{r}'] = word.numpy()
                    report['results'].append(row)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                    np.savez_compressed(args.output.with_suffix('.npz'), **raw_outputs)
                    print(json.dumps({k:row[k] for k in ('suite','kind','fractional_bits','backend','elapsed_seconds','repeats_identical_words')}), flush=True)
    sources = [Path(__file__), Path('experiments/benchmarks/activations/float_bit_metrics.py'),
               Path('experiments/benchmarks/activations/seeded_tee_simulation.py'),
               Path('src/rep4gpu/arithmetic/numerical.py')]
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
