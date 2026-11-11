"""Compare bounded activation fits on CPU; no secure execution is timed.

Run from the repository root with ``.venv/bin/python -m
experiments.benchmarks.activations.activation_approximation_quality``. Results include floating-point
approximation error and a deterministic Q14 floor-truncation simulation.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from threadpoolctl import threadpool_limits
from rep4gpu.arithmetic.activations import _GELU, _SILU, quadratic_plan

SCALE = 1 << 14


def reference(x, kind):
    values = torch.from_numpy(np.asarray(x, dtype=np.float64))
    gate = .5 * (1 + torch.erf(values / np.sqrt(2))) if kind == 'gelu' else torch.sigmoid(values)
    return x * gate.numpy()


def metrics(actual, expected):
    error = actual - expected
    return dict(mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                max_abs=float(np.max(abs(error))),
                relative_l2=float(np.linalg.norm(error.ravel()) / np.linalg.norm(expected.ravel())),
                **(dict(top1_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1)))) if actual.ndim == 2 else {}))


def fit(kind, degree, profile):
    x = np.linspace(-4, 4, 16001)
    t = x / 4
    design = np.stack([t**power for power in range(2, degree + 1, 2)], axis=1)
    weights = np.ones_like(x) if profile == 'uniform' else np.exp(-.5 * (x / .5)**2)
    if profile == 'mixed':
        weights = .01 + .99 * weights
    # Gaussian weighting intentionally sacrifices tails; all fits retain x/2.
    root = np.sqrt(weights)
    even = np.linalg.lstsq(design * root[:, None], (reference(x, kind) - x / 2) * root,
                           rcond=None)[0]
    coefficients = np.zeros(degree + 1)
    coefficients[1] = 2
    coefficients[2::2] = even
    return coefficients


def evaluate(x, coefficients, fixed=False):
    if not fixed:
        return np.polynomial.polynomial.polyval(x / 4, coefficients), {}
    if not np.all(np.isfinite(x)) or np.max(abs(x)) > 4:
        raise ValueError("fixed evaluation requires finite inputs in [-4,4]")
    encoded = np.rint(x * SCALE).astype(np.int64)
    internal_bits = 14
    coefficient_bound = float(np.sum(abs(coefficients[2::2])))
    while (coefficient_bound + 2) * 2**(2 * internal_bits) >= 2**30:
        internal_bits -= 1
    internal_scale = 1 << internal_bits
    t = encoded // (4 * (SCALE // internal_scale))
    raw_square = t * t
    square = raw_square // internal_scale
    c = np.rint(coefficients * internal_scale).astype(np.int64)
    value = np.full_like(t, c[-1])
    max_product = int(abs(raw_square).max())
    max_state = int(abs(value).max())
    for coefficient in c[-3:1:-2]:
        product = value * square
        max_product = max(max_product, int(abs(product).max()))
        value = product // internal_scale + coefficient
        max_state = max(max_state, int(abs(value).max()))
    product = value * square
    max_product = max(max_product, int(abs(product).max()))
    result = encoded // 2 + (product // internal_scale + c[0]) * (SCALE // internal_scale)
    return result / SCALE, dict(internal_fractional_bits=internal_bits,
                               conservative_product_bound=(coefficient_bound + 2) * 2**(2 * internal_bits),
                               conservative_strict_quarter_ring32_fit=True,
                               max_abs_raw_product=max_product,
                               max_abs_horner_state=max_state,
                               signed_ring32_products_fit=max_product < 2**31,
                               strict_quarter_ring32_products_fit=max_product < 2**30,
                               signed_ring64_products_fit=max_product < 2**63)


def evaluate_quadratic(x, kind, profile):
    if not np.all(np.isfinite(x)) or np.max(abs(x)) > 4:
        raise ValueError("quadratic evaluation requires finite inputs in [-4,4]")
    plan = quadratic_plan(kind, 4, fractional_bits=14, bits=32, profile=profile)
    encoded = np.rint(x * SCALE).astype(np.int64)
    normalized = encoded // (1 << plan.prescale)
    numerator = (plan.quadratic_coeff * normalized * normalized +
                 plan.linear_coeff * normalized + plan.constant_numerator)
    assert np.max(abs(numerator)) <= plan.numerator_abs_bound < 2**30
    return (numerator // (1 << plan.shift)) / SCALE, dict(
        schedule='quadratic_plan', prescale=plan.prescale, shift=plan.shift,
        conservative_numerator_bound=plan.numerator_abs_bound,
        max_abs_numerator=int(np.max(abs(numerator))),
        conservative_strict_quarter_ring32_fit=True,
        strict_quarter_ring32_products_fit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/results/activations/activation_approximation_quality.json')
    args = parser.parse_args()
    rng = np.random.default_rng(20261110)
    fixture_rng = np.random.default_rng(20261109)
    batch, width, hidden_width, out = 128, 2048, 8192, 2048
    values = [fixture_rng.normal(0, .5, (batch, width)),
              fixture_rng.normal(0, 1 / np.sqrt(width), (width, hidden_width)),
              fixture_rng.normal(0, .05, hidden_width),
              fixture_rng.normal(0, 1 / np.sqrt(hidden_width), (hidden_width, out)),
              fixture_rng.normal(0, .05, out)]
    # Match fixture()'s float32 conversion, then isolate activation errors in float64.
    x, w1, b1, w2, b2 = [v.astype(np.float32).astype(np.float64) for v in values]
    hidden = x @ w1 + b1
    del x, w1, values
    gate = rng.normal(0, .5, hidden.shape)
    projection = w2
    samples = {'dense_grid': np.linspace(-4, 4, 65537),
               'gaussian_std_0_5': rng.normal(0, .5, 100000), 'fixture_hidden': hidden}
    report = dict(seed=20261110, interval=[-4, 4], fractional_bits=14,
                  fit_grid_points=16001, fixture_shapes=[[batch,width],[width,hidden_width],[hidden_width],[hidden_width,out],[out]],
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path("experiments/benchmarks/inference/inference.py"), Path("src/rep4gpu/arithmetic/activations.py")]},
                  projection_precision="Float64 projection and bias; q14_activation_only quantizes only the activation input/evaluation, not matmuls or weights",
                  distributions={name: dict(count=int(v.size), min=float(v.min()), max=float(v.max()),
                                             std=float(v.std())) for name, v in samples.items()},
                  method='Least squares of f(x)-x/2 in (x/4)^2,...,(x/4)^degree; constant is zero.',
                  fixed_point='Q14 nearest-even input encoding; adaptive internal precision chosen by (sum(abs(even coefficients))+2)*2^(2*f)<2^30. Nearest-even coefficients and floor normalization, squares and every Horner product, then rescale even output to Q14. CPU integer model, not protocol execution.',
                  caveats=['Quadratic baselines use quadratic_plan ring32 prescaling and floor truncation; polynomial candidates use normalized even Horner.',
                           'Measured intermediates are sample bounds. The conservative coefficient-sum bound applies only to abs(x)<=4 and includes two real units of rounding slack; external Q14 products without internal downscaling can overflow ring32.',
                           'Probabilistic truncation is not simulated; its rounding distribution can change errors.',
                           'FFN and SwiGLU are synthetic, untrained fixtures; output errors do not establish model accuracy.',
                           'SwiGLU uses independent Gaussian gate/value fixtures and a random projection; no secure runtime timings.'],
                  activations={})
    for kind, existing in [('gelu', _GELU), ('silu', _SILU)]:
        local = np.array([0, 2, 6 if kind == 'gelu' else 4.])
        uniform = np.array([951 / 4096 if kind == 'gelu' else 427 / 4096, 2, 2.])
        candidates = {'local_quadratic': local, 'uniform_quadratic': uniform,
                      'existing_degree10': np.array(existing)}
        candidates.update({f'{profile}_degree{degree}': fit(kind, degree, profile)
                           for profile in ['uniform', 'gaussian_std_0_5', 'mixed'] for degree in [4, 6, 8, 10]})
        rows = {}
        for name, coefficients in candidates.items():
            row = dict(normalized_coefficients=coefficients.tolist(), degree=len(coefficients)-1,
                       fixed_schedule=('repository quadratic_plan ring32 floor' if name.endswith('_quadratic') else 'bounded normalized even Horner, adaptive internal precision, floor'),
                       coefficient_source=('rep4gpu.arithmetic.activations existing constants; evaluated with bounded adaptive schedule, not _smooth clipping/comparison path' if name == 'existing_degree10' else 'local definition' if name.endswith('_quadratic') else 'least squares fit'),
                       output_at_zero=float(coefficients[0]), datasets={})
            for dataset, inputs in samples.items():
                expected = reference(inputs, kind)
                real, _ = evaluate(inputs, coefficients)
                fixed, bounds = (evaluate_quadratic(inputs, kind, name.split('_')[0])
                                 if name.endswith('_quadratic') else evaluate(inputs, coefficients, True))
                row['datasets'][dataset] = dict(float=metrics(real, expected), q14=metrics(fixed, expected),
                                                q14_vs_float=metrics(fixed, real), intermediates=bounds)
                if dataset == 'fixture_hidden':
                    row['ffn_output'] = dict(float=metrics(real @ w2 + b2, expected @ w2 + b2),
                                             q14_activation_only=metrics(fixed @ w2 + b2, expected @ w2 + b2))
            if kind == 'silu':
                exact = (reference(gate, kind) * hidden) @ projection
                real, _ = evaluate(gate, coefficients)
                fixed, _ = (evaluate_quadratic(gate, kind, name.split('_')[0])
                            if name.endswith('_quadratic') else evaluate(gate, coefficients, True))
                row['synthetic_swiglu_output'] = dict(float=metrics((real * hidden) @ projection, exact),
                                                      q14_activation_only=metrics((fixed * hidden) @ projection, exact))
            rows[name] = row
        report['activations'][kind] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)
    for kind, rows in report['activations'].items():
        print(kind)
        for name, row in rows.items():
            print(name, 'grid max', row['datasets']['dense_grid']['q14']['max_abs'],
                  'gaussian rmse', row['datasets']['gaussian_std_0_5']['q14']['rmse'],
                  'FFN rmse', row['ffn_output']['q14_activation_only']['rmse'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
