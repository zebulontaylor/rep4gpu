"""End-to-end private Sequential MLP inference with GPU Rep4 and CUDA graphs.

Trains a small two-layer ReLU MLP (16 -> 64 -> 8) on 64 synthetic two-token
contexts, shares the PyTorch Sequential model once via ``share_sequential``,
captures fixed-shape CUDA phase graphs, and verifies checked private inference
against the exact integer and floating-point references.

Run from the repository root:
    python -m examples.mlp_demo
    python -m examples.mlp_demo --rounding probabilistic --bits 32
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import torch
from torch import nn

from rep4gpu import (
    FixedPoint,
    IdealNetwork,
    PhaseGraphParty,
    DeviceIdealEndpoint,
    make_seeds,
    sequential_spec,
    share_sequential,
)


def make_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    contexts = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    inputs = torch.nn.functional.one_hot(contexts, num_classes=8).reshape(64, 16).float()
    targets = contexts.sum(dim=1) % 8
    return inputs, targets


def train_mlp(steps: int = 600) -> nn.Sequential:
    if steps < 1:
        raise ValueError('steps must be positive')
    torch.manual_seed(20261109)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 8))
    inputs, targets = make_dataset()
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(inputs), targets).backward()
        opt.step()
    return model.eval()


def integer_reference(model: nn.Sequential, inputs: torch.Tensor,
                      fractional_bits: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1 << fractional_bits
    encode = lambda t: torch.round(t.detach().double() * scale).long()
    x = encode(inputs)
    w1, b1 = encode(model[0].weight.T), encode(model[0].bias)
    w2, b2 = encode(model[2].weight.T), encode(model[2].bias)
    raw1 = x @ w1 + b1 * scale
    hidden = (raw1 >> fractional_bits).clamp_min(0)
    raw2 = hidden @ w2
    expected = (raw2 >> fractional_bits) + b2
    limit = 1 << (bits - 1)
    for name, tensor in (('raw1', raw1), ('raw2', raw2), ('expected', expected)):
        if int(tensor.min()) < -limit or int(tensor.max()) >= limit:
            raise ValueError(f'{name} exceeds signed ring{bits}')
    return x, expected


def run_demo(*, steps: int = 600, bits: int = 32, fractional_bits: int = 12,
             rounding: str = 'floor', capture_graphs: bool = True) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PhaseGraphParty')
    model = train_mlp(steps)
    inputs, targets = make_dataset()
    x_int, expected = integer_reference(model, inputs, fractional_bits, bits)
    spec = sequential_spec(model)
    storage = 'words32' if bits == 32 else 'limbs'

    network = IdealNetwork(timeout=120)
    seeds = make_seeds()
    roles = [
        PhaseGraphParty(
            p,
            DeviceIdealEndpoint(network.endpoint(p), device='cuda'),
            {e: s for e, s in seeds.items() if e != p},
            device='cuda',
            bits=bits,
            binary_bits=bits,
            optimized=True,
            gpu_open=True,
            rounding=rounding,
            bounded_sign=(rounding == 'probabilistic'),
            sparse_products=True,
            narrow_products=(rounding == 'probabilistic' and fractional_bits <= 16),
        )
        for p in range(4)
    ]

    def share_worker(p: int):
        try:
            party = roles[p]
            shared_model = share_sequential(
                party, spec, model if p == 1 else None, owner=1,
                fractional_bits=fractional_bits, weight_storage=storage,
            )
            shared_x = party.input(0, x_int if p == 0 else None, x_int.shape)
            return shared_model, shared_x
        except BaseException as err:
            network.abort(err)
            raise

    with ThreadPoolExecutor(4) as pool:
        shared = list(pool.map(share_worker, range(4)))

    functions = [
        (lambda x, r=roles[p], m=shared[p][0]: r.open(m(x)))
        for p in range(4)
    ]
    plans = [
        roles[p].prepare_inference(functions[p], shared[p][1], check_at_send=True)
        for p in range(4)
    ] if capture_graphs else None

    def infer_worker(p: int):
        try:
            fn = plans[p].run if plans is not None else functions[p]
            return fn(shared[p][1]).cpu()
        except BaseException as err:
            network.abort(err)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(infer_worker, range(4)))

    if any(not torch.equal(outputs[0], out) for out in outputs[1:]):
        raise AssertionError('parties disagreed on opened logits')
    if rounding == 'floor' and not torch.equal(outputs[0], expected):
        raise AssertionError('floor output differs from exact integer reference')

    with torch.no_grad():
        floating = model(inputs)
    decoded = outputs[0].double() / (1 << fractional_bits)
    return {
        'bits': bits,
        'fractional_bits': fractional_bits,
        'rounding': rounding,
        'captured_graphs': capture_graphs,
        'exact_integer_match': bool(torch.equal(outputs[0], expected)),
        'float_accuracy': float((floating.argmax(-1) == targets).float().mean()),
        'private_accuracy': float((decoded.argmax(-1) == targets).float().mean()),
        'max_logit_error': float((decoded - floating.double()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--bits', type=int, choices=(32, 64), default=32)
    parser.add_argument('--fractional-bits', type=int, default=12)
    parser.add_argument('--rounding', choices=('floor', 'probabilistic'), default='floor')
    parser.add_argument('--no-graphs', action='store_true')
    args = parser.parse_args()
    result = run_demo(
        steps=args.steps,
        bits=args.bits,
        fractional_bits=args.fractional_bits,
        rounding=args.rounding,
        capture_graphs=not args.no_graphs,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
