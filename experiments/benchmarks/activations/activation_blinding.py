"""Explore activation leakage and toy replica costs on public fixtures.

Run from the repository root with ``python -m experiments.benchmarks.activations.activation_blinding``.
Add ``--gpu`` to measure ordinary PyTorch CUDA activations. No enclave, remote
attestation, private input, or secure blinding protocol runs in this benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch
import torch.nn.functional as F


def rms_normalize(x, epsilon=1e-6):
    return x*torch.rsqrt(x.square().mean(dim=-1, keepdim=True)+epsilon)


def leakage_witnesses():
    """Return numerical witnesses to information retained by the proposed blinds."""
    x = torch.tensor([-2., -0.5, 1., 3.], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])
    shifted = (x+37)[permutation]
    probabilities = torch.softmax(x, -1)
    blinded_probabilities = torch.softmax(shifted, -1)
    half_logits = torch.tensor([-1., 0., 1.], dtype=torch.float16)
    half_shifted = half_logits+8192
    half_shift_error = (torch.softmax(half_logits, -1)-torch.softmax(half_shifted, -1)).abs().max()
    scale, epsilon = 7., 0.25
    scaled = (scale*x)[permutation]
    normalized = rms_normalize(x, 0)
    blinded_normalized = rms_normalize(scaled, 0)
    disclosed_epsilon = epsilon*scale*scale
    dummies = torch.tensor([-9., 8.], dtype=x.dtype)
    padded = torch.cat((x, dummies))[torch.tensor([5, 2, 0, 4, 3, 1])]
    return {
        'public_input': x.tolist(),
        'softmax': {
            'sorted_pairwise_gaps_preserved': bool(torch.allclose(torch.diff(shifted.sort().values), torch.diff(x.sort().values))),
            'probability_multiset_preserved': bool(torch.allclose(probabilities.sort().values, blinded_probabilities.sort().values)),
            'fp16_large_shift_logits': half_shifted.tolist(),
            'fp16_large_shift_max_probability_error': float(half_shift_error),
            'disclosed_probability_multiset': blinded_probabilities.sort().values.tolist(),
            'interpretation': 'A row shift and permutation retain all sorted gaps and the probability multiset; they do not hide the distribution.',
        },
        'rms': {
            'zero_epsilon_normalized_multiset_preserved': bool(torch.allclose(normalized.sort().values, blinded_normalized.sort().values)),
            'absolute_pairwise_ratio_preserved': math.isclose(float(scaled.abs().sort().values[-1]/scaled.abs().sort().values[0]), float(x.abs().max()/x.abs().min())),
            'fixed_epsilon_max_error': float((rms_normalize(scale*x, epsilon)-rms_normalize(x, epsilon)).abs().max()),
            'scaled_epsilon_max_error': float((rms_normalize(scale*x, disclosed_epsilon)-rms_normalize(x, epsilon)).abs().max()),
            'scale_recovered_from_public_epsilons': math.sqrt(disclosed_epsilon/epsilon),
            'interpretation': 'Positive scaling preserves the normalized vector up to permutation when epsilon is zero. Public epsilon*r^2 with known nonzero epsilon reveals r.',
        },
        'gelu': {
            'all_original_values_still_visible': all(bool((padded==value).any()) for value in x),
            'visible_padded_multiset': padded.sort().values.tolist(),
            'interpretation': 'Permutation and dummy padding retain the original values in a visible multiset; this is not cryptographic masking.',
        },
    }


def _activation(name, x, epsilon):
    if name == 'softmax':
        return torch.softmax(x, dim=-1)
    if name == 'rms':
        return rms_normalize(x, epsilon)
    if name == 'gelu':
        return F.gelu(x, approximate='none')
    raise ValueError('activation must be softmax, rms, or gelu')


def _fixture_shares(output, seed, nonce, context, fractional_bits):
    """Deterministically split a public output into four ring32 components."""
    encoded = np.rint(output.detach().cpu().numpy()*2**fractional_bits)
    if not np.isfinite(encoded).all() or (np.abs(encoded)>=2**31).any():
        raise ValueError('fixture output does not fit signed ring32')
    target = encoded.astype(np.int64)&0xffffffff
    domain = b'ActivationPublicReplicaFixture-v1\0'+struct.pack('<Q', nonce)+context
    masks = np.frombuffer(hashlib.shake_256(seed+domain).digest(3*target.size*4), dtype='<u4').astype(np.int64).reshape((3, *target.shape))
    final = (target-masks.sum(axis=0))&0xffffffff
    return np.concatenate((masks, final[None]), axis=0)


class ReplicaSimulation:
    """Compare two software replicas on public fixtures with a public PRF seed.

    This checks agreement under a shared deterministic implementation. It does
    not authenticate hardware, hide inputs, or detect a common error in both
    replicas. Both replicas see the complete input, output, and fixture PRF seed. A nonce is consumed even when the two replicas disagree.
    """
    def __init__(self, seed=bytes(range(32)), fractional_bits=16):
        if not isinstance(seed, bytes) or len(seed)<32:
            raise ValueError('fixture seed must contain at least 32 bytes')
        if not isinstance(fractional_bits, int) or not 0<=fractional_bits<=24:
            raise ValueError('fractional_bits must be between 0 and 24')
        self.seed = seed
        self.fractional_bits = fractional_bits
        self.used_nonces = set()

    def run(self, public_input, activation, nonce, *, epsilon=1e-6, corrupt_second=False):
        if not isinstance(nonce, int) or not 0<=nonce<2**64 or nonce in self.used_nonces:
            raise ValueError('each fixture call requires a fresh uint64 nonce')
        if public_input.device.type != 'cpu' or not public_input.is_floating_point() or not public_input.numel():
            raise ValueError('replica simulation requires a nonempty public CPU floating tensor')
        self.used_nonces.add(nonce)
        context = json.dumps([activation, list(public_input.shape), epsilon, self.fractional_bits], separators=(',', ':')).encode()
        outputs = [_activation(activation, public_input.clone(), epsilon) for _ in range(2)]
        if corrupt_second:
            outputs[1].reshape(-1)[0] += 0.125
        shares = [_fixture_shares(output, self.seed, nonce, context, self.fractional_bits) for output in outputs]
        digests = [hashlib.sha256(np.asarray(parts, dtype='<u4').tobytes()).digest() for parts in shares]
        return {'accepted': digests[0]==digests[1], 'components': shares[0],
                'quantized_output': shares[0].sum(axis=0)&0xffffffff,
                'corrupt_second': bool(corrupt_second)}


def wire_model(elements, ring_bits=32):
    """Count directed application bytes, excluding transport and setup costs.

    Rep4's six checked joint inputs each send one tensor and one 32-byte tag.
    The optional balanced product retains four such joints. The hypothetical
    hybrid route assumes helpers colocated with parties 0/1: each receives its
    missing component, helper 0 sends three output components to each other
    party, and helper 1 sends one bundled output tag to parties 0/2/3. This
    route is a cost assumption. The separate implemented benchmark route
    keeps both helpers' output bundles local and sends bundles only to hosts
    2/3, totaling eight ring words per element and four tags.
    """
    if not isinstance(elements, int) or elements<0 or ring_bits not in (32,64):
        raise ValueError('nonnegative element count and ring32/ring64 are required')
    word_bytes = ring_bits//8
    def counts(words, tags):
        return {'ring_words': words*elements, 'payload_bytes': words*elements*word_bytes,
                'tag_bytes': tags*32, 'application_bytes': words*elements*word_bytes+tags*32}
    return {
        'elements': elements, 'ring_bits': ring_bits, 'accounting': 'sum of directed WAN application bytes across all parties',
        'rep4_generic_product': counts(6, 6), 'rep4_balanced_cross_product': counts(4, 4),
        'simulated_two_helper_relu': {
            **counts(8, 4), 'checked_rounds': 2,
            'implemented_ring_bits': 32, 'supported_by_benchmark': ring_bits==32,
            'input': counts(2, 2), 'output': counts(6, 2),
            'source': 'experiments/benchmarks/activations/simulated_activation.py',
            'scope': 'implemented public-fixture ring32 software simulation; ring64 values are byte scaling only; no enclave or activation privacy',
            'route': 'helpers 0/1 each receive one missing input component; both retain local output bundles; helper 0 sends three-component bundles to hosts 2/3 checked against helper 1 tags',
        },
        'hypothetical_two_helper_activation': {
            **counts(11, 3),
            'scope': 'earlier hypothetical route; not the implemented simulation',
        },
        'hybrid_assumptions': ['standalone helpers colocated with parties 0/1 assume trusted local isolation; it is not implemented',
            'helpers reuse their colocated party three-component inputs locally',
            'two missing input components cross the network',
            'primary output sends three components to each of three other parties',
            'secondary output sends a 32-byte bundle tag to each of three other parties'],
        'excluded': ['blinding and secret permutation protocols', 'input consistency checks',
            'transport framing and authentication', 'enclave entry, attestation, and protected-memory costs',
            'secure fixed-point encoding and nonlinear approximation costs'],
        'tee_latency_ms': None,
    }


def measure_plain_gpu(rows=64, width=128):
    """Measure captured ordinary CUDA kernels; no TEE or MPC is involved."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for --gpu')
    import triton.testing
    x = torch.linspace(-3, 3, rows*width, device='cuda').reshape(rows, width)
    measured = {}
    for activation in ('softmax', 'rms', 'gelu'):
        function = lambda: _activation(activation, x, 1e-6)
        function()
        measured[activation] = triton.testing.do_bench_cudagraph(function, rep=200)
    return {'device': torch.cuda.get_device_name(), 'shape': [rows,width], 'dtype': str(x.dtype),
            'graph_ms': measured, 'method': 'Triton CUDA graph event benchmark over ordinary PyTorch operations',
            'tee_latency_ms': None, 'includes_network_or_tee': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.rows<=0 or args.width<=0:
        parser.error('rows and width must be positive')
    simulator = ReplicaSimulation()
    x = torch.tensor([[-2., -0.5, 1., 3.]], dtype=torch.float64)
    simulations = {}
    for i,name in enumerate(('softmax','rms','gelu')):
        matching = simulator.run(x, name, 2*i)
        corrupted = simulator.run(x, name, 2*i+1, corrupt_second=True)
        simulations[name] = {'matching_accepted': matching['accepted'], 'corrupted_accepted': corrupted['accepted']}
    result = {'scope': 'public-fixture mathematical leakage and software replica simulation; no TEE or privacy claim',
              'leakage': leakage_witnesses(), 'replicas': simulations,
              'wire_model': wire_model(args.rows*args.width),
              'plain_gpu': measure_plain_gpu(args.rows,args.width) if args.gpu else None}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
