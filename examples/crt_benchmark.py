"""Four-process active-secure odd-prime CRT Rep4 inference benchmark.

Runs four independent OS processes over CUDA IPC (with mutually authenticated
TLS control) or host-staged mTLS, deriving fresh session seeds via
``correlated_seeds`` and executing captured CUDA phase graphs with fresh nonces:

  - ``--workload mlp``: 3-lane CRT (primes 251, 241, 239; M=14,457,349) ReLU MLP
    with 1-product INT8 Tensor Core CRT matmuls, active-secure ``crt_trunc_relu``,
    and private bit-sliced sign opening.
  - ``--workload cnn``: 4-lane CRT (primes 251, 241, 239, 233; M=3,368,562,317)
    1D CNN (Conv1d -> Square+GlobalSumPool -> Linear -> private sign).

Uses deterministic synthetic integer models that satisfy certified CRT modulus
bounds (no external dataset or checkpoint download required).

Run from the repository root:
    python -m examples.crt_benchmark --workload mlp --quick
    python -m examples.crt_benchmark --workload cnn --quick
    python -m examples.crt_benchmark --workload cnn --batch 2048 --repeats 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from examples._mtls import MeteredSocketTransport, mtls_certificates, reserve_addresses, ssl_contexts
from rep4gpu import (
    CrtPhaseGraphParty,
    CudaIpcTransport,
    PinnedDeviceByteBridge,
    correlated_seeds,
    crt_trunc_relu,
)
from rep4gpu.gpu.matmul.crt_conv import DEFAULT_CRT_PRIMES

CRT3_PRIMES = (251, 241, 239)
CRT4_PRIMES = DEFAULT_CRT_PRIMES  # (251, 241, 239, 233)


def make_mlp_fixture(batch: int, d_in: int, h1: int, h2: int, shift: int, primes: tuple[int, ...]):
    """Build a synthetic 3-layer ReLU MLP satisfying strict CRT quarter-ring bounds."""
    rng = np.random.default_rng(2026)
    m_mod = math.prod(primes)
    scale = 1 << shift
    off_q = (m_mod // 4) // scale
    trunc_limit = off_q * scale
    half_limit = m_mod // 2

    x = rng.integers(-4, 5, size=(batch, d_in), dtype=np.int64)
    w1 = rng.integers(-3, 4, size=(d_in, h1), dtype=np.int64)
    b1 = rng.integers(-8, 9, size=(h1,), dtype=np.int64)
    w2 = rng.integers(-3, 4, size=(h1, h2), dtype=np.int64)
    b2 = rng.integers(-8, 9, size=(h2,), dtype=np.int64)
    w3 = rng.integers(-3, 4, size=(h2, 1), dtype=np.int64)
    b3 = rng.integers(-8, 9, size=(1,), dtype=np.int64)

    raw1 = x @ w1 + b1 * scale
    h1_lo = np.maximum((raw1 >> shift) - 1, 0)
    h1_hi = np.maximum((raw1 >> shift) + 1, 0)
    w2_pos, w2_neg = np.maximum(w2, 0), np.minimum(w2, 0)
    raw2_lo = h1_lo @ w2_pos + h1_hi @ w2_neg + b2 * scale
    raw2_hi = h1_hi @ w2_pos + h1_lo @ w2_neg + b2 * scale
    h2_lo = np.maximum((raw2_lo >> shift) - 1, 0)
    h2_hi = np.maximum((raw2_hi >> shift) + 1, 0)
    w3_pos, w3_neg = np.maximum(w3, 0), np.minimum(w3, 0)
    raw3_lo = h2_lo @ w3_pos + h2_hi @ w3_neg + b3 * scale
    raw3_hi = h2_hi @ w3_pos + h2_lo @ w3_neg + b3 * scale

    max_raw1 = int(np.max(np.abs(raw1)))
    max_raw2 = int(max(np.max(np.abs(raw2_lo)), np.max(np.abs(raw2_hi))))
    max_raw3 = int(max(np.max(np.abs(raw3_lo)), np.max(np.abs(raw3_hi))))
    if max(max_raw1, max_raw2) >= trunc_limit:
        raise ValueError(f'MLP accumulator max({max_raw1}, {max_raw2}) >= CRT limit {trunc_limit}')
    if max_raw3 >= half_limit:
        raise ValueError(f'MLP score {max_raw3} >= CRT half modulus {half_limit}')

    params = {'x': x, 'w1': w1, 'b1': b1 * scale, 'w2': w2, 'b2': b2 * scale, 'w3': w3, 'b3': b3 * scale}
    return params, raw3_lo.reshape(-1), raw3_hi.reshape(-1)


def make_cnn_fixture(batch: int, seq_len: int, c_in: int, c_out: int, kernel_size: int,
                     primes: tuple[int, ...]):
    """Build a synthetic dense 1D CNN (Conv1d -> Square+SumPool -> Linear) within CRT bounds."""
    rng = np.random.default_rng(2026)
    m_mod = math.prod(primes)
    half_limit = m_mod // 2

    x = rng.choice(np.array([-1, 1], dtype=np.int64), size=(batch, seq_len, c_in))
    conv_w = rng.integers(-2, 3, size=(kernel_size, c_in, c_out), dtype=np.int64)
    conv_b = rng.integers(-8, 9, size=(c_out,), dtype=np.int64)
    head_w = rng.integers(-2, 3, size=(c_out, 1), dtype=np.int64)
    head_b = rng.integers(-8, 9, size=(1,), dtype=np.int64)

    # Exact integer reference via PyTorch CPU int64 unfold+matmul.
    x_t = torch.from_numpy(x)  # (B, L, C_in)
    windows = x_t.unfold(1, kernel_size, 1).permute(0, 1, 3, 2).reshape(batch, -1, kernel_size * c_in)
    w_flat = torch.from_numpy(conv_w.reshape(kernel_size * c_in, c_out))
    conv_out = windows @ w_flat + torch.from_numpy(conv_b)
    pooled = (conv_out * conv_out).sum(dim=1)
    score = (pooled @ torch.from_numpy(head_w) + torch.from_numpy(head_b)).reshape(-1).numpy()

    if int(np.max(np.abs(score))) >= half_limit:
        raise ValueError(f'CNN score {int(np.max(np.abs(score)))} >= CRT half modulus {half_limit}')

    params = {
        'x': x,
        'conv_w': conv_w,
        'conv_b': conv_b,
        'head_w': head_w,
        'head_b': head_b,
    }
    return params, score, score


def native_baseline_ms(workload: str, params: dict, repeats: int) -> float:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if workload == 'mlp':
        x = torch.from_numpy(params['x']).to(device='cuda', dtype=dtype)
        w1 = torch.from_numpy(params['w1']).to(device='cuda', dtype=dtype)
        b1 = torch.from_numpy(params['b1']).to(device='cuda', dtype=dtype)
        w2 = torch.from_numpy(params['w2']).to(device='cuda', dtype=dtype)
        b2 = torch.from_numpy(params['b2']).to(device='cuda', dtype=dtype)
        w3 = torch.from_numpy(params['w3']).to(device='cuda', dtype=dtype)
        b3 = torch.from_numpy(params['b3']).to(device='cuda', dtype=dtype)
        fn = lambda: (torch.relu(torch.relu(x @ w1 + b1) @ w2 + b2) @ w3 + b3 >= 0)
    else:
        x = torch.from_numpy(params['x']).permute(0, 2, 1).contiguous().to(device='cuda', dtype=dtype)
        w = torch.from_numpy(params['conv_w']).permute(2, 1, 0).contiguous().to(device='cuda', dtype=dtype)
        b = torch.from_numpy(params['conv_b']).to(device='cuda', dtype=dtype)
        hw = torch.from_numpy(params['head_w']).to(device='cuda', dtype=dtype)
        hb = torch.from_numpy(params['head_b']).to(device='cuda', dtype=dtype)
        fn = lambda: (F.conv1d(x, w, b).square().sum(dim=2) @ hw + hb >= 0)

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def _worker(config_path: str, result_path: str):
    cfg = json.loads(Path(config_path).read_text())
    p = cfg['party']
    transport = None
    dev_transport = None
    torch.set_num_threads(1)
    try:
        server, client = ssl_contexts(cfg['ca'], cfg['identity'])
        transport = MeteredSocketTransport(
            p, [tuple(a) for a in cfg['addresses']],
            timeout=cfg['timeout'],
            server_context=server, client_context=client,
            peer_hostnames={q: f'party{q}' for q in range(4)},
        )
        seeds = correlated_seeds(p, transport, context=b'Rep4GPU-crt-benchmark-v1')
        if cfg['transport'] == 'cuda-ipc':
            dev_transport = CudaIpcTransport(transport, device='cuda')
        else:
            dev_transport = PinnedDeviceByteBridge(transport, device='cuda')

        primes = tuple(cfg['primes'])
        party = CrtPhaseGraphParty(p, dev_transport, seeds, device='cuda', primes=primes)

        data = np.load(cfg['data_path']) if cfg['data_path'] is not None else None
        shapes = {k: tuple(v) for k, v in cfg['shapes'].items()}
        workload = cfg['workload']

        if workload == 'mlp':
            x_sh = party.crt_input(0, data['x'] if p == 0 else None, shapes['x'])
            w1 = party.crt_prepare_weight(party.crt_input(1, data['w1'] if p == 1 else None, shapes['w1']))
            b1 = party.crt_input(1, data['b1'] if p == 1 else None, shapes['b1'])
            w2 = party.crt_prepare_weight(party.crt_input(1, data['w2'] if p == 1 else None, shapes['w2']))
            b2 = party.crt_input(1, data['b2'] if p == 1 else None, shapes['b2'])
            w3 = party.crt_prepare_weight(party.crt_input(1, data['w3'] if p == 1 else None, shapes['w3']))
            b3 = party.crt_input(1, data['b3'] if p == 1 else None, shapes['b3'])
            shift = cfg['shift']

            def evaluate_gpu(x_in):
                h1 = crt_trunc_relu(party, party.crt_matmul(x_in, w1, b1), shift)
                h2 = crt_trunc_relu(party, party.crt_matmul(h1, w2, b2), shift)
                return party.crt_matmul(h2, w3, b3)

            plan = party.prepare_inference(evaluate_gpu, x_sh, check_at_send=True)
            run_fn = lambda: party.crt_nonnegative_label_open_cpu(plan.run(x_sh))
        else:
            x_sh = party.crt_input(0, data['x'] if p == 0 else None, shapes['x'])
            cw = party.crt_prepare_weight(party.crt_input(1, data['conv_w'] if p == 1 else None, shapes['conv_w']))
            cb = party.crt_input(1, data['conv_b'] if p == 1 else None, shapes['conv_b'])
            hw = party.crt_prepare_weight(party.crt_input(1, data['head_w'] if p == 1 else None, shapes['head_w']))
            hb = party.crt_input(1, data['head_b'] if p == 1 else None, shapes['head_b'])

            def evaluate_gpu(x_in):
                h = party.crt_conv1d(x_in, cw)
                pooled = party.crt_square_sum_pool(h, cb)
                return party.crt_matmul(pooled, hw, hb)

            plan = party.prepare_inference(evaluate_gpu, x_sh, check_at_send=True)
            run_fn = lambda: party.crt_nonnegative_label_open_cpu(plan.run(x_sh))

        for _ in range(cfg['warmup']):
            labels = run_fn()
        torch.cuda.synchronize()

        tls_before = transport.bytes_sent
        dev_before = getattr(dev_transport, 'device_payload_bytes_sent', 0)
        times_ms = []
        for _ in range(cfg['repeats']):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            labels = run_fn()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)

        Path(result_path).write_text(json.dumps({
            'party': p,
            'passed': True,
            'median_ms': float(statistics.median(times_ms)),
            'tls_bytes_per_iter': (transport.bytes_sent - tls_before) / cfg['repeats'],
            'device_bytes_per_iter': (getattr(dev_transport, 'device_payload_bytes_sent', 0) - dev_before) / cfg['repeats'],
            'labels': labels.reshape(-1).tolist(),
        }))
    except BaseException as err:
        Path(result_path).write_text(json.dumps({
            'party': p,
            'passed': False,
            'error': f'{type(err).__name__}: {err}',
            'traceback': traceback.format_exc(),
        }))
        raise
    finally:
        if dev_transport is not None:
            dev_transport.close()
        elif transport is not None:
            transport.close()


def run_crt_benchmark(*, workload: str = 'mlp', batch: int = 256,
                      transport: str = 'cuda-ipc', repeats: int = 5,
                      warmup: int = 2, quick: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for crt_benchmark')
    if workload == 'mlp':
        primes = CRT3_PRIMES
        shift = 9
        d_in, h1, h2 = (256, 128, 64) if quick else (4096, 1024, 512)
        params, lo, hi = make_mlp_fixture(batch, d_in, h1, h2, shift, primes)
    else:
        primes = CRT4_PRIMES
        shift = 0
        seq_len, c_in, c_out, k = (16, 64, 32, 5) if quick else (48, 256, 128, 5)
        params, lo, hi = make_cnn_fixture(batch, seq_len, c_in, c_out, k, primes)

    native_ms = native_baseline_ms(workload, params, repeats)
    shapes = {k: list(v.shape) for k, v in params.items()}
    addresses = reserve_addresses()

    with tempfile.TemporaryDirectory(prefix='rep4gpu-crt-') as tmp:
        tmp_dir = Path(tmp)
        ca, identities = mtls_certificates(tmp_dir)
        p0_path = tmp_dir / 'party0.npz'
        p1_path = tmp_dir / 'party1.npz'
        np.savez(p0_path, x=params['x'])
        np.savez(p1_path, **{k: v for k, v in params.items() if k != 'x'})

        procs, res_paths = [], []
        for p in range(4):
            cfg_path = tmp_dir / f'party{p}.config.json'
            res_path = tmp_dir / f'party{p}.result.json'
            data_path = str(p0_path) if p == 0 else (str(p1_path) if p == 1 else None)
            cfg_path.write_text(json.dumps({
                'party': p,
                'ca': ca,
                'identity': identities[p],
                'addresses': addresses,
                'timeout': 120.0,
                'transport': transport,
                'workload': workload,
                'primes': list(primes),
                'shift': shift,
                'shapes': shapes,
                'data_path': data_path,
                'warmup': warmup,
                'repeats': repeats,
            }))
            res_paths.append(res_path)
            procs.append(subprocess.Popen(
                [sys.executable, '-m', 'examples.crt_benchmark',
                 '--worker', str(cfg_path), str(res_path)],
                env=dict(os.environ),
            ))

        codes = [proc.wait(timeout=300) for proc in procs]
        reports = [json.loads(path.read_text()) for path in res_paths]
        if any(c != 0 or not r.get('passed') for c, r in zip(codes, reports)):
            raise RuntimeError(f'CRT worker failure: {reports}')

    labels = np.array(reports[0]['labels'], dtype=np.int64)
    for r in reports[1:]:
        if r['labels'] != reports[0]['labels']:
            raise AssertionError('parties disagreed on opened labels')
    # Every unambiguous example must match the certified sign interval.
    definite_pos = lo >= 0
    definite_neg = hi < 0
    if np.any(labels[definite_pos] != 1) or np.any(labels[definite_neg] != 0):
        raise AssertionError('opened labels violated certified CRT sign bounds')

    live_ms = max(r['median_ms'] for r in reports)
    return {
        'workload': workload,
        'primes': list(primes),
        'crt_modulus': math.prod(primes),
        'batch': batch,
        'transport': transport,
        'verified_labels': True,
        'live_4proc_ms': round(live_ms, 3),
        'native_gpu_ms': round(native_ms, 4),
        'live_4proc_vs_native': round(live_ms / native_ms, 2),
        'global_device_mb': round(sum(r['device_bytes_per_iter'] for r in reports) / (1024 * 1024), 3),
        'global_tls_kb': round(sum(r['tls_bytes_per_iter'] for r in reports) / 1024, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('CONFIG', 'RESULT'))
    parser.add_argument('--workload', choices=('mlp', 'cnn'), default='mlp')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--transport', choices=('cuda-ipc', 'pinned-tls'), default='cuda-ipc')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker[0], args.worker[1])
        return

    batch = args.batch if args.batch is not None else (64 if args.quick else 2048)
    repeats = 3 if args.quick else args.repeats
    summary = run_crt_benchmark(
        workload=args.workload,
        batch=batch,
        transport=args.transport,
        repeats=repeats,
        warmup=args.warmup,
        quick=args.quick,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
