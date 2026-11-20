"""Run checked four-process private inference over mutually authenticated TLS.

Spawns four independent OS processes with distinct ephemeral TLS identities,
derives correlated session seeds via commit/reveal (``correlated_seeds``),
executes a secret-weight 2-layer ReLU MLP, and verifies exact output agreement.

Run from the repository root:
    python -m examples.mtls_inference --transport cuda-ipc
    python -m examples.mtls_inference --transport pinned-tls
    python -m examples.mtls_inference --transport cpu-tls
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from examples._mtls import MeteredSocketTransport, mtls_certificates, reserve_addresses, ssl_contexts
from rep4gpu import (
    CudaIpcTransport,
    DeviceInferenceParty,
    FixedPoint,
    Party,
    PinnedDeviceByteBridge,
    correlated_seeds,
)
from rep4gpu.arithmetic.exact_truncation import trunc, trunc_relu


class _CpuExactParty(Party):
    def trunc(self, x, shift):
        return trunc(self, x, shift)

    def trunc_relu(self, x, shift):
        return trunc_relu(self, x, shift)


def make_fixture(batch: int, width: int, hidden: int, bits: int, fractional_bits: int):
    g = torch.Generator().manual_seed(2026)
    scale = 1 << fractional_bits
    x = torch.randint(-64, 65, (batch, width), generator=g, dtype=torch.int64)
    w1 = torch.randint(-16, 17, (width, hidden), generator=g, dtype=torch.int64)
    b1 = torch.randint(-32, 33, (hidden,), generator=g, dtype=torch.int64)
    w2 = torch.randint(-16, 17, (hidden, width), generator=g, dtype=torch.int64)
    b2 = torch.randint(-32, 33, (width,), generator=g, dtype=torch.int64)
    raw1 = x @ w1 + b1 * scale
    h = (raw1 >> fractional_bits).clamp_min(0)
    raw2 = h @ w2
    expected = (raw2 >> fractional_bits) + b2
    limit = 1 << (bits - 1)
    for tensor in (raw1, raw2, expected):
        if int(tensor.min()) < -limit or int(tensor.max()) >= limit:
            raise ValueError(f'fixture exceeds signed ring{bits}')
    return (x, w1, b1, w2, b2), expected


def _worker(config_path: str, result_path: str):
    config = json.loads(Path(config_path).read_text())
    p = config['party']
    transport = None
    device_transport = None
    torch.set_num_threads(1)
    try:
        server, client = ssl_contexts(config['ca'], config['identity'])
        t0 = time.perf_counter()
        transport = MeteredSocketTransport(
            p, [tuple(a) for a in config['addresses']],
            timeout=config['timeout'],
            server_context=server, client_context=client,
            peer_hostnames={q: f'party{q}' for q in range(4)},
        )
        handshake_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        seeds = correlated_seeds(p, transport, context=b'Rep4GPU-mtls-example-v1')
        setup_ms = (time.perf_counter() - t0) * 1000

        mode = config['transport']
        if mode == 'cpu-tls':
            runtime = _CpuExactParty(p, transport, seeds, device='cpu', bits=config['bits'])
            sync = lambda: None
        else:
            if mode == 'cuda-ipc':
                device_transport = CudaIpcTransport(transport, device='cuda')
            elif mode == 'pinned-tls':
                device_transport = PinnedDeviceByteBridge(transport, device='cuda')
            else:
                raise ValueError(f'unsupported transport mode: {mode}')
            from rep4gpu import DeviceInputMixin

            class _DeviceRuntime(DeviceInputMixin, DeviceInferenceParty):
                pass

            runtime = _DeviceRuntime(
                p, device_transport, seeds, device='cuda',
                bits=config['bits'], binary_bits=config['bits'],
                optimized=True, gpu_open=True, rounding='floor',
                sparse_products=True,
            )
            sync = torch.cuda.synchronize

        t0 = time.perf_counter()
        shared = []
        for idx, (owner, shape) in enumerate(zip((0, 1, 1, 1, 1), config['shapes'])):
            own = config['inputs'].get(str(idx))
            val = torch.tensor(own, dtype=torch.int64) if p == owner else None
            shared.append(runtime.input(owner, val, shape))
        x, w1, b1, w2, b2 = shared
        if mode != 'cpu-tls':
            storage = 'words32' if config['bits'] == 32 else 'limbs'
            w1 = runtime.prepare_weight_share(w1, storage=storage)
            w2 = runtime.prepare_weight_share(w2, storage=storage)
        sync()
        input_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fp = FixedPoint(runtime, config['fractional_bits'])
        hidden = fp.matmul_relu(x, w1, b1)
        output = runtime.open(fp.add(fp.matmul(hidden, w2), b2))
        sync()
        infer_ms = (time.perf_counter() - t0) * 1000

        report = {
            'party': p,
            'passed': True,
            'handshake_ms': handshake_ms,
            'setup_ms': setup_ms,
            'input_ms': input_ms,
            'infer_ms': infer_ms,
            'tls_bytes_sent': transport.bytes_sent,
            'tls_rounds': transport.rounds,
            'output': output.cpu().tolist(),
        }
        Path(result_path).write_text(json.dumps(report))
    except BaseException as err:
        Path(result_path).write_text(json.dumps({
            'party': p,
            'passed': False,
            'error': f'{type(err).__name__}: {err}',
            'traceback': traceback.format_exc(),
        }))
        raise
    finally:
        if device_transport is not None:
            device_transport.close()
        elif transport is not None:
            transport.close()


def run_mtls_inference(*, transport: str = 'cpu-tls', bits: int = 32,
                       batch: int = 4, width: int = 32, hidden: int = 64,
                       fractional_bits: int = 12, timeout: float = 60.0) -> dict:
    if transport != 'cpu-tls' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA is required for transport={transport}')
    tensors, expected = make_fixture(batch, width, hidden, bits, fractional_bits)
    shapes = [list(t.shape) for t in tensors]
    addresses = reserve_addresses()

    with tempfile.TemporaryDirectory(prefix='rep4gpu-mtls-') as tmp:
        tmp_dir = Path(tmp)
        ca, identities = mtls_certificates(tmp_dir)
        procs = []
        result_paths = []
        for p in range(4):
            own_inputs = {
                str(idx): tensors[idx].tolist()
                for idx, owner in enumerate((0, 1, 1, 1, 1))
                if owner == p
            }
            cfg_path = tmp_dir / f'party{p}.config.json'
            res_path = tmp_dir / f'party{p}.result.json'
            cfg_path.write_text(json.dumps({
                'party': p,
                'ca': ca,
                'identity': identities[p],
                'addresses': addresses,
                'timeout': timeout,
                'transport': transport,
                'bits': bits,
                'fractional_bits': fractional_bits,
                'shapes': shapes,
                'inputs': own_inputs,
            }))
            result_paths.append(res_path)
            env = dict(os.environ)
            procs.append(subprocess.Popen(
                [sys.executable, '-m', 'examples.mtls_inference',
                 '--worker', str(cfg_path), str(res_path)],
                env=env,
            ))

        exit_codes = [proc.wait(timeout=timeout + 30) for proc in procs]
        reports = [json.loads(path.read_text()) for path in result_paths]
        if any(code != 0 or not r.get('passed') for code, r in zip(exit_codes, reports)):
            raise RuntimeError(f'worker failure: {reports}')

    for report in reports:
        actual = torch.tensor(report['output'], dtype=torch.int64)
        if not torch.equal(actual, expected):
            raise AssertionError(f"party {report['party']} output differs from integer reference")

    return {
        'transport': transport,
        'bits': bits,
        'shape': {'batch': batch, 'width': width, 'hidden': hidden},
        'exact_match': True,
        'max_infer_ms': max(r['infer_ms'] for r in reports),
        'total_tls_bytes': sum(r['tls_bytes_sent'] for r in reports),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('CONFIG', 'RESULT'))
    parser.add_argument('--transport', choices=('cpu-tls', 'pinned-tls', 'cuda-ipc'), default='cuda-ipc')
    parser.add_argument('--bits', type=int, choices=(32, 64), default=32)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--fractional-bits', type=int, default=12)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker[0], args.worker[1])
        return
    summary = run_mtls_inference(
        transport=args.transport,
        bits=args.bits,
        batch=args.batch,
        width=args.width,
        hidden=args.hidden,
        fractional_bits=args.fractional_bits,
        timeout=args.timeout,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
