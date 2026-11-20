"""Measure private checked 4-party Text CNN inference against native PyTorch.

Party 0 owns the private dense token-sequence features X; Party 1 owns all
learned CNN weights and biases (W1, b1, w2, b2); Parties 2 and 3 hold no
plaintext. Only the final binary classification labels are opened after all
active-security consistency checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from experiments.benchmarks.inference.socket_inference import MeteredTransport, certificates, reserve_addresses
from experiments.examples.text_cnn_classifier import global_bounds, integer_forward
from rep4gpu.protocol.setup import correlated_seeds


def load_cnn_model(checkpoint: Path, batch: int) -> dict:
    with np.load(checkpoint, allow_pickle=False) as data:
        if str(data['architecture']) != 'text_cnn_square':
            raise ValueError('unsupported Text CNN checkpoint architecture')
        primes = tuple(int(p) for p in data['primes'])
        x0 = data['x_test']
        weights = {name: data[name].astype(np.int64) for name in ('w1', 'b1', 'w2', 'b2')}
        floats = {name: data['float_' + name].astype(np.float32) for name in ('w1', 'b1', 'w2', 'b2')}
    if x0.ndim != 3 or not len(x0):
        raise ValueError('x_test must be a nonempty 3D tensor (N, L, D)')
    _, seq_len, dim = x0.shape
    k_size, d_w, channels = weights['w1'].shape
    seq_out = seq_len - k_size + 1
    if d_w != dim or seq_out <= 0:
        raise ValueError('w1 shape is incompatible with x_test')
    if weights['b1'].shape != (channels,) or weights['w2'].shape != (channels, 1) or weights['b2'].shape != (1,):
        raise ValueError('invalid Text CNN parameter shapes')
    if not np.all((x0 == -1) | (x0 == 1)):
        raise ValueError('dense Text CNN inputs must have entries in {-1, +1}')
    bounds = global_bounds(weights['w1'], weights['b1'], weights['w2'], weights['b2'], seq_out)
    limit = math.prod(primes) // 2
    if bounds['score_max_abs'] >= limit:
        raise ValueError(f'checkpoint bound {bounds["score_max_abs"]} exceeds CRT limit {limit}')
    x = x0[np.arange(batch) % len(x0)].astype(np.int64)
    scores = integer_forward(x, weights['w1'], weights['b1'], weights['w2'], weights['b2'])
    expected = (scores >= 0).reshape(-1).astype(int).tolist()
    return dict(
        architecture='text_cnn_square',
        primes=primes,
        x=x,
        weights=weights,
        float_weights=floats,
        bounds=bounds,
        expected=expected,
    )


def native_cnn_baselines(x: np.ndarray, float_weights: dict, device: str = 'cuda', repeats: int = 100) -> dict:
    """Time fastest honest native PyTorch baselines (FP32/FP16/BF16, conv1d & unfold+mm, eager & graph)."""
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    results = {}
    batch, seq_len, dim = x.shape
    k_size, _, channels = float_weights['w1'].shape
    seq_out = seq_len - k_size + 1
    for dtype_name, dtype in (('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)):
        x_blc = torch.as_tensor(x, device=device, dtype=dtype).contiguous()
        x_bcl = x_blc.permute(0, 2, 1).contiguous()
        w1_conv = torch.as_tensor(float_weights['w1'], device=device, dtype=dtype).permute(2, 1, 0).contiguous()
        w1_mm = torch.as_tensor(float_weights['w1'], device=device, dtype=dtype).reshape(k_size * dim, channels).contiguous()
        b1 = torch.as_tensor(float_weights['b1'], device=device, dtype=dtype).contiguous()
        w2 = torch.as_tensor(float_weights['w2'], device=device, dtype=dtype).contiguous()
        b2 = torch.as_tensor(float_weights['b2'], device=device, dtype=dtype).contiguous()

        def fn_conv():
            h = F.conv1d(x_bcl, w1_conv, b1)
            return ((h * h).sum(dim=2) @ w2 + b2) >= 0

        def fn_mm():
            u = x_blc.unfold(1, k_size, 1).permute(0, 1, 3, 2).reshape(batch * seq_out, k_size * dim)
            h = (u @ w1_mm + b1).view(batch, seq_out, channels)
            return ((h * h).sum(dim=1) @ w2 + b2) >= 0

        for impl, fn in (('conv1d', fn_conv), ('unfold_mm', fn_mm)):
            for _ in range(10):
                fn()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize(device)
            results[f'{dtype_name}_{impl}_eager'] = (time.perf_counter() - t0) * 1000 / repeats

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _ = fn()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                graph.replay()
            torch.cuda.synchronize(device)
            results[f'{dtype_name}_{impl}_graph'] = (time.perf_counter() - t0) * 1000 / repeats
    return results


def child(config_path: Path, result_path: Path) -> int:
    cfg = json.loads(Path(config_path).read_text())
    p, device = cfg['party'], cfg['device']
    mode = cfg['device_transport']
    direct = mode == 'cuda-ipc'
    torch.set_num_threads(1)
    transport = None
    data_transport = None
    try:
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(**cfg['identity'])
        server.load_verify_locations(cfg['ca'])
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cfg['ca'])
        client.load_cert_chain(**cfg['identity'])
        transport = MeteredTransport(
            p, [tuple(a) for a in cfg['addresses']], timeout=cfg['timeout'],
            server_context=server, client_context=client,
            peer_hostnames={q: f'party{q}' for q in range(4)},
        )
        start = time.perf_counter()
        seeds = correlated_seeds(p, transport, context=b'Rep4-text-cnn-v1')
        if direct:
            from rep4gpu.protocol.cuda_ipc_transport import CudaIpcTransport
            data_transport = CudaIpcTransport(transport, device)
        else:
            from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge
            data_transport = PinnedDeviceByteBridge(transport, device)
        from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
        rt = CrtPhaseGraphParty(p, data_transport, seeds, device=device,
                                primes=tuple(cfg['primes']), optimized=True, gpu_open=True)
        own = np.load(cfg['input']) if 'input' in cfg else None
        w1 = rt.crt_prepare_weight(rt.crt_input(1, torch.from_numpy(own['w1'].astype(np.int64)) if p == 1 else None,
                                                cfg['weight_shapes']['w1']))
        b1 = rt.crt_input(1, torch.from_numpy(own['b1'].astype(np.int64)) if p == 1 else None,
                          cfg['weight_shapes']['b1'])
        w2 = rt.crt_prepare_weight(rt.crt_input(1, torch.from_numpy(own['w2'].astype(np.int64)) if p == 1 else None,
                                                cfg['weight_shapes']['w2']))
        b2 = rt.crt_input(1, torch.from_numpy(own['b2'].astype(np.int64)) if p == 1 else None,
                          cfg['weight_shapes']['b2'])
        params = (w1, b1, w2, b2)

        def forward_score(fx_in, prm):
            h = rt.crt_conv1d_from_eval(fx_in, prm[0])
            pooled = rt.crt_square_sum_pool(h, bias=prm[1])
            return rt.crt_matmul(pooled, prm[2], prm[3])

        input_value = torch.from_numpy(own['x'].astype(np.int8)).to(device) if p == 0 else None
        input_shape = tuple(cfg['input_shape'])
        plan = None
        if cfg['phase_graphs']:
            initial = rt.crt_input(0, input_value, input_shape)
            initial_fx = rt.crt_eval_input(initial)
            del initial
            torch.cuda.synchronize(device)
            plan = rt.prepare_inference(forward_score, initial_fx, params, check_at_send=True)
            torch.cuda.synchronize(device)
            del initial_fx
        resident = rt.crt_input(0, input_value, input_shape) if cfg.get('resident_inputs') else None
        torch.cuda.synchronize(device)
        setup_ms = (time.perf_counter() - start) * 1000
        measurements = []
        predictions = None
        for iteration in range(cfg['warmup'] + cfg['repeats']):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dev_byte_start = data_transport.bytes_sent if direct else 0
            dev_ctrl_start = data_transport.control_bytes_sent if direct else 0
            sock_byte_start = transport.bytes_sent
            sock_round_start = transport.rounds
            x_share = resident if resident is not None else rt.crt_input(0, input_value, input_shape)
            torch.cuda.synchronize(device)
            t_shared = time.perf_counter()
            input_dev_bytes = (data_transport.bytes_sent - dev_byte_start) if direct else 0
            input_sock_bytes = transport.bytes_sent - sock_byte_start
            input_bytes = input_dev_bytes if direct else input_sock_bytes
            if plan is not None:
                fx = rt.crt_eval_input(x_share, out=plan.inputs[0])
                score_share = plan.run(fx, params)
            else:
                score_share = forward_score(rt.crt_eval_input(x_share), params)
            opened = rt.crt_nonnegative_label_open_cpu(score_share)
            torch.cuda.synchronize(device)
            if resident is None:
                del x_share
            t_end = time.perf_counter()
            dev_bytes = (data_transport.bytes_sent - dev_byte_start) if direct else 0
            ctrl_bytes = (data_transport.control_bytes_sent - dev_ctrl_start) if direct else 0
            sock_bytes = transport.bytes_sent - sock_byte_start
            if direct:
                payload_bytes = dev_bytes
                host_payload_bytes = sock_bytes - ctrl_bytes
            else:
                payload_bytes = sock_bytes
                host_payload_bytes = sock_bytes
            rounds = transport.rounds - sock_round_start
            predictions = opened.cpu().numpy().reshape(-1)
            if iteration >= cfg['warmup']:
                measurements.append(dict(
                    start_monotonic=t0,
                    end_monotonic=t_end,
                    prediction_sha256=hashlib.sha256(predictions.astype('<i8').tobytes()).hexdigest(),
                    input_ms=(t_shared - t0) * 1000,
                    checked_inference_ms=(t_end - t_shared) * 1000,
                    online_ms=(t_end - t0) * 1000,
                    input_payload_bytes=input_bytes,
                    payload_bytes=payload_bytes,
                    inference_payload_bytes=payload_bytes - input_bytes,
                    host_payload_bytes=host_payload_bytes,
                    socket_application_bytes=sock_bytes,
                    rounds=rounds,
                ))
        control_auth = transport.authenticated
        if direct:
            data_transport.close()
            data_transport = transport = None
        Path(result_path).write_text(json.dumps(dict(
            party=p, passed=True, setup_ms=setup_ms, measurements=measurements,
            predictions=predictions.tolist(), device_transport=mode,
            control_authenticated_tls=control_auth,
        ), indent=2) + '\n')
        return 0
    except Exception as error:
        traceback.print_exc()
        Path(result_path).write_text(json.dumps(dict(
            party=p, passed=False, error=f'{type(error).__name__}: {error}',
        )) + '\n')
        return 1
    finally:
        if data_transport is not None and direct:
            getattr(data_transport, 'abort', data_transport.close)()
        elif transport is not None:
            transport.close()


def isolated_party_gpu_estimate(model: dict, device: str = 'cuda', repeats: int = 30) -> dict:
    """Measure uncontended single-party GPU compute per role (zero-network 4-GPU estimate).

    Replays each party's exact captured CUDA graph segments (all local arithmetic,
    BLAKE3 XOF masking, third-difference checks, BLAKE3 digests, and tag-check
    transfers) plus input share evaluation without peer contention or network
    transport. This is a compute-only lower bound; live 4-process timings remain primary.
    """
    from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
    from rep4gpu.protocol.runtime import Share, make_seeds

    class _ZeroNetworkStub:
        staging = 'isolated_zero_network'
        authenticated = False

        def __init__(self, party):
            self.party = party

        def exchange_device(self, *args, **kwargs):
            raise RuntimeError('isolated estimate cannot use live transport')

    primes = model['primes']
    n_lanes = len(primes)
    x_shape = tuple(model['x'].shape)
    w_shapes = {k: tuple(v.shape) for k, v in model['weights'].items()}
    seeds = make_seeds()
    per_party = {}
    for p in range(4):
        rt = CrtPhaseGraphParty(p, _ZeroNetworkStub(p), {e: seeds[e] for e in range(4) if e != p},
                                device=device, primes=primes, optimized=True, gpu_open=True)
        zero_share = lambda shape: Share(
            tuple(torch.zeros((n_lanes, *shape), dtype=torch.uint8, device=device) for _ in range(3)), False
        )
        w1 = rt.crt_prepare_weight(zero_share(w_shapes['w1']))
        b1 = zero_share(w_shapes['b1'])
        w2 = rt.crt_prepare_weight(zero_share(w_shapes['w2']))
        b2 = zero_share(w_shapes['b2'])
        params = (w1, b1, w2, b2)
        x_share = zero_share(x_shape)

        def forward_score(fx_in, prm):
            h = rt.crt_conv1d_from_eval(fx_in, prm[0])
            pooled = rt.crt_square_sum_pool(h, bias=prm[1])
            return rt.crt_matmul(pooled, prm[2], prm[3])

        fx0 = rt.crt_eval_input(x_share)
        plan = rt.prepare_inference(forward_score, fx0, params, check_at_send=True)
        del fx0
        graphs = [val for kind, val in plan.actions if kind == 'graph']

        def replay_local():
            with torch.cuda.stream(plan.stream):
                plan.nonces.copy_(plan.host_nonces, non_blocking=True)
                rt.crt_eval_input(x_share, out=plan.inputs[0])
                for graph in graphs:
                    graph.replay()

        for _ in range(5):
            replay_local()
        plan.stream.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            replay_local()
            plan.stream.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        per_party[f'party{p}'] = float(np.median(samples))
        del graphs, plan, x_share, params, w1, b1, w2, b2, rt
        torch.cuda.empty_cache()
    return dict(per_party_ms=per_party, bottleneck_party_ms=max(per_party.values()))


def run(checkpoint: Path, output: Path, *, batch: int = 1024, repeats: int = 10, warmup: int = 3,
        timeout: float = 300, phase_graphs: bool = True, device_transport: str = 'cuda-ipc',
        resident_inputs: bool = False, include_isolated_estimate: bool = False) -> dict:
    if device_transport not in ('host', 'cuda-ipc'):
        raise ValueError('device_transport must be host or cuda-ipc')
    model = load_cnn_model(checkpoint, batch)
    x, weights, expected = model['x'], model['weights'], model['expected']
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem + '_logs')
    logs.mkdir(exist_ok=True)
    baseline = native_cnn_baselines(x, model['float_weights'], device='cuda')
    fastest_native_key = min(baseline, key=baseline.get)
    fastest_native_ms = baseline[fastest_native_key]

    processes, handles = [], []
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='rep4-textcnn-') as folder:
        folder = Path(folder)
        ca, identities = certificates(folder)
        addresses = reserve_addresses()
        np.savez(folder / 'owner0.npz', x=x.astype(np.int8))
        np.savez(folder / 'owner1.npz', **weights)
        try:
            for p in range(4):
                result_path = logs / f'party{p}.json'
                result_path.unlink(missing_ok=True)
                cfg = dict(
                    party=p, device='cuda', primes=list(model['primes']),
                    device_transport=device_transport, phase_graphs=phase_graphs,
                    resident_inputs=resident_inputs, ca=ca,
                    identity=dict(certfile=identities[p]['certificate'], keyfile=identities[p]['key']),
                    addresses=addresses, timeout=timeout,
                    input_shape=list(x.shape),
                    weight_shapes={k: list(v.shape) for k, v in weights.items()},
                    repeats=repeats, warmup=warmup,
                )
                if p in (0, 1):
                    cfg['input'] = str(folder / f'owner{p}.npz')
                cfg_path = folder / f'party{p}.json'
                cfg_path.write_text(json.dumps(cfg))
                handle = open(logs / f'party{p}.log', 'w')
                handles.append(handle)
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                processes.append(subprocess.Popen(
                    [sys.executable, '-m', 'experiments.benchmarks.inference.text_cnn_classifier',
                     '--party-config', str(cfg_path), '--party-result', str(result_path)],
                    stdout=handle, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[3], env=env,
                ))
            deadline = time.monotonic() + timeout
            for proc in processes:
                proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for handle in handles:
                handle.close()

    parties = [json.loads((logs / f'party{p}.json').read_text()) if (logs / f'party{p}.json').exists()
               else dict(party=p, passed=False, error='no result') for p in range(4)]
    codes = [proc.returncode for proc in processes]
    passed = not timed_out and all(code == 0 for code in codes) and all(p['passed'] for p in parties)
    expected_hash = hashlib.sha256(np.asarray(expected, dtype='<i8').tobytes()).hexdigest()
    matched = all(
        p.get('predictions') == expected
        and all(m['prediction_sha256'] == expected_hash for m in p.get('measurements', []))
        for p in parties
    )
    passed = passed and matched
    report = dict(
        passed=passed,
        exact_integer_match=matched,
        architecture='text_cnn_square',
        primes=list(model['primes']),
        device_transport=device_transport,
        phase_graphs=phase_graphs,
        resident_inputs=resident_inputs,
        batch=batch,
        input_shape=list(x.shape),
        weight_shapes={k: list(v.shape) for k, v in weights.items()},
        repeats=repeats,
        warmup=warmup,
        checkpoint=str(Path(checkpoint).resolve()),
        hardware=torch.cuda.get_device_name(),
        native_ms=baseline,
        fastest_native_key=fastest_native_key,
        fastest_native_ms=fastest_native_ms,
        range_bounds=model['bounds'],
        return_codes=codes,
        timed_out=timed_out,
        parties=parties,
    )
    if passed:
        for metric in ('input_ms', 'checked_inference_ms', 'online_ms'):
            samples = [max(p['measurements'][i][metric] for p in parties) for i in range(repeats)]
            report[metric] = float(np.median(samples))
        report['inference_texts_per_second'] = batch * 1000 / report['checked_inference_ms']
        report['online_texts_per_second'] = batch * 1000 / report['online_ms']
        report['slowdown_resident'] = report['checked_inference_ms'] / fastest_native_ms
        report['slowdown_with_input_sharing'] = report['online_ms'] / fastest_native_ms
        report['target_met_resident'] = report['slowdown_resident'] < 20
        report['target_met_online'] = report['slowdown_with_input_sharing'] < 20
        for key in ('payload_bytes', 'input_payload_bytes', 'inference_payload_bytes',
                    'host_payload_bytes', 'socket_application_bytes'):
            report['global_' + key + '_per_batch'] = sum(p['measurements'][0][key] for p in parties)
        report['rounds_per_batch'] = parties[0]['measurements'][0]['rounds']
    if include_isolated_estimate:
        iso = isolated_party_gpu_estimate(model, device='cuda')
        report['isolated_party_gpu_estimate'] = dict(
            **iso,
            slowdown_vs_fastest_native=iso['bottleneck_party_ms'] / fastest_native_ms,
        )
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('experiments/results/examples/text_cnn_classifier/model.npz'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/inference/text_cnn_classifier_cuda.json'))
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--no-phase-graphs', action='store_true')
    parser.add_argument('--device-transport', choices=('host', 'cuda-ipc'), default='cuda-ipc')
    parser.add_argument('--resident-inputs', action='store_true')
    parser.add_argument('--isolated-estimate', action='store_true')
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config:
        return child(args.party_config, args.party_result)
    report = run(
        args.checkpoint, args.output, batch=args.batch, repeats=args.repeats,
        warmup=args.warmup, timeout=args.timeout, phase_graphs=not args.no_phase_graphs,
        device_transport=args.device_transport, resident_inputs=args.resident_inputs,
        include_isolated_estimate=args.isolated_estimate,
    )
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
