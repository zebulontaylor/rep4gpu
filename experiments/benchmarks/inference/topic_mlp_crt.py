"""Measure active-secure 3-lane CRT 4-party Topic MLP ReLU inference against native PyTorch.

Evaluates the same 4,720,641-parameter AG News Sci/Tech ReLU MLP
(4096 -> 1024 -> 512 -> 1) in Q9 3-lane CRT (primes 251, 241, 239; M=14,457,349)
with 1-product INT8 Tensor Core CRT matmuls, active-secure CRT trunc_relu
(compact uint32 bit-sliced CSA + carry tree + 4-joint complementary CRT product),
and CPU bit-sliced private sign opening.
"""
from __future__ import annotations

import argparse
import gc
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

from experiments.benchmarks.inference.socket_inference import MeteredTransport, certificates, reserve_addresses
from experiments.benchmarks.inference.topic_mlp_classifier import WEIGHT_NAMES, native_mlp_baselines
from experiments.examples.topic_mlp_classifier import binary_metrics, checked_int_matmul
from rep4gpu.gpu.bits.crt_relu_gpu import crt_trunc_relu
from rep4gpu.gpu.matmul.crt_conv import crt_matmul_local
from rep4gpu.protocol.setup import correlated_seeds


DEFAULT_CRT3_PRIMES = (251, 241, 239)


def propagate_crt_bounds(x_q: np.ndarray, q_params: dict[str, np.ndarray],
                         shift: int, primes: tuple[int, ...]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Validate CRT quarter-ring contracts and propagate exact [floor-1, floor+1] ReLU intervals."""
    m_mod = math.prod(primes)
    scale = 1 << shift
    off_q = (m_mod // 4) // scale
    trunc_limit = off_q * scale  # strict |x| < trunc_limit
    half_limit = m_mod // 2

    w1, b1 = q_params['w1'].astype(np.int64), q_params['b1'].astype(np.int64)
    w2, b2 = q_params['w2'].astype(np.int64), q_params['b2'].astype(np.int64)
    w3, b3 = q_params['w3'].astype(np.int64), q_params['b3'].astype(np.int64)

    raw1 = checked_int_matmul(x_q, w1) + b1 * scale
    h1_fl = np.maximum(raw1 >> shift, 0)
    h1_lo = np.maximum((raw1 >> shift) - 1, 0)
    h1_hi = np.maximum((raw1 >> shift) + 1, 0)

    w2_pos, w2_neg = np.maximum(w2, 0), np.minimum(w2, 0)
    raw2_fl = checked_int_matmul(h1_fl, w2) + b2 * scale
    raw2_lo = checked_int_matmul(h1_lo, w2_pos) + checked_int_matmul(h1_hi, w2_neg) + b2 * scale
    raw2_hi = checked_int_matmul(h1_hi, w2_pos) + checked_int_matmul(h1_lo, w2_neg) + b2 * scale

    h2_fl = np.maximum(raw2_fl >> shift, 0)
    h2_lo = np.maximum((raw2_lo >> shift) - 1, 0)
    h2_hi = np.maximum((raw2_hi >> shift) + 1, 0)

    w3_pos, w3_neg = np.maximum(w3, 0), np.minimum(w3, 0)
    raw3_fl = checked_int_matmul(h2_fl, w3) + b3 * scale
    raw3_lo = checked_int_matmul(h2_lo, w3_pos) + checked_int_matmul(h2_hi, w3_neg) + b3 * scale
    raw3_hi = checked_int_matmul(h2_hi, w3_pos) + checked_int_matmul(h2_lo, w3_neg) + b3 * scale

    max_raw1 = int(np.max(np.abs(raw1)))
    max_raw2 = int(max(np.max(np.abs(raw2_lo)), np.max(np.abs(raw2_hi))))
    max_raw3 = int(max(np.max(np.abs(raw3_lo)), np.max(np.abs(raw3_hi))))
    if max(max_raw1, max_raw2) >= trunc_limit:
        raise ValueError(f'CRT trunc_relu accumulator max({max_raw1}, {max_raw2}) >= limit {trunc_limit}')
    if max_raw3 >= half_limit:
        raise ValueError(f'CRT output score accumulator {max_raw3} >= half modulus {half_limit}')

    bounds = dict(
        primes=list(primes),
        crt_modulus=m_mod,
        shift=shift,
        trunc_limit=trunc_limit,
        half_limit=half_limit,
        raw1_max_abs=max_raw1,
        raw2_max_abs=max_raw2,
        raw3_max_abs=max_raw3,
        ambiguous_sign_examples=int(np.count_nonzero((raw3_lo < 0) & (raw3_hi >= 0))),
    )
    return bounds, raw3_fl, raw3_lo, raw3_hi


def load_crt_topic_mlp(checkpoint: Path, batch: int, *, shift: int = 9,
                       primes: tuple[int, ...] = DEFAULT_CRT3_PRIMES) -> dict:
    with np.load(checkpoint, allow_pickle=False) as data:
        if str(data['architecture']) != 'topic_mlp_relu':
            raise ValueError('unsupported topic MLP checkpoint architecture')
        src_fb = int(data['fractional_bits'])
        x_f = data['x_test'].astype(np.float64) / (1 << src_fb)
        y_test = data['y_test'].astype(np.int64)
        floats = {k: data['float_' + k].astype(np.float32) for k in WEIGHT_NAMES}

    scale = 1 << shift
    x_q_all = np.rint(x_f * scale).astype(np.int64)
    weights = {k: np.rint(floats[k].astype(np.float64) * scale).astype(np.int64) for k in WEIGHT_NAMES}
    # Pre-scale biases by 2**shift so bias addition matches accumulated matmul scale directly.
    full_bounds, raw3_fl_all, _, _ = propagate_crt_bounds(x_q_all, weights, shift, primes)
    test_q_metrics = binary_metrics(y_test, raw3_fl_all)

    idx = np.arange(batch) % len(x_q_all)
    x_q = x_q_all[idx]
    bounds, raw3_fl, raw3_lo, raw3_hi = propagate_crt_bounds(x_q, weights, shift, primes)
    scaled_weights = dict(weights)
    for bname in ('b1', 'b2', 'b3'):
        scaled_weights[bname] = weights[bname] * scale
    return dict(
        architecture='topic_mlp_relu_crt',
        primes=primes,
        shift=shift,
        x=x_q,
        y_batch=y_test[idx],
        weights=scaled_weights,
        float_weights=floats,
        bounds=bounds,
        full_test_bounds=full_bounds,
        full_test_q_metrics=test_q_metrics,
        expected_floor=(raw3_fl >= 0).reshape(-1).astype(int).tolist(),
        lower_label=(raw3_lo >= 0).reshape(-1).astype(int).tolist(),
        upper_label=(raw3_hi >= 0).reshape(-1).astype(int).tolist(),
    )


def _forward_crt_score(rt, fx_in, params, shift: int):
    z1 = rt.crt_add(rt._crt_poly_exchange(crt_matmul_local(fx_in, params['w1'])), params['b1'])
    h1 = crt_trunc_relu(rt, z1, shift)
    z2 = rt.crt_matmul(h1, params['w2'], params['b2'])
    h2 = crt_trunc_relu(rt, z2, shift)
    return rt.crt_matmul(h2, params['w3'], params['b3'])


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
        seeds = correlated_seeds(p, transport, context=b'Rep4-topic-mlp-crt-v1')
        if direct:
            from rep4gpu.protocol.cuda_ipc_transport import CudaIpcTransport
            data_transport = CudaIpcTransport(transport, device)
        else:
            from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge
            data_transport = PinnedDeviceByteBridge(transport, device)

        from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
        primes = tuple(cfg['primes'])
        rt = CrtPhaseGraphParty(p, data_transport, seeds, device=device,
                                primes=primes, optimized=True, gpu_open=True)
        own = np.load(cfg['input']) if 'input' in cfg else None
        params = {}
        for name in WEIGHT_NAMES:
            shape = tuple(cfg['weight_shapes'][name])
            val = torch.from_numpy(own[name].astype(np.int64)) if p == 1 else None
            sh = rt.crt_input(1, val, shape)
            params[name] = rt.crt_prepare_weight(sh) if name.startswith('w') else sh

        shift = cfg['shift']
        input_value = torch.from_numpy(own['x'].astype(np.int64)).to(device) if p == 0 else None
        input_shape = tuple(cfg['input_shape'])
        plan = None
        if cfg['phase_graphs']:
            initial = rt.crt_input(0, input_value, input_shape)
            initial_fx = rt.crt_eval_input(initial)
            del initial
            torch.cuda.synchronize(device)
            plan = rt.prepare_inference(lambda fx_in, prm: _forward_crt_score(rt, fx_in, prm, shift),
                                        initial_fx, params, check_at_send=True)
            torch.cuda.synchronize(device)
            del initial_fx
            torch.cuda.empty_cache()

        resident = rt.crt_input(0, input_value, input_shape) if cfg['resident_inputs'] else None
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
                score_share = _forward_crt_score(rt, rt.crt_eval_input(x_share), params, shift)
            opened = rt.crt_nonnegative_label_open_cpu(score_share)
            torch.cuda.synchronize(device)
            if resident is None:
                del x_share
            t_end = time.perf_counter()
            dev_bytes = (data_transport.bytes_sent - dev_byte_start) if direct else 0
            ctrl_bytes = (data_transport.control_bytes_sent - dev_ctrl_start) if direct else 0
            sock_bytes = transport.bytes_sent - sock_byte_start
            payload_bytes = dev_bytes if direct else sock_bytes
            host_payload_bytes = (sock_bytes - ctrl_bytes) if direct else sock_bytes
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
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if direct:
            data_transport.close()
            data_transport = transport = None
        Path(result_path).write_text(json.dumps(dict(
            party=p, passed=True, setup_ms=setup_ms, peak_vram_mb=peak_vram_mb,
            measurements=measurements, predictions=predictions.tolist(),
            device_transport=mode, control_authenticated_tls=control_auth,
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


def isolated_party_gpu_estimate(model: dict, device: str = 'cuda', repeats: int = 15, warmup: int = 3) -> dict:
    from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
    from rep4gpu.protocol.runtime import Share, make_seeds

    class _ZeroNetworkStub:
        staging = 'isolated_zero_network'
        authenticated = False

        def __init__(self, party: int):
            self.party = party

        def exchange_device(self, *args, **kwargs):
            raise RuntimeError('isolated estimate cannot use live transport')

    primes = model['primes']
    n_lanes = len(primes)
    shift = model['shift']
    x_shape = tuple(model['x'].shape)
    w_shapes = {k: tuple(v.shape) for k, v in model['weights'].items()}
    seeds = make_seeds()
    per_party_ms = {}
    per_party_phase_medians_ms = []
    per_party_peak_mb = {}
    total_payload_bytes = 0
    rounds = 0

    for p in range(4):
        torch.cuda.reset_peak_memory_stats(device)
        rt = CrtPhaseGraphParty(p, _ZeroNetworkStub(p), {e: seeds[e] for e in range(4) if e != p},
                                device=device, primes=primes, optimized=True, gpu_open=True)
        z = lambda sh: Share(tuple(torch.zeros((n_lanes, *sh), dtype=torch.uint8, device=device) for _ in range(3)), False)
        params = {k: (rt.crt_prepare_weight(z(sh)) if k.startswith('w') else z(sh)) for k, sh in w_shapes.items()}
        x_share = z(x_shape)
        fx0 = rt.crt_eval_input(x_share)
        plan = rt.prepare_inference(lambda fx_in, prm: _forward_crt_score(rt, fx_in, prm, shift),
                                    fx0, params, check_at_send=True)
        del fx0
        torch.cuda.empty_cache()

        graphs = [v for k, v in plan.actions if k == 'graph']
        exchanges = [v for k, v in plan.actions if k == 'exchange']
        rounds = len(exchanges)
        total_payload_bytes += sum(sum(t.numel() for t in o.values()) for o, _, _ in exchanges)

        with torch.cuda.stream(plan.stream):
            for _ in range(warmup):
                plan.nonces.copy_(plan.host_nonces, non_blocking=True)
                rt.crt_eval_input(x_share, out=plan.inputs[0])
                for g in graphs:
                    g.replay()
        plan.stream.synchronize()

        total_samples = []
        phase_samples = [[] for _ in range(len(graphs) + 1)]
        se = [torch.cuda.Event(enable_timing=True) for _ in range(len(graphs) + 1)]
        ee = [torch.cuda.Event(enable_timing=True) for _ in range(len(graphs) + 1)]
        ts, te = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        for _ in range(repeats):
            with torch.cuda.stream(plan.stream):
                ts.record(plan.stream)
                plan.nonces.copy_(plan.host_nonces, non_blocking=True)
                se[0].record(plan.stream)
                rt.crt_eval_input(x_share, out=plan.inputs[0])
                ee[0].record(plan.stream)
                for idx, g in enumerate(graphs):
                    se[idx + 1].record(plan.stream)
                    g.replay()
                    ee[idx + 1].record(plan.stream)
                te.record(plan.stream)
            plan.stream.synchronize()
            total_samples.append(ts.elapsed_time(te))
            for idx in range(len(graphs) + 1):
                phase_samples[idx].append(se[idx].elapsed_time(ee[idx]))

        per_party_ms[f'party{p}'] = float(np.median(total_samples))
        per_party_phase_medians_ms.append([float(np.median(s)) for s in phase_samples])
        per_party_peak_mb[f'party{p}'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        del graphs, exchanges, plan, x_share, params, rt
        gc.collect()
        torch.cuda.empty_cache()

    return dict(
        per_party_ms=per_party_ms,
        bottleneck_party_ms=max(per_party_ms.values()),
        sum_4_parties_ms=sum(per_party_ms.values()),
        phase_max_sum_ms=float(np.sum(np.max(np.asarray(per_party_phase_medians_ms), axis=0))),
        rounds=rounds,
        global_inference_payload_bytes=total_payload_bytes,
        per_party_peak_vram_mb=per_party_peak_mb,
    )


def run(checkpoint: Path, output: Path, *, batch: int = 2048, repeats: int = 5, warmup: int = 2,
        timeout: float = 300, shift: int = 9, primes: tuple[int, ...] = DEFAULT_CRT3_PRIMES,
        phase_graphs: bool = True, device_transport: str = 'cuda-ipc',
        resident_inputs: bool = False, include_isolated_estimate: bool = False) -> dict:
    model = load_crt_topic_mlp(checkpoint, batch, shift=shift, primes=primes)
    x, weights = model['x'], model['weights']
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem + '_logs')
    logs.mkdir(exist_ok=True)
    baseline = native_mlp_baselines(x, model['float_weights'], shift, device='cuda')
    fastest_native_key = min(baseline, key=baseline.get)
    fastest_native_ms = baseline[fastest_native_key]

    processes, handles = [], []
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='rep4-topicmlp-crt-') as folder:
        folder = Path(folder)
        ca, identities = certificates(folder)
        addresses = reserve_addresses()
        np.savez(folder / 'owner0.npz', x=x.astype(np.int32))
        np.savez(folder / 'owner1.npz', **{k: v.astype(np.int64) for k, v in weights.items()})
        try:
            for p in range(4):
                result_path = logs / f'party{p}.json'
                result_path.unlink(missing_ok=True)
                cfg = dict(
                    party=p, device='cuda', primes=list(primes), shift=shift,
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
                    [sys.executable, '-m', 'experiments.benchmarks.inference.topic_mlp_crt',
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

    parties = [
        json.loads((logs / f'party{p}.json').read_text()) if (logs / f'party{p}.json').exists()
        else dict(party=p, passed=False, error='no result')
        for p in range(4)
    ]
    codes = [proc.returncode for proc in processes]
    passed = not timed_out and all(code == 0 for code in codes) and all(p['passed'] for p in parties)
    preds0 = parties[0].get('predictions')
    all_agree = passed and all(
        p.get('predictions') == preds0
        and all(p['measurements'][i]['prediction_sha256'] == parties[0]['measurements'][i]['prediction_sha256']
                for i in range(repeats))
        for p in parties
    )
    interval_valid = False
    floor_agree_frac = 0.0
    if all_agree and preds0 is not None:
        pred_arr = np.asarray(preds0, dtype=np.int64)
        lo_arr = np.asarray(model['lower_label'], dtype=np.int64)
        hi_arr = np.asarray(model['upper_label'], dtype=np.int64)
        fl_arr = np.asarray(model['expected_floor'], dtype=np.int64)
        interval_valid = bool(np.all((pred_arr >= lo_arr) & (pred_arr <= hi_arr)))
        floor_agree_frac = float(np.mean(pred_arr == fl_arr))
    valid = passed and all_agree and interval_valid

    report = dict(
        passed=valid,
        all_parties_agree=all_agree,
        interval_valid=interval_valid,
        floor_label_agreement=floor_agree_frac,
        architecture='topic_mlp_relu_crt',
        primes=list(primes),
        shift=shift,
        device_transport=device_transport,
        phase_graphs=phase_graphs,
        resident_inputs=resident_inputs,
        batch=batch,
        input_shape=list(x.shape),
        weight_shapes={k: list(v.shape) for k, v in weights.items()},
        total_parameters=sum(int(v.size) for v in weights.values()),
        repeats=repeats,
        warmup=warmup,
        checkpoint=str(Path(checkpoint).resolve()),
        hardware=torch.cuda.get_device_name(),
        full_test_q_metrics=model['full_test_q_metrics'],
        native_ms=baseline,
        fastest_native_key=fastest_native_key,
        fastest_native_ms=fastest_native_ms,
        range_bounds=model['bounds'],
        return_codes=codes,
        timed_out=timed_out,
        parties=parties,
    )
    if valid:
        for metric in ('input_ms', 'checked_inference_ms', 'online_ms'):
            samples = [max(p['measurements'][i][metric] for p in parties) for i in range(repeats)]
            report[metric] = float(np.median(samples))
        report['inference_texts_per_second'] = batch * 1000 / report['checked_inference_ms']
        report['online_texts_per_second'] = batch * 1000 / report['online_ms']
        report['slowdown_resident'] = report['checked_inference_ms'] / fastest_native_ms
        report['slowdown_with_input_sharing'] = report['online_ms'] / fastest_native_ms
        report['slowdown_resident_vs_fp32_notf32_graph'] = report['checked_inference_ms'] / baseline['fp32_notf32_graph']
        report['slowdown_resident_vs_fp16_graph'] = report['checked_inference_ms'] / baseline['fp16_graph']
        for key in ('payload_bytes', 'input_payload_bytes', 'inference_payload_bytes',
                    'host_payload_bytes', 'socket_application_bytes'):
            report['global_' + key + '_per_batch'] = sum(p['measurements'][0][key] for p in parties)
        report['rounds_per_batch'] = parties[0]['measurements'][0]['rounds']
        report['max_party_peak_vram_mb'] = max(p['peak_vram_mb'] for p in parties)
    if include_isolated_estimate:
        iso = isolated_party_gpu_estimate(model, device='cuda')
        report['isolated_party_gpu_estimate'] = dict(
            **iso,
            slowdown_phase_max_vs_fastest_native=iso['phase_max_sum_ms'] / fastest_native_ms,
            slowdown_phase_max_vs_fp32_notf32_graph=iso['phase_max_sum_ms'] / baseline['fp32_notf32_graph'],
            throughput_texts_per_second=batch * 1000 / iso['phase_max_sum_ms'],
        )
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('experiments/results/examples/topic_mlp_classifier/model.npz'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/inference/topic_mlp_crt_cuda.json'))
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--shift', type=int, default=9)
    parser.add_argument('--primes', type=int, nargs='+', default=list(DEFAULT_CRT3_PRIMES))
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
        warmup=args.warmup, timeout=args.timeout, shift=args.shift,
        primes=tuple(args.primes), phase_graphs=not args.no_phase_graphs,
        device_transport=args.device_transport, resident_inputs=args.resident_inputs,
        include_isolated_estimate=args.isolated_estimate,
    )
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
