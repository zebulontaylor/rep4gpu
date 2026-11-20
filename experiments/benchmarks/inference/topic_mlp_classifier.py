"""Measure private checked 4-party Ring32 ReLU Topic MLP inference against native PyTorch.

Party 0 owns the private hashed n-gram features X (shape B x 4096); Party 1 owns
all 4,720,641 learned parameters (w1, b1, w2, b2, w3, b3); Parties 2 and 3 hold
no plaintext. Only the final binary topic labels (Sci/Tech vs other) are opened
after all active-security BLAKE3 consistency checks pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from experiments.examples.topic_mlp_classifier import propagate_bounds
from rep4gpu.arithmetic.activations import FixedPoint
from rep4gpu.models.text_classifier import nonnegative_label
from rep4gpu.protocol.setup import correlated_seeds


WEIGHT_NAMES = ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')


def load_topic_mlp(checkpoint: Path, batch: int, *, rounding: str = 'probabilistic') -> dict:
    """Load checkpoint, build timed batch from held-out test set, and validate all ring bounds."""
    if batch < 1:
        raise ValueError('batch must be positive')
    with np.load(checkpoint, allow_pickle=False) as data:
        if str(data['architecture']) != 'topic_mlp_relu':
            raise ValueError('unsupported topic MLP checkpoint architecture')
        bits = int(data['ring_bits'])
        f_bits = int(data['fractional_bits'])
        if bits != 32:
            raise ValueError('topic MLP benchmark requires ring32 checkpoint')
        x0 = data['x_test']
        weights = {k: data[k].astype(np.int64) for k in WEIGHT_NAMES}
        floats = {k: data['float_' + k].astype(np.float32) for k in WEIGHT_NAMES}

    if x0.ndim != 2 or not len(x0):
        raise ValueError('x_test must be a nonempty 2D matrix')
    _, d = x0.shape
    h1 = weights['w1'].shape[1]
    h2 = weights['w2'].shape[1]
    expected_shapes = dict(w1=(d, h1), b1=(h1,), w2=(h1, h2), b2=(h2,), w3=(h2, 1), b3=(1,))
    for name, shape in expected_shapes.items():
        if weights[name].shape != shape or floats[name].shape != shape:
            raise ValueError(f'parameter {name} shape mismatch: expected {shape}')

    idx = np.arange(batch) % len(x0)
    x = x0[idx].astype(np.int64)
    bounds, raw3_floor, raw3_lo, raw3_hi = propagate_bounds(x, weights, f_bits, bits=bits, rounding=rounding)
    expected_floor = (raw3_floor >= 0).reshape(-1).astype(np.int64)
    lower_label = (raw3_lo >= 0).reshape(-1).astype(np.int64)
    upper_label = (raw3_hi >= 0).reshape(-1).astype(np.int64)
    return dict(
        architecture='topic_mlp_relu',
        bits=bits,
        fractional_bits=f_bits,
        rounding=rounding,
        x=x,
        weights=weights,
        float_weights=floats,
        bounds=bounds,
        expected_floor=expected_floor.tolist(),
        lower_label=lower_label.tolist(),
        upper_label=upper_label.tolist(),
    )


def native_mlp_baselines(x_int: np.ndarray, float_weights: dict, fractional_bits: int,
                         device: str = 'cuda', repeats: int = 100) -> dict:
    """Time honest native PyTorch baselines (FP32 no-TF32, FP32 TF32, FP16, BF16; eager & graph; plus sparse L1)."""
    torch.set_num_threads(1)
    scale = float(1 << fractional_bits)
    x_f32 = torch.from_numpy(x_int.astype(np.float32) / scale).to(device).contiguous()
    results = {}

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        configs = (
            ('fp32_notf32', torch.float32, False),
            ('fp32_tf32', torch.float32, True),
            ('fp16', torch.float16, True),
            ('bf16', torch.bfloat16, True),
        )
        for name, dtype, allow_tf32 in configs:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            x_d = x_f32.to(dtype).contiguous()
            w1 = torch.as_tensor(float_weights['w1'], device=device, dtype=dtype).contiguous()
            b1 = torch.as_tensor(float_weights['b1'], device=device, dtype=dtype).contiguous()
            w2 = torch.as_tensor(float_weights['w2'], device=device, dtype=dtype).contiguous()
            b2 = torch.as_tensor(float_weights['b2'], device=device, dtype=dtype).contiguous()
            w3 = torch.as_tensor(float_weights['w3'], device=device, dtype=dtype).contiguous()
            b3 = torch.as_tensor(float_weights['b3'], device=device, dtype=dtype).contiguous()

            def fn():
                h1 = F.relu(torch.addmm(b1, x_d, w1))
                h2 = F.relu(torch.addmm(b2, h1, w2))
                return torch.addmm(b3, h2, w3) >= 0

            for _ in range(10):
                fn()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize(device)
            results[f'{name}_eager'] = (time.perf_counter() - t0) * 1000 / repeats

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
            results[f'{name}_graph'] = (time.perf_counter() - t0) * 1000 / repeats

        # Also time sparse-CSR first layer (since hashed n-grams are sparse) to ensure honest fastest baseline.
        torch.backends.cuda.matmul.allow_tf32 = True
        for name, dtype in (('sparse_l1_fp32', torch.float32), ('sparse_l1_fp16', torch.float16)):
            x_csr = x_f32.to(dtype).to_sparse_csr()
            w1 = torch.as_tensor(float_weights['w1'], device=device, dtype=dtype).contiguous()
            b1 = torch.as_tensor(float_weights['b1'], device=device, dtype=dtype).contiguous()
            w2 = torch.as_tensor(float_weights['w2'], device=device, dtype=dtype).contiguous()
            b2 = torch.as_tensor(float_weights['b2'], device=device, dtype=dtype).contiguous()
            w3 = torch.as_tensor(float_weights['w3'], device=device, dtype=dtype).contiguous()
            b3 = torch.as_tensor(float_weights['b3'], device=device, dtype=dtype).contiguous()

            def fn_sparse():
                h1 = F.relu(torch.sparse.mm(x_csr, w1) + b1)
                h2 = F.relu(torch.addmm(b2, h1, w2))
                return torch.addmm(b3, h2, w3) >= 0

            for _ in range(10):
                fn_sparse()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn_sparse()
            torch.cuda.synchronize(device)
            results[f'{name}_eager'] = (time.perf_counter() - t0) * 1000 / repeats
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    torch.cuda.empty_cache()
    return results


def _build_forward(rt, fractional_bits: int):
    fp = FixedPoint(rt, fractional_bits)
    scale = 1 << fractional_bits

    def forward(x, params):
        h1 = fp.matmul_relu(x, params['w1'], params['b1'])
        h2 = fp.matmul_relu(h1, params['w2'], params['b2'])
        score = rt.add(rt.matmul(h2, params['w3']), rt.mul_public(params['b3'], scale))
        return rt.open(nonnegative_label(rt, score))

    return forward


def _prepare_weights(rt, raw_weights: dict, weight_storage: str) -> dict:
    prepared = dict(raw_weights)
    # w1 (4096x1024) and w2 (1024x512) have even dimensions (compatible with strassen32 or words32);
    # w3 (512x1) has odd output dim 1 and uses words32.
    prepared['w1'] = rt.prepare_weight_share(raw_weights['w1'], storage=weight_storage)
    prepared['w2'] = rt.prepare_weight_share(raw_weights['w2'], storage=weight_storage)
    prepared['w3'] = rt.prepare_weight_share(raw_weights['w3'], storage='words32')
    return prepared


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
        seeds = correlated_seeds(p, transport, context=b'Rep4-topic-mlp-v1')
        if direct:
            from rep4gpu.protocol.cuda_ipc_transport import CudaIpcTransport
            data_transport = CudaIpcTransport(transport, device)
        else:
            from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge
            data_transport = PinnedDeviceByteBridge(transport, device)

        from rep4gpu.models.phase_graph import PhaseGraphParty
        from rep4gpu.protocol.device_input import DeviceInputMixin

        class TopicMlpParty(DeviceInputMixin, PhaseGraphParty):
            pass

        rounding = cfg['rounding']
        prob = rounding == 'probabilistic'
        rt = TopicMlpParty(
            p, data_transport, seeds, device=device, bits=cfg['bits'], binary_bits=32,
            optimized=True, gpu_open=True, rounding=rounding,
            bounded_sign=prob, sparse_products=prob, narrow_products=prob,
        )
        own = np.load(cfg['input']) if 'input' in cfg else None
        raw_weights = {}
        for name in WEIGHT_NAMES:
            shape = tuple(cfg['weight_shapes'][name])
            val = torch.from_numpy(own[name].astype(np.int64)) if p == 1 else None
            raw_weights[name] = rt.input(1, val, shape)
        weights = _prepare_weights(rt, raw_weights, cfg['weight_storage'])
        del raw_weights
        torch.cuda.empty_cache()

        forward = _build_forward(rt, cfg['fractional_bits'])
        input_value = torch.from_numpy(own['x'].astype(np.int64)).to(device) if p == 0 else None
        input_shape = tuple(cfg['input_shape'])
        plan = None
        if cfg['phase_graphs']:
            initial = rt.input(0, input_value, input_shape)
            torch.cuda.synchronize(device)
            plan = rt.prepare_inference(forward, initial, weights, check_at_send=True)
            torch.cuda.synchronize(device)
            del initial
            torch.cuda.empty_cache()

        resident = rt.input(0, input_value, input_shape) if cfg['resident_inputs'] else None
        torch.cuda.synchronize(device)
        setup_ms = (time.perf_counter() - start) * 1000

        meter = data_transport if direct else transport
        measurements = []
        predictions = None
        for iteration in range(cfg['warmup'] + cfg['repeats']):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            byte_start, round_start = meter.bytes_sent, meter.rounds
            control_start = transport.bytes_sent
            host_start = getattr(meter, 'host_payload_bytes_sent', 0) if direct else byte_start
            x_share = resident if resident is not None else rt.input(0, input_value, input_shape)
            torch.cuda.synchronize(device)
            t_shared = time.perf_counter()
            input_bytes = meter.bytes_sent - byte_start
            opened = plan.run(x_share, weights) if plan is not None else forward(x_share, weights)
            torch.cuda.synchronize(device)
            if resident is None:
                del x_share
            t_end = time.perf_counter()
            host_payload_bytes = (
                getattr(meter, 'host_payload_bytes_sent', 0) - host_start
                if direct else meter.bytes_sent - byte_start
            )
            if direct and host_payload_bytes:
                raise RuntimeError('CUDA IPC transport used host protocol payload during timed batch')
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
                    payload_bytes=meter.bytes_sent - byte_start,
                    inference_payload_bytes=(meter.bytes_sent - byte_start) - input_bytes,
                    host_payload_bytes=host_payload_bytes,
                    socket_application_bytes=transport.bytes_sent - control_start,
                    rounds=meter.rounds - round_start,
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


def isolated_party_gpu_estimate(model: dict, *, weight_storage: str = 'strassen32',
                                device: str = 'cuda', repeats: int = 15, warmup: int = 3) -> dict:
    """Measure uncontended single-party GPU compute per role (zero-network 4-GPU estimate).

    Builds one party at a time on the GPU (avoiding 4-process VRAM multiplication),
    captures the exact PhaseGraphParty CUDA graph segments (all Ring32/Strassen
    matmuls, ChaCha20 masks, BLAKE3 hashes, SWAR/prefix carry trees, B2A
    conversions, and checked output opening), and times per-phase replay medians.
    """
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.runtime import Share, make_seeds

    class _ZeroNetworkStub:
        staging = 'isolated_zero_network'
        authenticated = False

        def __init__(self, party: int):
            self.party = party

        def exchange_device(self, *args, **kwargs):
            raise RuntimeError('isolated estimate cannot use live transport')

    rounding = model['rounding']
    prob = rounding == 'probabilistic'
    x_shape = tuple(model['x'].shape)
    w_shapes = {k: tuple(v.shape) for k, v in model['weights'].items()}
    seeds = make_seeds()
    per_party_ms = {}
    per_party_phase_medians_ms = []
    per_party_peak_mb = {}
    total_payload_bytes = 0
    rounds_per_party = []

    for p in range(4):
        torch.cuda.reset_peak_memory_stats(device)
        rt = PhaseGraphParty(
            p, _ZeroNetworkStub(p), {e: seeds[e] for e in range(4) if e != p},
            device=device, bits=model['bits'], binary_bits=32,
            optimized=True, gpu_open=True, rounding=rounding,
            bounded_sign=prob, sparse_products=prob, narrow_products=prob,
        )
        zero_share = lambda shape: Share(
            tuple(torch.zeros(shape, dtype=torch.int64, device=device) for _ in range(3)), False
        )
        raw_weights = {name: zero_share(shape) for name, shape in w_shapes.items()}
        weights = _prepare_weights(rt, raw_weights, weight_storage)
        del raw_weights
        x_share = zero_share(x_shape)
        forward = _build_forward(rt, model['fractional_bits'])
        plan = rt.prepare_inference(forward, x_share, weights, check_at_send=True)
        del x_share
        torch.cuda.empty_cache()

        graphs = [val for kind, val in plan.actions if kind == 'graph']
        exchanges = [val for kind, val in plan.actions if kind == 'exchange']
        rounds_per_party.append(len(exchanges))
        total_payload_bytes += sum(sum(t.numel() for t in outgoing.values()) for outgoing, _, _ in exchanges)

        # Warmup all graph segments
        with torch.cuda.stream(plan.stream):
            for _ in range(warmup):
                plan.nonces.copy_(plan.host_nonces, non_blocking=True)
                for g in graphs:
                    g.replay()
        plan.stream.synchronize()

        # Time total per-party compute AND per-phase compute using CUDA events.
        total_samples = []
        phase_samples = [[] for _ in graphs]
        start_evs = [torch.cuda.Event(enable_timing=True) for _ in graphs]
        end_evs = [torch.cuda.Event(enable_timing=True) for _ in graphs]
        t_start_ev = torch.cuda.Event(enable_timing=True)
        t_end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(repeats):
            with torch.cuda.stream(plan.stream):
                t_start_ev.record(plan.stream)
                plan.nonces.copy_(plan.host_nonces, non_blocking=True)
                for idx, g in enumerate(graphs):
                    start_evs[idx].record(plan.stream)
                    g.replay()
                    end_evs[idx].record(plan.stream)
                t_end_ev.record(plan.stream)
            plan.stream.synchronize()
            total_samples.append(t_start_ev.elapsed_time(t_end_ev))
            for idx in range(len(graphs)):
                phase_samples[idx].append(start_evs[idx].elapsed_time(end_evs[idx]))

        per_party_ms[f'party{p}'] = float(np.median(total_samples))
        per_party_phase_medians_ms.append([float(np.median(s)) for s in phase_samples])
        per_party_peak_mb[f'party{p}'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        del graphs, exchanges, plan, weights, rt
        gc.collect()
        torch.cuda.empty_cache()

    phase_max_sum_ms = float(np.sum(np.max(np.asarray(per_party_phase_medians_ms), axis=0)))
    return dict(
        per_party_ms=per_party_ms,
        bottleneck_party_ms=max(per_party_ms.values()),
        phase_max_sum_ms=phase_max_sum_ms,
        rounds=rounds_per_party[0],
        global_inference_payload_bytes=total_payload_bytes,
        per_party_peak_vram_mb=per_party_peak_mb,
    )


def run(checkpoint: Path, output: Path, *, batch: int = 512, repeats: int = 5, warmup: int = 2,
        timeout: float = 300, rounding: str = 'probabilistic', weight_storage: str = 'strassen32',
        phase_graphs: bool = True, device_transport: str = 'cuda-ipc',
        resident_inputs: bool = False, include_isolated_estimate: bool = False,
        isolated_only: bool = False) -> dict:
    if device_transport not in ('host', 'cuda-ipc'):
        raise ValueError('device_transport must be host or cuda-ipc')
    if weight_storage not in ('words32', 'strassen32'):
        raise ValueError('weight_storage must be words32 or strassen32')
    if weight_storage == 'strassen32' and batch % 2 != 0:
        raise ValueError('strassen32 requires even batch size')

    model = load_topic_mlp(checkpoint, batch, rounding=rounding)
    x, weights = model['x'], model['weights']
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = native_mlp_baselines(x, model['float_weights'], model['fractional_bits'], device='cuda')
    fastest_native_key = min(baseline, key=baseline.get)
    fastest_native_ms = baseline[fastest_native_key]

    report = dict(
        architecture='topic_mlp_relu',
        bits=model['bits'],
        fractional_bits=model['fractional_bits'],
        rounding=rounding,
        weight_storage=weight_storage,
        device_transport=device_transport if not isolated_only else 'isolated_only',
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
        native_ms=baseline,
        fastest_native_key=fastest_native_key,
        fastest_native_ms=fastest_native_ms,
        range_bounds=model['bounds'],
    )

    if not isolated_only:
        logs = output.parent / (output.stem + '_logs')
        logs.mkdir(exist_ok=True)
        processes, handles = [], []
        timed_out = False
        with tempfile.TemporaryDirectory(prefix='rep4-topicmlp-') as folder:
            folder = Path(folder)
            ca, identities = certificates(folder)
            addresses = reserve_addresses()
            np.savez(folder / 'owner0.npz', x=x.astype(np.int32))
            np.savez(folder / 'owner1.npz', **{k: v.astype(np.int32) for k, v in weights.items()})
            try:
                for p in range(4):
                    result_path = logs / f'party{p}.json'
                    result_path.unlink(missing_ok=True)
                    cfg = dict(
                        party=p, device='cuda', bits=model['bits'],
                        fractional_bits=model['fractional_bits'],
                        rounding=rounding, weight_storage=weight_storage,
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
                        [sys.executable, '-m', 'experiments.benchmarks.inference.topic_mlp_classifier',
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
        exact_floor_match = False
        if all_agree and preds0 is not None:
            pred_arr = np.asarray(preds0, dtype=np.int64)
            lo_arr = np.asarray(model['lower_label'], dtype=np.int64)
            hi_arr = np.asarray(model['upper_label'], dtype=np.int64)
            fl_arr = np.asarray(model['expected_floor'], dtype=np.int64)
            interval_valid = bool(np.all((pred_arr >= lo_arr) & (pred_arr <= hi_arr)))
            exact_floor_match = bool(np.array_equal(pred_arr, fl_arr))
        valid = passed and all_agree and (exact_floor_match if rounding == 'floor' else interval_valid)

        report.update(
            passed=valid,
            all_parties_agree=all_agree,
            interval_valid=interval_valid,
            exact_floor_match=exact_floor_match,
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
            report['slowdown_resident_vs_fp32_notf32_graph'] = (
                report['checked_inference_ms'] / baseline['fp32_notf32_graph']
            )
            report['slowdown_resident_vs_fp16_graph'] = (
                report['checked_inference_ms'] / baseline['fp16_graph']
            )
            for key in ('payload_bytes', 'input_payload_bytes', 'inference_payload_bytes',
                        'host_payload_bytes', 'socket_application_bytes'):
                report['global_' + key + '_per_batch'] = sum(p['measurements'][0][key] for p in parties)
            report['rounds_per_batch'] = parties[0]['measurements'][0]['rounds']
            report['max_party_peak_vram_mb'] = max(p['peak_vram_mb'] for p in parties)
    else:
        report['passed'] = True

    if include_isolated_estimate or isolated_only:
        iso = isolated_party_gpu_estimate(model, weight_storage=weight_storage, device='cuda')
        report['isolated_party_gpu_estimate'] = dict(
            **iso,
            slowdown_phase_max_vs_fastest_native=iso['phase_max_sum_ms'] / fastest_native_ms,
            slowdown_bottleneck_vs_fastest_native=iso['bottleneck_party_ms'] / fastest_native_ms,
            slowdown_phase_max_vs_fp32_notf32_graph=iso['phase_max_sum_ms'] / baseline['fp32_notf32_graph'],
            slowdown_phase_max_vs_fp16_graph=iso['phase_max_sum_ms'] / baseline['fp16_graph'],
            throughput_texts_per_second=batch * 1000 / iso['phase_max_sum_ms'],
        )

    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('experiments/results/examples/topic_mlp_classifier/model.npz'))
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/inference/topic_mlp_cuda.json'))
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--rounding', choices=('probabilistic', 'floor'), default='probabilistic')
    parser.add_argument('--weight-storage', choices=('words32', 'strassen32'), default='strassen32')
    parser.add_argument('--no-phase-graphs', action='store_true')
    parser.add_argument('--device-transport', choices=('host', 'cuda-ipc'), default='cuda-ipc')
    parser.add_argument('--resident-inputs', action='store_true')
    parser.add_argument('--isolated-estimate', action='store_true')
    parser.add_argument('--isolated-only', action='store_true')
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config:
        return child(args.party_config, args.party_result)
    report = run(
        args.checkpoint, args.output, batch=args.batch, repeats=args.repeats,
        warmup=args.warmup, timeout=args.timeout, rounding=args.rounding,
        weight_storage=args.weight_storage, phase_graphs=not args.no_phase_graphs,
        device_transport=args.device_transport, resident_inputs=args.resident_inputs,
        include_isolated_estimate=args.isolated_estimate, isolated_only=args.isolated_only,
    )
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
