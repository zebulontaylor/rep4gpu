"""Measure a quantized text classifier in four mutually authenticated processes.

The checkpoint contains public evaluation fixtures. Each worker receives only its
owner's plaintext: party 0 receives features and party 1 receives model weights.
Only the final class is opened. The launcher checks it against integer inference.
"""
import argparse
from contextlib import nullcontext
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

from experiments.benchmarks.inference.socket_inference import certificates, reserve_addresses, MeteredTransport
from rep4gpu.protocol.runtime import Party
from rep4gpu.protocol.setup import correlated_seeds
from experiments.benchmarks.inference.text_classifier_model import load_model, shared_labels, plain_labels, encoder_ring_bounds


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def child(config_path, result_path, *, prepare_barrier=None, prepare_lock=None, sync_session=sync):
    cfg = json.loads(Path(config_path).read_text())
    p, device = cfg['party'], cfg['device']
    transport = None
    data_transport = None
    mode = cfg.get('device_transport', 'host')
    direct = mode in ('cuda-ipc', 'nccl')
    if prepare_barrier is None:
        torch.set_num_threads(1)
    try:
        if mode not in ('host', 'cuda-ipc', 'nccl'):
            raise ValueError(f'unknown device transport {mode!r}')
        if direct and torch.device(device).type != 'cuda':
            raise ValueError('a device-to-device transport requires CUDA')
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(**cfg['identity'])
        server.load_verify_locations(cfg['ca'])
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cfg['ca'])
        client.load_cert_chain(**cfg['identity'])
        transport = MeteredTransport(p, [tuple(a) for a in cfg['addresses']], timeout=cfg['timeout'],
                                    server_context=server, client_context=client,
                                    peer_hostnames={q: f'party{q}' for q in range(4)})
        start = time.perf_counter()
        seeds = correlated_seeds(p, transport, context=b'Rep4-text-classifier-v1')
        if torch.device(device).type == 'cuda':
            from rep4gpu.protocol.device_transport import DeviceByteBridge
            from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge
            if mode == 'cuda-ipc':
                from rep4gpu.protocol.cuda_ipc_transport import CudaIpcTransport
                data_transport = CudaIpcTransport(transport, device)
            elif mode == 'nccl':
                from rep4gpu.protocol.nccl_transport import NcclDeviceTransport
                data_transport = NcclDeviceTransport(transport, torch.distributed.group.WORLD, device)
            else:
                bridge = PinnedDeviceByteBridge if cfg.get('pinned_staging', False) else DeviceByteBridge
                data_transport = bridge(transport, device)
            from rep4gpu.models.classifier_runtime import ClassifierParty
            from rep4gpu.models.phase_graph import PhaseGraphParty
            runtime = PhaseGraphParty if cfg['generic_kernel'] or cfg['architecture'] != 'linear' or cfg.get('bits', 32) != 32 else ClassifierParty
            if direct or cfg.get('device_input', False):
                from rep4gpu.protocol.device_input import DeviceInputMixin
                runtime = type('DeviceInputClassifierParty', (DeviceInputMixin, runtime), {})
            rt = runtime(p, data_transport, seeds,
                                      device=device, bits=cfg.get('bits', 32), binary_bits=32, optimized=True, gpu_open=True,
                                      packed_arithmetic=cfg.get('packed_arithmetic', False),
                                      encoder_pipeline_stages=cfg.get('encoder_pipeline_stages', 0))
        else:
            rt = Party(p, transport, seeds, device=device, bits=cfg.get('bits', 32), binary_bits=32, prg='aes')
        own = np.load(cfg['input']) if 'input' in cfg else None
        encoder_bits = cfg.get('encoder_ring_bits')

        def encoder_ring():
            return rt.arithmetic_ring(encoder_bits) if encoder_bits is not None else nullcontext()

        def share_input(value):
            with encoder_ring():
                return rt.input(0, value, cfg['input_shape'])

        weights = {}
        for name, shape in cfg['weight_shapes'].items():
            with encoder_ring() if name in ('w1', 'b1') else nullcontext():
                value = rt.input(1, torch.from_numpy(own[name].astype(np.int64)) if p == 1 else None, shape)
                if torch.device(device).type == 'cuda' and (cfg['generic_kernel'] or cfg['architecture'] != 'linear' or cfg.get('bits', 32) == 16) and len(shape) == 2:
                    storage = 'limbs' if cfg.get('bits', 32) == 16 else 'words32'
                    if name == 'w1' and cfg.get('prepare_encoder'):
                        storage = 'pipelined32'
                    value = rt.prepare_weight_share(value, storage=storage)
            weights[name] = value
        def forward(x, parameters):
            return rt.open(shared_labels(rt, x, parameters, cfg['architecture'], cfg['fractional_bits'], cfg.get('public_signs', ()), encoder_bits))
        sync_session(device)
        setup_ms = (time.perf_counter()-start)*1000
        input_value = torch.from_numpy(own['x'].astype(np.int64)).to(device) if p == 0 else None
        plan = None
        if cfg['phase_graphs']:
            begin = time.perf_counter()
            initial = share_input(input_value)
            sync_session(device)
            if prepare_barrier is not None:
                prepare_barrier.wait(timeout=cfg['timeout'])
            with prepare_lock if prepare_lock is not None else nullcontext():
                plan = rt.prepare_inference(forward, initial, weights, check_at_send=True)
                sync_session(device)
            # The plan owns cloned input buffers; retaining this example keeps
            # another full batch of replicated shares alive between requests.
            del initial
            if prepare_barrier is not None:
                prepare_barrier.wait(timeout=cfg['timeout'])
            setup_ms += (time.perf_counter()-begin)*1000
        resident = share_input(input_value) if cfg.get('resident_inputs') else None
        meter = data_transport if direct else transport
        measurements = []
        for iteration in range(cfg['warmup']+cfg['repeats']):
            sync_session(device)
            start = time.perf_counter()
            byte_start, round_start = meter.bytes_sent, meter.rounds
            control_start = transport.bytes_sent
            host_start = getattr(meter, 'host_payload_bytes_sent', 0) if direct else byte_start
            x = resident if resident is not None else share_input(input_value)
            sync_session(device)
            shared = time.perf_counter()
            input_bytes = meter.bytes_sent-byte_start
            opened = plan.run(x, weights) if plan is not None else forward(x, weights)
            sync_session(device)
            del x
            end = time.perf_counter()
            host_payload_bytes = (getattr(meter, 'host_payload_bytes_sent', 0) - host_start
                                  if direct else meter.bytes_sent - byte_start)
            if direct and host_payload_bytes:
                raise RuntimeError('device transport used a host protocol payload during the timed batch')
            predictions = opened.cpu().numpy().reshape(-1)
            if iteration >= cfg['warmup']:
                measurements.append(dict(start_monotonic=start, end_monotonic=end,
                                         prediction_sha256=hashlib.sha256(predictions.astype('<i8').tobytes()).hexdigest(), input_ms=(shared-start)*1000, checked_inference_ms=(end-shared)*1000,
                                         online_ms=(end-start)*1000, input_payload_bytes=input_bytes,
                                         payload_bytes=meter.bytes_sent-byte_start,
                                         inference_payload_bytes=meter.bytes_sent-byte_start-input_bytes,
                                         host_payload_bytes=host_payload_bytes,
                                         socket_application_bytes=transport.bytes_sent-control_start,
                                         rounds=meter.rounds-round_start))
        control_authenticated = transport.authenticated
        if direct:
            data_transport.close()
            data_transport = transport = None
        Path(result_path).write_text(json.dumps(dict(
            party=p, passed=True, setup_ms=setup_ms, measurements=measurements,
            predictions=predictions.tolist(), device_transport=mode,
            authenticated_tls=control_authenticated and not direct,
            control_authenticated_tls=control_authenticated), indent=2)+'\n')
        return 0
    except Exception as error:
        if prepare_barrier is not None:
            prepare_barrier.abort()
        traceback.print_exc()
        Path(result_path).write_text(json.dumps(dict(party=p, passed=False, error=f'{type(error).__name__}: {error}'))+'\n')
        return 1
    finally:
        if data_transport is not None and direct:
            abort = getattr(data_transport, 'abort', data_transport.close)
            abort()
        elif transport is not None:
            transport.close()


def native(x, w, device, repeats=100, architecture='linear', fractional_bits=0, input_divisor=None, public_signs=()):
    """Time the full floating-point network and threshold, with one CPU thread."""
    torch.set_num_threads(1)
    results = {}
    for dtype in ([torch.float32, torch.float16] if torch.device(device).type == 'cuda' else [torch.float32]):
        values = w if isinstance(w, dict) else {'weights': w}
        a = torch.as_tensor(x, device=device).to(dtype) / (input_divisor if input_divisor is not None else 1 << fractional_bits)
        b = {name: torch.as_tensor(value, device=device).to(dtype) for name, value in values.items()}
        signs = torch.as_tensor(public_signs, device=device, dtype=dtype)
        for _ in range(10):
            out = plain_labels(a, b, architecture, signs)
        sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            out = plain_labels(a, b, architecture, signs)
        sync(device)
        name = str(dtype).split('.')[-1]
        results[name+'_eager'] = (time.perf_counter()-start)*1000/repeats
        if torch.device(device).type == 'cuda':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = plain_labels(a, b, architecture, signs)
            for _ in range(10):
                graph.replay()
            sync(device)
            start = time.perf_counter()
            for _ in range(repeats):
                graph.replay()
            sync(device)
            results[name+'_graph'] = (time.perf_counter()-start)*1000/repeats
    return results


def run(checkpoint, output, *, batch=128, device='cpu', repeats=3, warmup=1, timeout=300, generic_kernel=False, phase_graphs=False, pinned_staging=False, device_input=False, bits=32, native_baseline=None, packed_arithmetic=False, encoder_ring_bits=None, device_transport='host'):
    if min(batch, repeats) < 1 or warmup < 0 or timeout <= 0:
        raise ValueError('batch, repeats, timeout must be positive; warmup must be nonnegative')
    if device_transport not in ('host', 'cuda-ipc'):
        raise ValueError('local classifier transport must be host or cuda-ipc')
    if device_transport == 'cuda-ipc':
        if pinned_staging:
            raise ValueError('use either cuda-ipc or pinned host staging')
        device_input = True
    if phase_graphs and torch.device(device).type != 'cuda':
        raise ValueError('phase graphs require CUDA')
    if device_input and torch.device(device).type != 'cuda':
        raise ValueError('device input requires CUDA')
    if packed_arithmetic and (torch.device(device).type != 'cuda' or bits != 16):
        raise ValueError('packed arithmetic requires CUDA ring16')
    if encoder_ring_bits is not None and (torch.device(device).type != 'cuda' or bits != 16 or not packed_arithmetic):
        raise ValueError('mixed encoder requires CUDA packed ring16')
    model = load_model(checkpoint, batch, bits=bits)
    encoder_bounds = encoder_ring_bounds(model, encoder_ring_bits) if encoder_ring_bits is not None else None
    x, weights, expected = model['x'], model['weights'], model['expected']
    architecture, fractional_bits = model['architecture'], model['fractional_bits']
    bound = max(model['bounds'].values())
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem+'_logs')
    logs.mkdir(exist_ok=True)
    baseline = native_baseline if native_baseline is not None else native(x, model['float_weights'], device, architecture=architecture, fractional_bits=fractional_bits, input_divisor=model['input_divisor'], public_signs=model['public_signs'])
    dense_baseline = {k: v for k, v in baseline.items() if 'sparse' not in k}
    sparse_baseline = {k: v for k, v in baseline.items() if 'sparse' in k}
    if model['sparse_native_eligible'] and native_baseline is None:
        from experiments.benchmarks.inference.sparse_classifier_native import native_sparse_baselines
        sparse_baseline = native_sparse_baselines(x, model['sparse_native_weights'], device=device)
        baseline = {**baseline, **sparse_baseline}
    processes, handles = [], []
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='rep4-text-') as folder:
        folder = Path(folder)
        ca, identities = certificates(folder)
        addresses = reserve_addresses()
        np.savez(folder/'owner0.npz', x=x)
        np.savez(folder/'owner1.npz', **weights)
        try:
            for p in range(4):
                result_path = logs/f'party{p}.json'
                result_path.unlink(missing_ok=True)
                cfg = dict(party=p, device=device, bits=bits, device_transport=device_transport, encoder_ring_bits=encoder_ring_bits, packed_arithmetic=packed_arithmetic, pinned_staging=pinned_staging, device_input=device_input, generic_kernel=generic_kernel, phase_graphs=phase_graphs, ca=ca, identity=dict(certfile=identities[p]['certificate'], keyfile=identities[p]['key']),
                           addresses=addresses, timeout=timeout, input_shape=list(x.shape),
                           weight_shapes={name: list(value.shape) for name, value in weights.items()},
                           architecture=architecture, fractional_bits=fractional_bits, public_signs=model['public_signs'],
                           repeats=repeats, warmup=warmup)
                if p in (0, 1):
                    cfg['input'] = str(folder/f'owner{p}.npz')
                config_path = folder/f'party{p}.json'
                config_path.write_text(json.dumps(cfg))
                handle = open(logs/f'party{p}.log', 'w')
                handles.append(handle)
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                processes.append(subprocess.Popen([sys.executable, '-m', __name__ if __name__ != '__main__' else
                    'experiments.benchmarks.inference.text_classifier', '--party-config', str(config_path),
                    '--party-result', str(result_path)], stdout=handle, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[3], env=env))
            deadline = time.monotonic()+timeout
            for proc in processes:
                proc.wait(timeout=max(0.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for handle in handles:
                handle.close()
    parties = [json.loads((logs/f'party{p}.json').read_text()) if (logs/f'party{p}.json').exists()
               else dict(party=p, passed=False, error='no result') for p in range(4)]
    codes = [proc.returncode for proc in processes]
    passed = not timed_out and all(code == 0 for code in codes) and all(p['passed'] for p in parties)
    expected_hash = hashlib.sha256(np.asarray(expected, dtype='<i8').tobytes()).hexdigest()
    matched = all(p.get('predictions') == expected and all(m['prediction_sha256'] == expected_hash for m in p.get('measurements', [])) for p in parties)
    passed = passed and matched
    report = dict(passed=passed, exact_integer_match=matched, device_transport=device_transport, architecture=architecture, fractional_bits=fractional_bits, range_bounds=model['bounds'], encoder_ring_bits=encoder_ring_bits, encoder_range_certificate=encoder_bounds, packed_arithmetic=packed_arithmetic, pinned_staging=pinned_staging, device_input=device_input, device=device, generic_kernel=generic_kernel, phase_graphs=phase_graphs, batch=batch, features=x.shape[1], bits=bits,
                  repeats=repeats, warmup=warmup, checkpoint=str(Path(checkpoint).resolve()),
                  method='Four processes colocated on one host; mutual TLS; distributed seeds; private weights and features; only classes opened.',
                  hardware=torch.cuda.get_device_name() if torch.device(device).type == 'cuda' else 'CPU',
                  native_dense_ms=dense_baseline, native_sparse_ms=sparse_baseline, native_ms=baseline, native_scope='Resident features and weights; full dense floating-point classifier plus threshold; one CPU thread; eager and CUDA graph GPU. Fastest measured baseline defines slowdown; eligible sparse integer models also use a validated sparse native baseline.',
                  deployment_scope='Loopback TLS, four parties share one device. Does not measure independent hosts or WAN.',
                  security='Rep4, at most one corrupt party; two colluding parties can reconstruct secrets.',
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_adapter_sha256=hashlib.sha256(Path(__file__).with_name('text_classifier_model.py').read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  range_bound=bound, return_codes=codes, timed_out=timed_out, parties=parties)
    if device_transport == 'cuda-ipc':
        report['method'] = 'Four local processes exchange masked frames and tags through CUDA IPC. TLS carries seed setup and IPC coordination.'
        report['deployment_scope'] = 'Four processes use the same GPU and trust the local host and CUDA IPC peers.'
        report['security'] = 'Rep4 checks run on every batch. Local IPC peers must be trusted with exported memory handles.'
    if encoder_ring_bits is not None:
        report['mixed_ring_scope'] = ('Private input and first-layer weights use ring8; '
            'checked inference includes the exact signed hidden-state lift and ring16 head. '
            'The launcher validates binary fixture inputs; the sharing protocol does not '
            'prove that an arbitrary input owner supplied binary values.')
    if passed:
        for metric in ('input_ms', 'checked_inference_ms', 'online_ms'):
            samples = [max(p['measurements'][i][metric] for p in parties) for i in range(repeats)]
            report[metric] = float(np.median(samples))
        report['inference_texts_per_second'] = batch*1000/report['checked_inference_ms']
        report['online_texts_per_second'] = batch*1000/report['online_ms']
        report['slowdown_resident'] = report['checked_inference_ms']/min(baseline.values())
        report['slowdown_with_input_sharing'] = report['online_ms']/min(baseline.values())
        report['target_met_resident'] = report['slowdown_resident'] < 20
        for key in ('payload_bytes', 'input_payload_bytes', 'inference_payload_bytes', 'host_payload_bytes', 'socket_application_bytes'):
            report['global_'+key+'_per_batch'] = sum(p['measurements'][0][key] for p in parties)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/inference/text_classifier.json'))
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bits', type=int, choices=[16, 32], default=32)
    parser.add_argument('--encoder-ring-bits', type=int, choices=[8], help='use certified ring8 first-layer arithmetic, then securely lift hidden values to ring16')
    parser.add_argument('--packed-arithmetic', action='store_true', help='store CUDA ring16 arithmetic shares as uint16; binary shares remain int64')
    parser.add_argument('--phase-graphs', action='store_true', help='capture local CUDA phases and exchange fresh live protocol messages')
    parser.add_argument('--pinned-staging', action='store_true', help='reuse pinned host buffers for device protocol messages')
    parser.add_argument('--device-transport', choices=('host', 'cuda-ipc'), default='host',
                        help='Use CUDA IPC for protocol payloads between trusted local CUDA workers.')
    parser.add_argument('--device-input', action='store_true', help='check private input corrections with GPU BLAKE3')
    parser.add_argument('--generic-kernel', action='store_true', help='use generic matrix products instead of the ring32 classifier kernel')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config:
        return child(args.party_config, args.party_result)
    if args.checkpoint is None:
        parser.error('--checkpoint is required')
    report = run(args.checkpoint, args.output, batch=args.batch, device=args.device,
                 repeats=args.repeats, warmup=args.warmup, timeout=args.timeout, generic_kernel=args.generic_kernel, phase_graphs=args.phase_graphs, pinned_staging=args.pinned_staging, device_input=args.device_input, device_transport=args.device_transport, bits=args.bits, packed_arithmetic=args.packed_arithmetic, encoder_ring_bits=args.encoder_ring_bits)
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
