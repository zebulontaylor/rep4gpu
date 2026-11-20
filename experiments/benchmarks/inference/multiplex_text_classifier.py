"""Measure independent MPC sessions sharing each party's CUDA context.

Four OS processes remain four protocol parties. Each session has its own TLS
mesh, distributed seeds, runtime counters, input shares, and inference plan.
Only CUDA context and process resources are shared within a party.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

from .concurrent_text_classifier import overlapping_throughput
from .socket_inference import certificates, reserve_addresses
from .text_classifier import child, load_model, native


def party_bundle(path):
    bundle = json.loads(Path(path).read_text())
    torch.set_num_threads(1)
    configs = bundle['sessions']
    barrier = threading.Barrier(len(configs))
    lock = threading.Lock()
    device = bundle['device']

    def worker(item):
        if torch.device(device).type == 'cuda':
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                return child(item['config'], item['result'], prepare_barrier=barrier,
                             prepare_lock=lock,
                             sync_session=lambda _: torch.cuda.current_stream(device).synchronize())
        return child(item['config'], item['result'], prepare_barrier=barrier,
                     prepare_lock=lock)

    with ThreadPoolExecutor(len(configs)) as pool:
        codes = list(pool.map(worker, configs))
    return 0 if all(code == 0 for code in codes) else 1


def run_multiplex(checkpoint, output, *, sessions=2, batch=4096, repeats=30,
                  warmup=3, device='cuda', bits=32, timeout=600, device_transport='host',
                  resident_inputs=False, encoder_pipeline_stages=0, prepare_encoder=False):
    if sessions < 1 or batch < 1 or repeats < 3 or warmup < 0 or timeout <= 0:
        raise ValueError('require positive sessions/batch/timeout, repeats>=3 and warmup>=0')
    if device_transport not in ('host', 'cuda-ipc'):
        raise ValueError('local classifier transport must be host or cuda-ipc')
    if device_transport == 'cuda-ipc' and torch.device(device).type != 'cuda':
        raise ValueError('CUDA IPC requires a CUDA device')
    limit = 5 if prepare_encoder else 4
    if encoder_pipeline_stages not in range(limit + 1):
        raise ValueError(f'encoder pipeline depth must be zero through {limit} for this weight format')
    if prepare_encoder and not encoder_pipeline_stages:
        raise ValueError('preparing the encoder requires a positive pipeline depth')
    model = load_model(checkpoint, batch*sessions, bits=bits)
    if encoder_pipeline_stages and (device != 'cuda' or bits != 32
            or 'w1' not in model['weights'] or model['weights']['w1'].shape != (512, 128)
            or batch < 16 or torch.cuda.get_device_capability(device) != (8, 9)):
        raise ValueError('encoder pipelining requires SM89, ring32, 512×128 weights and at least 16 rows')
    baseline = native(model['x'], model['float_weights'], device,
                      architecture=model['architecture'], input_divisor=model['input_divisor'],
                      public_signs=model['public_signs'])
    if model['sparse_native_eligible']:
        from .sparse_classifier_native import native_sparse_baselines
        baseline.update(native_sparse_baselines(model['x'], model['sparse_native_weights'], device=device))
    output = Path(output).resolve()
    directory = output.parent/(output.stem+'_sessions')
    directory.mkdir(parents=True, exist_ok=True)
    gpu = torch.device(device).type == 'cuda'
    processes, handles = [], []
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='rep4-multiplex-') as temporary:
        root = Path(temporary)
        ca, identities = certificates(root)
        addresses = [reserve_addresses() for _ in range(sessions)]
        np.savez(root/'owner1.npz', **model['weights'])
        for s in range(sessions):
            np.savez(root/f'owner0_s{s}.npz', x=model['x'][s*batch:(s+1)*batch])
        try:
            for p in range(4):
                entries = []
                for s in range(sessions):
                    result = directory/f'session{s}_party{p}.json'
                    result.unlink(missing_ok=True)
                    cfg = dict(party=p, device=device, bits=bits, pinned_staging=gpu and device_transport == 'host',
                               device_transport=device_transport,
                               device_input=gpu, generic_kernel=False, phase_graphs=gpu,
                               resident_inputs=resident_inputs, encoder_pipeline_stages=encoder_pipeline_stages,
                               prepare_encoder=prepare_encoder,
                               ca=ca, identity=dict(certfile=identities[p]['certificate'],
                                                    keyfile=identities[p]['key']),
                               addresses=addresses[s], timeout=timeout,
                               input_shape=[batch, model['x'].shape[1]],
                               weight_shapes={name: list(value.shape) for name, value in model['weights'].items()},
                               architecture=model['architecture'], fractional_bits=model['fractional_bits'],
                               public_signs=model['public_signs'], repeats=repeats, warmup=warmup)
                    if p == 0:
                        cfg['input'] = str(root/f'owner0_s{s}.npz')
                    elif p == 1:
                        cfg['input'] = str(root/'owner1.npz')
                    config = root/f'session{s}_party{p}.json'
                    config.write_text(json.dumps(cfg))
                    entries.append(dict(config=str(config), result=str(result)))
                bundle = root/f'party{p}.json'
                bundle.write_text(json.dumps(dict(device=device, sessions=entries)))
                handle = open(directory/f'party{p}.log', 'w')
                handles.append(handle)
                processes.append(subprocess.Popen(
                    [sys.executable, '-m', 'experiments.benchmarks.inference.multiplex_text_classifier',
                     '--party-bundle', str(bundle)], stdout=handle, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[3],
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')))
            deadline = time.monotonic()+timeout
            for process in processes:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
            for handle in handles:
                handle.close()
    codes = [process.returncode for process in processes]
    reports = []
    for s in range(sessions):
        expected = model['expected'][s*batch:(s+1)*batch]
        digest = hashlib.sha256(np.asarray(expected, dtype='<i8').tobytes()).hexdigest()
        parties = []
        for p in range(4):
            path = directory/f'session{s}_party{p}.json'
            parties.append(json.loads(path.read_text()) if path.exists()
                           else dict(party=p, passed=False, error='missing worker result'))
        valid = all(p['passed'] and p.get('predictions') == expected
                    and len(p.get('measurements', [])) == repeats
                    and all(row['prediction_sha256'] == digest for row in p['measurements'])
                    for p in parties)
        report = dict(session=s, passed=valid, parties=parties)
        (directory/f'session{s}.json').write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
    passed = (len(codes) == 4 and codes == [0]*4 and not timed_out
              and all(report['passed'] for report in reports))
    result = dict(passed=passed, return_codes=codes, timed_out=timed_out,
                  sessions=sessions, party_processes=4, batch_per_session=batch,
                  device_transport=device_transport,
                  resident_inputs=resident_inputs, encoder_pipeline_stages=encoder_pipeline_stages,
                  prepare_encoder=prepare_encoder, repeats=repeats, warmup=warmup, device=device, bits=bits,
                  architecture=model['architecture'], checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  worker_sha256=hashlib.sha256(Path(__file__).with_name('text_classifier.py').read_bytes()).hexdigest(),
                  native_ms=baseline, native_batch=batch*sessions,
                  native_texts_per_second=batch*sessions*1000/min(baseline.values()),
                  scope='Four parties on one host/device; independent checked TLS sessions share each party process and CUDA context.',
                  timing='Separate CUDA streams; stream-local synchronization; native baseline precedes all MPC workers.')
    if device_transport == 'cuda-ipc':
        result['scope'] = ('Four trusted local GPU processes. Each session uses CUDA IPC for masked '
                           'payloads and tags and mutually authenticated TLS for setup and coordination.')
    if passed:
        try:
            result.update(overlapping_throughput(reports, batch))
            metric = 'slowdown_resident' if resident_inputs else 'slowdown_including_input_sharing'
            result[metric] = result['native_texts_per_second']/result['texts_per_second']
            result['target_met'] = result[metric] < 20
        except ValueError as error:
            result['passed'] = False
            result['measurement_error'] = str(error)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bits', type=int, choices=[16, 32], default=32)
    parser.add_argument('--encoder-pipeline-stages', type=int, choices=range(6), default=0,
                        help='Pipeline the SM89 ring32 encoder K loop. Zero selects the baseline kernel.')
    parser.add_argument('--prepare-encoder', action='store_true',
                        help='Precompute local ring32 weight sums and signed byte planes before timing.')
    parser.add_argument('--device-transport', choices=('host', 'cuda-ipc'), default='host',
                        help='Use CUDA IPC for protocol payloads between trusted local CUDA workers.')
    parser.add_argument('--resident-inputs', action='store_true',
                        help='Share each session fixture once before timing. Inference masks stay fresh.')
    parser.add_argument('--party-bundle', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_bundle:
        return party_bundle(args.party_bundle)
    if args.checkpoint is None or args.output is None:
        parser.error('--checkpoint and --output are required')
    result = run_multiplex(args.checkpoint, args.output, sessions=args.sessions, batch=args.batch,
                           repeats=args.repeats, warmup=args.warmup, timeout=args.timeout,
                           device=args.device, bits=args.bits, device_transport=args.device_transport,
                           resident_inputs=args.resident_inputs, encoder_pipeline_stages=args.encoder_pipeline_stages,
                           prepare_encoder=args.prepare_encoder)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
