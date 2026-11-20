"""Attribute live classifier wall time to socket exchange and device staging.

This opt-in profiler wraps the existing TLS benchmark without changing its
protocol. Staging times include waits for preceding CUDA work; socket times
include waits for peers. Nested buckets must not be added to their parents.
"""
import argparse
import json
from pathlib import Path
import time

from experiments.benchmarks.inference import text_classifier as benchmark


def profile_child(config, result):
    from rep4gpu.protocol import device_transport
    from rep4gpu.protocol import runtime
    from rep4gpu.protocol.runtime import Party
    from rep4gpu.protocol.transport import SocketTransport
    from rep4gpu.models.phase_graph import InferencePlan

    active = None
    phases = []

    def measured(owner, name, bucket):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if active is not None:
                    elapsed = (time.perf_counter() - start) * 1000
                    active[bucket + '_ms'] = active.get(bucket + '_ms', 0) + elapsed
                    active[bucket + '_calls'] = active.get(bucket + '_calls', 0) + 1
        setattr(owner, name, wrapped)

    measured(SocketTransport, 'exchange', 'socket')
    measured(device_transport, '_bytes', 'device_to_host')
    measured(device_transport, '_copy_bytes_into', 'host_to_device')
    measured(device_transport, '_tensor', 'host_to_device')
    measured(runtime, 'tensor_bytes', 'input_device_to_host')
    measured(runtime, 'bytes_tensor', 'input_host_to_device')
    measured(device_transport.DeviceByteBridge, 'exchange_device_into', 'bridge')
    measured(device_transport.DeviceByteBridge, 'exchange_device', 'bridge')

    def phase(owner, name, kind):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            nonlocal active
            previous, active = active, {'phase': kind}
            start = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                benchmark.sync(args[0].rt.device if kind == 'inference' else args[0].device)
                return value
            finally:
                active['total_ms'] = (time.perf_counter() - start) * 1000
                phases.append(active)
                active = previous
        setattr(owner, name, wrapped)

    phase(Party, 'input', 'input')
    phase(InferencePlan, 'run', 'inference')
    code = benchmark.child(config, result)
    Path(result).with_suffix('.profile.json').write_text(json.dumps(phases, indent=2) + '\n')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('experiments/results/examples/text_classifier/model.npz'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/inference/text_classifier_profile.json'))
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config:
        return profile_child(args.party_config, args.party_result)
    benchmark.__name__ = 'experiments.benchmarks.inference.profile_text_classifier'
    report = benchmark.run(args.checkpoint, args.output, batch=args.batch, device='cuda',
                           repeats=args.repeats, warmup=2, phase_graphs=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
