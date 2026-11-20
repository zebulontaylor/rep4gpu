"""Profile live captured polynomial inference, including GPU stage durations.

Event durations include scheduling delays from the other colocated CUDA roles.
Socket durations include peer waiting. Nested measurements are not additive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.benchmarks.inference import text_classifier as benchmark


def profile_child(config, result):
    from rep4gpu.protocol import runtime
    from rep4gpu.protocol.runtime import Party
    from rep4gpu.protocol.device_input import DeviceInputMixin
    from rep4gpu.protocol.device_runtime import DeviceParty
    from rep4gpu.protocol.transport import SocketTransport
    from rep4gpu.protocol.pinned_transport import PinnedDeviceByteBridge
    from rep4gpu.models.phase_graph import InferencePlan, _Builder
    from rep4gpu.models.text_classifier import nonnegative_label

    active = None
    phases = []
    stage = 'setup'
    graph_stages = {}

    def measured(owner, name, bucket):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if active is not None:
                    active[bucket + '_ms'] = active.get(bucket + '_ms', 0) + (time.perf_counter()-start)*1000
                    active[bucket + '_calls'] = active.get(bucket + '_calls', 0) + 1
        setattr(owner, name, wrapped)

    for owner, name, bucket in ((SocketTransport, 'exchange', 'socket'),
            (runtime, 'tensor_bytes', 'input_device_to_host'),
            (runtime, 'bytes_tensor', 'input_host_to_device'),
            (DeviceParty, '_random', 'input_mask_enqueue'),
            (PinnedDeviceByteBridge, 'exchange_device_into', 'bridge')):
        measured(owner, name, bucket)

    original_shared = benchmark.shared_labels
    def shared(rt, x, weights, architecture, fractional_bits, public_signs=()):
        nonlocal stage
        if architecture not in ('square_unscaled', 'fixed_square_head'):
            return original_shared(rt, x, weights, architecture, fractional_bits, public_signs)
        stage = 'first_matmul'
        first = rt.add(rt.matmul(x, weights['w1']), weights['b1'])
        if architecture == 'fixed_square_head':
            from rep4gpu.models.square_sum import signed_square_sum
            stage = 'square_sum'
            score = rt.add(signed_square_sum(rt, first, public_signs), weights['b2'])
        else:
            stage = 'square'
            hidden = rt.mul(first, first)
            stage = 'second_matmul'
            score = rt.add(rt.matmul(hidden, weights['w2']), weights['b2'])
        stage = 'sign_and_open'
        return nonnegative_label(rt, score)
    benchmark.shared_labels = shared

    original_end = _Builder.end_segment
    def end_segment(builder):
        original_end(builder)
        graph_stages[id(builder.graph)] = stage
    _Builder.end_segment = end_segment

    class GraphTimer:
        def __init__(self, graph):
            self.graph = graph
        def replay(self):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            self.graph.replay()
            end.record()
            active['_events'].append((graph_stages[id(self.graph)], begin, end))

    original_init = InferencePlan.__init__
    def plan_init(plan, *args, **kwargs):
        original_init(plan, *args, **kwargs)
        plan.actions = [(kind, GraphTimer(value) if kind == 'graph' else value) for kind, value in plan.actions]
    InferencePlan.__init__ = plan_init

    def phase(owner, name, kind):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            nonlocal active
            previous, active = active, {'phase': kind, '_events': []}
            start = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                benchmark.sync(args[0].rt.device if kind == 'inference' else args[0].device)
                return value
            finally:
                active['total_ms'] = (time.perf_counter()-start)*1000
                events = active.pop('_events')
                for label, begin, end in events:
                    active[label + '_gpu_event_ms'] = active.get(label + '_gpu_event_ms', 0) + begin.elapsed_time(end)
                phases.append(active)
                active = previous
        setattr(owner, name, wrapped)
    phase(Party, 'input', 'input')
    phase(DeviceInputMixin, 'input', 'input')
    phase(InferencePlan, 'run', 'inference')
    code = benchmark.child(config, result)
    Path(result).with_suffix('.profile.json').write_text(json.dumps(phases, indent=2)+'\n')
    return code


def summarize(output, repeats):
    """Save per-party medians without adding overlapping timing buckets."""
    logs = output.parent / (output.stem + '_logs')
    parties = {}
    for party in range(4):
        phases = json.loads((logs / f'party{party}.profile.json').read_text())
        kinds = {}
        for kind in ('input', 'inference'):
            rows = [row for row in phases if row['phase'] == kind][-repeats:]
            keys = set().union(*(row.keys() for row in rows)) - {'phase'}
            kinds[kind] = {key: statistics.median(row.get(key, 0) for row in rows)
                           for key in sorted(keys)}
        parties[party] = kinds
    summary = dict(scope='Instrumented live TLS; nested buckets are not additive. GPU events include competing role scheduling.',
                   statistic=f'Median of last {repeats} iterations per party.',
                   profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), parties=parties)
    output.with_name(output.stem + '_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('experiments/results/examples/unscaled_text_classifier/model.npz'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/inference/unscaled_text_classifier_profile.json'))
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--bits', type=int, choices=(16, 32), default=32)
    parser.add_argument('--device-input', action='store_true')
    parser.add_argument('--packed-arithmetic', action='store_true')
    parser.add_argument('--party-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--party-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.party_config:
        return profile_child(args.party_config, args.party_result)
    benchmark.__name__ = 'experiments.benchmarks.inference.profile_neural_classifier'
    report = benchmark.run(args.checkpoint, args.output, batch=args.batch, device='cuda', repeats=args.repeats,
                           warmup=2, phase_graphs=True, pinned_staging=True, bits=args.bits,
                           device_input=args.device_input, packed_arithmetic=args.packed_arithmetic)
    if report['passed']:
        summarize(args.output, args.repeats)
    print(json.dumps({k: v for k, v in report.items() if k != 'parties'}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
