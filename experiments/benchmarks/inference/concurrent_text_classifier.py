"""Measure aggregate throughput of independent live four-party sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .text_classifier import load_model, native, run


def overlapping_throughput(reports, batch):
    """Count complete batches within the common measured steady-state window.

    All workers run on this host and use its monotonic clock. A batch starts
    when its first party starts and completes when its final party finishes.
    Boundary-crossing batches are excluded, giving a conservative finite-window
    throughput instead of adding per-session rates measured at different times.
    """
    if not reports or any(not report['passed'] for report in reports):
        raise ValueError('every session must finish and pass its output checks')
    sessions = []
    for report in reports:
        parties = report['parties']
        sizes = {len(p['measurements']) for p in parties}
        if len(parties) != 4 or len(sizes) != 1 or min(sizes) < 2:
            raise ValueError('require four parties and at least two matching measurements')
        sessions.append([(min(p['measurements'][i]['start_monotonic'] for p in parties),
                          max(p['measurements'][i]['end_monotonic'] for p in parties))
                         for i in range(next(iter(sizes)))])
    begin = max(rows[0][0] for rows in sessions)
    end = min(rows[-1][1] for rows in sessions)
    if end <= begin:
        raise ValueError('sessions have no overlapping measured window')
    counts = [sum(begin <= start < finish <= end for start, finish in rows) for rows in sessions]
    if not all(counts):
        raise ValueError('common window must contain complete batches from every session')
    return dict(window_seconds=end-begin, complete_batches_per_session=counts,
                texts_per_second=sum(counts)*batch/(end-begin),
                counting='Only batches wholly contained in the common steady-state window.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.sessions < 1 or args.batch < 1 or args.repeats < 3:
        parser.error('require positive sessions/batch and at least three repeats')
    model = load_model(args.checkpoint, args.batch*args.sessions)
    # Measure native once before launching any MPC workers. Use the combined
    # batch so splitting MPC work cannot inflate the native launch overhead.
    baseline = native(model['x'], model['float_weights'], 'cuda',
                      architecture=model['architecture'], input_divisor=model['input_divisor'], public_signs=model['public_signs'])
    if model['sparse_native_eligible']:
        from .sparse_classifier_native import native_sparse_baselines
        baseline.update(native_sparse_baselines(model['x'], model['sparse_native_weights'], device='cuda'))
    native_rate = args.batch*args.sessions*1000/min(baseline.values())
    session_model = load_model(args.checkpoint, args.batch)
    session_baseline = native(session_model['x'], session_model['float_weights'], 'cuda',
                              architecture=session_model['architecture'],
                              input_divisor=session_model['input_divisor'], public_signs=session_model['public_signs'])
    if session_model['sparse_native_eligible']:
        from .sparse_classifier_native import native_sparse_baselines
        session_baseline.update(native_sparse_baselines(session_model['x'], session_model['sparse_native_weights'], device='cuda'))
    directory = args.output.parent / (args.output.stem + '_sessions')
    directory.mkdir(parents=True, exist_ok=True)

    def launch(index):
        return run(args.checkpoint, directory/f'session{index}.json', batch=args.batch,
                   device='cuda', repeats=args.repeats, warmup=3, timeout=600,
                   phase_graphs=True, pinned_staging=True, device_input=True,
                   native_baseline=session_baseline)

    with ThreadPoolExecutor(args.sessions) as pool:
        reports = list(pool.map(launch, range(args.sessions)))
    result = dict(passed=all(report['passed'] for report in reports), sessions=args.sessions,
                  batch_per_session=args.batch, repeats=args.repeats, native_ms=baseline,
                  native_texts_per_second=native_rate,
                  scope='Independent seeds, models and sessions; all parties share one host and GPU.',
                  comparison='Both native baselines are measured before launching sessions; aggregate comparison uses the combined native batch.')
    if result['passed']:
        result.update(overlapping_throughput(reports, args.batch))
        result['slowdown_including_input_sharing'] = native_rate/result['texts_per_second']
        result['target_met'] = result['slowdown_including_input_sharing'] < 20
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
