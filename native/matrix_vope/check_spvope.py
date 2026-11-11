"""Check candidate matrix correlations and demonstrate its beta privacy limit."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import uuid

SOURCE = Path(__file__).resolve().parent
WORK = SOURCE.parent.parent / 'work/spvope_candidate'


def case(name, backend, field, beta_mode='uniform', fault='none', fault_role=0, accepted=True, sessions=1):
    # Keep beta diagnostic labels out of the counterpart's output-path argument.
    folder = WORK / 'artifacts' / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, logs = [], []
    a, b = 16, 128
    try:
        for role in (1, 2):
            log = open(folder / f'role{role}.log', 'w')
            logs.append(log)
            command = [str(WORK / 'spvope_candidate'), str(role), str(port),
                       str(folder / f'role{role}.json'), str(sessions),
                       fault if role == fault_role else 'none', beta_mode if role == 1 else 'private',
                       backend, field, str(a), str(b)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=45)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    codes = [p.returncode for p in processes]
    result = {'name': name, 'backend': backend, 'field': field, 'returncodes': codes,
              'security_reviewed': False, 'arbitrary_beta_malicious_privacy': False,
              'artifact_directory': str(folder.relative_to(WORK))}
    if not accepted:
        assert codes == [1, 1], result
        assert all('outer single-point VOPE consistency check failed' in (folder / f'role{role}.log').read_text()
                   for role in (1, 2)), result
        assert all('entropy_retired=true' in (folder / f'role{role}.log').read_text()
                   for role in (1, 2)), result
        result['outer_check_rejected'] = True
        return result
    assert codes == [0, 0], result
    left = json.loads((folder / 'role1.json').read_text())
    right = json.loads((folder / 'role2.json').read_text())
    assert len(left['sessions']) == len(right['sessions']) == sessions
    q = (1 << {'fp61': 61, 'fp127': 127}[field]) - 1
    scalar_keys = []
    for index, (x, y) in enumerate(zip(left['sessions'], right['sessions'])):
        assert x['index'] == y['index'] == index
        assert len(x['matrix_share']) == len(y['matrix_share']) == a * b
        assert len(y['v']) == b
        assert all(0 <= value < q for value in x['matrix_share'] + y['matrix_share'] + y['v'])
        scalar_keys.extend([x['first_delta'], x['second_delta']])
        for row in range(a):
            for col in range(b):
                expected = x['beta'] * y['v'][col] if row == x['alpha'] else 0
                assert (x['matrix_share'][row*b+col] - y['matrix_share'][row*b+col]) % q == expected % q
    assert len(set(scalar_keys)) == len(scalar_keys)
    assert all(s['v'] == right['sessions'][0]['v'] for s in right['sessions'])
    result['matrix_dimensions'] = [a, b]
    result['matrix_relations_checked'] = sessions * a * b
    result['fresh_scalar_keys'] = True
    result['same_seed_same_right_vector'] = True
    result['protocol_seconds'] = [[s['seconds'] for s in owner['sessions']] for owner in (left, right)]
    result['sent_bytes'] = [sum(s['sent_bytes'] for s in owner['sessions']) for owner in (left, right)]
    return result


def selective_abort_algebra():
    q, v1, v2, r = 101, [3, 19, 44], [8, 11, 72], 1
    for beta, accepts in [(0, False), (1, True), (2, False)]:
        gap = [r * (beta - 1) * (x-y) % q for x, y in zip(v1, v2)]
        assert (not any(gap)) == accepts
    assert not any(0 * (x-y) % q for x, y in zip(v1, v2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['pcg', 'direct', 'both'], default='both')
    parser.add_argument('--field', choices=['fp61', 'fp127'], default='fp127')
    args = parser.parse_args()
    selective_abort_algebra()
    results = []
    for backend in (['pcg', 'direct'] if args.backend == 'both' else [args.backend]):
        results.append(case('honest', backend, args.field, sessions=2))
        for fault, role in [('tree', 2), ('offset', 2), ('mac', 1), ('check', 1), ('seed', 2)]:
            results.append(case('bad_' + fault, backend, args.field, fault=fault,
                                fault_role=role, accepted=False))
        results.append(case('zero_beta_seed_equivocation', backend, args.field,
                            beta_mode='zero', fault='seed', fault_role=2))
        results.append(case('selective_abort_beta1_accepts', backend, args.field,
                            beta_mode='one', fault='selective', fault_role=2))
        results.append(case('selective_abort_beta2_rejects', backend, args.field,
                            beta_mode='two', fault='selective', fault_role=2, accepted=False))
    report = {'security_reviewed': False, 'arbitrary_beta_malicious_privacy': False,
              'selective_abort_limitation_demonstrated': True, 'cases': results}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f'results_{args.field}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
