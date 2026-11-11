"""Exercise separate roles and verify their saved outputs after both exit."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

SOURCE = Path(__file__).resolve().parent
HERE = SOURCE.parent.parent / 'work/matrix_vope'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def run_case(name, fault_role=0, fault='none', b=128, t=8, log_block=6, sessions=2, mode='persistent', role_cpus=None, field='fp61'):
    folder = HERE / 'artifacts' / name
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    processes = []
    logs = []
    started = time.monotonic()
    try:
        for role in (1, 2):
            log = open(folder / f'role{role}.log', 'w')
            logs.append(log)
            command = [str(HERE / 'seeded_vole'), str(role), str(port),
                       str(folder / f'role{role}.json'), str(sessions),
                       fault if role == fault_role else ('force-cope-bit' if fault == 'cope-high' and role == 1 else 'none'), str(b), str(t), str(log_block), mode, field]
            if role_cpus:
                command = ['taskset', '-c', role_cpus[role - 1], *command]
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
    codes = [process.returncode for process in processes]
    result = {'name': name, 'returncodes': codes, 'seconds': time.monotonic() - started,
              'security_reviewed': False}
    if fault_role:
        assert codes == [1, 1], result
        assert all('consistency check failed' in (folder / f'role{role}.log').read_text()
                   for role in (1, 2)), result
        assert all('entropy_retired=true' in (folder / f'role{role}.log').read_text() for role in (1, 2)), result
        result['entropy_retired'] = True
        result['fault_detected'] = True
        return result
    assert codes == [0, 0], result
    scalar = json.loads((folder / 'role1.json').read_text())
    programmer = json.loads((folder / 'role2.json').read_text())
    assert scalar['field'] == programmer['field'] == field
    bits = {'fp61': 61, 'fp127': 127}[field]
    assert scalar['prime_bits'] == programmer['prime_bits'] == bits
    q = (1 << bits) - 1
    assert int(scalar['prime']) == int(programmer['prime']) == q
    assert scalar['security_reviewed'] is False
    assert programmer['network_debug_disclosure'] is False
    assert len(scalar['sessions']) == len(programmer['sessions']) == sessions
    assert [s['index'] for s in scalar['sessions']] == list(range(sessions))
    assert [s['index'] for s in programmer['sessions']] == list(range(sessions))
    assert all(s['value'] == programmer['sessions'][0]['value'] for s in programmer['sessions'])
    assert len({s['delta'] for s in scalar['sessions']}) == sessions
    for role in (scalar, programmer):
        counters = [s['entropy_substreams_consumed'] for s in role['sessions']]
        assert all(a < b for a, b in zip(counters, counters[1:]))
    result['advancing_entropy_substreams'] = True
    assert any(programmer['sessions'][0]['value'])
    assert len({tuple(s['share']) for s in scalar['sessions']}) == sessions
    for receiver, sender in zip(scalar['sessions'], programmer['sessions']):
        assert all(0 <= value < q for value in receiver['share'] + sender['value'] + sender['share'])
        assert 0 < receiver['delta'] < q
        assert len(receiver['share']) == len(sender['share']) == len(sender['value']) == b
        assert all((key + receiver['delta'] * value) % q == mac
                   for key, value, mac in zip(receiver['share'], sender['value'], sender['share']))
    if bits > 64:
        assert any(s['delta'] > (1 << 64) for s in scalar['sessions'])
        assert any(v > (1 << 64) for v in programmer['sessions'][0]['value'])
    result['field'] = field
    result['prime_bits'] = bits
    result['full_width_and_canonical_values'] = True
    result['base_check_polynomial_degree'] = scalar['base_check_polynomial_degree']
    result['base_check_error_bound'] = f"{scalar['base_check_polynomial_degree']}/{q}"
    result['same_seed_same_value'] = True
    result['fresh_scalar_keys'] = True
    result['correlation_checks'] = sessions * b
    result['mode'] = mode
    result['sent_bytes'] = [sum(s['sent_bytes'] for s in role['sessions'])
                            for role in (scalar, programmer)]
    result['role_cpus'] = role_cpus
    result['process_cpu_seconds'] = [[s['process_cpu_seconds'] for s in r['sessions']] for r in (scalar, programmer)]
    result['base_wall_seconds'] = [[s['base_wall_seconds'] for s in r['sessions']] for r in (scalar, programmer)]
    result['base_cpu_seconds'] = [[s['base_cpu_seconds'] for s in r['sessions']] for r in (scalar, programmer)]
    result['context_setup_seconds'] = [r['context_setup_seconds'] for r in (scalar, programmer)]
    result['context_sent_bytes'] = [r['context_sent_bytes'] for r in (scalar, programmer)]
    result['context_received_bytes'] = [r['context_received_bytes'] for r in (scalar, programmer)]
    result['phase_seconds'] = [[s['phase_seconds'] for s in r['sessions']] for r in (scalar, programmer)]
    result['protocol_seconds'] = [[s['seconds'] for s in role['sessions']]
                                  for role in (scalar, programmer)]
    return result


def reflection_case(size):
    folder = HERE / 'artifacts' / f'reflection_{size}'
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    processes, logs = [], []
    try:
        for role in (1, 2):
            log = open(folder / f'role{role}.log', 'w')
            logs.append(log)
            command = [str(HERE / 'seeded_vole'), 'reflection', str(role), str(port), str(size)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=15)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    codes = [p.returncode for p in processes]
    assert codes == [1, 0], codes
    assert 'reflection rejected: commitment opening failed' in (folder / 'role1.log').read_text()
    return {'name': f'reflection_{size}', 'returncodes': codes,
            'reflection_rejected': True, 'security_reviewed': False}


def wire_case(field, encoding):
    folder = HERE / 'artifacts' / f'wire_{field}_{encoding}'
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    processes, logs = [], []
    try:
        for role in (1, 2):
            log = open(folder / f'role{role}.log', 'w')
            logs.append(log)
            command = [str(HERE / 'seeded_vole'), 'wire-test', str(role), str(port), field, encoding]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=15)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    codes = [p.returncode for p in processes]
    expected = [0, 0] if encoding == 'canonical' else [1, 0]
    assert codes == expected, (field, encoding, codes)
    return {'name': f'wire_{field}_{encoding}', 'returncodes': codes,
            'canonical_boundary_checked': True, 'security_reviewed': False}


def deterministic_mode_case():
    folder = HERE / 'artifacts' / 'deterministic_mode'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'must_not_exist.json'
    if output.exists(): output.unlink()
    environment = dict(os.environ, EMP_TEST_MODE='1')
    command = [str(HERE / 'seeded_vole'), '1', str(free_port()), str(output), '1']
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    (folder / 'startup.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 2
    assert 'EMP deterministic test mode is not allowed' in result.stderr
    assert not output.exists()
    return {'name': 'deterministic_mode_rejected', 'returncodes': [result.returncode],
            'rejected_before_connection': True, 'security_reviewed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-only', action='store_true')
    parser.add_argument('--field', choices=['fp61', 'fp127'], default='fp61')
    parser.add_argument('--sessions', type=int, default=2)
    parser.add_argument('--mode', choices=['persistent', 'fresh'], default='persistent')
    parser.add_argument('--role-cpus', nargs=2, metavar=('SCALAR_CPUS', 'PROGRAMMER_CPUS'))
    args = parser.parse_args()
    results = [run_case('honest_' + args.mode + '_' + args.field, sessions=args.sessions, mode=args.mode, role_cpus=args.role_cpus, field=args.field)]
    if not args.honest_only:
        results.extend([deterministic_mode_case(), wire_case(args.field, 'canonical'),
                        wire_case(args.field, 'modulus'), wire_case(args.field, 'highbit'),
                        reflection_case(16), reflection_case(32),
                        run_case('bad_tree_' + args.field, 1, 'tree', mode=args.mode, field=args.field),
                        run_case('bad_mac_' + args.field, 2, 'mac', mode=args.mode, field=args.field),
                        run_case('bad_correction_' + args.field, 2, 'correction', mode=args.mode, field=args.field),
                        run_case('offset_errors_' + args.field, 2, 'offset', mode=args.mode, field=args.field)])
        if args.mode == 'persistent':
            results.extend(run_case(fault + '_' + args.field, 2, fault, mode=args.mode, field=args.field)
                           for fault in (['cope-high'] if args.field == 'fp127' else []) + ['base-response0', 'base-response1', 'base-mac'])
    summary = HERE / f'artifacts/summary_{args.mode}_{args.field}.json'
    summary.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
