"""Exercise the private audit with five isolated TLS subprocesses."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from rep4gpu.proofs.private_audit_service import init_private, copy_role_setup, aggregate_traffic


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
ROLES = ('prover', 'party0', 'party1', 'party2', 'party3')


def launch(tmp_path, *, samples=None, fault=None):
    setup = tmp_path / 'setup'
    init_private(setup)
    reservations = [socket.socket() for _ in range(5)]
    try:
        for sock in reservations:
            sock.bind(('127.0.0.1', 0))
        addresses = [list(sock.getsockname()) for sock in reservations]
    finally:
        for sock in reservations:
            sock.close()
    config_path = setup / 'deployment.json'
    config = json.loads(config_path.read_text())
    config.update(prover=addresses[0], verifiers=addresses[1:], timeout=15, samples=samples)
    config_path.write_text(json.dumps(config))
    processes, logs, codes = {}, {}, {}
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for role in ROLES:
            role_setup = tmp_path / ('setup-' + role)
            role_setup.mkdir(mode=0o700)
            copy_role_setup(setup, role_setup, role)
            if role == 'prover':
                assert not list(role_setup.glob('party*-model.json'))
            else:
                assert not (role_setup / 'model.json').exists()
                assert {p.name for p in role_setup.glob('party*-model.json')} == {role + '-model.json'}
            assert {p.name for p in role_setup.glob('*.key')} == {role + '.key'}
            args = ['prover' if role == 'prover' else 'verifier', '--setup', str(role_setup),
                    '--output-dir', str(tmp_path / role)]
            if role != 'prover':
                args += ['--party', role[-1]]
            command = [PYTHON, '-m', 'rep4gpu.proofs.private_audit_service', *args]
            if role == 'prover' and fault:
                if fault == 'computation':
                    patch = '''
class FaultTrace(original):
    def __init__(self, model, inputs, **kwargs):
        first = model.layers[0]
        from rep4gpu.proofs.layer_commitments import Layer
        def altered(state):
            output = first.forward(state)
            output['x'] = output['x'] + 1
            return output
        model.layers = (Layer(first.name, altered, preserves_inputs=True), *model.layers[1:])
        super().__init__(model, inputs, **kwargs)
'''
                elif fault == 'weights':
                    patch = '''
class FaultTrace(original):
    def __init__(self, model, inputs, **kwargs):
        model.parameters[0][1].add_(1)
        super().__init__(model, inputs, **kwargs)
'''
                elif fault == 'path':
                    patch = '''
class FaultTrace(original):
    def opening(self, boundary, party):
        value = super().opening(boundary, party)
        value['components'][0]['siblings'][0] = '00' * 32
        return value
'''
                else:
                    raise AssertionError(fault)
                code = ('import rep4gpu.proofs.private_audit_service as service\n'
                        'original = service.PrivateTrace\n' + patch +
                        '\nservice.PrivateTrace = FaultTrace\n'
                        'raise SystemExit(service.main())\n')
                command = [PYTHON, '-c', code, *args]
            logs[role] = (tmp_path / (role + '.log')).open('w')
            processes[role] = subprocess.Popen(command, cwd=ROOT, env=env,
                                                stdout=logs[role], stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 45
        while any(p.poll() is None for p in processes.values()):
            if time.monotonic() >= deadline:
                pytest.fail('private audit subprocess deadline exceeded')
            time.sleep(0.05)
    finally:
        for role, process in processes.items():
            if process.poll() is None:
                process.kill()
            codes[role] = process.wait(timeout=5)
        for log in logs.values():
            log.close()
        (tmp_path / 'exit-codes.json').write_text(json.dumps(codes, sort_keys=True))
    return codes


@pytest.mark.parametrize('samples', [None, 2])
def test_full_and_sampled_five_process_audit(tmp_path, samples):
    codes = launch(tmp_path, samples=samples)
    assert codes == dict.fromkeys(ROLES, 0), {role: (tmp_path / (role + '.log')).read_text() for role in ROLES}
    reports = [json.loads((tmp_path / role / 'result.json').read_text()) for role in ROLES]
    assert all(report == reports[0] for report in reports)
    report = reports[0]
    assert report['accepted'] is True
    assert report['output'] == {'x': [[91, 0, 0, 2]]}
    assert len(report['checked_transitions']) == (4 if samples is None else samples)
    assert report['verification_mode'] == ('full' if samples is None else 'sampled')
    assert set(report) == {'accepted', 'protocol', 'model_id', 'run_id', 'commitment',
                           'checked_transitions', 'total_transitions', 'verification_mode', 'output', 'privacy'}
    assert set(report['commitment']) == {'version', 'scheme', 'model_id', 'run_id', 'layer_names', 'shape', 'roots'}
    assert not any((tmp_path / role / 'failure.json').exists() for role in ROLES)
    traffic = aggregate_traffic(tmp_path)
    assert traffic['total_sent'] > traffic['transition_checks_and_output_sent'] > 0
    assert traffic['channels']['mesh_identity'] == {'sent': 6, 'received': 6}
    assert traffic['total_sent'] == sum(traffic['per_role_sent'].values())


@pytest.mark.parametrize('fault', ['computation', 'weights', 'path'])
def test_prover_fault_has_no_accepted_output(tmp_path, fault):
    codes = launch(tmp_path, fault=fault)
    assert all(code != 0 for code in codes.values()), codes
    for role in ROLES:
        assert not (tmp_path / role / 'result.json').exists()
        failure = json.loads((tmp_path / role / 'failure.json').read_text())
        assert failure['accepted'] is False and set(failure) == {'accepted', 'error'}
    verifier_logs = '\n'.join((tmp_path / (role + '.log')).read_text() for role in ROLES[1:])
    assert ('witness batch rejected' if fault in ('computation', 'weights') else
            'private component does not match its commitment') in verifier_logs
