"""Exercise separate prover/verifier processes and TLS role identities."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from rep4gpu.proofs.audit_model import AuditModel, example_spec
from rep4gpu.proofs.audit_service import init_example


@pytest.fixture(autouse=True)
def subprocess_environment(monkeypatch):
    source = str(Path(__file__).resolve().parents[2] / 'src')
    monkeypatch.setenv('PYTHONPATH', source + os.pathsep + os.environ.get('PYTHONPATH', ''))


def command(*args):
    return [sys.executable, '-m', 'rep4gpu.proofs.audit_service', *map(str, args)]


@pytest.mark.parametrize('samples', [None, 2])
def test_demo_processes(tmp_path, samples):
    output = tmp_path / 'run'
    args = ['demo', '--output-dir', output, '--timeout', '15']
    if samples is not None:
        args += ['--samples', str(samples)]
    result = subprocess.run(command(*args), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary == {'passed': True, 'exit_codes': {'prover': 0, 'verifier': 0}, 'error': None}
    report = json.loads((output / 'verifier.json').read_text())
    assert report == json.loads((output / 'prover/session-000001.json').read_text())
    assert report['accepted'] is True
    assert report['verification_mode'] == ('full' if samples is None else 'sampled')
    assert len(report['checked_transitions']) == (4 if samples is None else samples)
    assert 'plaintext' in report['plaintext_semantics']
    model = AuditModel(example_spec())
    state = {'x': torch.tensor([[1, -2, 3, 4]], dtype=torch.int64)}
    for layer in model.layers:
        state = layer.forward(state)
    assert report['output'] == {'x': state['x'].tolist()}
    assert report['output'] == {'x': [[91, 0, 0, 2]]}


@pytest.mark.parametrize('wrong_role', ['prover', 'verifier'])
def test_tls_rejects_wrong_pinned_identity(tmp_path, wrong_role):
    setup = tmp_path / 'setup'
    init_example(setup)
    ready = tmp_path / 'ready.json'

    def role_args(role):
        peer = 'verifier' if role == 'prover' else 'prover'
        if role == wrong_role:
            peer = role
        return ['--model', setup / 'model.json', '--ca', setup / 'ca.pem',
                '--cert', setup / f'{role}.pem', '--key', setup / f'{role}.key',
                '--peer-cert', setup / f'{peer}.pem', '--timeout', '5']

    prover = subprocess.Popen(command('serve', *role_args('prover'), '--port', '0',
        '--sessions', '1', '--ready-file', ready, '--output-dir', tmp_path / 'prover'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            assert prover.poll() is None, prover.communicate()[0]
            assert time.monotonic() < deadline, 'prover startup timed out'
            time.sleep(0.05)
        port = json.loads(ready.read_text())['port']
        result = subprocess.run(command('verify', *role_args('verifier'), '--port', port,
            '--input', setup / 'input.json', '--output', tmp_path / 'verifier.json'),
            capture_output=True, text=True, timeout=15)
        prover_output = prover.communicate(timeout=15)[0]
        assert result.returncode != 0
        assert prover.returncode != 0
        assert 'configured role identity' in result.stderr + prover_output
        assert json.loads((tmp_path / 'verifier.json').read_text())['accepted'] is False
        assert json.loads((tmp_path / 'prover/session-000001.json').read_text())['accepted'] is False
    finally:
        if prover.poll() is None:
            prover.kill()
            prover.communicate()


def test_model_identity_binds_parameters_and_accepts_key_order():
    spec = example_spec()
    reordered = dict(reversed(list(spec.items())))
    assert AuditModel(spec).model_id == AuditModel(reordered).model_id
    changed = copy.deepcopy(spec)
    changed['layers'][0]['bias'][0] += 1
    assert AuditModel(changed).model_id != AuditModel(spec).model_id


@pytest.mark.parametrize('change', ['bool', 'range', 'shape', 'extra'])
def test_model_rejects_invalid_schema(change):
    spec = example_spec()
    if change == 'bool':
        spec['layers'][0]['bias'][0] = True
    elif change == 'range':
        spec['layers'][0]['bias'][0] = 1 << 63
    elif change == 'shape':
        spec['layers'][0]['weight'].pop()
    else:
        spec['extra'] = 1
    with pytest.raises(ValueError):
        AuditModel(spec)


@pytest.mark.parametrize('tensor', [torch.ones(1, 4, dtype=torch.int32),
                                    torch.ones(0, 4, dtype=torch.int64),
                                    torch.ones(1, 3, dtype=torch.int64)])
def test_model_rejects_invalid_state(tensor):
    with pytest.raises(ValueError):
        AuditModel(example_spec()).validate_state({'x': tensor})


def test_model_int64_wrap_precedes_relu():
    spec = {'format': 'siv-int64-relu-v1', 'width': 1,
            'layers': [{'weight': [[1]], 'bias': [1]}]}
    model = AuditModel(spec)
    output = model.layers[0].forward({'x': torch.tensor([[(1 << 63) - 1]], dtype=torch.int64)})
    assert output['x'].tolist() == [[0]]
    spec['layers'][0] = {'weight': [[-1]], 'bias': [-1]}
    model = AuditModel(spec)
    output = model.layers[0].forward({'x': torch.tensor([[-(1 << 63)]], dtype=torch.int64)})
    assert output['x'].tolist() == [[(1 << 63) - 1]]
