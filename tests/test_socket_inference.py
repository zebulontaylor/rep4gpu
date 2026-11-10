"""End-to-end isolated-role TLS inference and coordinator failure checks."""
import json
import shutil
import subprocess
import sys

import pytest
import torch

from benchmarks.socket_inference import run


@pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl needed for ephemeral TLS identities')
@pytest.mark.parametrize('bits,batch', [(32, 1), (64, 16)])
def test_four_process_tls_inference(tmp_path, bits, batch):
    output = tmp_path / 'inference.json'
    report = run(output, device='cpu', bits=bits, width=8, batch=batch, timeout=40)
    assert report['passed']
    assert report['return_codes'] == [0, 0, 0, 0]
    assert report['exact_integer_match']
    assert report['range_validated']
    assert report['global_payload_bytes'] > report['global_inference_payload_bytes'] > 0
    assert len({party['party'] for party in report['parties']}) == 4
    assert all(party['authenticated_tls'] and party['distributed_setup'] for party in report['parties'])
    assert all(party['output'] == report['expected'] for party in report['parties'])
    assert all(party['inference_rounds'] > 0 for party in report['parties'])
    assert json.loads(output.read_text()) == report
    logs = tmp_path / 'inference_logs'
    assert all((logs / f'party{p}.log').exists() for p in range(4))
    assert not list(tmp_path.rglob('*.key'))


def test_invalid_fused_cpu_is_clear():
    with pytest.raises(ValueError, match='requires a CUDA'):
        run('/tmp/unused-rep4-result.json', device='cpu', fused=True)


@pytest.mark.skipif(not torch.cuda.is_available() or shutil.which('openssl') is None,
                    reason='CUDA and openssl required')
@pytest.mark.parametrize('bits,optimized,gpu_open', [(32, False, False), (64, False, False),
                                                  (32, True, False), (64, True, False),
                                                  (32, True, True), (64, True, True)])
def test_device_protocol_four_process_tls(tmp_path, bits, optimized, gpu_open):
    report = run(tmp_path / 'device.json', device='cuda', bits=bits, width=8,
                 batch=2, timeout=90, device_protocol=True, optimized=optimized, gpu_open=gpu_open)
    assert report['passed'] and report['exact_integer_match']
    assert report['return_codes'] == [0, 0, 0, 0]
    assert report['online_prg'] == 'chacha20' and report['online_hash'] == 'blake3'
    assert report['staging'] == 'host'
    assert all(party['authenticated_tls'] and party['distributed_setup']
               for party in report['parties'])


def test_cli_help_has_no_environment_requirement():
    result = subprocess.run([sys.executable, '-m', 'benchmarks.socket_inference', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--bits' in result.stdout


@pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl needed for ephemeral TLS identities')
def test_timeout_cleans_children_and_preserves_exit_codes(tmp_path):
    report = run(tmp_path / 'timeout.json', timeout=.001)
    assert report['timed_out']
    assert not report['passed']
    assert all(isinstance(code, int) and code != 0 for code in report['return_codes'])
    assert not list(tmp_path.rglob('*.key'))
