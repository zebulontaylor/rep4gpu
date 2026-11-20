"""End-to-end four-process mTLS and CUDA IPC inference tests."""
import shutil
import subprocess
import sys

import pytest
import torch

from examples.mtls_inference import run_mtls_inference


@pytest.mark.skipif(shutil.which('openssl') is None, reason='openssl needed for ephemeral TLS identities')
@pytest.mark.parametrize('bits', [32, 64])
def test_four_process_cpu_tls_inference(bits):
    report = run_mtls_inference(transport='cpu-tls', bits=bits, batch=2, width=8, hidden=16, timeout=40)
    assert report['exact_match']
    assert report['total_tls_bytes'] > 0


@pytest.mark.skipif(not torch.cuda.is_available() or shutil.which('openssl') is None,
                    reason='CUDA and openssl required')
@pytest.mark.parametrize('transport', ['pinned-tls', 'cuda-ipc'])
def test_four_process_gpu_mtls_inference(transport):
    report = run_mtls_inference(transport=transport, bits=32, batch=4, width=16, hidden=32, timeout=90)
    assert report['exact_match']
    assert report['total_tls_bytes'] > 0


def test_mtls_cli_help():
    result = subprocess.run([sys.executable, '-m', 'examples.mtls_inference', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--transport' in result.stdout
