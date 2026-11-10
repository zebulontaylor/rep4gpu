"""Validate warmed clear baselines without running an authenticated protocol."""
import math
import statistics

import pytest
import torch
from threadpoolctl import threadpool_limits

from benchmarks.spdz_inference import baselines,fixture
from benchmarks.spdz_native_replay import trained_baselines
from examples.spdz_trained_toy import Square


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('workload',['matmul','square_mlp'])
def test_generic_captured_baseline_outputs_and_group_statistics(workload):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    values,_=fixture(16,128,128,512,workload)
    with threadpool_limits(limits=1,user_api='blas'):
        timings,errors,details=baselines(values,workload,'cuda',20,return_details=True)
    for label in ('fp32','fp16'):
        device_grouped=details[label+'_device_batch_graph']
        assert len(device_grouped['group_samples_ms'])==5
        assert timings[label+'_device_batch_graph_ms']==statistics.median(device_grouped['group_samples_ms'])
        grouped=details[label+'_graph']
        assert len(grouped['group_samples_ms'])==5
        assert grouped['replays_per_group']==200
        assert grouped['actual_warmup_wall_ms']>=100
        assert timings[label+'_graph_ms']==statistics.median(grouped['group_samples_ms'])
        assert timings[label+'_graph_fastest_group_ms']==min(grouped['group_samples_ms'])
        assert math.isfinite(errors[label+'_graph']) and errors[label+'_graph']<1e-3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_trained_model_captured_baseline_outputs():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(244)
    model=torch.nn.Sequential(torch.nn.Linear(16,64),Square(),torch.nn.Linear(64,8)).eval()
    inputs=torch.randint(-1,2,(64,16))
    with threadpool_limits(limits=1,user_api='blas'):
        report=trained_baselines(model,dict(encoded=[inputs]),'cuda',20)
    for label,tolerance in [('fp32',1e-4),('fp16',.02)]:
        assert report['accuracy'][label+'_graph']['max_logit_error']<tolerance
        assert report['timings'][label+'_device_batch_graph_ms']>0
        assert len(report['measurements'][label+'_device_batch_graph']['group_samples_ms'])==5
        assert report['timings'][label+'_graph_fastest_group_ms']<=report['timings'][label+'_graph_ms']
        assert len(report['measurements'][label+'_graph']['group_samples_ms'])==5
