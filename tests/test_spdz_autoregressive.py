import torch
from benchmarks.spdz_autoregressive import integer_stream,clear_baselines
from examples.spdz_micro_toy import train,quantized_fixture


def test_dependent_integer_stream_and_matching_cpu_baselines():
    torch.set_num_threads(1)
    model=train(1);model.load_state_dict(torch.load('results/spdz_micro_toy/model.pt',weights_only=True))
    fixture=quantized_fixture(model,batch=1)
    expected=integer_stream([x.tolist() for x in fixture['encoded'][1:]],3,12)
    assert expected['tokens']==[(3+i)%8 for i in range(13)]
    clear=clear_baselines(model,3,12,repeats=2)
    assert clear['fp32_reference']['tokens']==expected['tokens']
    assert clear['accuracy']['numpy_cpu']['matches_fp32_tokens']
    assert all(t>0 for t in clear['timings'].values())


def test_gpu_resident_step_graph_preserves_dependent_trace():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=train(1);model.load_state_dict(torch.load('results/spdz_micro_toy/model.pt',weights_only=True))
    result=clear_baselines(model,5,17,repeats=2,gpu_device='cuda')
    expected=[(5+i)%8 for i in range(18)]
    for name in ('fp32_eager','fp16_eager','fp32_step_graph','fp16_step_graph'):
        assert result['accuracy'][name]['tokens']==expected
        assert result['timings'][name+'_resident_sequence_ms']>0
        assert result['timings'][name+'_host_logits_sequence_ms']>0
    assert len(result['measurements']['fp32_step_graph_resident_events']['group_samples_ms'])==5
