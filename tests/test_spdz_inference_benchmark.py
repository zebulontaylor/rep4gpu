"""CPU checks for complete authenticated benchmark accounting and replay."""
import pytest
import torch
from benchmarks.spdz_inference import run_case,fixture


@pytest.mark.parametrize('workload',['matmul','square_mlp'])
@pytest.mark.parametrize('prepared',[False,True])
def test_checked_fresh_runs_and_salted_replay(workload,tmp_path,prepared):
    torch.set_num_threads(1)
    row=run_case(m=1,k=3,n=2,hidden=4,workload=workload,device='cpu',repeats=1,mac_keys=10,profile_path=tmp_path/'warmup.pstats',prepare_model=prepared,replay_warmups=1)
    assert (tmp_path/'warmup.pstats').stat().st_size>0
    assert row['exact_field_match'] and row['exact_recorded_replay_match']
    assert row['mac_keys']==10
    assert row['prepared_model']==prepared
    if prepared:assert row['model_dependent_preparation_median_ms']>0
    assert row['salts_per_inference']==(2 if workload=='matmul' else 4)
    assert len(row['replay_modes']['host']['replay_phase_samples_seconds'])==4
    assert len(row['replay_modes']['host']['replay_phase_samples_seconds'][0][0])==row['rounds']+1
    assert row['preprocessing_median_ms']>0
    assert set(row['ideal_slowdowns'])==set(row['baselines'])


def test_invalid_workload_rejected():
    with pytest.raises(ValueError,match='workload'):
        fixture(1,3,2,4,'transformer')


def test_streamed_incoming_replay_preserves_messages_without_preloading():
    from rep4gpu.device_transport import DeviceFrame
    from benchmarks.spdz_inference import StreamedIncomingReplay
    frame=DeviceFrame('device',{1:b'abc'},{2:b'defg'})
    replay=StreamedIncomingReplay.create([frame],'cpu',True)
    assert replay._incoming_device==[None]
    replay.start()
    result=replay.exchange_device({1:torch.tensor([97,98,99],dtype=torch.uint8)},{2:4})
    replay.finish();replay.assert_consumed()
    assert torch.equal(result[2],torch.tensor([100,101,102,103],dtype=torch.uint8))
    assert len(replay.phase_times)==2


def test_streamed_replay_has_one_canonical_class():
    import runpy
    from benchmarks.spdz_streamed_replay import StreamedDeviceReplay
    alternate=runpy.run_module('benchmarks.spdz_inference',run_name='benchmark_identity_smoke')
    assert alternate['StreamedDeviceReplay'] is StreamedDeviceReplay
