"""Failure retirement and fixture setup checks for the dependent consumer."""
from pathlib import Path

import pytest

from scripts import run_spdz_goldilocks_stream as runner
from rep4gpu import lowgear_key_cache


@pytest.mark.parametrize('claimed',[False,True])
def test_failed_consumer_retires_epoch_if_any_role_was_claimed(tmp_path,monkeypatch,claimed):
    if claimed:
        role=tmp_path/'party0';role.mkdir();(role/'consumed').write_text('test')
    calls=[]
    monkeypatch.setattr(lowgear_key_cache,'retire_epoch',lambda root,reason:calls.append((Path(root),reason)))
    def failure(*args,**kwargs):raise RuntimeError('post-consumption failure')
    monkeypatch.setattr(runner,'_run',failure)
    with pytest.raises(RuntimeError):runner.run(tmp_path/'out.json',tmp_path)
    assert len(calls)==int(claimed)


def test_stream_checkpoint_uses_exact_square_architecture_without_training():
    import torch
    from examples import spdz_micro_toy as api
    model=torch.nn.Sequential(torch.nn.Linear(8,2),api.Square(),torch.nn.Linear(2,8)).eval()
    model.load_state_dict(torch.load('results/spdz_micro_toy/model.pt',map_location='cpu',weights_only=True))
    fixture=api.quantized_fixture(model,batch=8)
    assert fixture['expected'].argmax(-1).tolist()==[1,2,3,4,5,6,7,0]
