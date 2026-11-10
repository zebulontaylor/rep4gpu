"""Role-local replay guards, using an explicitly trusted CPU unit fixture."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from benchmarks.spdz_native_replay import (begin_recording,combine_replay_metrics,
    end_recording,entropy_fixture_class,run_local_replay)
from examples.spdz_trained_toy import infer_authenticated,quantized_fixture,train
from rep4gpu.spdz import TrustedDealer,PRIME
from rep4gpu.spdz_distributed import DistributedParty,split_share,split_triple
from rep4gpu.spdz_polynomial import make_square_masks
from rep4gpu.transport import IdealNetwork


@pytest.mark.parametrize('prepared',[False,True])
def test_role_replay_checks_outputs_and_preserves_live_state(tmp_path,prepared):
    torch.set_num_threads(1)
    fixture=quantized_fixture(train(steps=20),fractional_bits=4,batch=2)
    dealer=TrustedDealer(4,10,'cpu')
    all_values=[split_share(dealer.share(value)) for value in fixture['encoded']]
    first=split_triple(dealer.triple(2,16,64))
    second=split_triple(dealer.triple(2,64,8))
    squares=make_square_masks(dealer,(2,64))
    network=IdealNetwork()
    cls=entropy_fixture_class(DistributedParty)
    parties=[cls(p,network.endpoint(p),dealer.alpha_shares[:,p],session=b'unit-fixture',
                 stack_macs=True,_allow_insecure_test_transport=True) for p in range(4)]
    materials=[dict(matrix_triples=dict(first=first[p],second=second[p]),
                    square_masks=dict(hidden=squares[p])) for p in range(4)]
    values=[[items[p] for items in all_values] for p in range(4)]
    packs=None
    if prepared:
        from benchmarks.spdz_prepared_model import prepare_model
        with ThreadPoolExecutor(4) as pool:
            packs=list(pool.map(lambda p:prepare_model(p,values[p.party],
                              (first[p.party],squares[p.party],second[p.party])),parties))
    states=[begin_recording(party,'host') for party in parties]
    def work(p):
        try:
            if packs is None:
                result=infer_authenticated(p,values[p.party],first[p.party],second[p.party],squares[p.party])
            else:
                from benchmarks.spdz_prepared_model import infer_prepared
                result=infer_prepared(p,values[p.party],packs[p.party],squares[p.party])
            return p.open(result)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:outputs=list(pool.map(work,parties))
    assert all(torch.equal(output,fixture['expected'].remainder(PRIME)) for output in outputs)
    reports=[]
    for p,state in zip(parties,states):
        end_recording(p,state)
        before=(p.round,set(p.spent_triples),set(p._spent_square_masks))
        reports.append(run_local_replay(p,materials[p.party],values[p.party],state,outputs[p.party],
                                       repeats=1,gpu_lock_path=tmp_path/'replay.lock',
                                       profile_path=tmp_path/'native.pstats' if p.party==0 else None,
                                       prepared_model=packs[p.party] if packs is not None else None,replay_warmups=1))
        assert before==(p.round,p.spent_triples,p._spent_square_masks)
    assert (tmp_path/'native.pstats').stat().st_size>0
    summary=combine_replay_metrics(reports,{'test_baseline_ms':1.})
    assert summary['exact_recorded_output_match']
    assert summary['role_local_preprocessing']
    assert summary['ideal_network_phase_max_ms']>0


def test_public_combiner_rejects_duplicate_roles():
    with pytest.raises(ValueError,match='one public timing record'):
        combine_replay_metrics([dict(party=0)]*4,{'baseline':1})


def test_folded_cli_and_clear_baseline_hook():
    import subprocess
    import sys
    from scripts.run_spdz_native import fixture_api
    from benchmarks.spdz_native_replay import trained_baselines
    from examples.spdz_trained_toy import Square
    help_result=subprocess.run([sys.executable,'-m','scripts.run_spdz_native','--help'],
                               capture_output=True,text=True,timeout=30)
    assert help_result.returncode==0
    assert '{tiny,dense,folded,compact,micro}' in help_result.stdout
    assert '--online-mac-keys {7,10}' in help_result.stdout
    torch.set_num_threads(1)
    model=torch.nn.Sequential(torch.nn.Linear(512,2048),Square(),torch.nn.Linear(2048,8)).eval()
    model.register_buffer('embedding_table',torch.randint(0,2,(8,256)).float()*2-1)
    folded=fixture_api('folded').baseline_model(model)
    assert folded[0].in_features==16 and folded[0].out_features==2048
    encoded=torch.nn.functional.one_hot(torch.tensor([[0,1],[2,3]]),8).reshape(2,16).long()
    report=trained_baselines(folded,dict(encoded=[encoded]),'cpu',2)
    assert 'Linear(16,2048)' in report['model']
    assert report['timings']['fp32_eager_ms']>0
    assert report['timings']['numpy_cpu_ms']>0
    assert report['accuracy']['fp32']['max_logit_error']==0
    assert report['accuracy']['numpy']['max_logit_error']<1e-4


def test_native_provider_dispatch_and_security_status():
    from scripts.run_spdz_native import preprocessing_provider
    old=preprocessing_provider({'format':'mama-prime31-role-v1'},10)
    assert old['privacy_audit_status']=='failed' and old['security_target_met'] is False
    assert old['native_mac_keys']==10
    new=preprocessing_provider({'format':'mascot256-prime31-role-v1'},7)
    assert new['native_mac_keys']==1 and new['application_mac_keys']==7
    assert new['internal_mascot_prime']==2**256-189 and new['amplification']==4
    assert new['privacy_audit_status']=='pending' and not new['security_target_met']
    import pytest
    with pytest.raises(ValueError,match='seven application'):
        preprocessing_provider({'format':'mascot256-prime31-role-v1'},10)
    with pytest.raises(ValueError,match='unsupported'):
        preprocessing_provider({'format':'unknown'},7)
