"""CPU validation of the device benchmark's fixture proof and estimator."""
import pytest
import torch
from experiments.benchmarks.inference.device_inference import fixture_range,phase_estimate,run_case
from experiments.benchmarks.inference.inference import fixture


def test_public_fixture_bounds():
    _,values,_=fixture(1,8,8,4)
    proof=fixture_range(values,32,12)
    assert proof['accumulator_fits_signed_ring']
    assert proof['maximum_accumulator'] < 2**31


def test_fixture_rejects_ring_overflow():
    values=[torch.tensor([[4096]]),torch.tensor([[2**30]]),torch.tensor([0]),
            torch.tensor([[4096]]),torch.tensor([0])]
    with pytest.raises(ValueError,match='exceeds signed ring32'):
        fixture_range(values,32,12)


def test_phase_estimate_uses_maxima_after_medians():
    samples=[[[1.,2.],[3.,4.]],[[5.,1.],[7.,3.]]]
    assert phase_estimate(samples)==9000


def test_device_runtime_requirement_precedes_cuda_imports():
    with pytest.raises(ValueError,match='requires CUDA'):
        run_case(1,8,8,4,device='cpu')


def test_payload_profile_matches_timing_phase_convention():
    from types import SimpleNamespace
    from rep4gpu.protocol.device_transport import DeviceFrame
    from experiments.benchmarks.inference.device_inference import phase_payloads
    records=[SimpleNamespace(transcript=[
        DeviceFrame('device',{(party+1)%4: b'abc'},{(party-1)%4: b'def'}),
        DeviceFrame('bytes',{(party+1)%4: b'ab'},{(party-1)%4: b'cd'})]) for party in range(4)]
    profile=phase_payloads(records)
    assert profile['round_kinds']==['device','bytes']
    assert profile['outgoing_bytes_per_party_per_phase']==[[3,2,0]]*4
    assert profile['incoming_bytes_per_party_per_phase']==[[3,2,0]]*4


@pytest.mark.parametrize('bits,prepared',[(64,True),(32,False)])
def test_five32_rejected_before_cuda_setup(bits,prepared):
    with pytest.raises(ValueError,match='five32 requires bits=32 and prepared=True'):
        run_case(1,8,8,4,device='cpu',bits=bits,prepared=prepared,weight_storage='five32')


def test_weight_storage_counts_retained_buffers_once():
    from types import SimpleNamespace
    from experiments.benchmarks.inference.device_inference import weight_storage_bytes
    words=torch.zeros(5,7,dtype=torch.int32)
    wrapper=SimpleNamespace(parts=(SimpleNamespace(words=words),SimpleNamespace(words=words.view(5,7))))
    assert weight_storage_bytes([wrapper])==5*7*4


def test_probabilistic_bounds_propagate_both_weight_signs():
    from experiments.benchmarks.inference.device_inference import fixture_bounds
    values=[torch.tensor([[3]]),torch.tensor([[3]]),torch.tensor([0]),
            torch.tensor([[-3,5]]),torch.tensor([0,0])]
    proof,lower,upper=fixture_bounds(values,32,2,'probabilistic')
    assert torch.equal(lower,torch.tensor([[-3,2]]))
    assert torch.equal(upper,torch.tensor([[-1,4]]))
    for hidden in [2,3]:
        raw=hidden*values[3]
        floor=raw >> 2
        ceil=(raw+3) >> 2
        assert ((floor>=lower)&(floor<=upper)).all()
        assert ((ceil>=lower)&(ceil<=upper)).all()
    assert proof['rounding_bounds_validated']


def test_probabilistic_contract_is_strict():
    from experiments.benchmarks.inference.device_inference import fixture_bounds
    values=[torch.tensor([[32768]]),torch.tensor([[32768]]),torch.tensor([0]),
            torch.tensor([[0]]),torch.tensor([0])]
    with pytest.raises(ValueError,match='strict probabilistic'):
        fixture_bounds(values,32,12,'probabilistic')


def test_floor_bounds_equal_exact_integer_reference():
    from experiments.benchmarks.inference.device_inference import fixture_bounds
    _,values,expected=fixture(1,8,8,4)
    _,lower,upper=fixture_bounds(values,32,12,'floor')
    assert torch.equal(lower,expected) and torch.equal(upper,expected)


def test_probability_intervals_do_not_replace_agreement_or_exact_replay():
    from experiments.benchmarks.inference.device_inference import validate_outputs
    lower,upper=torch.tensor([0]),torch.tensor([2])
    zero,one=torch.tensor([0]),torch.tensor([1])
    assert torch.equal(validate_outputs([one,one],lower,upper),one)
    with pytest.raises(AssertionError,match='parties disagree'):
        validate_outputs([zero,one],lower,upper)
    with pytest.raises(AssertionError,match='exact recorded'):
        validate_outputs([one],lower,upper,expected=zero)
    with pytest.raises(AssertionError,match='rounding bounds'):
        validate_outputs([torch.tensor([3])],lower,upper)


def test_local_matmul_route_metadata_matches_validated_dispatch():
    from experiments.benchmarks.inference.device_inference import matmul_backend
    tuned=matmul_backend(128,32,True,'words32',True,(8,9))
    assert tuned['tile_mnk']==[32,128,32] and tuned['num_stages']==1
    assert matmul_backend(1,32,True,'words32',True,(8,9))['route']=='baseline direct decode'
    assert matmul_backend(1,64,True,'limbs',True,(8,9))['route']=='baseline packed decode'
    for batch,bits,storage,optimized,capability in [(8,32,'words32',True,(8,9)),
            (128,64,'limbs',True,(8,9)),(128,32,'words32',False,(8,9)),
            (128,32,'words32',True,(9,0))]:
        assert matmul_backend(batch,bits,True,storage,optimized,capability)['tile_mnk']==[16,32,64]
