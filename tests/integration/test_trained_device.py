"""Validate GPU-crypto inference on all 64 trained toy contexts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
import torch
from torch import nn
from experiments.examples.toy_next_token import quantized_fixture
from experiments.benchmarks.inference.device_inference import fixture_bounds
from rep4gpu.arithmetic.activations import FixedPoint
from rep4gpu.models.device_inference import DeviceInferenceParty
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork


CHECKPOINT = Path(__file__).resolve().parents[2]/'experiments/results/examples/toy_next_token/model.pt'
FRACTIONAL_BITS = 12


@pytest.fixture(scope='module')
def trained_model():
    assert CHECKPOINT.is_file(),f'trained toy checkpoint missing: {CHECKPOINT}'
    model = nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,8))
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    return model.eval()


def _public_fixture(model,bits,rounding):
    integers,expected,targets,_ = quantized_fixture(model,FRACTIONAL_BITS,bits)
    proof,lower,upper = fixture_bounds(integers,bits,FRACTIONAL_BITS,rounding)
    # Establish the probabilistic protocol's strict range even for floor tests.
    assert proof['maximum_accumulator'] < 1 << (bits-2)
    rows = torch.arange(targets.numel())
    target_lower = lower[rows,targets]
    competitors_upper = upper.clone()
    competitors_upper[rows,targets] = -(1 << 63)
    classification_margin = target_lower-competitors_upper.max(dim=1).values
    assert bool(torch.all(classification_margin > 0))
    assert torch.equal(expected.argmax(dim=1),targets)
    return integers,expected,targets,lower,upper


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('rounding',['floor','probabilistic'])
def test_trained_device_public_range_and_classification_margin(trained_model,bits,rounding):
    integers,expected,targets,lower,upper = _public_fixture(trained_model,bits,rounding)
    assert integers[0].shape == (64,16)
    assert expected.shape == (64,8) and targets.shape == (64,)
    assert bool(torch.all(lower <= upper))
    if rounding == 'floor':
        assert torch.equal(lower,expected) and torch.equal(upper,expected)


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('rounding',['floor','probabilistic'])
def test_trained_fast_device_all_contexts(trained_model,bits,rounding):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    # All public range and strict classification-margin checks precede coins.
    integers,expected,targets,lower,upper = _public_fixture(trained_model,bits,rounding)
    device = torch.device('cuda',torch.cuda.current_device())
    values = tuple(value.to(device) for value in integers)
    network,seeds = IdealNetwork(timeout=120),make_seeds()
    def role(party):
        try:
            transport = DeviceIdealEndpoint(network.endpoint(party),device=device)
            runtime = DeviceInferenceParty(party,transport,
                                          {e:seed for e,seed in seeds.items() if e != party},
                                          device=device,bits=bits,optimized=True,
                                          gpu_open=True,rounding=rounding)
            shared = [runtime.input(owner,value if party == owner else None,value.shape)
                      for owner,value in zip((0,1,1,1,1),values)]
            x,w1,b1,w2,b2 = shared
            storage = 'words32' if bits == 32 else 'limbs'
            w1 = runtime.prepare_weight_share(w1,storage=storage)
            w2 = runtime.prepare_weight_share(w2,storage=storage)
            fp = FixedPoint(runtime,FRACTIONAL_BITS)
            hidden = fp.matmul_relu(x,w1,b1)
            output = fp.add(fp.matmul(hidden,w2),b2)
            return runtime.open(output).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(role,range(4)))
    assert all(torch.equal(value,outputs[0]) for value in outputs[1:])
    if rounding == 'floor':
        assert torch.equal(outputs[0],expected)
    else:
        assert bool(torch.all((outputs[0] >= lower) & (outputs[0] <= upper)))
    assert all(torch.equal(value.argmax(dim=1),targets) for value in outputs)
