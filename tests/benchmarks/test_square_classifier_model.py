from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from experiments.benchmarks.inference.text_classifier_model import shared_scores, load_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('architecture', ['square', 'square_unscaled'])
def test_private_cuda_square_scores_and_bias_broadcast(architecture):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    x = torch.tensor([[0, 7, 3], [8, 0, 4], [2, 11, 8]], dtype=torch.int64)
    parameters = dict(w1=torch.tensor([[3, -1], [-2, 2], [4, 1]]),
                      b1=torch.tensor([-5, 3]), w2=torch.tensor([[3], [-7]]),
                      b2=torch.tensor([13]))
    shift = 2 if architecture == 'square' else 0
    first = ((x @ parameters['w1']) >> shift) + parameters['b1']
    hidden = first.square() >> shift
    expected = ((hidden @ parameters['w2']) >> shift) + parameters['b2']
    network, seeds = IdealNetwork(timeout=60), make_seeds()
    roles = [PhaseGraphParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                            {i: s for i, s in seeds.items() if i != p},
                            bits=32, binary_bits=32, gpu_open=True) for p in range(4)]

    def share(p):
        rt = roles[p]
        features = rt.input(0, x if p == 0 else None, x.shape)
        weights = {name: rt.input(1, value if p == 1 else None, value.shape)
                   for name, value in parameters.items()}
        return features, weights

    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(share, range(4)))
    plans = [rt.prepare_inference(
        lambda x, w, rt=rt: rt.open(shared_scores(rt, x, w, architecture, shift)),
        *value, check_at_send=True) for rt, value in zip(roles, values)]

    def evaluate(p):
        try:
            return plans[p].run(*values[p]).cpu()
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(evaluate, range(4)))
    assert all(torch.equal(output, expected) for output in outputs)


def test_unscaled_square_rejects_overflow_before_integer_reference(tmp_path):
    model = tmp_path/'unsafe.npz'
    weights = dict(w1=np.array([[65535]], dtype=np.int32), b1=np.array([0], dtype=np.int32),
                   w2=np.array([[1]], dtype=np.int32), b2=np.array([0], dtype=np.int32))
    np.savez(model, x_test=np.array([[65535]], dtype=np.int32),
             architecture=np.array('square_unscaled'), input_divisor=np.array(8), **weights,
             **{'float_'+k: v.astype(np.float32) for k, v in weights.items()})
    with pytest.raises(ValueError, match='accumulator range'):
        load_model(model, 1)
