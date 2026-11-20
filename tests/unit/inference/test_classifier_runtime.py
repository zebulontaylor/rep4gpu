from concurrent.futures import ThreadPoolExecutor

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_classifier_graph_accepts_fresh_private_inputs():
    from rep4gpu.models.classifier_runtime import ClassifierParty, prepare_classifier
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    network, seeds = IdealNetwork(timeout=60), make_seeds()
    roles = [ClassifierParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                            {e: s for e, s in seeds.items() if e != p},
                            bits=32, binary_bits=32, gpu_open=True) for p in range(4)]
    x = torch.tensor([[0, 0, 1], [1, 3, 1], [3, 1, 1]], dtype=torch.int64, device='cuda')
    w = torch.tensor([[2], [-1], [-1]], dtype=torch.int64, device='cuda')

    def share(p):
        rt = roles[p]
        return (rt.input(0, x if p == 0 else None, x.shape),
                rt.input(1, w if p == 1 else None, w.shape))

    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(share, range(4)))
    before = [dict(rt.counters) for rt in roles]
    plans = [prepare_classifier(rt, *value) for rt, value in zip(roles, values)]
    assert before == [dict(rt.counters) for rt in roles]

    def work(p):
        rt, plan = roles[p], plans[p]
        try:
            first = plan.run(*values[p])
            counters = dict(rt.counters)
            fresh = rt.input(0, -x if p == 0 else None, x.shape)
            second = plan.run(fresh, values[p][1])
            assert any(rt.counters[k] > v for k, v in counters.items())
            return first.cpu(), second.cpu()
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    for first, second in outputs:
        assert torch.equal(first, (x.cpu() @ w.cpu() >= 0).long())
        assert torch.equal(second, (-x.cpu() @ w.cpu() >= 0).long())
