"""Public-fixture trusted-helper simulation; no enclave isolation is tested."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from experiments.benchmarks.activations.simulated_activation import simulated_activation_party
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _inputs(values, roles):
    generator = torch.Generator().manual_seed(591)
    components = [torch.randint(0, 2**32, values.shape, generator=generator,
                                dtype=torch.int64) for _ in range(3)]
    components.append((values-sum(components)) & 0xffffffff)
    return [Share(tuple(components[(p+j+1) % 4].to(role.device) for j in range(3)), False)
            for p, role in enumerate(roles)]


@pytest.mark.parametrize('captured', [False, True])
def test_two_fresh_inputs_followed_by_normal_mpc(captured):
    network, seeds = IdealNetwork(timeout=30), make_seeds()
    kind = simulated_activation_party(b'simulated activation fixture key!')
    roles = [kind(p, DeviceIdealEndpoint(network.endpoint(p)),
                  {e:s for e,s in seeds.items() if e != p}, bits=32, gpu_open=True)
             for p in range(4)]
    values = [torch.tensor([[-2147483648, -17, -1], [0, 17, 2147483647]], dtype=torch.int64),
              torch.tensor([[-80, 0, 8], [24, 32, 40]], dtype=torch.int64)]
    inputs = [_inputs(value, roles) for value in values]
    def function(role, value):
        result = role.trunc_relu(value, 3)
        opened = role.open(role.mul(result, role.public(3, like=result)))
        return result, opened
    plans = ([role.prepare_inference(lambda x, role=role: function(role, x), inputs[0][p],
                                     check_at_send=True) for p, role in enumerate(roles)]
             if captured else None)
    assert all(not any(role.counters.values()) for role in roles)
    def worker(p):
        try:
            outputs = []
            for run in range(2):
                result, opened = (plans[p].run(inputs[run][p]) if plans
                                  else function(roles[p], inputs[run][p]))
                outputs.append((tuple(part.cpu() for part in result.parts), opened.cpu()))
            return outputs
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(worker, range(4)))
    for p in range(4):
        for run in range(2):
            expected = (((values[run] >> 3).clamp_min(0))*3) & 0xffffffff
            torch.testing.assert_close(outputs[p][run][1], expected, rtol=0, atol=0)
        assert any(not torch.equal(a,b) for a,b in zip(outputs[p][0][0], outputs[p][1][0]))
    for run in range(2):
        for component in range(4):
            copies = [outputs[p][run][0][(component-p-1) % 4] for p in range(4) if p != component]
            assert all(torch.equal(copies[0], value) for value in copies[1:])
            assert torch.any(copies[0] != 0)
    for excluded in range(4):
        counts = [roles[p].counters[excluded] for p in range(4) if p != excluded]
        assert len(set(counts)) == 1
    assert roles[0].counters[2] >= 2
    assert all(roles[p]._fixture_helper_key is None for p in (2,3))


@pytest.mark.parametrize('corrupt_round', [0, 1])
@pytest.mark.parametrize('captured', [False, True])
def test_corrupt_helper_input_or_output_rejected(corrupt_round, captured):
    network, seeds = IdealNetwork(timeout=5), make_seeds()
    class CorruptEndpoint(DeviceIdealEndpoint):
        def __init__(self, underlying, party):
            super().__init__(underlying)
            self.party, self.calls = party, 0
        def _corrupt(self, outgoing):
            selected = (self.party == 1 and corrupt_round == 0 or
                        self.party == 0 and corrupt_round == 1)
            if selected and self.calls == corrupt_round:
                peer = 0 if corrupt_round == 0 else 2
                outgoing = dict(outgoing)
                outgoing[peer] = outgoing[peer].clone()
                outgoing[peer][0] ^= 1
            self.calls += 1
            return outgoing
        def exchange_device(self, outgoing, sizes):
            return super().exchange_device(self._corrupt(outgoing), sizes)
    kind = simulated_activation_party(b'simulated activation fixture key!')
    roles = [kind(p, CorruptEndpoint(network.endpoint(p), p),
                  {e:s for e,s in seeds.items() if e != p}, bits=32, gpu_open=True)
             for p in range(4)]
    inputs = _inputs(torch.tensor([[-8, 0, 8]], dtype=torch.int64), roles)
    plans = ([role.prepare_inference(lambda x, role=role: role.trunc_relu(x, 3), inputs[p],
                                     check_at_send=True) for p, role in enumerate(roles)]
             if captured else None)
    def worker(p):
        try:
            return plans[p].run(inputs[p]) if plans else roles[p].trunc_relu(inputs[p], 3)
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(worker, range(4)))
    assert any(isinstance(value, ValueError) and 'hash mismatch' in str(value) for value in outputs)
    if captured:
        assert any(plan.failed for plan in plans)
