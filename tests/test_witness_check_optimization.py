"""Public-bound checks retain full-ring semantics with fewer secret operations."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork
from rep4gpu.witness_activations import InvalidWitness, WitnessSession, numerical_program


class LegacyChecks(WitnessSession):
    """Original checked schedule, retained only as a communication comparator."""

    def _interval(self, value, lower, upper):
        self._active()
        self._arithmetic(value)
        rt = self.runtime
        below = rt.lt_public(value, lower)
        above = rt.sub(rt.public(1, like=value), rt.lt_public(value, upper+1))
        self._violation(below)
        self._violation(above)
        return below, above

    def nonnegative(self, value):
        self._active()
        self._arithmetic(value)
        self._violation(self.runtime.lt_public(value, 0))
        self._residuals += 1

    def finish(self):
        self._active()
        self._closed = True
        accepted = self.runtime.lt_public(self._count, 1)
        outputs, self._outputs = tuple(self._outputs), []
        if int(self.runtime.open(accepted).item()) != 1:
            raise InvalidWitness('activation witness batch rejected')
        return outputs


def _run_four(work, device='cpu'):
    torch.set_num_threads(1)
    network = IdealNetwork(timeout=120)
    seeds = make_seeds()

    def run(role):
        try:
            selected = {i: s for i, s in seeds.items() if i != role}
            if device == 'cpu':
                party = Party(role, network.endpoint(role), selected)
            else:
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceIdealEndpoint
                target = torch.device(device)
                party = DeviceInferenceParty(
                    role, DeviceIdealEndpoint(network.endpoint(role), device=target), selected,
                    device=target, bits=64, binary_bits=64, optimized=True,
                    gpu_open=True, rounding='floor')
            return work(party)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, range(4)))


def _secret(party, value):
    return party.input(0, value if party.party == 0 else None, value.shape)


@pytest.mark.parametrize('bounds', [(-2**63, -2**63+1), (-5, -1), (-4, 4),
                                    (0, 1), (3, 5), (2**63-4, 2**63-2)])
def test_interval_matches_signed_comparison_at_ring_boundaries(bounds):
    lower, upper = bounds
    points = sorted({-2**63, 2**63-1, -5, -4, -1, 0, 1, 3, 4, 5,
                     lower, upper, max(-2**63, lower-1), min(2**63-1, upper+1)})
    encoded = torch.tensor(points, dtype=torch.int64)

    def work(party):
        session = WitnessSession(party)
        below, above = session._interval(_secret(party, encoded), lower, upper)
        return party.open(below), party.open(above)

    for below, above in _run_four(work):
        assert torch.equal(below, (encoded < lower).long())
        assert torch.equal(above, (encoded > upper).long())


def measure_reciprocal_checks(device='cpu', elements=17):
    """Measure only execution and acceptance, excluding input/output sharing."""
    scale = 1 << 24
    x = torch.round(torch.linspace(1, 2, elements, dtype=torch.float64)*scale).long()
    w = torch.round(scale*scale / x.double()).long()
    measurements = {}
    for session_type in (LegacyChecks, WitnessSession):
        def work(party):
            inputs = {'x': _secret(party, x)}
            witnesses = {'reciprocal': _secret(party, w)}
            transport = party.transport
            # DeviceIdealEndpoint tracks bytes in its underlying CPU endpoint.
            underlying = getattr(transport, 'underlying', transport)
            start = underlying.bytes_sent
            session = session_type(party)
            session.evaluate(numerical_program('reciprocal'), inputs, witnesses)
            output, = session.finish()
            size = underlying.bytes_sent-start
            return party.open(output).cpu(), size
        rows = _run_four(work, device)
        assert all(torch.equal(actual, w) for actual, _ in rows)
        measurements[session_type.__name__] = sum(size for _, size in rows)
    return measurements


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_checks_reduce_actual_traffic_without_changing_output(device):
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    measurements = measure_reciprocal_checks(device)
    assert measurements['WitnessSession'] < measurements['LegacyChecks']
