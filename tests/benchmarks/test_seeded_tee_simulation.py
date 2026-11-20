"""Check the seeded ring64 helper simulation, not enclave isolation."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from experiments.benchmarks.activations.seeded_tee_simulation import seeded_tee_party
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder
from rep4gpu.protocol.runtime import Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _roles(endpoint=DeviceIdealEndpoint, **options):
    torch.set_num_threads(1)
    network, seeds = IdealNetwork(timeout=15), make_seeds()
    kind = seeded_tee_party(b'seeded helper simulation test key')
    roles = [kind(p, endpoint(network.endpoint(p)),
                  {e: s for e, s in seeds.items() if e != p},
                  bits=64, gpu_open=True, **options) for p in range(4)]
    return network, roles


def _shares(value, roles):
    generator = torch.Generator().manual_seed(8149)
    components = [torch.randint(-(1 << 62), 1 << 62, value.shape,
                                generator=generator, dtype=torch.int64)
                  for _ in range(3)]
    components.append(value - sum(components))
    return [Share(tuple(components[(p + j + 1) % 4].to(role.device)
                        for j in range(3))) for p, role in enumerate(roles)]


def _parallel(network, roles, function):
    def run(role):
        try:
            return function(role)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, roles))


@pytest.mark.parametrize('captured', [False, True])
def test_fresh_outputs_payload_and_following_mpc(captured):
    network, roles = _roles()
    scale = 1 << 24
    gate = torch.tensor([[-4., -1., 0.], [.125, 1., 4.]])
    value = torch.tensor([[.5, -1., 2.], [-2., .25, 1.]])
    raw = [(torch.round(gate.double() * scale).long() * scale + 17,
            torch.round(value.double() * scale).long() * scale + 31),
           (torch.round(gate.double() * scale / 2).long() * scale + 13,
            torch.round(value.double() * scale / 2).long() * scale + 7)]
    inputs = [(_shares(a, roles), _shares(b, roles)) for a, b in raw]
    plans = ([role.prepare_inference(lambda a, b, role=role: role.tee_swiglu(a, b),
                                     inputs[0][0][p], inputs[0][1][p],
                                     check_at_send=True)
              for p, role in enumerate(roles)] if captured else None)
    assert all(not any(role.counters.values()) for role in roles)
    snapshots = []
    for index in (0, 1, 0):
        records = [DeviceRecorder(role.transport) for role in roles]
        for role, record in zip(roles, records):
            role.transport = record
        def evaluate(role):
            p = role.party
            a, b = inputs[index][0][p], inputs[index][1][p]
            return plans[p].run(a, b) if plans else role.tee_swiglu(a, b)
        output = _parallel(network, roles, evaluate)
        assert all(len(record.transcript) == 2 for record in records)
        assert sum(record.bytes_sent for record in records) == 32 * gate.numel() + 32
        for role, record in zip(roles, records):
            role.transport = record.underlying
        snapshots.append([tuple(t.cpu() for t in share.parts) for share in output])
        opened = _parallel(network, roles, lambda role: role.open(role.mul(
            output[role.party], role.public(3, like=output[role.party]))))
        a, b = raw[index]
        expected = torch.round(torch.nn.functional.silu((a >> 24).double() / scale)
                               * ((b >> 24).double() / scale) * scale).long() * 3
        assert all(torch.equal(result.cpu(), expected) for result in opened)
        for component in range(4):
            copies = [snapshots[-1][p][(component - p - 1) % 4]
                      for p in range(4) if p != component]
            assert all(torch.equal(copies[0], copy) for copy in copies[1:])
    assert all(any(not torch.equal(a, b) for a, b in zip(first, last))
               for first, last in zip(snapshots[0], snapshots[2]))
    for excluded in range(4):
        assert len({role.counters[excluded] for role in roles
                    if role.party != excluded}) == 1


@pytest.mark.parametrize('captured', [False, True])
def test_corrupt_missing_component_rejected(captured):
    class CorruptEndpoint(DeviceIdealEndpoint):
        def exchange_device(self, outgoing, sizes):
            if self.underlying.party == 1 and 0 in outgoing:
                outgoing = dict(outgoing)
                outgoing[0] = outgoing[0].clone()
                outgoing[0][0] ^= 1
            return super().exchange_device(outgoing, sizes)
    network, roles = _roles(CorruptEndpoint)
    inputs = _shares(torch.tensor([[1 << 48]], dtype=torch.int64), roles)
    plans = ([role.prepare_inference(lambda x, role=role: role.tee_swiglu(x, x),
                                     inputs[p], check_at_send=True)
              for p, role in enumerate(roles)] if captured else None)
    def run(role):
        p = role.party
        try:
            return plans[p].run(inputs[p],) if plans else role.tee_swiglu(inputs[p], inputs[p])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, roles))
    assert any(isinstance(result, ValueError) and 'hash mismatch' in str(result)
               for result in results)
    if plans:
        assert plans[0].failed


def test_invalid_inputs_rejected_before_exchange():
    _, roles = _roles()
    role = roles[0]
    x = role.public(torch.ones((2, 3), dtype=torch.int64, device=role.device))
    with pytest.raises(ValueError):
        role.tee_swiglu(x, x.reshape(3, 2))
    with pytest.raises(ValueError):
        role.tee_swiglu(Share(x.parts, True), x)
    with pytest.raises(ValueError):
        role.tee_trunc(Share(x.parts, True))
    for fractional_bits in (-1, 0, 19, 25, 64, 1.5):
        with pytest.raises(ValueError):
            role.tee_swiglu(x, x, fractional_bits=fractional_bits)
        with pytest.raises(ValueError):
            role.tee_trunc(x, fractional_bits=fractional_bits)


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('fractional_bits', [20, 24])
def test_trunc_signed_edges_and_payload(captured, fractional_bits):
    network, roles = _roles()
    scale = 1 << fractional_bits
    clear = torch.tensor([[-(1 << 63), -(1 << 63) + 1, -scale - 1, -scale],
                          [-1, 0, scale + 1, (1 << 63) - 1]], dtype=torch.int64)
    inputs = [_shares(clear, roles), _shares(clear.roll(1, dims=1), roles)]
    plans = ([role.prepare_inference(
        lambda x, role=role: role.tee_trunc(x, fractional_bits), inputs[0][p],
        check_at_send=True) for p, role in enumerate(roles)] if captured else None)
    for index, shares in enumerate(inputs):
        records = [DeviceRecorder(role.transport) for role in roles]
        for role, record in zip(roles, records):
            role.transport = record
        def truncate(role):
            p = role.party
            return (plans[p].run(shares[p]) if plans else
                    role.tee_trunc(shares[p], fractional_bits))
        outputs = _parallel(network, roles, truncate)
        assert all(len(record.transcript) == 2 for record in records)
        assert sum(record.bytes_sent for record in records) == 24 * clear.numel() + 32
        for role, record in zip(roles, records):
            role.transport = record.underlying
        opened = _parallel(network, roles, lambda role: role.open(outputs[role.party]))
        expected = (clear if index == 0 else clear.roll(1, dims=1)) >> fractional_bits
        assert all(torch.equal(value.cpu(), expected) for value in opened)


def test_empty_inputs_do_not_send_or_consume_masks():
    _, roles = _roles()
    for role in roles:
        x = role.public(torch.empty((2, 0), dtype=torch.int64, device=role.device))
        counters = dict(role.counters)
        record = DeviceRecorder(role.transport)
        role.transport = record
        result = role.tee_swiglu(x, x)
        assert result.shape == x.shape
        result = role.tee_trunc(x)
        assert result.shape == x.shape
        assert role.counters == counters
        assert not record.transcript


def test_ring32_rejected():
    kind = seeded_tee_party(b'seeded helper simulation test key')
    network, seeds = IdealNetwork(), make_seeds()
    with pytest.raises(ValueError):
        kind(0, DeviceIdealEndpoint(network.endpoint(0)),
             {e: seed for e, seed in seeds.items() if e != 0}, bits=32)


@pytest.mark.parametrize('tee_down', [False, True])
def test_full_ffn_graph_refresh(tee_down):
    from experiments.benchmarks.inference.numerical_inference_common import fixture
    from experiments.benchmarks.inference.seeded_tee_inference import oracle
    from experiments.benchmarks.activations.seeded_tee_simulation import infer

    network, roles = _roles(optimized=True, rounding='probabilistic', sparse_products=True)
    _, integers = fixture('swiglu', batch=1, width=8, hidden=16, out=8)
    by_value = [_shares(value, roles) for value in integers]
    shares = [[value[p] for value in by_value] for p in range(4)]
    plans = [role.prepare_inference(
        lambda x, role=role: infer(role, [x, *shares[role.party][1:]], tee_down=tee_down),
        shares[p][0], check_at_send=True) for p, role in enumerate(roles)]
    for refresh in range(2):
        if refresh:
            integers[0] = integers[0] // 2
            updated = _shares(integers[0], roles)
            for p in range(4):
                shares[p][0] = updated[p]
        interval, _ = oracle(integers, tee_down=tee_down)
        outputs = _parallel(network, roles, lambda role: plans[role.party].run(
            shares[role.party][0]))
        for result in outputs:
            result = result.cpu()
            assert torch.equal(result, outputs[0].cpu())
            assert bool(((result >= interval.lo) & (result <= interval.hi)).all())
            if tee_down:
                assert torch.equal(interval.lo, interval.hi)
                assert torch.equal(result, interval.lo)


@pytest.mark.parametrize('kind', ['relu', 'silu', 'gelu', 'swiglu'])
@pytest.mark.parametrize('captured', [False, True])
def test_generic_activation_exact_words_and_payload(kind, captured):
    network, roles = _roles()
    scale = 1 << 24
    # Include signed raw fractions: the boundary floors before evaluation.
    encoded = torch.tensor([[-4*scale, -scale, -1, 0, 1, scale//8, scale, 4*scale]],
                           dtype=torch.int64)
    raw = encoded*scale + torch.tensor([[1, scale-1, 1, 0, scale-1, 17, 31, 0]])
    value_raw = -raw.flip(-1)
    gates, values = _shares(raw, roles), _shares(value_raw, roles)
    def call(role, gate, value):
        return role.tee_activation(kind, gate, value if kind == 'swiglu' else None)
    plans = ([role.prepare_inference(lambda a, b, role=role: call(role, a, b),
                                     gates[p], values[p], check_at_send=True)
              for p, role in enumerate(roles)] if captured else None)
    records = [DeviceRecorder(role.transport) for role in roles]
    for role, record in zip(roles, records):
        role.transport = record
    outputs = _parallel(network, roles, lambda role: (
        plans[role.party].run(gates[role.party], values[role.party]) if captured
        else call(role, gates[role.party], values[role.party])))
    assert all(len(record.transcript) == 2 for record in records)
    assert sum(record.bytes_sent for record in records) == (32 if kind == 'swiglu' else 24)*raw.numel()+32
    for role, record in zip(roles, records):
        role.transport = record.underlying
    actual = _parallel(network, roles, lambda role: role.open(outputs[role.party]))
    gate = (raw >> 24).double()/scale
    if kind == 'relu':
        real = gate.clamp_min(0)
    elif kind == 'gelu':
        real = .5*gate*(1+torch.erf(gate/(2**.5)))
    else:
        real = gate/(1+torch.exp(-gate))
        if kind == 'swiglu':
            real = real*((value_raw >> 24).double()/scale)
    expected = torch.round(real*scale).long()
    assert all(torch.equal(output.cpu(), expected) for output in actual)


def test_generic_activation_invalid_arity_and_kind():
    _, roles = _roles()
    role = roles[0]
    x = role.public(torch.ones((2, 3), dtype=torch.int64, device=role.device))
    for kind, value in [('sigmoid', None), ('swiglu', None), ('relu', x), ('gelu', x)]:
        with pytest.raises(ValueError):
            role.tee_activation(kind, x, value)
