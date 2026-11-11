"""Check ring32 helper arithmetic and traffic, not hardware enclave isolation."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from experiments.benchmarks.activations.seeded_relu_simulation import infer, seeded_relu_party
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder
from rep4gpu.protocol.runtime import Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _roles(endpoint=DeviceIdealEndpoint, **options):
    torch.set_num_threads(1)
    network, seeds = IdealNetwork(timeout=15), make_seeds()
    kind = seeded_relu_party(b'ring32 seeded helper test seed!!')
    roles = [kind(p, endpoint(network.endpoint(p)),
                  {e: s for e, s in seeds.items() if e != p},
                  bits=32, gpu_open=True, **options) for p in range(4)]
    return network, roles


def _shares(value, roles):
    generator = torch.Generator().manual_seed(157)
    components = [torch.randint(0, 1 << 32, value.shape, generator=generator,
                                dtype=torch.int64) for _ in range(3)]
    components.append((value - sum(components)) & 0xffffffff)
    return [Share(tuple(components[(p + j + 1) % 4].to(role.device)
                        for j in range(3))) for p, role in enumerate(roles)]


def _parallel(network, roles, function):
    def guarded(role):
        try:
            return function(role)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(guarded, roles))


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('method', ['tee_relu', 'tee_trunc'])
@pytest.mark.parametrize('shift', [0, 14, 31])
def test_signed_boundaries_fresh_masks_and_exact_payload(captured, method, shift):
    network, roles = _roles()
    clear = torch.tensor([[-(1 << 31), -(1 << 31) + 1, -16385],
                          [-16384, -1, 0], [1, 16385, (1 << 31) - 1]], dtype=torch.int64)
    inputs = [_shares(clear, roles), _shares(clear.roll(1, dims=1), roles)]
    plans = ([role.prepare_inference(
        lambda x, role=role: getattr(role, method)(x, shift), inputs[0][p],
        check_at_send=True) for p, role in enumerate(roles)] if captured else None)
    assert all(not any(role.counters.values()) for role in roles)
    snapshots = []
    for index in (0, 1, 0):
        records = [DeviceRecorder(role.transport) for role in roles]
        for role, record in zip(roles, records):
            role.transport = record
        outputs = _parallel(network, roles, lambda role: (
            plans[role.party].run(inputs[index][role.party]) if plans else
            getattr(role, method)(inputs[index][role.party], shift)))
        assert all(len(record.transcript) == 2 for record in records)
        assert sum(record.bytes_sent for record in records) == 12 * clear.numel() + 32
        for role, record in zip(roles, records):
            role.transport = record.underlying
        snapshots.append([tuple(part.cpu().clone() for part in output.parts) for output in outputs])
        for snapshot in snapshots[-1]:
            assert all(bool(((part >= 0) & (part < (1 << 32))).all()) for part in snapshot)
        for component in range(4):
            copies = [snapshots[-1][p][(component - p - 1) % 4]
                      for p in range(4) if p != component]
            assert all(torch.equal(copies[0], copy) for copy in copies[1:])
        opened = _parallel(network, roles, lambda role: role.open(outputs[role.party]))
        expected = (clear if index == 0 else clear.roll(1, dims=1)) >> shift
        if method == 'tee_relu':
            expected = expected.clamp_min(0)
        assert all(torch.equal(value.cpu(), expected) for value in opened)
    assert all(any(not torch.equal(a, b) for a, b in zip(first, last))
               for first, last in zip(snapshots[0], snapshots[2]))
    for stream in (4, 5, 6):
        assert len({role.counters[stream] for role in roles}) == 1


@pytest.mark.parametrize('captured', [False, True])
def test_helper_output_can_feed_checked_mpc(captured):
    network, roles = _roles()
    clear = torch.tensor([[-50000, -1, 0, 16384, 32769]], dtype=torch.int64)
    inputs = _shares(clear, roles)
    def forward(role, share):
        hidden = role.tee_relu(share, 14)
        return role.open(role.mul(hidden, role.public(3, like=hidden)))
    plans = ([role.prepare_inference(lambda x, role=role: forward(role, x),
                                     inputs[p], check_at_send=True)
              for p, role in enumerate(roles)] if captured else None)
    output = _parallel(network, roles, lambda role: (
        plans[role.party].run(inputs[role.party]) if plans else forward(role, inputs[role.party])))
    assert all(torch.equal(value.cpu(), (clear >> 14).clamp_min(0) * 3) for value in output)


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('sender', [1, 2])
def test_corrupt_missing_component_or_hash_rejected(captured, sender):
    class CorruptEndpoint(DeviceIdealEndpoint):
        def exchange_device(self, outgoing, sizes):
            if self.underlying.party == sender and 0 in outgoing:
                outgoing = dict(outgoing)
                outgoing[0] = outgoing[0].clone()
                outgoing[0][0] ^= 1
            return super().exchange_device(outgoing, sizes)
    network, roles = _roles(CorruptEndpoint)
    inputs = _shares(torch.tensor([[1 << 28]], dtype=torch.int64), roles)
    plans = ([role.prepare_inference(lambda x, role=role: role.tee_relu(x),
                                     inputs[p], check_at_send=True)
              for p, role in enumerate(roles)] if captured else None)
    def run(role):
        try:
            return plans[role.party].run(inputs[role.party]) if plans else role.tee_relu(inputs[role.party])
        except BaseException as error:
            network.abort(error)
            return error
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, roles))
    assert any(isinstance(result, ValueError) and 'hash mismatch' in str(result) for result in results)
    if plans:
        assert plans[0].failed


def test_invalid_inputs_rejected_before_exchange():
    _, roles = _roles()
    role = roles[0]
    x = role.public(torch.ones((2, 3), dtype=torch.int64, device=role.device))
    for method in (role.tee_relu, role.tee_trunc):
        with pytest.raises(ValueError):
            method(Share(x.parts, True))
        for shift in (-1, 32, 64, 1.5, True):
            with pytest.raises(ValueError):
                method(x, shift)
        with pytest.raises(ValueError):
            method(Share(tuple(part.cpu() for part in x.parts)))


def test_empty_inputs_consume_no_masks_or_payload():
    _, roles = _roles()
    for role in roles:
        empty = role.public(torch.empty((2, 0), dtype=torch.int64, device=role.device))
        counters = dict(role.counters)
        record = DeviceRecorder(role.transport)
        role.transport = record
        for method in (role.tee_relu, role.tee_trunc):
            assert method(empty).shape == empty.shape
        assert role.counters == counters
        assert not record.transcript


def test_ring64_rejected():
    kind = seeded_relu_party(b'ring32 seeded helper test seed!!')
    network, seeds = IdealNetwork(), make_seeds()
    with pytest.raises(ValueError):
        kind(0, DeviceIdealEndpoint(network.endpoint(0)),
             {e: seed for e, seed in seeds.items() if e != 0}, bits=64)


@pytest.mark.parametrize('tee_down', [False, True])
def test_full_ffn_graph_refresh(tee_down):
    from experiments.benchmarks.inference.inference import fixture
    network, roles = _roles(optimized=True, rounding='probabilistic', sparse_products=True,
                            narrow_products=True)
    _, integers, _ = fixture(1, 8, 16, 8, 14)
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
        x, w1, b1, w2, b2 = integers
        hidden = (((x @ w1) + (b1 << 14)) >> 14).clamp_min(0)
        raw = hidden @ w2
        assert int(raw.abs().max()) < 1 << 30
        lower = (raw >> 14) + b2
        upper = lower if tee_down else ((raw + (1 << 14) - 1) >> 14) + b2
        outputs = _parallel(network, roles, lambda role: plans[role.party].run(shares[role.party][0]))
        for result in outputs:
            result = result.cpu()
            assert torch.equal(result, outputs[0].cpu())
            assert bool(((result >= lower) & (result <= upper)).all())
