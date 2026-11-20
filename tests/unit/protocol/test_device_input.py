from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.models.device_inference import DeviceInferenceParty
from rep4gpu.protocol.device_input import DeviceInputMixin
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork


class InputParty(DeviceInputMixin, DeviceInferenceParty):
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('owner,bits,binary', [(0, 32, False), (1, 64, True), (2, 32, True), (3, 64, False)])
@pytest.mark.parametrize('empty', [False, True])
def test_device_input_matches_reference_shares(owner, bits, binary, empty):
    seeds = make_seeds()
    x = torch.tensor([-(1 << 63), -1, 0, (1 << 63) - 1], device='cuda')
    if empty:
        x = x[:0]

    def execute(kind):
        network = IdealNetwork(timeout=30)
        roles = [kind(p, DeviceIdealEndpoint(network.endpoint(p)),
                      {e: s for e, s in seeds.items() if e != p}, bits=bits,
                      binary_bits=bits) for p in range(4)]

        def worker(p):
            try:
                share = roles[p].input(owner, x if p == owner else None, x.shape, binary)
                return tuple(t.cpu() for t in share.parts), roles[p].open(share).cpu()
            except BaseException as error:
                network.abort(error)
                raise

        with ThreadPoolExecutor(4) as pool:
            return list(pool.map(worker, range(4)))

    reference, device = execute(DeviceInferenceParty), execute(InputParty)
    for (a, opened_a), (b, opened_b) in zip(reference, device):
        assert all(torch.equal(left, right) for left, right in zip(a, b))
        assert torch.equal(opened_a, opened_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_inconsistent_input_correction_is_rejected():
    seeds, network = make_seeds(), IdealNetwork(timeout=30)
    roles = []
    for p in range(4):
        endpoint = DeviceIdealEndpoint(network.endpoint(p))
        if p == 1:
            original = endpoint.exchange_device
            count = [0]

            def corrupt(outgoing, sizes):
                received = original(outgoing, sizes)
                if count[0] == 0:
                    received[0] = received[0].clone()
                    received[0][0] ^= 1
                count[0] += 1
                return received

            endpoint.exchange_device = corrupt
        roles.append(InputParty(p, endpoint, {e: s for e, s in seeds.items() if e != p},
                                bits=32, binary_bits=32))

    def worker(p):
        try:
            roles[p].input(0, torch.ones(8, dtype=torch.int64, device='cuda') if p == 0 else None, (8,))
            return None
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(4) as pool:
        errors = list(pool.map(worker, range(4)))
    assert all('hash mismatch' in errors[p] for p in (1, 3))
