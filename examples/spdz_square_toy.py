"""Run a bounded integer square-activation network with trusted preprocessing.

Run from the repository root: python -m examples.spdz_square_toy
Four role threads use an ideal test transport; use scripts.run_spdz_distributed
for isolated-role TLS matrix-product measurements. This is an algebra and
protocol smoke test, not a network or LLM performance benchmark.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os

import torch

from rep4gpu.spdz import PRIME, TrustedDealer
from rep4gpu.spdz_distributed import DistributedParty, split_triple
from rep4gpu.spdz_inputs import make_input_masks, private_input
from rep4gpu.spdz_polynomial import make_square_masks, square
from rep4gpu.transport import IdealNetwork


def run():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(123)
    x = torch.randint(-2, 3, (4, 8), generator=generator)
    w1 = torch.randint(-2, 3, (8, 16), generator=generator)
    w2 = torch.randint(-2, 3, (16, 4), generator=generator)
    expected = (x @ w1).square() @ w2
    assert expected.abs().max().item() < PRIME // 2
    dealer = TrustedDealer()
    masks = [make_input_masks(dealer, owner, tuple(value.shape))
             for owner, value in zip([0, 1, 1], [x, w1, w2])]
    first = split_triple(dealer.triple(4, 8, 16))
    second = split_triple(dealer.triple(4, 16, 4))
    squares = make_square_masks(dealer, (4, 16))
    network = IdealNetwork(timeout=30)
    session = os.urandom(32)
    parties = [DistributedParty(p, network.endpoint(p), dealer.alpha_shares[:, p],
                                session=session, _allow_insecure_test_transport=True)
               for p in range(4)]
    def role(p):
        party = parties[p]
        values = [private_input(party, mask[p], value if p == owner else None)
                  for mask, value, owner in zip(masks, [x, w1, w2], [0, 1, 1])]
        hidden = square(party, party.matmul(values[0], values[1], first[p]), squares[p])
        result = party.open(party.matmul(hidden, values[2], second[p]))
        return torch.where(result > PRIME // 2, result - PRIME, result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(role, range(4)))
    passed = all(torch.equal(value, expected) for value in outputs)
    return dict(passed=passed, architecture='8 -> 16 -> square -> 4', batch=4,
                field_prime=PRIME, mac_keys=5, preprocessing='trusted offline dealer',
                transport='ideal in-process test transport', output=outputs[0].tolist(),
                expected=expected.tolist(), global_payload_bytes=sum(p.transport.bytes_sent for p in parties))


if __name__ == '__main__':
    report = run()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
