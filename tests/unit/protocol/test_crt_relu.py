"""Unit and 4-party protocol tests for active-secure CRT trunc_relu."""
from concurrent.futures import ThreadPoolExecutor
import math

import pytest
import torch

from rep4gpu.gpu.bits.crt_relu_gpu import crt_elem_mul, crt_joint_input_pair, crt_trunc_relu
from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork


PRIMES_3 = (251, 241, 239)


def _run_4party(fn, primes=PRIMES_3):
    net = IdealNetwork()
    seeds = make_seeds()
    parties = [
        CrtPhaseGraphParty(
            p, DeviceIdealEndpoint(net.endpoint(p), 'cuda'),
            {e: seeds[e] for e in range(4) if e != p},
            device='cuda', primes=primes, optimized=True, gpu_open=True,
        )
        for p in range(4)
    ]
    with ThreadPoolExecutor(4) as pool:
        def guarded(rt):
            try:
                return fn(rt)
            except BaseException as err:
                net.abort(err)
                raise
        return list(pool.map(guarded, parties))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_crt_joint_input_and_elem_mul_exact():
    g = torch.Generator().manual_seed(123)
    m = math.prod(PRIMES_3)
    a = torch.randint(-m // 4, m // 4, (64, 33), generator=g, dtype=torch.int64)
    b = torch.randint(-100, 100, (64, 33), generator=g, dtype=torch.int64)

    def body(rt):
        sa = rt.crt_input(0, a if rt.party == 0 else None, a.shape)
        sb = rt.crt_input(1, b if rt.party == 1 else None, b.shape)
        sc = crt_elem_mul(rt, sa, sb)
        return rt.crt_open(sc).cpu()

    outs = _run_4party(body)
    expected = (a * b) % m
    expected = torch.where(expected > m // 2, expected - m, expected)
    for out in outs:
        assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('primes,shift', [
    ((251, 241, 239), 9),   # general CSA path (q_lo_mask != 0)
    ((241, 229, 19), 8),    # zero-CSA path (M = 2^20 + 15, q_M = 2^12, q_mask == 0)
    ((233, 211), 6),        # 2-lane zero-CSA top-bit path (M = 49163, q_M = 768 = 2^9 + 2^8)
])
def test_crt_trunc_relu_eager_and_phase_graph(primes, shift):
    m = math.prod(primes)
    scale = 1 << shift
    off_q = (m // 4) // scale
    limit = off_q * scale - 1

    g = torch.Generator().manual_seed(7)
    rand = torch.randint(-limit, limit + 1, (32, 64), generator=g, dtype=torch.int64)
    edges = torch.tensor(
        [0, 1, -1, scale - 1, -(scale - 1), scale, -scale, scale + 1, -(scale + 1),
         -2 * scale, 2 * scale, limit, -limit, limit - 5, -(limit - 5)],
        dtype=torch.int64,
    )
    rand.view(-1)[:len(edges)] = edges

    def body_eager(rt):
        sx = rt.crt_input(0, rand if rt.party == 0 else None, rand.shape)
        sy = crt_trunc_relu(rt, sx, shift=shift)
        return rt.crt_open(sy).cpu()

    outs_eager = _run_4party(body_eager, primes=primes)
    for out in outs_eager[1:]:
        assert torch.equal(out, outs_eager[0])

    # Check every output lies strictly in [max(floor(x/scale) - 1, 0), max(floor(x/scale) + 1, 0)].
    fl = torch.div(rand, scale, rounding_mode='floor')
    lo = torch.clamp(fl - 1, min=0)
    hi = torch.clamp(fl + 1, min=0)
    assert bool(((outs_eager[0] >= lo) & (outs_eager[0] <= hi)).all())
    assert bool((outs_eager[0][rand <= -scale] == 0).all())

    # Now test inside PhaseGraphParty prepare_inference (captured CUDA graphs with fresh nonces)!
    net = IdealNetwork()
    seeds = make_seeds()
    parties = [
        CrtPhaseGraphParty(
            p, DeviceIdealEndpoint(net.endpoint(p), 'cuda'),
            {e: seeds[e] for e in range(4) if e != p},
            device='cuda', primes=primes, optimized=True, gpu_open=True,
        )
        for p in range(4)
    ]
    with ThreadPoolExecutor(4) as pool:
        sx_list = list(pool.map(
            lambda rt: rt.crt_input(0, rand if rt.party == 0 else None, rand.shape), parties,
        ))
        plans = [
            rt.prepare_inference(lambda s, r=rt: crt_trunc_relu(r, s, shift=shift), sx_list[rt.party], check_at_send=True)
            for rt in parties
        ]
        outs_graph = list(pool.map(
            lambda rt: rt.crt_open(plans[rt.party].run(sx_list[rt.party])).cpu(), parties,
        ))
    for out in outs_graph[1:]:
        assert torch.equal(out, outs_graph[0])
    assert bool(((outs_graph[0] >= lo) & (outs_graph[0] <= hi)).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('corrupt_round,corrupt_sender', [
    (0, 0),   # Round 1 masked m = (x23 + r) mod M from P0
    (1, 0),   # Bit-sliced joint input from P0
    (2, 0),   # Complementary uint32 binary AND from P0
    (7, 0),   # CRT joint input pair from P0
    (8, 0),   # 4-joint complementary CRT product (W, SUV) from P0
    (9, 0),   # Final CRT polynomial exchange Y * KEEP from P0
])
def test_crt_trunc_relu_active_faults_abort(corrupt_round, corrupt_sender):
    """Flipping one byte in any crt_trunc_relu round must trigger a BLAKE3 check failure."""
    primes = (241, 229, 19)
    shift = 8
    m = math.prod(primes)
    limit = ((m // 4) // (1 << shift)) * (1 << shift) - 1
    g = torch.Generator().manual_seed(99)
    rand = torch.randint(-limit, limit + 1, (16, 64), generator=g, dtype=torch.int64)

    class _FaultEndpoint(DeviceIdealEndpoint):
        def __init__(self, ep, device, party):
            super().__init__(ep, device)
            self._party = party
            self._r = 0
            self._armed = False

        def exchange_device(self, outgoing, incoming_sizes):
            if self._armed:
                if self._party == corrupt_sender and self._r == corrupt_round and outgoing:
                    out_mod = {}
                    for peer, buf in outgoing.items():
                        c = buf.clone()
                        if c.numel() > 0:
                            c[0] ^= 1
                        out_mod[peer] = c
                    outgoing = out_mod
                self._r += 1
            return super().exchange_device(outgoing, incoming_sizes)

    net = IdealNetwork()
    seeds = make_seeds()
    eps = [_FaultEndpoint(net.endpoint(p), 'cuda', p) for p in range(4)]
    parties = [
        CrtPhaseGraphParty(
            p, eps[p], {e: seeds[e] for e in range(4) if e != p},
            device='cuda', primes=primes, optimized=True, gpu_open=True,
        )
        for p in range(4)
    ]

    def body(rt):
        sx = rt.crt_input(0, rand if rt.party == 0 else None, rand.shape)
        eps[rt.party]._armed = True
        sy = crt_trunc_relu(rt, sx, shift=shift)
        return rt.crt_open(sy)

    with pytest.raises((ValueError, Exception)):
        with ThreadPoolExecutor(4) as pool:
            def guarded(rt):
                try:
                    return body(rt)
                except BaseException as err:
                    net.abort(err)
                    raise
            list(pool.map(guarded, parties))
    assert isinstance(net.failure, ValueError)
    msg = str(net.failure).lower()
    assert 'mismatch' in msg or 'check failed' in msg
