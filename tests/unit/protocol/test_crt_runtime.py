"""Unit, oracle, phase-graph, and active-fault tests for GPU CRT polynomial MPC."""
import concurrent.futures as ft
import math
import threading

import pytest
import torch
import torch.nn.functional as F

from rep4gpu.protocol.crt_runtime import CrtPhaseGraphParty
from rep4gpu.transports.device import DeviceIdealEndpoint
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.transports.byte import IdealNetwork


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _run_four(worker_fn, primes=(251, 241, 239, 233), fault=None):
    seeds = make_seeds()
    net = IdealNetwork(timeout=60)

    def run_one(p):
        endpoint = DeviceIdealEndpoint(net.endpoint(p), 'cuda:0')
        if fault is not None:
            orig_ex = endpoint.exchange_device
            orig_into = endpoint.exchange_device_into
            round_box = [0]

            def wrap(outgoing, incoming_sizes, buffers=None):
                r = round_box[0]
                round_box[0] += 1
                mod_out = fault(p, r, outgoing)
                if buffers is None:
                    return orig_ex(mod_out, incoming_sizes)
                return orig_into(mod_out, incoming_sizes, buffers)

            endpoint.exchange_device = lambda out, sizes: wrap(out, sizes)
            endpoint.exchange_device_into = lambda out, sizes, bufs: wrap(out, sizes, bufs)
        rt = CrtPhaseGraphParty(p, endpoint, {e: seeds[e] for e in range(4) if e != p},
                                device='cuda:0', primes=primes)
        try:
            return worker_fn(rt)
        except BaseException as err:
            net.abort(err)
            raise

    with ft.ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(run_one, p) for p in range(4)]
        return [f.result() for f in futs]


def _oracle_text_cnn(x, w1, b1, w2, b2):
    # x: (B, L, D), w1: (k, D, C), b1: (C,), w2: (C, 1), b2: (1,)
    xt = x.permute(0, 2, 1).to(torch.float64)
    wt = w1.permute(2, 1, 0).to(torch.float64)
    h = F.conv1d(xt, wt, b1.to(torch.float64)).round().to(torch.int64).permute(0, 2, 1)
    p = (h * h).sum(dim=1)
    score = p @ w2.to(torch.int64) + b2.to(torch.int64)
    return h, p, score, (score >= 0).to(torch.int64)


@pytest.mark.parametrize('primes', [(251, 241), (251, 241, 239), (251, 241, 239, 233)])
def test_crt_text_cnn_exact_oracle(primes):
    torch.manual_seed(123)
    B, L, D, k, C = 17, 13, 19, 5, 11
    x = torch.randint(-1, 2, (B, L, D), dtype=torch.int64)
    w1 = torch.randint(-2, 3, (k, D, C), dtype=torch.int64)
    b1 = torch.randint(-3, 4, (C,), dtype=torch.int64)
    w2 = torch.randint(-2, 3, (C, 1), dtype=torch.int64)
    b2 = torch.randint(-50, 51, (1,), dtype=torch.int64)
    if len(primes) == 2:
        # Keep scores within signed 2-prime bound (+-30,245)
        w1 = torch.randint(-1, 2, (k, D, C), dtype=torch.int64)
        b1 = torch.zeros((C,), dtype=torch.int64)
        w2 = torch.randint(-1, 2, (C, 1), dtype=torch.int64)
        b2 = torch.tensor([-500], dtype=torch.int64)
    h_ref, p_ref, sc_ref, lbl_ref = _oracle_text_cnn(x, w1, b1, w2, b2)
    M = math.prod(primes)
    assert int(sc_ref.abs().max()) < M // 2

    def worker(rt):
        xs = rt.crt_input(0, x if rt.party == 0 else None, x.shape)
        w1s = rt.crt_prepare_weight(rt.crt_input(1, w1 if rt.party == 1 else None, w1.shape))
        b1s = rt.crt_input(1, b1 if rt.party == 1 else None, b1.shape)
        w2s = rt.crt_prepare_weight(rt.crt_input(1, w2 if rt.party == 1 else None, w2.shape))
        b2s = rt.crt_input(1, b2 if rt.party == 1 else None, b2.shape)
        h = rt.crt_conv1d(xs, w1s, b1s)
        p = rt.crt_square_sum_pool(h)
        sc = rt.crt_matmul(p, w2s, b2s)
        lbl = rt.open(rt.crt_nonnegative_label(sc))
        return rt.crt_open(h).cpu(), rt.crt_open(p).cpu(), rt.crt_open(sc).cpu(), lbl.cpu()

    results = _run_four(worker, primes=primes)
    for party, (h_got, p_got, sc_got, lbl_got) in enumerate(results):
        assert torch.equal(h_got, h_ref), f'party {party} conv mismatch'
        assert torch.equal(p_got, p_ref), f'party {party} pool mismatch'
        assert torch.equal(sc_got, sc_ref), f'party {party} score mismatch'
        assert torch.equal(lbl_got, lbl_ref), f'party {party} label mismatch'


def test_crt_phase_graph_capture_and_fresh_inputs():
    torch.manual_seed(7)
    B, L, D, k, C = 32, 16, 64, 5, 32
    x_a = torch.randint(-1, 2, (B, L, D), dtype=torch.int64)
    x_b = torch.randint(-1, 2, (B, L, D), dtype=torch.int64)
    w1 = torch.randint(-3, 4, (k, D, C), dtype=torch.int64)
    b1 = torch.randint(-5, 6, (C,), dtype=torch.int64)
    w2 = torch.randint(-3, 4, (C, 1), dtype=torch.int64)
    b2 = torch.tensor([-20000], dtype=torch.int64)
    _, _, _, lbl_a = _oracle_text_cnn(x_a, w1, b1, w2, b2)
    _, _, _, lbl_b = _oracle_text_cnn(x_b, w1, b1, w2, b2)

    prep_lock = threading.Lock()

    def worker(rt):
        w1s = rt.crt_prepare_weight(rt.crt_input(1, w1 if rt.party == 1 else None, w1.shape))
        b1s = rt.crt_input(1, b1 if rt.party == 1 else None, b1.shape)
        w2s = rt.crt_prepare_weight(rt.crt_input(1, w2 if rt.party == 1 else None, w2.shape))
        b2s = rt.crt_input(1, b2 if rt.party == 1 else None, b2.shape)
        params = (w1s, b1s, w2s, b2s)

        def forward(xs, prm):
            h = rt.crt_conv1d(xs, prm[0])
            p = rt.crt_square_sum_pool(h, bias=prm[1])
            sc = rt.crt_matmul(p, prm[2], prm[3])
            return rt.open(rt.crt_nonnegative_label(sc))

        xa_s = rt.crt_input(0, x_a if rt.party == 0 else None, x_a.shape)
        with prep_lock:
            plan = rt.prepare_inference(forward, xa_s, params, check_at_send=True)
        counters_before = dict(rt.counters)
        out_a = plan.run(xa_s, params).cpu()
        counters_mid = dict(rt.counters)
        xb_s = rt.crt_input(0, x_b if rt.party == 0 else None, x_b.shape)
        out_b = plan.run(xb_s, params).cpu()
        counters_after = dict(rt.counters)
        assert all(counters_mid[e] > counters_before[e] for e in counters_before)
        assert all(counters_after[e] > counters_mid[e] for e in counters_mid)
        return out_a, out_b

    results = _run_four(worker)
    for out_a, out_b in results:
        assert torch.equal(out_a, lbl_a)
        assert torch.equal(out_b, lbl_b)


def test_crt_private_sign_boundaries():
    primes = (251, 241, 239, 233)
    M = math.prod(primes)
    vals = torch.tensor(
        [0, 1, -1, 2, -2, M // 2, -(M // 2), M // 2 - 1, -(M // 2) + 1, 1234567, -1234567],
        dtype=torch.int64,
    ).reshape(-1, 1)
    expected = (vals >= 0).to(torch.int64)

    def worker(rt):
        s = rt.crt_input(0, vals if rt.party == 0 else None, vals.shape)
        gpu_lbl = rt.open(rt.crt_nonnegative_label(s)).cpu()
        cpu_lbl = rt.crt_nonnegative_label_open_cpu(s).cpu()
        return gpu_lbl, cpu_lbl

    for gpu_lbl, cpu_lbl in _run_four(worker, primes=primes):
        assert torch.equal(gpu_lbl, expected)
        assert torch.equal(cpu_lbl, expected)


@pytest.mark.parametrize('corrupt_mode', [
    'y0_bad', 'y1_bad', 'y2_bad', 'y3_bad',
    'p0_equivocate', 'p3_equivocate',
    'p1_bad_c', 'p2_bad_dc', 'p1_bad_dall_p0',
    'input_equivocate',
])
def test_crt_active_malicious_faults_abort(corrupt_mode):
    torch.manual_seed(99)
    B, L, D, k, C = 4, 8, 8, 3, 4
    x = torch.randint(-1, 2, (B, L, D), dtype=torch.int64)
    w1 = torch.randint(-2, 3, (k, D, C), dtype=torch.int64)

    def flip_byte(t, idx=0):
        c = t.clone()
        c[idx] = (c[idx].to(torch.int32) + 1).to(torch.uint8)
        return c

    # Input rounds: x uses rounds 0..1, w1 uses rounds 2..3; conv1d poly uses rounds 4 (R1) & 5 (R2).
    def fault(party, r, outgoing):
        out = dict(outgoing)
        if corrupt_mode == 'input_equivocate' and party == 0 and r == 0:
            out[1] = flip_byte(out[1])
        elif corrupt_mode == 'y0_bad' and party == 0 and r == 4:
            out[1] = flip_byte(out[1])
            out[2] = flip_byte(out[2])
        elif corrupt_mode == 'y1_bad' and party == 1 and r == 4:
            out[2] = flip_byte(out[2])
        elif corrupt_mode == 'y2_bad' and party == 2 and r == 4:
            out[1] = flip_byte(out[1])
        elif corrupt_mode == 'y3_bad' and party == 3 and r == 4:
            out[1] = flip_byte(out[1])
            out[2] = flip_byte(out[2])
        elif corrupt_mode == 'p0_equivocate' and party == 0 and r == 4:
            out[1] = flip_byte(out[1])
        elif corrupt_mode == 'p3_equivocate' and party == 3 and r == 4:
            out[2] = flip_byte(out[2])
        elif corrupt_mode == 'p1_bad_c' and party == 1 and r == 5:
            out[3] = flip_byte(out[3], 0)
        elif corrupt_mode == 'p2_bad_dc' and party == 2 and r == 5:
            out[3] = flip_byte(out[3], 0)
        elif corrupt_mode == 'p1_bad_dall_p0' and party == 1 and r == 5:
            out[0] = flip_byte(out[0], 0)
        return out

    def worker(rt):
        xs = rt.crt_input(0, x if rt.party == 0 else None, x.shape)
        w1s = rt.crt_prepare_weight(rt.crt_input(1, w1 if rt.party == 1 else None, w1.shape))
        h = rt.crt_conv1d(xs, w1s)
        return rt.crt_open(h).cpu()

    with pytest.raises((ValueError, ConnectionError)):
        _run_four(worker, fault=fault)


def test_crt_blake3_keyed_xof_matches_reference():
    """Verify GPU BLAKE3-256 keyed XOF + base-p extraction matches pure-Python reference."""
    import struct
    from rep4gpu.gpu.matmul.crt_conv import crt_input_masks

    iv = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
          0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
    perm = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

    def rotr(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def g(s, a, b, c, d, mx, my):
        s[a] = (s[a] + s[b] + mx) & 0xFFFFFFFF
        s[d] = rotr(s[d] ^ s[a], 16)
        s[c] = (s[c] + s[d]) & 0xFFFFFFFF
        s[b] = rotr(s[b] ^ s[c], 12)
        s[a] = (s[a] + s[b] + my) & 0xFFFFFFFF
        s[d] = rotr(s[d] ^ s[a], 8)
        s[c] = (s[c] + s[d]) & 0xFFFFFFFF
        s[b] = rotr(s[b] ^ s[c], 7)

    def ref_b3_xof_block(key_words, nonce, counter):
        m = [nonce & 0xFFFFFFFF, (nonce >> 32) & 0xFFFFFFFF] + [0] * 14
        s = list(key_words) + iv[:4] + [counter & 0xFFFFFFFF, (counter >> 32) & 0xFFFFFFFF, 8, 27]
        for _ in range(7):
            for i, (a, b, c, d) in enumerate(((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
                                              (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))):
                g(s, a, b, c, d, m[2 * i], m[2 * i + 1])
            m = [m[j] for j in perm]
        return [s[i] ^ s[i + 8] for i in range(8)] + [s[i + 8] ^ key_words[i] for i in range(8)]

    key_bytes = bytes(range(32))
    key_words = list(struct.unpack('<8I', key_bytes))
    key_t = torch.tensor(key_words, dtype=torch.uint32, device='cuda:0')
    nonce = 0x123456789ABCDEF0
    nonces = torch.tensor([nonce, 0], dtype=torch.uint64, device='cuda:0')
    primes = (251, 241, 239, 233)
    for n in (37, 64):
        out0, _ = crt_input_masks((key_t, None), nonces, primes, (n,), 'cuda:0')
        got = out0.cpu()
        blocks_per_lane = (n + 31) // 32
        for lane, p in enumerate(primes):
            expected = []
            for blk in range(blocks_per_lane):
                words = ref_b3_xof_block(key_words, nonce, lane * blocks_per_lane + blk)
                for j in range(8):
                    u = words[2 * j] | (words[2 * j + 1] << 32)
                    rem = u % (p ** 4)
                    for _ in range(4):
                        expected.append(rem % p)
                        rem //= p
            assert got[lane].tolist() == expected[:n]


@pytest.mark.parametrize('tamper_call,target_peer', [(4, 3), (10, 1), (10, 3)])
def test_crt_cpu_sign_active_fault_aborts(tamper_call, target_peer):
    """Verify tampering with CPU bit-sliced sign mul or open messages triggers a BLAKE2b abort."""
    primes = (251, 241, 239, 233)
    vals = torch.tensor([[10], [-10], [0], [42]], dtype=torch.int64)
    seeds = make_seeds()
    net = IdealNetwork(timeout=30)
    tampered = [0]

    def run_one(p):
        raw_ep = net.endpoint(p)
        if p == 0:
            orig_ex = raw_ep.exchange
            calls = [0]

            def tamper(outgoing):
                calls[0] += 1
                if calls[0] == tamper_call and target_peer in outgoing:
                    b = bytearray(outgoing[target_peer])
                    b[0] ^= 1
                    tampered[0] += 1
                    outgoing = {**outgoing, target_peer: bytes(b)}
                return orig_ex(outgoing)

            raw_ep.exchange = tamper
        rt = CrtPhaseGraphParty(p, DeviceIdealEndpoint(raw_ep, 'cuda:0'),
                                {e: seeds[e] for e in range(4) if e != p},
                                device='cuda:0', primes=primes)
        try:
            s = rt.crt_input(0, vals if p == 0 else None, vals.shape)
            return rt.crt_nonnegative_label_open_cpu(s)
        except BaseException as err:
            net.abort(err)
            raise

    with pytest.raises((ValueError, ConnectionError)):
        with ft.ThreadPoolExecutor(4) as pool:
            futs = [pool.submit(run_one, p) for p in range(4)]
            [f.result() for f in futs]
    assert tampered[0] == 1
