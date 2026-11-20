"""Active-secure GPU CRT probabilistic truncation + exact ReLU (3- or 4-lane CRT).

Given a CRT share of x in (-M/4, +M/4) mod M = prod_l p_l (odd 24..32-bit M),
computes ReLU(trunc_pr(x, shift)) directly back into CRT shares using:
  1. 1-round CRT masking m = (x23 + r) mod M (checked via BLAKE3 tag from P1)
     yielding exact wrap identity w = 1[r > M//2] * 1[c <= M//2].
  2. Compact bit-sliced uint32 GPU binary Rep4 CSA + carry prefix tree for the
     sign bit of y = c_q - r_q + w * q_M in W-bit signed two's complement.
  3. 1-round checked 2-sender CRT joint input + batched 1-product CRT gate for
     (w, s_u * s_v) + fused register-level Y * (1 - s) CRT gate.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from ..crypto.gpu_hash import digest, digest_batch
from ..matmul.crt_conv import (
    _fast_mod_u32,
    _lane_barrett_u32,
    _pad_primes,
    _party_coeffs,
    crt_comp_mul_prep,
    crt_input_masks,
    crt_joint_input_sub,
)
from ...protocol.runtime import SCHEDULE, Share

# Complementary 4-joint schedule for a supported on {2,3} and b supported on {0,1}:
# (s, b, r, o) -> (2,1,0,3): a3&b0; (3,0,1,2): a2&b1; (1,3,2,0): a2&b0; (0,2,3,1): a3&b1.
_COMP_BIN_SCHEDULE = ((2, 1, 0, 3), (3, 0, 1, 2), (1, 3, 2, 0), (0, 2, 3, 1))


def _crt_tr_domain(kind: bytes, primes: tuple[int, ...], excluded: int) -> bytes:
    return kind + bytes((excluded, len(primes), *primes))


@triton.jit
def _crt_round1_p01_warp_kernel(
    X2, X3, R2, R3, M_OUT, U_PLANES, CRT_01,
    N: tl.constexpr, N_WORDS: tl.constexpr,
    SHIFT: tl.constexpr, W: tl.constexpr,
    M_MOD: tl.constexpr, HALF_M: tl.constexpr,
    N_LANES: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    w_idx = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    bit = tl.arange(0, 32)
    idx = w_idx[:, None] * 32 + bit[None, :]
    valid = idx < N
    valid_w = w_idx < N_WORDS

    r_sum = tl.zeros((BLOCK_W, 32), dtype=tl.int64)
    m_sum = tl.zeros((BLOCK_W, 32), dtype=tl.int64)
    for lane in tl.static_range(N_LANES):
        p = P0 if lane == 0 else (P1 if lane == 1 else (P2 if lane == 2 else P3))
        coeff = C0 if lane == 0 else (C1 if lane == 1 else (C2 if lane == 2 else C3))
        off = lane * N + idx
        x2v = tl.load(X2 + off, valid, 0).to(tl.int32)
        x3v = tl.load(X3 + off, valid, 0).to(tl.int32)
        r2v = tl.load(R2 + off, valid, 0).to(tl.int32)
        r3v = tl.load(R3 + off, valid, 0).to(tl.int32)
        rl = r2v + r3v
        rl = tl.where(rl >= p, rl - p, rl)
        ml = x2v + x3v + rl
        ml = tl.where(ml >= p, ml - p, ml)
        ml = tl.where(ml >= p, ml - p, ml)
        r_sum += rl.to(tl.int64) * coeff
        m_sum += ml.to(tl.int64) * coeff

    r_val = (r_sum % M_MOD).to(tl.uint32)
    m_val = (m_sum % M_MOD).to(tl.uint32)
    if M_MOD < 65536:
        tl.store(M_OUT + idx, m_val.to(tl.uint16), valid)
    else:
        tl.store(M_OUT + idx, m_val, valid)

    r_hi = tl.where(valid & (r_val > HALF_M), 1, 0).to(tl.uint32)
    r_q_i32 = (r_val >> SHIFT).to(tl.int32)
    u_val = (0 - r_q_i32.to(tl.uint32)) & ((1 << W) - 1)
    u_val = tl.where(valid, u_val, 0).to(tl.uint32)

    # Bit-pack (r_hi, u_val[0..W-1]) into (1+W, N_WORDS) uint32 via axis=1 reduction.
    shift_b = bit[None, :].to(tl.uint32)
    tl.store(U_PLANES + w_idx, tl.sum(r_hi << shift_b, axis=1), valid_w)
    for k in tl.static_range(W):
        bk = (u_val >> k) & 1
        tl.store(U_PLANES + (k + 1) * N_WORDS + w_idx, tl.sum(bk << shift_b, axis=1), valid_w)

    # Store CRT residues of r_q (slice 0) and r_hi (slice 1) into CRT_01: (N_LANES, 3, N).
    for lane in tl.static_range(N_LANES):
        p = P0 if lane == 0 else (P1 if lane == 1 else (P2 if lane == 2 else P3))
        base = lane * (3 * N)
        tl.store(CRT_01 + base + idx, (r_q_i32 % p).to(tl.uint8), valid)
        tl.store(CRT_01 + base + N + idx, r_hi.to(tl.uint8), valid)


@triton.jit
def _crt_round1_p23_warp_kernel(
    M_IN, X0, X1, V_PLANES, CRT_23,
    N: tl.constexpr, N_WORDS: tl.constexpr,
    SHIFT: tl.constexpr, W: tl.constexpr,
    M_MOD: tl.constexpr, HALF_M: tl.constexpr,
    OFF_M: tl.constexpr, OFF_Q: tl.constexpr,
    N_LANES: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    w_idx = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    bit = tl.arange(0, 32)
    idx = w_idx[:, None] * 32 + bit[None, :]
    valid = idx < N
    valid_w = w_idx < N_WORDS

    x01_sum = tl.zeros((BLOCK_W, 32), dtype=tl.int64)
    for lane in tl.static_range(N_LANES):
        p = P0 if lane == 0 else (P1 if lane == 1 else (P2 if lane == 2 else P3))
        coeff = C0 if lane == 0 else (C1 if lane == 1 else (C2 if lane == 2 else C3))
        off = lane * N + idx
        x0v = tl.load(X0 + off, valid, 0).to(tl.int32)
        x1v = tl.load(X1 + off, valid, 0).to(tl.int32)
        xl = x0v + x1v
        xl = tl.where(xl >= p, xl - p, xl)
        x01_sum += xl.to(tl.int64) * coeff

    x01 = x01_sum % M_MOD
    m_val = tl.load(M_IN + idx, valid, 0).to(tl.int64)
    c_val = ((m_val + x01 + OFF_M) % M_MOD).to(tl.uint32)

    c_lo = tl.where(valid & (c_val <= HALF_M), 1, 0).to(tl.uint32)
    c_q = (c_val >> SHIFT).to(tl.int32) - OFF_Q
    v_val = c_q.to(tl.uint32) & ((1 << W) - 1)
    v_val = tl.where(valid, v_val, 0).to(tl.uint32)

    shift_b = bit[None, :].to(tl.uint32)
    tl.store(V_PLANES + w_idx, tl.sum(c_lo << shift_b, axis=1), valid_w)
    for k in tl.static_range(W):
        bk = (v_val >> k) & 1
        tl.store(V_PLANES + (k + 1) * N_WORDS + w_idx, tl.sum(bk << shift_b, axis=1), valid_w)

    # Store nonnegative CRT residues of c_q (slice 0) and c_lo (slice 1) into CRT_23: (N_LANES, 3, N).
    for lane in tl.static_range(N_LANES):
        p = P0 if lane == 0 else (P1 if lane == 1 else (P2 if lane == 2 else P3))
        # c_q >= -OFF_Q; add a multiple of p larger than OFF_Q so dividend is nonnegative.
        pad = ((OFF_Q // p) + 2) * p
        base = lane * (3 * N)
        tl.store(CRT_23 + base + idx, ((c_q + pad) % p).to(tl.uint8), valid)
        tl.store(CRT_23 + base + N + idx, c_lo.to(tl.uint8), valid)


@triton.jit
def _unpack_sign_to_crt_kernel(
    S0, S1, CRT_BUF, N: tl.constexpr, N_WORDS: tl.constexpr,
    N_LANES: tl.constexpr, BLOCK_W: tl.constexpr,
):
    w_idx = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    bit = tl.arange(0, 32)
    idx = w_idx[:, None] * 32 + bit[None, :]
    valid = idx < N
    valid_w = w_idx < N_WORDS
    word = tl.load(S0 + w_idx, valid_w, 0) ^ tl.load(S1 + w_idx, valid_w, 0)
    b = ((word[:, None] >> bit[None, :].to(tl.uint32)) & 1).to(tl.uint8)
    for lane in tl.static_range(N_LANES):
        tl.store(CRT_BUF + lane * (3 * N) + 2 * N + idx, b, valid)


@triton.jit
def _bin_and_prepare_kernel(
    A0, A1, A2, B0, B1, B2, MASKS, OUT_S, WIRES,
    N: tl.constexpr,
    PARTY: tl.constexpr,
    COMPLEMENTARY: tl.constexpr,
    S0_IS_ME: tl.constexpr, B0_IS_ME: tl.constexpr, R0_IS_ME: tl.constexpr, O0_SLOT: tl.constexpr, R0_SLOT: tl.constexpr,
    S1_IS_ME: tl.constexpr, B1_IS_ME: tl.constexpr, R1_IS_ME: tl.constexpr, O1_SLOT: tl.constexpr, R1_SLOT: tl.constexpr,
    S2_IS_ME: tl.constexpr, B2_IS_ME: tl.constexpr, R2_IS_ME: tl.constexpr, O2_SLOT: tl.constexpr, R2_SLOT: tl.constexpr,
    S3_IS_ME: tl.constexpr, B3_IS_ME: tl.constexpr, R3_IS_ME: tl.constexpr, O3_SLOT: tl.constexpr, R3_SLOT: tl.constexpr,
    S4_IS_ME: tl.constexpr, B4_IS_ME: tl.constexpr, R4_IS_ME: tl.constexpr, O4_SLOT: tl.constexpr, R4_SLOT: tl.constexpr,
    S5_IS_ME: tl.constexpr, B5_IS_ME: tl.constexpr, R5_IS_ME: tl.constexpr, O5_SLOT: tl.constexpr, R5_SLOT: tl.constexpr,
    W0_IDX: tl.constexpr, W1_IDX: tl.constexpr, W2_IDX: tl.constexpr,
    W3_IDX: tl.constexpr, W4_IDX: tl.constexpr, W5_IDX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    a0 = tl.load(A0 + i, valid, 0)
    a1 = tl.load(A1 + i, valid, 0)
    a2 = tl.load(A2 + i, valid, 0)
    b0 = tl.load(B0 + i, valid, 0)
    b1 = tl.load(B1 + i, valid, 0)
    b2 = tl.load(B2 + i, valid, 0)

    if COMPLEMENTARY:
        # Terms for _COMP_BIN_SCHEDULE = ((2,1,0,3):a3&b0, (3,0,1,2):a2&b1, (1,3,2,0):a2&b0, (0,2,3,1):a3&b1)
        # Party 0 holds (a1,a2,a3) & (b1,b2,b3): a2=a1, a3=a2, b1=b0
        # Party 1 holds (a2,a3,a0) & (b2,b3,b0): a2=a0, a3=a1, b0=b2
        # Party 2 holds (a3,a0,a1) & (b3,b0,b1): a3=a0, b0=b1, b1=b2
        # Party 3 holds (a0,a1,a2) & (b0,b1,b2): a2=a2, b0=b0, b1=b1
        if PARTY == 0:
            t0 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t1 = a1 & b0
            t2 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t3 = a2 & b0
        elif PARTY == 1:
            t0 = a1 & b2
            t1 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t2 = a0 & b2
            t3 = tl.zeros((BLOCK,), dtype=tl.uint32)
        elif PARTY == 2:
            t0 = a0 & b1
            t1 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t2 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t3 = a0 & b2
        else:
            t0 = tl.zeros((BLOCK,), dtype=tl.uint32)
            t1 = a2 & b1
            t2 = a2 & b0
            t3 = tl.zeros((BLOCK,), dtype=tl.uint32)
        t4 = tl.zeros((BLOCK,), dtype=tl.uint32)
        t5 = tl.zeros((BLOCK,), dtype=tl.uint32)
    else:
        c_prev = ((a0 ^ a1) & b0) ^ (a0 & b1)
        c_curr = ((a1 ^ a2) & b1) ^ (a1 & b2)
        c_diag = (a0 & b2) ^ (a2 & b0)
        z = tl.zeros((BLOCK,), dtype=tl.uint32)
        if PARTY == 0:
            t0, t1, t2, t3, t4, t5 = c_curr, z, z, c_prev, c_diag, c_diag
        elif PARTY == 1:
            t0, t1, t2, t3, t4, t5 = c_prev, c_curr, z, z, c_diag, c_diag
        elif PARTY == 2:
            t0, t1, t2, t3, t4, t5 = z, c_prev, c_curr, z, c_diag, c_diag
        else:
            t0, t1, t2, t3, t4, t5 = z, z, c_prev, c_curr, c_diag, c_diag

    o0 = tl.zeros((BLOCK,), dtype=tl.uint32)
    o1 = tl.zeros((BLOCK,), dtype=tl.uint32)
    o2 = tl.zeros((BLOCK,), dtype=tl.uint32)

    N_SCHED: tl.constexpr = 4 if COMPLEMENTARY else 6
    for k in tl.static_range(N_SCHED):
        s_me = (S0_IS_ME if k == 0 else S1_IS_ME if k == 1 else S2_IS_ME if k == 2
                else S3_IS_ME if k == 3 else S4_IS_ME if k == 4 else S5_IS_ME)
        b_me = (B0_IS_ME if k == 0 else B1_IS_ME if k == 1 else B2_IS_ME if k == 2
                else B3_IS_ME if k == 3 else B4_IS_ME if k == 4 else B5_IS_ME)
        r_me = (R0_IS_ME if k == 0 else R1_IS_ME if k == 1 else R2_IS_ME if k == 2
                else R3_IS_ME if k == 3 else R4_IS_ME if k == 4 else R5_IS_ME)
        o_sl = (O0_SLOT if k == 0 else O1_SLOT if k == 1 else O2_SLOT if k == 2
                else O3_SLOT if k == 3 else O4_SLOT if k == 4 else O5_SLOT)
        r_sl = (R0_SLOT if k == 0 else R1_SLOT if k == 1 else R2_SLOT if k == 2
                else R3_SLOT if k == 3 else R4_SLOT if k == 4 else R5_SLOT)
        w_id = (W0_IDX if k == 0 else W1_IDX if k == 1 else W2_IDX if k == 2
                else W3_IDX if k == 3 else W4_IDX if k == 4 else W5_IDX)
        if r_me == 0:
            mk = tl.load(MASKS + k * N + i, valid, 0)
            if r_sl == 0:
                o0 ^= mk
            elif r_sl == 1:
                o1 ^= mk
            else:
                o2 ^= mk
            if (s_me + b_me) != 0:
                tk = t0 if k == 0 else t1 if k == 1 else t2 if k == 2 else t3 if k == 3 else t4 if k == 4 else t5
                masked = tk ^ mk
                if o_sl == 0:
                    o0 ^= masked
                elif o_sl == 1:
                    o1 ^= masked
                else:
                    o2 ^= masked
                tl.store(WIRES + w_id * N + i, masked, valid)

    tl.store(OUT_S + i, o0, valid)
    tl.store(OUT_S + N + i, o1, valid)
    tl.store(OUT_S + 2 * N + i, o2, valid)


@triton.jit
def _crt_elem_mul_eval_kernel(
    A0, A1, A2, B0, B1, B2, OUT, N: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    shift = p * 16
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    off = lane * N + idx
    fa = _fast_mod_u32(
        (C0 * tl.load(A0 + off, valid, 0).to(tl.int32)
         + C1 * tl.load(A1 + off, valid, 0).to(tl.int32)
         + C2 * tl.load(A2 + off, valid, 0).to(tl.int32) + shift).to(tl.uint32),
        p_u32, m_u32,
    )
    fb = _fast_mod_u32(
        (C0 * tl.load(B0 + off, valid, 0).to(tl.int32)
         + C1 * tl.load(B1 + off, valid, 0).to(tl.int32)
         + C2 * tl.load(B2 + off, valid, 0).to(tl.int32) + shift).to(tl.uint32),
        p_u32, m_u32,
    )
    tl.store(OUT + off, _fast_mod_u32(fa * fb, p_u32, m_u32).to(tl.uint8), valid)


@triton.jit
def _crt_fused_y_keep_mul_kernel(
    T0, T1, T2, P_W0, P_W1, P_W2, W_RECV, OUT,
    N: tl.constexpr, Q_M: tl.constexpr,
    PARTY: tl.constexpr,
    Y_C0: tl.constexpr, Y_C1: tl.constexpr, Y_C2: tl.constexpr,
    K_C0: tl.constexpr, K_C1: tl.constexpr, K_C2: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C_RECV: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused: form Y = C_q - R_q + q_M * W and KEEP = 1 - S_v - S_u + 2 * SUV, eval F_Y * F_KEEP mod p."""
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    q_mod = (Q_M % p)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N

    # T0, T1, T2 are the 3 non-zero (L, 3, N) active tensors across (A, B): slice 0 = R_q/C_q, slice 2 = S_v/S_u.
    # P_W0..2 and W_RECV have shape (L, 2, N): slice 0 = W, slice 1 = SUV.
    base3 = lane * (3 * N)
    base2 = lane * (2 * N)
    off_q = base3 + idx
    off_s = base3 + 2 * N + idx
    off_w = base2 + idx
    off_suv = base2 + N + idx

    t0_q = tl.load(T0 + off_q, valid, 0).to(tl.int32)
    t1_q = tl.load(T1 + off_q, valid, 0).to(tl.int32)
    t2_q = tl.load(T2 + off_q, valid, 0).to(tl.int32)
    t0_s = tl.load(T0 + off_s, valid, 0).to(tl.int32)
    t1_s = tl.load(T1 + off_s, valid, 0).to(tl.int32)
    t2_s = tl.load(T2 + off_s, valid, 0).to(tl.int32)

    w_lin = (
        C0 * tl.load(P_W0 + off_w, valid, 0).to(tl.int32)
        + C1 * tl.load(P_W1 + off_w, valid, 0).to(tl.int32)
        + C2 * tl.load(P_W2 + off_w, valid, 0).to(tl.int32)
        + C_RECV * tl.load(W_RECV + off_w, valid, 0).to(tl.int32)
    )
    suv_lin = (
        C0 * tl.load(P_W0 + off_suv, valid, 0).to(tl.int32)
        + C1 * tl.load(P_W1 + off_suv, valid, 0).to(tl.int32)
        + C2 * tl.load(P_W2 + off_suv, valid, 0).to(tl.int32)
        + C_RECV * tl.load(W_RECV + off_suv, valid, 0).to(tl.int32)
    )

    fy_acc = Y_C0 * t0_q + Y_C1 * t1_q + Y_C2 * t2_q + q_mod * w_lin
    fk_acc = PARTY + K_C0 * t0_s + K_C1 * t1_s + K_C2 * t2_s + 2 * suv_lin

    # Shift into positive range before fast uint32 Barrett reduction (min fy_acc > -9*p^2 - 3*p > -4096*p).
    fy_pos = (fy_acc + p * 4096).to(tl.uint32)
    fk_pos = (fk_acc + p * 64).to(tl.uint32)
    fy = _fast_mod_u32(fy_pos, p_u32, m_u32)
    fk = _fast_mod_u32(fk_pos, p_u32, m_u32)
    tl.store(OUT + lane * N + idx, _fast_mod_u32(fy * fk, p_u32, m_u32).to(tl.uint8), valid)


def _bin_mask_batch_u32(rt, count: int, schedule: tuple) -> torch.Tensor:
    """Draw len(schedule) packed int32 mask rows (length count each) via PhaseGraph/DeviceParty."""
    from ...protocol.device_runtime import mask_domain
    if rt._phase_builder is not None:
        excluded = tuple(None if rt.party == r else r for _, _, r, _ in schedule)
        keys = tuple(
            None if e is None else rt._device_keys[e].key(mask_domain(rt.bits, rt.binary_bits, True, e))
            for e in excluded
        )
        return rt._phase_builder.masks(keys, excluded, (count,), 32, packed=True).view(torch.int32).reshape(-1)
    from ..crypto.gpu_prg_batch import generate_rows
    excluded = tuple(r for _, _, r, _ in schedule if rt.party != r)
    reservations = iter(rt._reserve_nonces(excluded))
    nonces = tuple(None if rt.party == r else next(reservations) for _, _, r, _ in schedule)
    keys = tuple(
        None if rt.party == r else rt._device_keys[r].key(mask_domain(rt.bits, rt.binary_bits, True, r))
        for _, _, r, _ in schedule
    )
    return generate_rows(keys, nonces, count, bits=32, packed=True).view(torch.int32).reshape(-1)


def _bin_joint_pair_u32(rt, u_01: torch.Tensor | None, v_23: torch.Tensor | None,
                        shape: tuple[int, ...]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """1-round checked 2-sender sharing of int32 bit-planes: u_01 from {P0,P1} and v_23 from {P2,P3}."""
    from ...protocol.device_runtime import mask_domain
    count = math.prod(shape)
    n_bytes = count * 4
    p = rt.party
    # Draw masks for excl=2 (known to P0,P1,P3) and excl=0 (known to P1,P2,P3) in one call.
    excl_pair = (None if p == 2 else 2, None if p == 0 else 0)
    if rt._phase_builder is not None:
        keys = tuple(
            None if e is None else rt._device_keys[e].key(mask_domain(rt.bits, rt.binary_bits, True, e))
            for e in excl_pair
        )
        rows = rt._phase_builder.masks(keys, excl_pair, (count,), 32, packed=True).view(torch.int32)
        m2, m0 = rows[0].reshape(shape), rows[1].reshape(shape)
    else:
        m2 = None if p == 2 else rt._random(2, shape, rt.device, True, packed=True).view(torch.int32)
        m0 = None if p == 0 else rt._random(0, shape, rt.device, True, packed=True).view(torch.int32)

    zeros = torch.zeros(shape, dtype=torch.int32, device=rt.device)
    outgoing, sizes = {}, {}
    u_wire = v_wire = None
    if p in (0, 1):
        raw_buf = rt._alloc_wire_buffer(n_bytes) if p == 0 else None
        if raw_buf is not None:
            u_wire = raw_buf.view(torch.int32).reshape(shape)
            torch.bitwise_xor(u_01, m2, out=u_wire)
            u_bytes = raw_buf
        else:
            u_wire = u_01 ^ m2
            u_bytes = u_wire.reshape(-1).view(torch.uint8)
        if p == 0:
            outgoing[2] = u_bytes
            sizes[2] = n_bytes
            sizes[3] = 32
        else:
            outgoing[2] = digest(u_bytes, block_tree=rt.hash_block_tree)
    else:
        raw_buf = rt._alloc_wire_buffer(n_bytes) if p == 2 else None
        if raw_buf is not None:
            v_wire = raw_buf.view(torch.int32).reshape(shape)
            torch.bitwise_xor(v_23, m0, out=v_wire)
            v_bytes = raw_buf
        else:
            v_wire = v_23 ^ m0
            v_bytes = v_wire.reshape(-1).view(torch.uint8)
        if p == 2:
            outgoing[0] = v_bytes
            sizes[0] = n_bytes
            sizes[1] = 32
        else:
            outgoing[0] = digest(v_bytes, block_tree=rt.hash_block_tree)

    incoming = rt._exchange_device(outgoing, sizes)
    if p == 0:
        v_bytes = incoming[2]
        rt._verify_tags(digest(v_bytes, block_tree=rt.hash_block_tree), incoming[3],
                        'CRT bit-sliced joint input BLAKE3 mismatch at party 0')
        v_wire = v_bytes.view(torch.int32).reshape(shape)
    elif p == 2:
        u_bytes = incoming[0]
        rt._verify_tags(digest(u_bytes, block_tree=rt.hash_block_tree), incoming[1],
                        'CRT bit-sliced joint input BLAKE3 mismatch at party 2')
        u_wire = u_bytes.view(torch.int32).reshape(shape)

    if p == 0:
        u_share = (zeros, m2, u_wire)
        v_share = (v_wire, zeros, zeros)
    elif p == 1:
        u_share = (m2, u_wire, zeros)
        v_share = (zeros, zeros, m0)
    elif p == 2:
        u_share = (u_wire, zeros, zeros)
        v_share = (zeros, m0, v_wire)
    else:
        u_share = (zeros, zeros, m2)
        v_share = (m0, v_wire, zeros)
    return u_share, v_share


def _bin_and_u32(rt, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...],
                 *, complementary: bool = False) -> tuple[torch.Tensor, ...]:
    """1-round checked int32 bit-sliced Rep4 binary AND (4-joint complementary or 6-joint general)."""
    shape = a[0].shape
    n = a[0].numel()
    n_bytes = n * 4
    p = rt.party
    schedule = _COMP_BIN_SCHEDULE if complementary else SCHEDULE
    masks = _bin_mask_batch_u32(rt, n, schedule)

    # Identify which schedule entries this party sends or hashes, assigning compact wire indices.
    wire_specs = []
    w_idx = [0] * 6
    for k, (s, bk, r, o) in enumerate(schedule):
        if p in (s, bk):
            w_idx[k] = len(wire_specs)
            wire_specs.append((k, s, bk, r, o))

    n_wires = len(wire_specs)
    wires_u32 = torch.empty((n_wires, n), dtype=torch.int32, device=rt.device)
    out_s = torch.empty((3, n), dtype=torch.int32, device=rt.device)

    pad_sched = list(schedule) + [(0, 0, p, 0)] * (6 - len(schedule))
    flags = []
    for k, (s, bk, r, o) in enumerate(pad_sched):
        flags.extend([
            int(p == s), int(p == bk), int(p == r),
            (o - p - 1) % 4, (r - p - 1) % 4,
        ])
    if n:
        with torch.cuda.device(rt.device):
            _bin_and_prepare_kernel[(triton.cdiv(n, 256),)](
                *(t.contiguous().reshape(-1).view(torch.uint32) for t in a),
                *(t.contiguous().reshape(-1).view(torch.uint32) for t in b),
                masks.view(torch.uint32), out_s.view(torch.uint32), wires_u32.view(torch.uint32),
                n, p, int(complementary),
                *flags, *w_idx, 256, num_warps=4,
            )

    # Stage payloads and BLAKE3 tags per peer.
    payloads, tag_src = {}, {}
    for slot_i, (_, s, bk, r, _) in enumerate(wire_specs):
        w_bytes = wires_u32[slot_i].view(torch.uint8)
        if p == s:
            payloads.setdefault(r, []).append(w_bytes)
        else:
            tag_src.setdefault(r, []).append((s, w_bytes))

    outgoing = {}
    for peer in set(payloads) | set(tag_src):
        chunks = list(payloads.get(peer, ()))
        if peer in tag_src:
            t_list = [wb for _, wb in sorted(tag_src[peer], key=lambda x: x[0])]
            tags = digest_batch(torch.stack(t_list), block_tree=rt.hash_block_tree).reshape(-1)
            chunks.append(tags)
        if len(chunks) == 1:
            outgoing[peer] = chunks[0]
        else:
            total_sz = sum(c.numel() for c in chunks)
            export_buf = rt._alloc_wire_buffer(total_sz)
            outgoing[peer] = torch.cat(chunks, out=export_buf) if export_buf is not None else torch.cat(chunks)

    recv_payloads, recv_tags = {}, {}
    for k, (s, bk, r, o) in enumerate(schedule):
        if p == r:
            recv_payloads.setdefault(s, []).append((k, o, bk))
            recv_tags.setdefault(bk, []).append(s)

    sizes = {}
    for peer in set(recv_payloads) | set(recv_tags):
        sizes[peer] = len(recv_payloads.get(peer, ())) * n_bytes + len(recv_tags.get(peer, ())) * 32

    incoming = rt._exchange_device(outgoing, sizes)

    actual_frames, wanted_tags = [], []
    out_parts = list(out_s.unbind(0))
    for peer, entries in recv_payloads.items():
        buf = incoming[peer]
        for idx_e, (_, o, bk) in enumerate(entries):
            frame = buf[idx_e * n_bytes:(idx_e + 1) * n_bytes]
            out_parts[(o - p - 1) % 4] ^= frame.view(torch.int32)
            actual_frames.append((bk, peer, frame))
    for bk, senders in recv_tags.items():
        buf = incoming[bk]
        off = len(recv_payloads.get(bk, ())) * n_bytes
        for idx_s, s in enumerate(sorted(senders)):
            tag = buf[off + idx_s * 32:off + (idx_s + 1) * 32]
            wanted_tags.append((bk, s, tag))

    if actual_frames:
        actual_sorted = sorted(actual_frames, key=lambda x: (x[0], x[1]))
        wanted_sorted = sorted(wanted_tags, key=lambda x: (x[0], x[1]))
        computed = digest_batch(torch.stack([f for _, _, f in actual_sorted]), block_tree=rt.hash_block_tree)
        expected = torch.stack([t for _, _, t in wanted_sorted])
        rt._verify_tags(computed, expected, 'CRT bit-sliced binary AND BLAKE3 mismatch')

    return tuple(t.reshape(shape) for t in out_parts)


def _bin_xor_u32(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])


def _bin_slice(s: tuple[torch.Tensor, ...], sl) -> tuple[torch.Tensor, ...]:
    return (s[0][sl], s[1][sl], s[2][sl])


def _bin_cat(items: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.cat([x[i] for x in items], dim=0) for i in range(3))


def _crt_joint_input_pair_raw(
    rt, a_01: torch.Tensor | None, b_23: torch.Tensor | None,
    logical_shape: tuple[int, ...],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """1-round checked 2-sender CRT sharing returning only the 3 active tensors (m2, a_wire, m0, b_wire)."""
    primes = rt.primes
    n_lanes = len(primes)
    n = math.prod(logical_shape)
    wire_bytes = n_lanes * n
    p = rt.party

    key2 = None if p == 2 else rt._device_keys[2].key(_crt_tr_domain(b'crt-ji-v1', primes, 2))
    key0 = None if p == 0 else rt._device_keys[0].key(_crt_tr_domain(b'crt-ji-v1', primes, 0))
    nonces = rt._crt_nonces((None if p == 2 else 2, None if p == 0 else 0),
                            tag=('crt_ji', logical_shape))

    outgoing, sizes = {}, {}
    m2 = a_wire = m0 = b_wire = None
    if p in (0, 1):
        buf = rt._alloc_wire_buffer(wire_bytes) if p == 0 else None
        a_wire = buf.reshape(n_lanes, *logical_shape) if buf is not None else torch.empty(
            (n_lanes, *logical_shape), dtype=torch.uint8, device=rt.device,
        )
        m2, m0 = crt_joint_input_sub(a_01, key2, key0, nonces, 0, primes, a_wire)
        wire_flat = a_wire.reshape(-1)
        if p == 0:
            outgoing[2] = wire_flat
            sizes[2] = wire_bytes
            sizes[3] = 32
        else:
            outgoing[2] = digest(wire_flat, block_tree=rt.hash_block_tree)
    else:
        buf = rt._alloc_wire_buffer(wire_bytes) if p == 2 else None
        b_wire = buf.reshape(n_lanes, *logical_shape) if buf is not None else torch.empty(
            (n_lanes, *logical_shape), dtype=torch.uint8, device=rt.device,
        )
        m0, m2 = crt_joint_input_sub(b_23, key0, key2, nonces, 1, primes, b_wire)
        wire_flat = b_wire.reshape(-1)
        if p == 2:
            outgoing[0] = wire_flat
            sizes[0] = wire_bytes
            sizes[1] = 32
        else:
            outgoing[0] = digest(wire_flat, block_tree=rt.hash_block_tree)

    incoming = rt._exchange_device(outgoing, sizes)
    if p == 0:
        b_flat = incoming[2]
        rt._verify_tags(digest(b_flat, block_tree=rt.hash_block_tree), incoming[3],
                        'CRT joint input BLAKE3 mismatch at party 0')
        b_wire = b_flat.reshape(n_lanes, *logical_shape)
    elif p == 2:
        a_flat = incoming[0]
        rt._verify_tags(digest(a_flat, block_tree=rt.hash_block_tree), incoming[1],
                        'CRT joint input BLAKE3 mismatch at party 2')
        a_wire = a_flat.reshape(n_lanes, *logical_shape)
    return m2, a_wire, m0, b_wire


def crt_joint_input_pair(rt, a_01: torch.Tensor | None, b_23: torch.Tensor | None,
                         logical_shape: tuple[int, ...]) -> tuple[Share, Share]:
    """1-round checked 2-sender CRT sharing: a_01 from {P0,P1} and b_23 from {P2,P3}."""
    m2, a_wire, m0, b_wire = _crt_joint_input_pair_raw(rt, a_01, b_23, logical_shape)
    zeros = torch.zeros((len(rt.primes), *logical_shape), dtype=torch.uint8, device=rt.device)
    p = rt.party
    if p == 0:
        return Share((zeros, m2, a_wire), False), Share((b_wire, zeros, zeros), False)
    if p == 1:
        return Share((m2, a_wire, zeros), False), Share((zeros, zeros, m0), False)
    if p == 2:
        return Share((a_wire, zeros, zeros), False), Share((zeros, m0, b_wire), False)
    return Share((zeros, zeros, m2), False), Share((m0, b_wire, zeros), False)


def _crt_comp_mul_raw(
    rt, c_in: torch.Tensor, xs_in: torch.Tensor, xh_in: torch.Tensor,
    logical_shape: tuple[int, ...], *,
    src_lane_stride: int | None = None, src_slice_off: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """1-round active-secure 4-joint CRT product returning (out0, out1, out2, recv_wire, recv_slot)."""
    primes = rt.primes
    n_lanes = len(primes)
    n = math.prod(logical_shape)
    wire_bytes = n_lanes * n
    p = rt.party

    e0, e1, e2 = (p + 1) % 4, (p + 2) % 4, (p + 3) % 4
    k0 = rt._device_keys[e0].key(_crt_tr_domain(b'crt-cm-v1', primes, e0))
    k1 = rt._device_keys[e1].key(_crt_tr_domain(b'crt-cm-v1', primes, e1))
    k2 = rt._device_keys[e2].key(_crt_tr_domain(b'crt-cm-v1', primes, e2))
    excl_canon = tuple(None if p == r else r for r in range(4))
    nonces_all = rt._crt_nonces(excl_canon, tag=('crt_cm', logical_shape))

    send_buf = rt._alloc_wire_buffer(wire_bytes)
    w_send = send_buf.reshape(n_lanes, *logical_shape) if send_buf is not None else torch.empty(
        (n_lanes, *logical_shape), dtype=torch.uint8, device=rt.device,
    )
    out0, out1, out2, w_hash = crt_comp_mul_prep(
        c_in.contiguous(), xs_in.contiguous(), xh_in.contiguous(),
        (k0, k1, k2), nonces_all, p, primes, logical_shape, w_send,
        src_lane_stride=src_lane_stride, src_slice_off=src_slice_off,
    )

    send_to = (3, 2, 0, 1)[p]
    hash_to = (1, 0, 3, 2)[p]
    recv_from = (2, 3, 1, 0)[p]
    tag_from = (1, 0, 3, 2)[p]
    recv_o = (3, 2, 0, 1)[p]
    recv_slot = (recv_o - p - 1) % 4

    tag_out = digest(w_hash.reshape(-1), block_tree=rt.hash_block_tree)
    incoming = rt._exchange_device(
        {send_to: w_send.reshape(-1), hash_to: tag_out},
        {recv_from: wire_bytes, tag_from: 32},
    )
    recv_wire = incoming[recv_from]
    rt._verify_tags(digest(recv_wire, block_tree=rt.hash_block_tree), incoming[tag_from],
                    'CRT complementary product BLAKE3 mismatch')
    return out0, out1, out2, recv_wire.reshape(n_lanes, *logical_shape), recv_slot


def crt_comp_mul(rt, a: Share, b: Share) -> Share:
    """1-round active-secure 4-joint CRT product for a on {2,3} and b on {0,1}."""
    rt._validate_crt_share(a)
    rt._validate_crt_share(b)
    if a.parts[0].shape != b.parts[0].shape:
        raise ValueError('crt_comp_mul operands must have matching shapes')
    logical_shape = tuple(a.parts[0].shape[1:])
    p = rt.party
    if p == 0:
        c_in, xs_in, xh_in = b.parts[0], a.parts[2], a.parts[1]
    elif p == 1:
        c_in, xs_in, xh_in = b.parts[2], a.parts[0], a.parts[1]
    elif p == 2:
        c_in, xs_in, xh_in = a.parts[0], b.parts[1], b.parts[2]
    else:
        c_in, xs_in, xh_in = a.parts[2], b.parts[1], b.parts[0]
    out0, out1, out2, recv_wire, recv_slot = _crt_comp_mul_raw(rt, c_in, xs_in, xh_in, logical_shape)
    outs = [out0, out1, out2]
    from ..matmul.crt_conv import crt_add_mod
    crt_add_mod(outs[recv_slot], recv_wire, rt.primes, inplace=True)
    return Share(tuple(outs), False)


def crt_elem_mul(rt, a: Share, b: Share) -> Share:
    """Elementwise private CRT product via 1-product quadratic polynomial gate."""
    rt._validate_crt_share(a)
    rt._validate_crt_share(b)
    if a.parts[0].shape != b.parts[0].shape:
        raise ValueError('crt_elem_mul operands must have matching shapes')
    primes = rt.primes
    n_lanes = len(primes)
    c0, c1, c2 = _party_coeffs(rt.party)
    p0, p1, p2, p3 = _pad_primes(primes)
    n = a.parts[0].numel() // n_lanes
    h_eval = torch.empty_like(a.parts[0])
    if n:
        with torch.cuda.device(rt.device):
            _crt_elem_mul_eval_kernel[(triton.cdiv(n, 256), n_lanes)](
                *(t.contiguous().reshape(-1) for t in a.parts),
                *(t.contiguous().reshape(-1) for t in b.parts),
                h_eval.reshape(-1), n, c0, c1, c2, p0, p1, p2, p3, 256, num_warps=4,
            )
    return rt._crt_poly_exchange(h_eval)


def crt_trunc_relu(rt, x: Share, shift: int) -> Share:
    """Active-secure CRT probabilistic truncation by 2**shift fused with exact ReLU.

    Contract: |signed(x)| <= ((M // 4) // 2**shift) * 2**shift - 1.
    Returns a CRT share of ReLU(y) where y in [floor(x/2**shift) - 1, floor(x/2**shift) + 1].
    """
    rt._validate_crt_share(x)
    primes = rt.primes
    n_lanes = len(primes)
    m_mod = rt.crt_modulus
    if m_mod >= (1 << 32):
        raise ValueError('crt_trunc_relu requires CRT modulus M < 2**32')
    scale = 1 << shift
    if shift < 1 or scale >= m_mod // 4:
        raise ValueError('invalid CRT truncation shift')

    logical_shape = tuple(x.parts[0].shape[1:])
    n = math.prod(logical_shape)
    if not n:
        return x

    half_m = m_mod // 2
    q_m = m_mod // scale
    off_q = (m_mod // 4) // scale
    off_m = off_q * scale
    # Signed bit width W covering |y| <= off_q + 2 < 2**(W-1).
    w_bits = (off_q + 2).bit_length() + 1
    q_mask = q_m & ((1 << w_bits) - 1)
    n_words = triton.cdiv(n, 32)
    p = rt.party
    p0, p1, p2, p3 = _pad_primes(primes)
    coeffs = [(m_mod // q) * pow(m_mod // q, -1, q) for q in primes]
    c0, c1, c2, c3 = tuple(coeffs) + (0,) * (4 - n_lanes)

    # --- Step 1: Round 1 (P0,P1 generate uniform r = (r2+r3) mod M, send m = (x23+r) mod M) ---
    key2 = None if p == 2 else rt._device_keys[2].key(_crt_tr_domain(b'crt-tr-v1', primes, 2))
    key3 = None if p == 3 else rt._device_keys[3].key(_crt_tr_domain(b'crt-tr-v1', primes, 3))
    nonces = rt._crt_nonces((None if p == 2 else 2, None if p == 3 else 3),
                            tag=('crt_tr_r1', logical_shape, shift))

    u_planes = v_planes = None
    crt_01 = crt_23 = None
    outgoing, sizes = {}, {}
    m_dtype = torch.uint16 if m_mod < 65536 else torch.uint32
    m_bytes_len = n * (2 if m_mod < 65536 else 4)
    if p in (0, 1):
        r2, r3 = crt_input_masks((key2, key3), nonces, primes, logical_shape, rt.device)
        x2_part = x.parts[1] if p == 0 else x.parts[0]
        x3_part = x.parts[2] if p == 0 else x.parts[1]
        m_buf = rt._alloc_wire_buffer(m_bytes_len) if p == 0 else None
        m_out = m_buf.view(m_dtype) if m_buf is not None else torch.empty(n, dtype=m_dtype, device=rt.device)
        u_planes = torch.empty((1 + w_bits, n_words), dtype=torch.int32, device=rt.device)
        crt_01 = torch.empty((n_lanes, 3, n), dtype=torch.uint8, device=rt.device)
        with torch.cuda.device(rt.device):
            _crt_round1_p01_warp_kernel[(triton.cdiv(n_words, 8),)](
                x2_part.contiguous().reshape(-1), x3_part.contiguous().reshape(-1),
                r2.reshape(-1), r3.reshape(-1),
                m_out, u_planes.view(torch.uint32), crt_01,
                n, n_words, shift, w_bits, m_mod, half_m,
                n_lanes, p0, p1, p2, p3, c0, c1, c2, c3, 8, num_warps=4,
            )
        m_bytes = m_buf if m_buf is not None else m_out.view(torch.uint8)
        if p == 0:
            outgoing = {2: m_bytes, 3: m_bytes}
        else:
            m_tag = digest(m_bytes, block_tree=rt.hash_block_tree)
            outgoing = {2: m_tag, 3: m_tag}
    else:
        sizes = {0: m_bytes_len, 1: 32}

    r1_in = rt._exchange_device(outgoing, sizes)
    if p in (2, 3):
        m_recv = r1_in[0]
        rt._verify_tags(digest(m_recv, block_tree=rt.hash_block_tree), r1_in[1],
                        'CRT trunc_relu Round 1 BLAKE3 mismatch')
        x0_part = x.parts[1] if p == 2 else x.parts[0]
        x1_part = x.parts[2] if p == 2 else x.parts[1]
        v_planes = torch.empty((1 + w_bits, n_words), dtype=torch.int32, device=rt.device)
        crt_23 = torch.empty((n_lanes, 3, n), dtype=torch.uint8, device=rt.device)
        with torch.cuda.device(rt.device):
            _crt_round1_p23_warp_kernel[(triton.cdiv(n_words, 8),)](
                m_recv.view(m_dtype),
                x0_part.contiguous().reshape(-1), x1_part.contiguous().reshape(-1),
                v_planes.view(torch.uint32), crt_23,
                n, n_words, shift, w_bits, m_mod, half_m, off_m, off_q,
                n_lanes, p0, p1, p2, p3, c0, c1, c2, c3, 8, num_warps=4,
            )

    # --- Step 2: Bit-sliced int32 sign extraction for y = c_q - r_q + w * q_M mod 2**W ---
    q_lo_mask = q_mask & ((1 << (w_bits - 1)) - 1)
    q_top_bit = (q_mask >> (w_bits - 1)) & 1
    if q_lo_mask == 0:
        # Exact zero-CSA path: q_M has zero bits in 0..W-2, so w * q_M never produces a carry
        # into bit W-1. If q_top_bit == 1, w simply XORs into the top sign bit via the same
        # 1-round complementary AND as the leaf layer.
        if q_top_bit == 0:
            u_bin, v_bin = _bin_joint_pair_u32(
                rt,
                None if u_planes is None else u_planes[1:].contiguous(),
                None if v_planes is None else v_planes[1:].contiguous(),
                (w_bits, n_words),
            )
            g_cur = _bin_and_u32(
                rt, _bin_slice(u_bin, slice(0, w_bits - 1)),
                _bin_slice(v_bin, slice(0, w_bits - 1)), complementary=True,
            )
            p_cur = _bin_xor_u32(
                _bin_slice(u_bin, slice(0, w_bits - 1)),
                _bin_slice(v_bin, slice(0, w_bits - 1)),
            )
            top_xor = _bin_xor_u32(
                _bin_slice(u_bin, slice(w_bits - 1, w_bits)),
                _bin_slice(v_bin, slice(w_bits - 1, w_bits)),
            )
        else:
            u_all, v_all = _bin_joint_pair_u32(rt, u_planes, v_planes, (1 + w_bits, n_words))
            w_and_g = _bin_and_u32(
                rt, _bin_slice(u_all, slice(0, w_bits)),
                _bin_slice(v_all, slice(0, w_bits)), complementary=True,
            )
            w_bit = _bin_slice(w_and_g, slice(0, 1))
            g_cur = _bin_slice(w_and_g, slice(1, w_bits))
            p_cur = _bin_xor_u32(
                _bin_slice(u_all, slice(1, w_bits)),
                _bin_slice(v_all, slice(1, w_bits)),
            )
            top_xor = _bin_xor_u32(
                _bin_xor_u32(
                    _bin_slice(u_all, slice(w_bits, 1 + w_bits)),
                    _bin_slice(v_all, slice(w_bits, 1 + w_bits)),
                ),
                w_bit,
            )
        width = w_bits - 1
    else:
        u_all, v_all = _bin_joint_pair_u32(rt, u_planes, v_planes, (1 + w_bits, n_words))
        # Complementary AND for [R_hi & C_lo, U[0..W-2] & V[0..W-2]] in 1 round (W planes).
        w_and_gab = _bin_and_u32(rt, _bin_slice(u_all, slice(0, w_bits)),
                                 _bin_slice(v_all, slice(0, w_bits)), complementary=True)
        w_bit = _bin_slice(w_and_gab, slice(0, 1))
        g_ab = _bin_slice(w_and_gab, slice(1, w_bits))  # W-1 planes (bits 0..W-2)
        p_ab = _bin_xor_u32(_bin_slice(u_all, slice(1, 1 + w_bits)),
                            _bin_slice(v_all, slice(1, 1 + w_bits)))  # W planes (bits 0..W-1)

        # CSA: S1 = P_ab ^ (W * q_mask), C1 = G_ab ^ (P_ab & (W * q_mask)) on bits 0..W-2.
        set_bits = [k for k in range(w_bits - 1) if (q_mask >> k) & 1]
        if set_bits:
            p_sel = _bin_cat([_bin_slice(p_ab, slice(k, k + 1)) for k in set_bits])
            w_rep = tuple(t.expand(len(set_bits), n_words).contiguous() for t in w_bit)
            pw_sel = _bin_and_u32(rt, p_sel, w_rep, complementary=False)
        else:
            pw_sel = None

        # Assemble S1 (W planes) and C1 (W-1 planes).
        s1_list = []
        c1_list = []
        sel_pos = {k: i for i, k in enumerate(set_bits)}
        for k in range(w_bits):
            pk = _bin_slice(p_ab, slice(k, k + 1))
            s1_list.append(_bin_xor_u32(pk, w_bit) if (q_mask >> k) & 1 else pk)
            if k < w_bits - 1:
                gk = _bin_slice(g_ab, slice(k, k + 1))
                c1_list.append(_bin_xor_u32(gk, _bin_slice(pw_sel, slice(sel_pos[k], sel_pos[k] + 1)))
                               if k in sel_pos else gk)
        s1 = _bin_cat(s1_list)
        c1 = _bin_cat(c1_list)

        # Bit (W-1) of S1 + (C1 << 1) = S1[W-1] ^ C1[W-2] ^ carry_into_{W-1}(S1[1..W-2], C1[0..W-3]).
        s_lo = _bin_slice(s1, slice(1, w_bits - 1))  # W-2 planes
        c_lo_s = _bin_slice(c1, slice(0, w_bits - 2))  # W-2 planes
        g_cur = _bin_and_u32(rt, s_lo, c_lo_s, complementary=False)
        p_cur = _bin_xor_u32(s_lo, c_lo_s)
        top_xor = _bin_xor_u32(
            _bin_slice(s1, slice(w_bits - 1, w_bits)),
            _bin_slice(c1, slice(w_bits - 2, w_bits - 1)),
        )
        width = w_bits - 2

    while width > 1:
        n_pairs = width // 2
        even_g = _bin_slice(g_cur, slice(0, 2 * n_pairs, 2))
        odd_g = _bin_slice(g_cur, slice(1, 2 * n_pairs, 2))
        even_p = _bin_slice(p_cur, slice(0, 2 * n_pairs, 2))
        odd_p = _bin_slice(p_cur, slice(1, 2 * n_pairs, 2))
        if n_pairs > 1:
            left = _bin_cat([odd_p, _bin_slice(odd_p, slice(1, n_pairs))])
            right = _bin_cat([even_g, _bin_slice(even_p, slice(1, n_pairs))])
            prod = _bin_and_u32(rt, left, right, complementary=False)
            g_next = _bin_xor_u32(odd_g, _bin_slice(prod, slice(0, n_pairs)))
            p_next = _bin_cat([_bin_slice(odd_p, slice(0, 1)), _bin_slice(prod, slice(n_pairs, 2 * n_pairs - 1))])
        else:
            prod = _bin_and_u32(rt, odd_p, even_g, complementary=False)
            g_next = _bin_xor_u32(odd_g, prod)
            p_next = odd_p  # unused at index 0
        if width % 2 == 1:
            g_next = _bin_cat([g_next, _bin_slice(g_cur, slice(width - 1, width))])
            p_next = _bin_cat([p_next, _bin_slice(p_cur, slice(width - 1, width))])
        g_cur, p_cur = g_next, p_next
        width = n_pairs + (width % 2)

    sign_bit = _bin_xor_u32(top_xor, g_cur)

    # --- Step 3: Unpack local sign parities (s_v at P0,P1; s_u at P2,P3) into CRT joint input ---
    if p in (0, 1):
        s2_part = sign_bit[1] if p == 0 else sign_bit[0]
        s3_part = sign_bit[2] if p == 0 else sign_bit[1]
        with torch.cuda.device(rt.device):
            _unpack_sign_to_crt_kernel[(triton.cdiv(n_words, 8),)](
                s2_part.reshape(-1).view(torch.uint32), s3_part.reshape(-1).view(torch.uint32),
                crt_01, n, n_words, n_lanes, 8, num_warps=4,
            )
    else:
        s0_part = sign_bit[1] if p == 2 else sign_bit[0]
        s1_part = sign_bit[2] if p == 2 else sign_bit[1]
        with torch.cuda.device(rt.device):
            _unpack_sign_to_crt_kernel[(triton.cdiv(n_words, 8),)](
                s0_part.reshape(-1).view(torch.uint32), s1_part.reshape(-1).view(torch.uint32),
                crt_23, n, n_words, n_lanes, 8, num_warps=4,
            )

    m2, a_wire, m0, b_wire = _crt_joint_input_pair_raw(rt, crt_01, crt_23, (3, n))
    # Zero-copy 1-round 4-joint complementary CRT product directly on slices 1 (W) & 2 (SUV).
    pc0, pc1, pc2 = _party_coeffs(p)
    if p == 0:
        c_in, xs_in, xh_in = b_wire, a_wire, m2
        t0, t1, t2 = m2, a_wire, b_wire
        yc = (-pc1, -pc2, pc0)
        kc = (-pc1, -pc2, -pc0)
    elif p == 1:
        c_in, xs_in, xh_in = m0, m2, a_wire
        t0, t1, t2 = m2, a_wire, m0
        yc = (-pc0, -pc1, pc2)
        kc = (-pc0, -pc1, -pc2)
    elif p == 2:
        c_in, xs_in, xh_in = a_wire, m0, b_wire
        t0, t1, t2 = a_wire, m0, b_wire
        yc = (-pc0, pc1, pc2)
        kc = (-pc0, -pc1, -pc2)
    else:
        c_in, xs_in, xh_in = m2, b_wire, m0
        t0, t1, t2 = m2, m0, b_wire
        yc = (-pc2, pc0, pc1)
        kc = (-pc2, -pc0, -pc1)

    w0, w1, w2, w_recv, recv_slot = _crt_comp_mul_raw(
        rt, c_in, xs_in, xh_in, (2, n), src_lane_stride=3 * n, src_slice_off=n,
    )

    # Fused register evaluation of F_Y(party) * F_KEEP(party) mod p_l (skipping zero slots + folding w_recv).
    c_recv = (pc0, pc1, pc2)[recv_slot]
    h_eval = torch.empty((n_lanes, *logical_shape), dtype=torch.uint8, device=rt.device)
    with torch.cuda.device(rt.device):
        _crt_fused_y_keep_mul_kernel[(triton.cdiv(n, 256), n_lanes)](
            t0.reshape(-1), t1.reshape(-1), t2.reshape(-1),
            w0.reshape(-1), w1.reshape(-1), w2.reshape(-1), w_recv.reshape(-1),
            h_eval.reshape(-1), n, q_m, p,
            yc[0], yc[1], yc[2], kc[0], kc[1], kc[2],
            pc0, pc1, pc2, c_recv, p0, p1, p2, p3, 256, num_warps=4,
        )
    return rt._crt_poly_exchange(h_eval)
