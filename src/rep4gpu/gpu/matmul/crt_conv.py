"""Exact GPU Triton kernels for odd-prime CRT 1-product polynomial Rep4 MPC."""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
import triton
import triton.language as tl

from ..crypto.gpu_hash import _check, _current_context

DEFAULT_CRT_PRIMES = (251, 241, 239, 233)


def validate_primes(primes) -> tuple[int, ...]:
    """Validate 2–4 distinct odd primes in (3, 256) for byte-lane CRT."""
    primes = tuple(int(p) for p in primes)
    if not 2 <= len(primes) <= 4 or len(set(primes)) != len(primes):
        raise ValueError('CRT requires 2 to 4 distinct primes')
    for p in primes:
        if not 5 <= p <= 255 or p % 2 == 0 or any(p % d == 0 for d in range(3, int(math.isqrt(p)) + 1, 2)):
            raise ValueError(f'CRT modulus {p} must be an odd prime in [5, 255]')
    return primes


def _pad_primes(primes: tuple[int, ...]) -> tuple[int, int, int, int]:
    return primes + (primes[0],) * (4 - len(primes))


def _party_coeffs(party: int) -> tuple[int, int, int]:
    if party not in range(4):
        raise ValueError('party must be in [0, 3]')
    return tuple(party - ((party + s + 1) % 4) for s in range(3))


@dataclass(frozen=True)
class PreparedCrtWeight:
    """Precomputed centered int8 polynomial evaluation G(party) for static weights."""
    g_eval: torch.Tensor
    primes: tuple[int, ...]
    party: int
    original_shape: tuple[int, ...]

    @property
    def shape(self):
        return self.original_shape

    @property
    def device(self):
        return self.g_eval.device


@triton.jit
def _lane_prime(lane, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr):
    return tl.where(lane == 0, P0, tl.where(lane == 1, P1, tl.where(lane == 2, P2, P3)))


@triton.jit
def _lane_barrett_u32(lane, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr):
    # Compile-time 32-bit Barrett multipliers floor(2^32 / p) without uint64 register pressure.
    M0: tl.constexpr = (1 << 32) // P0
    M1: tl.constexpr = (1 << 32) // P1
    M2: tl.constexpr = (1 << 32) // P2
    M3: tl.constexpr = (1 << 32) // P3
    p = _lane_prime(lane, P0, P1, P2, P3).to(tl.uint32)
    m = tl.where(lane == 0, M0, tl.where(lane == 1, M1, tl.where(lane == 2, M2, M3))).to(tl.uint32)
    return p, m


@triton.jit
def _fast_mod_u32(x, p, m):
    # Exact x % p for all uint32 x using mul.hi.u32 (zero hardware division).
    q = tl.inline_asm_elementwise(
        "mul.hi.u32 $0, $1, $2;", constraints="=r,r,r",
        args=[x.to(tl.uint32), m], dtype=tl.uint32, is_pure=True, pack=1,
    )
    r = x.to(tl.uint32) - q * p
    return tl.where(r >= p, r - p, r)


@triton.jit
def _eval_linear_kernel(X0, X1, X2, OUT, N: tl.constexpr,
                        C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                        BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    half = p >> 1
    # Shift by 16*p (> 6*255) so dividend is guaranteed nonnegative for fast uint32 reduction.
    shift = p * 16
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    off = lane * N + idx
    a0 = tl.load(X0 + off, valid, 0).to(tl.int32)
    a1 = tl.load(X1 + off, valid, 0).to(tl.int32)
    a2 = tl.load(X2 + off, valid, 0).to(tl.int32)
    val = _fast_mod_u32((C0 * a0 + C1 * a1 + C2 * a2 + shift).to(tl.uint32), p_u32, m_u32).to(tl.int32)
    centered = tl.where(val > half, val - p, val).to(tl.int8)
    tl.store(OUT + off, centered, valid)


@triton.jit
def _eval_linear_vec4_kernel(X0, X1, X2, OUT, N4: tl.constexpr,
                             C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                             P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                             BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    half = p >> 1
    shift = p * 16
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N4
    off = lane * N4 + idx
    w0 = tl.load(X0 + off, valid, 0)
    w1 = tl.load(X1 + off, valid, 0)
    w2 = tl.load(X2 + off, valid, 0)
    packed = tl.zeros((BLOCK,), dtype=tl.uint32)
    for sub in tl.static_range(4):
        sh = sub * 8
        a0 = ((w0 >> sh) & 0xFF).to(tl.int32)
        a1 = ((w1 >> sh) & 0xFF).to(tl.int32)
        a2 = ((w2 >> sh) & 0xFF).to(tl.int32)
        val = _fast_mod_u32((C0 * a0 + C1 * a1 + C2 * a2 + shift).to(tl.uint32), p_u32, m_u32).to(tl.int32)
        centered = tl.where(val > half, val - p, val) & 0xFF
        packed |= centered.to(tl.uint32) << sh
    tl.store(OUT + off, packed, valid)


@triton.jit
def _conv1d_eval_kernel(FX, GW, OUT,
                        B: tl.constexpr, L_IN: tl.constexpr, L_OUT: tl.constexpr,
                        C_IN: tl.constexpr, C_OUT: tl.constexpr, K_SIZE: tl.constexpr,
                        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    lane = tl.program_id(2)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    M: tl.constexpr = B * L_OUT
    CIN_BLOCKS: tl.constexpr = (C_IN + BK - 1) // BK
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    valid_r = r < M
    valid_c = c < C_OUT
    b_idx = r // L_OUT
    t_idx = r % L_OUT
    kk = tl.arange(0, BK)
    fx_base = FX + lane * (B * L_IN * C_IN)
    gw_base = GW + lane * (K_SIZE * C_IN * C_OUT)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for step in range(K_SIZE * CIN_BLOCKS):
        kd = step // CIN_BLOCKS
        start = step % CIN_BLOCKS
        ci = start * BK + kk
        row_base = (b_idx * L_IN + (t_idx + kd)) * C_IN
        a = tl.load(fx_base + row_base[:, None] + ci[None, :],
                    valid_r[:, None] & (ci[None, :] < C_IN), 0)
        w = tl.load(gw_base + (kd * C_IN + ci[:, None]) * C_OUT + c[None, :],
                    (ci[:, None] < C_IN) & valid_c[None, :], 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    # Reduce signed int32 acc mod p via fast unsigned Barrett on |acc|.
    neg = acc < 0
    mag = tl.where(neg, -acc, acc).to(tl.uint32)
    rem = _fast_mod_u32(mag, p_u32, m_u32).to(tl.int32)
    val = tl.where(neg & (rem != 0), p - rem, rem).to(tl.uint8)
    tl.store(OUT + lane * (M * C_OUT) + r[:, None] * C_OUT + c[None, :], val,
             valid_r[:, None] & valid_c[None, :])


@triton.jit
def _matmul_eval_kernel(FX, GW, OUT,
                        M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    lane = tl.program_id(2)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    valid_r = r < M
    valid_c = c < N
    fx_base = FX + lane * (M * K)
    gw_base = GW + lane * (K * N)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for start in range(tl.cdiv(K, BK)):
        ki = start * BK + kk
        a = tl.load(fx_base + r[:, None] * K + ki[None, :],
                    valid_r[:, None] & (ki[None, :] < K), 0)
        w = tl.load(gw_base + ki[:, None] * N + c[None, :],
                    (ki[:, None] < K) & valid_c[None, :], 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    neg = acc < 0
    mag = tl.where(neg, -acc, acc).to(tl.uint32)
    rem = _fast_mod_u32(mag, p_u32, m_u32).to(tl.int32)
    val = tl.where(neg & (rem != 0), p - rem, rem).to(tl.uint8)
    tl.store(OUT + lane * (M * N) + r[:, None] * N + c[None, :], val,
             valid_r[:, None] & valid_c[None, :])


@triton.jit
def _matvec_eval_kernel(FX, GW, OUT,
                        M: tl.constexpr, K: tl.constexpr,
                        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                        BM: tl.constexpr, BK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    kk = tl.arange(0, BK)
    valid_r = r < M
    fx_base = FX + lane * (M * K)
    gw_base = GW + lane * K
    acc = tl.zeros((BM,), dtype=tl.int32)
    for start in range(tl.cdiv(K, BK)):
        ki = start * BK + kk
        valid_k = ki < K
        a = tl.load(fx_base + r[:, None] * K + ki[None, :],
                    valid_r[:, None] & valid_k[None, :], 0).to(tl.int32)
        w = tl.load(gw_base + ki, valid_k, 0).to(tl.int32)
        acc += tl.sum(a * w[None, :], axis=1)
    neg = acc < 0
    mag = tl.where(neg, -acc, acc).to(tl.uint32)
    rem = _fast_mod_u32(mag, p_u32, m_u32).to(tl.int32)
    val = tl.where(neg & (rem != 0), p - rem, rem).to(tl.uint8)
    tl.store(OUT + lane * M + r, val, valid_r)


@triton.jit
def _square_sum_pool_kernel(H0, H1, H2, B0, B1, B2, OUT,
                            B: tl.constexpr, L_OUT: tl.constexpr, C: tl.constexpr,
                            HAS_BIAS: tl.constexpr,
                            C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                            P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                            BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    shift = p * 16
    total: tl.constexpr = B * C
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < total
    b = idx // C
    c = idx % C
    fb = tl.zeros((BLOCK,), dtype=tl.int32)
    if HAS_BIAS:
        boff = lane * C + c
        bv0 = tl.load(B0 + boff, valid, 0).to(tl.int32)
        bv1 = tl.load(B1 + boff, valid, 0).to(tl.int32)
        bv2 = tl.load(B2 + boff, valid, 0).to(tl.int32)
        fb = _fast_mod_u32((C0 * bv0 + C1 * bv1 + C2 * bv2 + shift).to(tl.uint32), p_u32, m_u32).to(tl.int32)
    base = lane * (B * L_OUT * C) + b * (L_OUT * C) + c
    acc = tl.zeros((BLOCK,), dtype=tl.uint32)
    for t in range(L_OUT):
        off = base + t * C
        h0 = tl.load(H0 + off, valid, 0).to(tl.int32)
        h1 = tl.load(H1 + off, valid, 0).to(tl.int32)
        h2 = tl.load(H2 + off, valid, 0).to(tl.int32)
        f = _fast_mod_u32((C0 * h0 + C1 * h1 + C2 * h2 + fb + shift).to(tl.uint32), p_u32, m_u32)
        acc += f * f
    val = _fast_mod_u32(acc, p_u32, m_u32).to(tl.uint8)
    tl.store(OUT + lane * total + idx, val, valid)


_B3_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)


def _blake3_xof_cuda_source() -> str:
    """Build NVRTC source for official BLAKE3-256 keyed XOF + CRT mask kernels."""
    b3_lines = [
        '__device__ __forceinline__ void blake3_xof_block(const U32* __restrict__ k, U64 nonce, U64 counter, U32 out[16]) {',
        '    U32 m0 = (U32)nonce, m1 = (U32)(nonce >> 32);',
        '    U32 s0=k[0], s1=k[1], s2=k[2], s3=k[3], s4=k[4], s5=k[5], s6=k[6], s7=k[7];',
        '    U32 s8=0x6A09E667u, s9=0xBB67AE85u, s10=0x3C6EF372u, s11=0xA54FF53Au;',
        '    U32 s12=(U32)counter, s13=(U32)(counter >> 32), s14=8u, s15=27u;',
    ]
    sched = list(range(16))
    pairs = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
             (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))
    for _ in range(7):
        for i, (a, b, c, d) in enumerate(pairs):
            mx, my = sched[2 * i], sched[2 * i + 1]
            addx = ' + m0' if mx == 0 else (' + m1' if mx == 1 else '')
            addy = ' + m0' if my == 0 else (' + m1' if my == 1 else '')
            b3_lines.append(
                f'    s{a} = s{a} + s{b}{addx}; s{d} = rotr32(s{d} ^ s{a}, 16);'
                f' s{c} += s{d}; s{b} = rotr32(s{b} ^ s{c}, 12);'
            )
            b3_lines.append(
                f'    s{a} = s{a} + s{b}{addy}; s{d} = rotr32(s{d} ^ s{a}, 8);'
                f' s{c} += s{d}; s{b} = rotr32(s{b} ^ s{c}, 7);'
            )
        sched = [sched[i] for i in _B3_PERM]
    b3_lines.extend([
        '    out[0]=s0^s8;   out[1]=s1^s9;   out[2]=s2^s10;  out[3]=s3^s11;',
        '    out[4]=s4^s12;  out[5]=s5^s13;  out[6]=s6^s14;  out[7]=s7^s15;',
        '    out[8]=s8^k[0]; out[9]=s9^k[1]; out[10]=s10^k[2]; out[11]=s11^k[3];',
        '    out[12]=s12^k[4]; out[13]=s13^k[5]; out[14]=s14^k[6]; out[15]=s15^k[7];',
        '}',
    ])
    return (
        'typedef unsigned char U8;\n'
        'typedef unsigned int U32;\n'
        'typedef unsigned long long U64;\n'
        '__device__ __forceinline__ U32 rotr32(U32 x, int n) { return __funnelshift_r(x, x, n); }\n'
        + '\n'.join(b3_lines)
        + r'''
__device__ __forceinline__ U32 fast_mod_u32(U32 x, U32 p, U32 m) {
    U32 q = __umulhi(x, m);
    U32 r = x - q * p;
    return (r >= p) ? (r - p) : r;
}
__device__ __forceinline__ void fast_divmod_u32(U32 x, U32 p, U32 m, U32 &q, U32 &r) {
    q = __umulhi(x, m);
    r = x - q * p;
    if (r >= p) { r -= p; q += 1; }
}
__device__ __forceinline__ void extract4_u64(U32 lo, U32 hi, U32 p, U32 m, U64 q4, U64 mq4,
                                             U32 &v0, U32 &v1, U32 &v2, U32 &v3) {
    U64 u = ((U64)hi << 32) | (U64)lo;
    U64 q = __umul64hi(u, mq4);
    U64 rem64 = u - q * q4;
    U32 rem = (U32)((rem64 >= q4) ? (rem64 - q4) : rem64);
    U32 t1, t2;
    fast_divmod_u32(rem, p, m, t1, v0);
    fast_divmod_u32(t1, p, m, t2, v1);
    fast_divmod_u32(t2, p, m, v3, v2);
}

extern "C" __global__ void __launch_bounds__(64) b3_poly_mask(
    const U8* __restrict__ h_eval,
    const U32* __restrict__ k0_ptr, const U32* __restrict__ k1_ptr, const U32* __restrict__ k2_ptr,
    const U64* __restrict__ nonces,
    U8* __restrict__ y_wire, U8* __restrict__ s_out,
    U64 n, U64 blocks_per_lane, int n_lanes, int aligned4,
    int party, int c0, int c1, int c2,
    const U32* __restrict__ primes_u32, const U64* __restrict__ params4
) {
    int lane = blockIdx.y;
    U32 p = primes_u32[lane];
    U32 m = (U32)params4[lane * 3 + 0];
    U64 q4 = params4[lane * 3 + 1], mq4 = params4[lane * 3 + 2];
    U64 blk = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= blocks_per_lane) return;
    U64 blk_global = (U64)lane * blocks_per_lane + blk;
    U64 base_elem = blk * 16;
    U64 lane_base = (U64)lane * n;
    U64 total_bytes = (U64)n_lanes * n;
    int shift = (int)p * 64;
    int full_vec = aligned4 && (base_elem + 16 <= n);

    __shared__ int ya[16][64];
    int tid = threadIdx.x;
    if (full_vec) {
        const U32* h32 = (const U32*)(h_eval + lane_base + base_elem);
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            U32 hw = h32[g];
            ya[g*4+0][tid] = (int)(hw & 0xFFu) + shift;
            ya[g*4+1][tid] = (int)((hw >> 8) & 0xFFu) + shift;
            ya[g*4+2][tid] = (int)((hw >> 16) & 0xFFu) + shift;
            ya[g*4+3][tid] = (int)((hw >> 24) & 0xFFu) + shift;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            U64 idx = base_elem + i;
            ya[i][tid] = (idx < n ? (int)h_eval[lane_base + idx] : 0) + shift;
        }
    }

    const U32* kptrs[3] = {k0_ptr, k1_ptr, k2_ptr};
    int coeffs[3] = {c0, c1, c2};
    for (int s = 0; s < 3; s++) {
        U32 w[16];
        blake3_xof_block(kptrs[s], nonces[s], blk_global, w);
        int cf = coeffs[s];
        U8* s_lane = s_out + (U64)s * total_bytes + lane_base;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            U32 r0, s0, r1, s1, r2, s2, r3, s3;
            extract4_u64(w[g*4+0], w[g*4+1], p, m, q4, mq4, r0, s0, r1, s1);
            extract4_u64(w[g*4+2], w[g*4+3], p, m, q4, mq4, r2, s2, r3, s3);
            ya[g*4+0][tid] += cf * ((int)r0 + party * (int)s0);
            ya[g*4+1][tid] += cf * ((int)r1 + party * (int)s1);
            ya[g*4+2][tid] += cf * ((int)r2 + party * (int)s2);
            ya[g*4+3][tid] += cf * ((int)r3 + party * (int)s3);
            U32 ns0 = s0 ? (p - s0) : 0u, ns1 = s1 ? (p - s1) : 0u;
            U32 ns2 = s2 ? (p - s2) : 0u, ns3 = s3 ? (p - s3) : 0u;
            if (full_vec) {
                ((U32*)(s_lane + base_elem))[g] = ns0 | (ns1 << 8) | (ns2 << 16) | (ns3 << 24);
            } else {
                U64 b0 = base_elem + g * 4;
                if (b0 + 0 < n) s_lane[b0 + 0] = (U8)ns0;
                if (b0 + 1 < n) s_lane[b0 + 1] = (U8)ns1;
                if (b0 + 2 < n) s_lane[b0 + 2] = (U8)ns2;
                if (b0 + 3 < n) s_lane[b0 + 3] = (U8)ns3;
            }
        }
    }
    if (full_vec) {
        U32* y32 = (U32*)(y_wire + lane_base + base_elem);
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            y32[g] = fast_mod_u32((U32)ya[g*4+0][tid], p, m)
                   | (fast_mod_u32((U32)ya[g*4+1][tid], p, m) << 8)
                   | (fast_mod_u32((U32)ya[g*4+2][tid], p, m) << 16)
                   | (fast_mod_u32((U32)ya[g*4+3][tid], p, m) << 24);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            U64 idx = base_elem + i;
            if (idx < n) y_wire[lane_base + idx] = (U8)fast_mod_u32((U32)ya[i][tid], p, m);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) b3_input_masks(
    const U32* __restrict__ k0_ptr, const U32* __restrict__ k1_ptr,
    const U64* __restrict__ nonces,
    U8* __restrict__ out0, U8* __restrict__ out1,
    U64 n, U64 blocks_per_lane, int aligned4, int has0, int has1,
    const U32* __restrict__ primes_u32, const U64* __restrict__ params4
) {
    int lane = blockIdx.y;
    U32 p = primes_u32[lane];
    U32 m = (U32)params4[lane * 3 + 0];
    U64 q4 = params4[lane * 3 + 1], mq4 = params4[lane * 3 + 2];
    U64 blk = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= blocks_per_lane) return;
    U64 blk_global = (U64)lane * blocks_per_lane + blk;
    U64 base_elem = blk * 32;
    U64 lane_base = (U64)lane * n;
    int full_vec = aligned4 && (base_elem + 32 <= n);

    for (int s = 0; s < 2; s++) {
        if ((s == 0 && !has0) || (s == 1 && !has1)) continue;
        U32 w[16];
        blake3_xof_block(s == 0 ? k0_ptr : k1_ptr, nonces[s], blk_global, w);
        U8* dst = (s == 0 ? out0 : out1) + lane_base;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            U32 v0, v1, v2, v3;
            extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, v0, v1, v2, v3);
            if (full_vec) {
                ((U32*)(dst + base_elem))[g] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
            } else {
                U64 b0 = base_elem + g * 4;
                if (b0 + 0 < n) dst[b0 + 0] = (U8)v0;
                if (b0 + 1 < n) dst[b0 + 1] = (U8)v1;
                if (b0 + 2 < n) dst[b0 + 2] = (U8)v2;
                if (b0 + 3 < n) dst[b0 + 3] = (U8)v3;
            }
        }
    }
}
__device__ __forceinline__ U32 sub_mod_u32(U32 a, U32 b, U32 p) {
    return a >= b ? (a - b) : (a + p - b);
}

__device__ __forceinline__ U32 add_mod_u32(U32 a, U32 b, U32 p) {
    U32 s = a + b;
    return s >= p ? (s - p) : s;
}

extern "C" __global__ void __launch_bounds__(128) b3_joint_input_sub(
    const U8* __restrict__ src_in,
    const U32* __restrict__ k_sub, const U32* __restrict__ k_aux,
    const U64* __restrict__ nonces, int sub_nonce_idx,
    U8* __restrict__ m_sub_out, U8* __restrict__ wire_out, U8* __restrict__ m_aux_out,
    U64 n, U64 blocks_per_lane, int aligned4, int has_aux,
    const U32* __restrict__ primes_u32, const U64* __restrict__ params4
) {
    int lane = blockIdx.y;
    U32 p = primes_u32[lane];
    U32 m = (U32)params4[lane * 3 + 0];
    U64 q4 = params4[lane * 3 + 1], mq4 = params4[lane * 3 + 2];
    U64 blk = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= blocks_per_lane) return;
    U64 blk_global = (U64)lane * blocks_per_lane + blk;
    U64 base_elem = blk * 32;
    U64 lane_base = (U64)lane * n;
    int full_vec = aligned4 && (base_elem + 32 <= n);

    U32 w[16];
    blake3_xof_block(k_sub, nonces[sub_nonce_idx], blk_global, w);
    const U8* src_lane = src_in + lane_base;
    U8* msub_lane = m_sub_out + lane_base;
    U8* wire_lane = wire_out + lane_base;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        U32 v0, v1, v2, v3;
        extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, v0, v1, v2, v3);
        if (full_vec) {
            ((U32*)(msub_lane + base_elem))[g] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
            U32 sw = ((const U32*)(src_lane + base_elem))[g];
            U32 d0 = sub_mod_u32(sw & 0xFFu, v0, p);
            U32 d1 = sub_mod_u32((sw >> 8) & 0xFFu, v1, p);
            U32 d2 = sub_mod_u32((sw >> 16) & 0xFFu, v2, p);
            U32 d3 = sub_mod_u32((sw >> 24) & 0xFFu, v3, p);
            ((U32*)(wire_lane + base_elem))[g] = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);
        } else {
            U64 b0 = base_elem + g * 4;
            U32 vs[4] = {v0, v1, v2, v3};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                if (b0 + k < n) {
                    msub_lane[b0 + k] = (U8)vs[k];
                    wire_lane[b0 + k] = (U8)sub_mod_u32((U32)src_lane[b0 + k], vs[k], p);
                }
            }
        }
    }
    if (has_aux) {
        blake3_xof_block(k_aux, nonces[1 - sub_nonce_idx], blk_global, w);
        U8* aux_lane = m_aux_out + lane_base;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            U32 v0, v1, v2, v3;
            extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, v0, v1, v2, v3);
            if (full_vec) {
                ((U32*)(aux_lane + base_elem))[g] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
            } else {
                U64 b0 = base_elem + g * 4;
                if (b0 + 0 < n) aux_lane[b0 + 0] = (U8)v0;
                if (b0 + 1 < n) aux_lane[b0 + 1] = (U8)v1;
                if (b0 + 2 < n) aux_lane[b0 + 2] = (U8)v2;
                if (b0 + 3 < n) aux_lane[b0 + 3] = (U8)v3;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) b3_comp_mul_prep(
    const U8* __restrict__ c_in, const U8* __restrict__ xs_in, const U8* __restrict__ xh_in,
    const U32* __restrict__ k0_ptr, const U32* __restrict__ k1_ptr, const U32* __restrict__ k2_ptr,
    const U64* __restrict__ nonces_all,
    U8* __restrict__ out0, U8* __restrict__ out1, U8* __restrict__ out2,
    U8* __restrict__ w_send, U8* __restrict__ w_hash,
    U64 n, U64 src_lane_stride, U64 src_slice_off, U64 blocks_per_lane,
    int aligned4, int party,
    const U32* __restrict__ primes_u32, const U64* __restrict__ params4
) {
    int lane = blockIdx.y;
    U32 p = primes_u32[lane];
    U32 m = (U32)params4[lane * 3 + 0];
    U64 q4 = params4[lane * 3 + 1], mq4 = params4[lane * 3 + 2];
    U64 blk = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= blocks_per_lane) return;
    U64 blk_global = (U64)lane * blocks_per_lane + blk;
    U64 base_elem = blk * 32;
    U64 dst_lane = (U64)lane * n;
    U64 src_lane = (U64)lane * src_lane_stride + src_slice_off;
    int full_vec = aligned4 && (base_elem + 32 <= n);

    U32 w[16];
    U32 m0_p[8], m1_p[8];
    blake3_xof_block(k0_ptr, nonces_all[(party + 1) & 3], blk_global, w);
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        U32 v0, v1, v2, v3;
        extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, v0, v1, v2, v3);
        m0_p[g] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
    }
    blake3_xof_block(k1_ptr, nonces_all[(party + 2) & 3], blk_global, w);
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        U32 v0, v1, v2, v3;
        extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, v0, v1, v2, v3);
        m1_p[g] = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
    }
    blake3_xof_block(k2_ptr, nonces_all[(party + 3) & 3], blk_global, w);

    const U8* c_ptr = c_in + src_lane;
    const U8* xs_ptr = xs_in + src_lane;
    const U8* xh_ptr = xh_in + src_lane;
    U8* o0_ptr = out0 + dst_lane;
    U8* o1_ptr = out1 + dst_lane;
    U8* o2_ptr = out2 + dst_lane;
    U8* ws_ptr = w_send + dst_lane;
    U8* wh_ptr = w_hash + dst_lane;

    #pragma unroll
    for (int g = 0; g < 8; g++) {
        U32 mv2[4];
        extract4_u64(w[g*2], w[g*2+1], p, m, q4, mq4, mv2[0], mv2[1], mv2[2], mv2[3]);
        U32 m0w = m0_p[g], m1w = m1_p[g];
        U32 cw = 0u, xsw = 0u, xhw = 0u;
        if (full_vec) {
            cw = ((const U32*)(c_ptr + base_elem))[g];
            xsw = ((const U32*)(xs_ptr + base_elem))[g];
            xhw = ((const U32*)(xh_ptr + base_elem))[g];
        }
        U32 po0 = 0u, po1 = 0u, po2 = 0u, pws = 0u, pwh = 0u;
        U64 b0 = base_elem + g * 4;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int sh = k * 8;
            U32 a0 = (m0w >> sh) & 0xFFu;
            U32 a1 = (m1w >> sh) & 0xFFu;
            U32 a2 = mv2[k];
            U32 cv, xsv, xhv;
            if (full_vec) {
                cv = (cw >> sh) & 0xFFu;
                xsv = (xsw >> sh) & 0xFFu;
                xhv = (xhw >> sh) & 0xFFu;
            } else if (b0 + k < n) {
                cv = (U32)c_ptr[b0 + k];
                xsv = (U32)xs_ptr[b0 + k];
                xhv = (U32)xh_ptr[b0 + k];
            } else {
                continue;
            }
            U32 ts = fast_mod_u32(xsv * cv, p, m);
            U32 th = fast_mod_u32(xhv * cv, p, m);
            U32 ws, wh, r0, r1, r2;
            if (party == 0) {
                ws = sub_mod_u32(ts, a2, p);
                wh = sub_mod_u32(th, a0, p);
                r0 = add_mod_u32(a0, ws, p);
                r1 = add_mod_u32(a1, wh, p);
                r2 = a2;
            } else if (party == 1) {
                ws = sub_mod_u32(ts, a0, p);
                wh = sub_mod_u32(th, a2, p);
                r0 = a0;
                r1 = add_mod_u32(a1, wh, p);
                r2 = add_mod_u32(a2, ws, p);
            } else if (party == 2) {
                ws = sub_mod_u32(ts, a1, p);
                wh = sub_mod_u32(th, a0, p);
                r0 = add_mod_u32(a0, ws, p);
                r1 = a1;
                r2 = add_mod_u32(a2, wh, p);
            } else {
                ws = sub_mod_u32(ts, a1, p);
                wh = sub_mod_u32(th, a2, p);
                r0 = add_mod_u32(a0, wh, p);
                r1 = a1;
                r2 = add_mod_u32(a2, ws, p);
            }
            if (full_vec) {
                po0 |= r0 << sh;
                po1 |= r1 << sh;
                po2 |= r2 << sh;
                pws |= ws << sh;
                pwh |= wh << sh;
            } else {
                o0_ptr[b0 + k] = (U8)r0;
                o1_ptr[b0 + k] = (U8)r1;
                o2_ptr[b0 + k] = (U8)r2;
                ws_ptr[b0 + k] = (U8)ws;
                wh_ptr[b0 + k] = (U8)wh;
            }
        }
        if (full_vec) {
            ((U32*)(o0_ptr + base_elem))[g] = po0;
            ((U32*)(o1_ptr + base_elem))[g] = po1;
            ((U32*)(o2_ptr + base_elem))[g] = po2;
            ((U32*)(ws_ptr + base_elem))[g] = pws;
            ((U32*)(wh_ptr + base_elem))[g] = pwh;
        }
    }
}
'''
    )


@lru_cache(None)
def _compile_b3_prg(major: int, minor: int) -> bytes:
    nvrtc = ct.CDLL(ct.util.find_library('nvrtc') or '/usr/local/cuda/lib64/libnvrtc.so')
    program = ct.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ct.byref(program), _blake3_xof_cuda_source().encode(),
                                    b'crt_b3_prg.cu', 0, None, None), 'nvrtcCreateProgram')
    options = (ct.c_char_p * 2)(f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++17')
    status = nvrtc.nvrtcCompileProgram(program, 2, options)
    size = ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
    log = ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if status:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
        raise RuntimeError(f'CRT BLAKE3 XOF NVRTC compilation failed: {log.value.decode()}')
    _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), 'nvrtcGetPTXSize')
    ptx = ct.create_string_buffer(size.value)
    _check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx.raw


@lru_cache(None)
def _b3_prg_module(device_index: int):
    with torch.cuda.device(device_index):
        driver = ct.CDLL(ct.util.find_library('cuda') or 'libcuda.so.1')
        context = ct.c_void_p()
        _check(driver.cuDevicePrimaryCtxRetain(ct.byref(context), ct.c_int(device_index)),
               'cuDevicePrimaryCtxRetain')
        with _current_context(driver, context):
            torch.cuda.current_stream()
            ptx = ct.create_string_buffer(_compile_b3_prg(*torch.cuda.get_device_capability(device_index)))
            module = ct.c_void_p()
            _check(driver.cuModuleLoadData(ct.byref(module), ptx), 'cuModuleLoadData')
            functions = {}
            for name in ('b3_poly_mask', 'b3_input_masks', 'b3_joint_input_sub', 'b3_comp_mul_prep'):
                fn = ct.c_void_p()
                _check(driver.cuModuleGetFunction(ct.byref(fn), module, name.encode()), 'cuModuleGetFunction')
                functions[name] = fn
        return driver, module, functions, context


@lru_cache(None)
def _prime_barrett_tables(device_index: int, primes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device('cuda', device_index)
    primes_u32 = torch.tensor(primes, dtype=torch.uint32, device=dev)
    params = []
    for p in primes:
        q4 = p ** 4
        params.extend(((1 << 32) // p, q4, (1 << 64) // q4))
    params4 = torch.tensor(params, dtype=torch.uint64, device=dev)
    return primes_u32, params4


def _launch_b3_prg(driver, fn, grid_x: int, grid_y: int, block_x: int, parameters, device, context):
    params = (ct.c_void_p * len(parameters))(*(ct.cast(ct.byref(p), ct.c_void_p) for p in parameters))
    with _current_context(driver, context):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        _check(driver.cuLaunchKernel(fn, grid_x, grid_y, 1, block_x, 1, 1, 0, stream, params, None),
               'CRT BLAKE3 PRG cuLaunchKernel')


@triton.jit
def _decode_check_vec4_kernel(Y0, Y1, Y2, Y3, SLOT0, C_OUT, ERR_FLAG,
                              N4: tl.constexpr,
                              P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                              BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    inv2 = (p + 1) >> 1
    shift = p * 8
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N4
    off = lane * N4 + idx
    w0 = tl.load(Y0 + off, valid, 0)
    w1 = tl.load(Y1 + off, valid, 0)
    w2 = tl.load(Y2 + off, valid, 0)
    w3 = tl.load(Y3 + off, valid, 0)
    ws = tl.load(SLOT0 + off, valid, 0)
    c_pack = tl.zeros((BLOCK,), dtype=tl.uint32)
    s_pack = tl.zeros((BLOCK,), dtype=tl.uint32)
    bad_acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for sub in tl.static_range(4):
        sh = sub * 8
        y0 = ((w0 >> sh) & 0xFF).to(tl.int32)
        y1 = ((w1 >> sh) & 0xFF).to(tl.int32)
        y2 = ((w2 >> sh) & 0xFF).to(tl.int32)
        y3 = ((w3 >> sh) & 0xFF).to(tl.int32)
        diff3 = _fast_mod_u32((y3 - 3 * y2 + 3 * y1 - y0 + shift).to(tl.uint32), p_u32, m_u32)
        bad_acc += tl.where(valid & (diff3 != 0), 1, 0)
        c_val = _fast_mod_u32(((y2 - 2 * y1 + y0 + shift) * inv2).to(tl.uint32), p_u32, m_u32)
        c_pack |= c_val << sh
        cur = ((ws >> sh) & 0xFF)
        upd = cur + c_val
        upd = tl.where(upd >= p_u32, upd - p_u32, upd)
        s_pack |= upd << sh
    if tl.sum(bad_acc, axis=0) != 0:
        tl.atomic_or(ERR_FLAG, 1)
    tl.store(C_OUT + off, c_pack, valid)
    tl.store(SLOT0 + off, s_pack, valid)


@triton.jit
def _decode_check_kernel(Y0, Y1, Y2, Y3, SLOT0, C_OUT, ERR_FLAG,
                         N: tl.constexpr,
                         P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                         BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    inv2 = (p + 1) >> 1
    shift = p * 8
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    off = lane * N + idx
    y0 = tl.load(Y0 + off, valid, 0).to(tl.int32)
    y1 = tl.load(Y1 + off, valid, 0).to(tl.int32)
    y2 = tl.load(Y2 + off, valid, 0).to(tl.int32)
    y3 = tl.load(Y3 + off, valid, 0).to(tl.int32)
    diff3 = _fast_mod_u32((y3 - 3 * y2 + 3 * y1 - y0 + shift).to(tl.uint32), p_u32, m_u32)
    bad = tl.sum(tl.where(valid & (diff3 != 0), 1, 0), axis=0)
    if bad != 0:
        tl.atomic_or(ERR_FLAG, 1)
    c_val = _fast_mod_u32(((y2 - 2 * y1 + y0 + shift) * inv2).to(tl.uint32), p_u32, m_u32)
    tl.store(C_OUT + off, c_val.to(tl.uint8), valid)
    cur = tl.load(SLOT0 + off, valid, 0).to(tl.uint32)
    updated = cur + c_val
    updated = tl.where(updated >= p_u32, updated - p_u32, updated).to(tl.uint8)
    tl.store(SLOT0 + off, updated, valid)


@triton.jit
def _add_mod_vec4_kernel(A, B, OUT, N4: tl.constexpr,
                         P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                         BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p = _lane_prime(lane, P0, P1, P2, P3).to(tl.uint32)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N4
    off = lane * N4 + idx
    wa = tl.load(A + off, valid, 0)
    wb = tl.load(B + off, valid, 0)
    packed = tl.zeros((BLOCK,), dtype=tl.uint32)
    for sub in tl.static_range(4):
        sh = sub * 8
        s = ((wa >> sh) & 0xFF) + ((wb >> sh) & 0xFF)
        s = tl.where(s >= p, s - p, s)
        packed |= s << sh
    tl.store(OUT + off, packed, valid)


@triton.jit
def _add_mod_kernel(A, B, OUT, N: tl.constexpr, B_STRIDE: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                    BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p = _lane_prime(lane, P0, P1, P2, P3)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    a = tl.load(A + lane * N + idx, valid, 0).to(tl.int32)
    b = tl.load(B + lane * B_STRIDE + (idx % B_STRIDE), valid, 0).to(tl.int32)
    s = a + b
    s = tl.where(s >= p, s - p, s).to(tl.uint8)
    tl.store(OUT + lane * N + idx, s, valid)


@triton.jit
def _input_corr_i8_vec4_kernel(VAL_U32, M0_U32, M2_U32, OUT_U32, N4: tl.constexpr,
                               P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                               BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p_u32, m_u32 = _lane_barrett_u32(lane, P0, P1, P2, P3)
    p = p_u32.to(tl.int32)
    shift = p * 4
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N4
    off = lane * N4 + idx
    vw = tl.load(VAL_U32 + idx, valid, 0)
    a_w = tl.load(M0_U32 + off, valid, 0)
    b_w = tl.load(M2_U32 + off, valid, 0)
    packed = tl.zeros((BLOCK,), dtype=tl.uint32)
    for sub in tl.static_range(4):
        sh = sub * 8
        # Sign-extend byte to int32 via arithmetic shift.
        v = ((vw << (24 - sh)).to(tl.int32)) >> 24
        a = ((a_w >> sh) & 0xFF).to(tl.int32)
        b = ((b_w >> sh) & 0xFF).to(tl.int32)
        r = _fast_mod_u32((v - a - b + shift).to(tl.uint32), p_u32, m_u32)
        packed |= r << sh
    tl.store(OUT_U32 + off, packed, valid)


@triton.jit
def _input_corr_kernel(VAL, M0, M2, OUT, N: tl.constexpr,
                       P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
                       BLOCK: tl.constexpr):
    lane = tl.program_id(1)
    p = _lane_prime(lane, P0, P1, P2, P3).to(tl.int64)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    off = lane * N + idx
    v = tl.load(VAL + idx, valid, 0).to(tl.int64)
    a = tl.load(M0 + off, valid, 0).to(tl.int64)
    b = tl.load(M2 + off, valid, 0).to(tl.int64)
    r = (v - a - b) % p
    r = tl.where(r < 0, r + p, r).to(tl.uint8)
    tl.store(OUT + off, r, valid)


def crt_eval_linear(parts: tuple[torch.Tensor, ...], party: int, primes: tuple[int, ...],
                    *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute centered int8 polynomial evaluation F(party) = sum_s (party - owner_s) * X_s mod p."""
    primes = validate_primes(primes)
    c0, c1, c2 = _party_coeffs(party)
    p0, p1, p2, p3 = _pad_primes(primes)
    n_lanes = len(primes)
    if len(parts) != 3 or any(t.dtype != torch.uint8 or not t.is_cuda or t.shape != parts[0].shape for t in parts):
        raise ValueError('CRT evaluation requires three matching uint8 CUDA tensors')
    if parts[0].shape[0] != n_lanes:
        raise ValueError('first dimension must match CRT prime count')
    xs = tuple(t.contiguous() for t in parts)
    n = xs[0].numel() // n_lanes
    if out is None:
        out = torch.empty(xs[0].shape, dtype=torch.int8, device=xs[0].device)
    elif out.shape != xs[0].shape or out.dtype != torch.int8 or out.device != xs[0].device or not out.is_contiguous():
        raise ValueError('crt_eval_linear output buffer must be contiguous int8 matching share shape and device')
    if n:
        with torch.cuda.device(out.device):
            if n % 4 == 0:
                n4 = n // 4
                _eval_linear_vec4_kernel[(triton.cdiv(n4, 256), n_lanes)](
                    *(x.reshape(-1).view(torch.uint32) for x in xs), out.reshape(-1).view(torch.uint32),
                    n4, c0, c1, c2, p0, p1, p2, p3, 256, num_warps=4)
            else:
                _eval_linear_kernel[(triton.cdiv(n, 256), n_lanes)](
                    *xs, out, n, c0, c1, c2, p0, p1, p2, p3, 256, num_warps=4)
    return out


def prepare_crt_weight(parts: tuple[torch.Tensor, ...], party: int, primes: tuple[int, ...]) -> PreparedCrtWeight:
    """Precompute centered int8 G(party) once for a static secret-shared weight tensor."""
    g_eval = crt_eval_linear(parts, party, primes)
    return PreparedCrtWeight(g_eval=g_eval, primes=validate_primes(primes),
                             party=party, original_shape=tuple(parts[0].shape[1:]))


def crt_conv1d_local(fx: torch.Tensor, gw: PreparedCrtWeight) -> torch.Tensor:
    """Compute exact 1D convolution FX * GW mod p_l without materializing im2col."""
    primes = gw.primes
    n_lanes = len(primes)
    if fx.dtype != torch.int8 or not fx.is_cuda or fx.ndim != 4 or fx.shape[0] != n_lanes:
        raise ValueError('FX must be a 4D int8 CUDA tensor (L_primes, B, L_in, C_in)')
    if gw.g_eval.ndim != 4 or gw.g_eval.shape[0] != n_lanes or gw.g_eval.shape[2] != fx.shape[3]:
        raise ValueError('GW must have shape (L_primes, k, C_in, C_out) matching FX')
    _, batch, l_in, c_in = fx.shape
    _, k_size, _, c_out = gw.g_eval.shape
    l_out = l_in - k_size + 1
    if l_out <= 0 or batch <= 0 or c_out <= 0:
        raise ValueError('invalid Conv1d dimensions')
    max_half = (max(primes) - 1) // 2
    if k_size * c_in * max_half * max_half >= 2**31:
        raise ValueError('Conv1d dot product could overflow int32 accumulator')
    fx = fx.contiguous()
    w = gw.g_eval.contiguous()
    out = torch.empty((n_lanes, batch, l_out, c_out), dtype=torch.uint8, device=fx.device)
    p0, p1, p2, p3 = _pad_primes(primes)
    m = batch * l_out
    bm, bn, bk, nw, ns = 128, 128, 64, 4, 4
    with torch.cuda.device(fx.device):
        _conv1d_eval_kernel[(triton.cdiv(m, bm), triton.cdiv(c_out, bn), n_lanes)](
            fx, w, out, batch, l_in, l_out, c_in, c_out, k_size,
            p0, p1, p2, p3, bm, bn, bk, num_warps=nw, num_stages=ns)
    return out


def crt_matmul_local(fx: torch.Tensor, gw: PreparedCrtWeight) -> torch.Tensor:
    """Compute exact matrix product FX @ GW mod p_l for (L_primes, M, K) @ (L_primes, K, N)."""
    primes = gw.primes
    n_lanes = len(primes)
    if fx.dtype != torch.int8 or not fx.is_cuda or fx.ndim != 3 or fx.shape[0] != n_lanes:
        raise ValueError('FX must be a 3D int8 CUDA tensor (L_primes, M, K)')
    if gw.g_eval.ndim != 3 or gw.g_eval.shape[0] != n_lanes or gw.g_eval.shape[1] != fx.shape[2]:
        raise ValueError('GW must have shape (L_primes, K, N) matching FX')
    _, m, k = fx.shape
    _, _, n = gw.g_eval.shape
    max_half = (max(primes) - 1) // 2
    if k * max_half * max_half >= 2**31:
        raise ValueError('matmul dot product could overflow int32 accumulator')
    fx = fx.contiguous()
    w = gw.g_eval.contiguous()
    out = torch.empty((n_lanes, m, n), dtype=torch.uint8, device=fx.device)
    p0, p1, p2, p3 = _pad_primes(primes)
    with torch.cuda.device(fx.device):
        if n == 1:
            _matvec_eval_kernel[(triton.cdiv(m, 64), n_lanes)](
                fx, w.reshape(n_lanes, k), out.reshape(n_lanes, m),
                m, k, p0, p1, p2, p3, 64, 64, num_warps=4)
        else:
            bm, bn, bk = (128, 128, 64)
            _matmul_eval_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), n_lanes)](
                fx, w, out, m, k, n, p0, p1, p2, p3, bm, bn, bk, num_warps=4, num_stages=4)
    return out


def crt_square_sum_pool_local(parts: tuple[torch.Tensor, ...], party: int,
                              primes: tuple[int, ...],
                              bias_parts: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
    """Compute sum_t (F_H(party)[b, t, c] + F_b(party)[c])^2 mod p_l directly into (L_primes, B, C)."""
    primes = validate_primes(primes)
    c0, c1, c2 = _party_coeffs(party)
    p0, p1, p2, p3 = _pad_primes(primes)
    n_lanes = len(primes)
    if len(parts) != 3 or any(t.dtype != torch.uint8 or not t.is_cuda or t.ndim != 4 or t.shape != parts[0].shape for t in parts):
        raise ValueError('square-sum-pool requires three matching 4D uint8 CUDA tensors (L_primes, B, L_out, C)')
    _, batch, l_out, channels = parts[0].shape
    if l_out * (max(primes) - 1) ** 2 >= 2**32:
        raise ValueError('square-sum-pool sequence length could overflow uint32 accumulator')
    hs = tuple(t.contiguous() for t in parts)
    has_bias = bias_parts is not None
    if has_bias:
        if len(bias_parts) != 3 or any(t.dtype != torch.uint8 or t.shape != (n_lanes, channels) for t in bias_parts):
            raise ValueError('square-sum-pool bias must be three uint8 tensors of shape (L_primes, C)')
        bs = tuple(t.contiguous() for t in bias_parts)
    else:
        bs = hs
    out = torch.empty((n_lanes, batch, channels), dtype=torch.uint8, device=hs[0].device)
    total = batch * channels
    if total:
        with torch.cuda.device(out.device):
            _square_sum_pool_kernel[(triton.cdiv(total, 256), n_lanes)](
                *hs, *bs, out, batch, l_out, channels, has_bias,
                c0, c1, c2, p0, p1, p2, p3, 256, num_warps=4)
    return out


def crt_poly_mask(h_eval: torch.Tensor, keys: tuple[torch.Tensor, ...], nonces: torch.Tensor,
                  party: int, primes: tuple[int, ...], *,
                  y_wire_out: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Fuse official BLAKE3-256 keyed XOF mask generation with polynomial masking and output shares."""
    primes = validate_primes(primes)
    c0, c1, c2 = _party_coeffs(party)
    n_lanes = len(primes)
    if h_eval.dtype != torch.uint8 or not h_eval.is_cuda or h_eval.shape[0] != n_lanes:
        raise ValueError('h_eval must be a uint8 CUDA tensor with leading prime dimension')
    if len(keys) != 3 or nonces.shape != (3,) or nonces.dtype != torch.uint64 or nonces.device != h_eval.device:
        raise ValueError('crt_poly_mask requires three 32-byte keys and a (3,) uint64 CUDA nonce vector')
    h_flat = h_eval.contiguous()
    shape = h_flat.shape
    n = h_flat.numel() // n_lanes
    blocks_per_lane = (n + 15) // 16
    if n_lanes * blocks_per_lane >= 2**63:
        raise ValueError('CRT mask draw exceeds BLAKE3 XOF counter limit')
    if y_wire_out is None:
        y_wire = torch.empty(h_flat.numel(), dtype=torch.uint8, device=h_eval.device)
    else:
        if y_wire_out.dtype != torch.uint8 or y_wire_out.device != h_eval.device or y_wire_out.numel() != h_flat.numel() or not y_wire_out.is_contiguous():
            raise ValueError('y_wire_out must be a contiguous uint8 CUDA buffer matching h_eval size')
        y_wire = y_wire_out.reshape(-1)
    s_out = torch.empty((3, *shape), dtype=torch.uint8, device=h_eval.device)
    if n:
        with torch.cuda.device(h_eval.device):
            dev_idx = h_eval.device.index
            driver, _, functions, context = _b3_prg_module(dev_idx)
            primes_u32, params4 = _prime_barrett_tables(dev_idx, primes)
            _launch_b3_prg(
                driver, functions['b3_poly_mask'], (blocks_per_lane + 63) // 64, n_lanes, 64,
                [ct.c_void_p(h_flat.data_ptr()), ct.c_void_p(keys[0].data_ptr()),
                 ct.c_void_p(keys[1].data_ptr()), ct.c_void_p(keys[2].data_ptr()),
                 ct.c_void_p(nonces.data_ptr()), ct.c_void_p(y_wire.data_ptr()),
                 ct.c_void_p(s_out.data_ptr()), ct.c_uint64(n), ct.c_uint64(blocks_per_lane),
                 ct.c_int(n_lanes), ct.c_int(int(n % 4 == 0)),
                 ct.c_int(party), ct.c_int(c0), ct.c_int(c1), ct.c_int(c2),
                 ct.c_void_p(primes_u32.data_ptr()), ct.c_void_p(params4.data_ptr())],
                h_eval.device, context,
            )
    return y_wire.reshape(-1), tuple(s_out.unbind(0))


def crt_input_masks(keys: tuple[torch.Tensor | None, ...], nonces: torch.Tensor,
                    primes: tuple[int, ...], shape: tuple[int, ...], device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Generate 1 or 2 uniform mod-p_l CUDA uint8 mask tensors via official BLAKE3-256 keyed XOF."""
    primes = validate_primes(primes)
    n_lanes = len(primes)
    n = math.prod(shape)
    has0, has1 = keys[0] is not None, keys[1] is not None
    out0 = torch.empty((n_lanes, *shape), dtype=torch.uint8, device=device) if has0 else None
    out1 = torch.empty((n_lanes, *shape), dtype=torch.uint8, device=device) if has1 else None
    if n and (has0 or has1):
        blocks_per_lane = (n + 31) // 32
        if n_lanes * blocks_per_lane >= 2**63:
            raise ValueError('CRT input mask draw exceeds BLAKE3 XOF counter limit')
        dummy = keys[0] if has0 else keys[1]
        dummy_out = out0 if has0 else out1
        dev = torch.device(device)
        with torch.cuda.device(dev):
            dev_idx = torch.cuda.current_device() if dev.index is None else dev.index
            driver, _, functions, context = _b3_prg_module(dev_idx)
            primes_u32, params4 = _prime_barrett_tables(dev_idx, primes)
            _launch_b3_prg(
                driver, functions['b3_input_masks'], (blocks_per_lane + 127) // 128, n_lanes, 128,
                [ct.c_void_p((keys[0] if has0 else dummy).data_ptr()),
                 ct.c_void_p((keys[1] if has1 else dummy).data_ptr()),
                 ct.c_void_p(nonces.data_ptr()),
                 ct.c_void_p((out0 if has0 else dummy_out).data_ptr()),
                 ct.c_void_p((out1 if has1 else dummy_out).data_ptr()),
                 ct.c_uint64(n), ct.c_uint64(blocks_per_lane), ct.c_int(int(n % 4 == 0)),
                 ct.c_int(int(has0)), ct.c_int(int(has1)),
                 ct.c_void_p(primes_u32.data_ptr()), ct.c_void_p(params4.data_ptr())],
                dev, context,
            )
    return out0, out1


def crt_joint_input_sub(src_in: torch.Tensor, k_sub: torch.Tensor, k_aux: torch.Tensor | None,
                        nonces: torch.Tensor, sub_nonce_idx: int, primes: tuple[int, ...],
                        wire_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused BLAKE3 mask draw + (src_in - m_sub) mod p_l + optional auxiliary mask draw."""
    primes = validate_primes(primes)
    n_lanes = len(primes)
    src = src_in.contiguous()
    shape = src.shape
    if src.dtype != torch.uint8 or not src.is_cuda or shape[0] != n_lanes:
        raise ValueError('crt_joint_input_sub requires uint8 CUDA src_in with leading prime dimension')
    if wire_out.dtype != torch.uint8 or wire_out.device != src.device or wire_out.numel() != src.numel() or not wire_out.is_contiguous():
        raise ValueError('wire_out must be a contiguous uint8 CUDA buffer matching src_in')
    n = src.numel() // n_lanes
    has_aux = k_aux is not None
    m_sub = torch.empty(shape, dtype=torch.uint8, device=src.device)
    m_aux = torch.empty(shape, dtype=torch.uint8, device=src.device) if has_aux else None
    if n:
        blocks_per_lane = (n + 31) // 32
        if n_lanes * blocks_per_lane >= 2**63:
            raise ValueError('CRT joint input mask draw exceeds BLAKE3 XOF counter limit')
        aligned4 = int(
            n % 4 == 0 and src.data_ptr() % 4 == 0 and m_sub.data_ptr() % 4 == 0
            and wire_out.data_ptr() % 4 == 0 and (not has_aux or m_aux.data_ptr() % 4 == 0)
        )
        with torch.cuda.device(src.device):
            dev_idx = src.device.index
            driver, _, functions, context = _b3_prg_module(dev_idx)
            primes_u32, params4 = _prime_barrett_tables(dev_idx, primes)
            _launch_b3_prg(
                driver, functions['b3_joint_input_sub'], (blocks_per_lane + 127) // 128, n_lanes, 128,
                [ct.c_void_p(src.data_ptr()),
                 ct.c_void_p(k_sub.data_ptr()),
                 ct.c_void_p((k_aux if has_aux else k_sub).data_ptr()),
                 ct.c_void_p(nonces.data_ptr()), ct.c_int(sub_nonce_idx),
                 ct.c_void_p(m_sub.data_ptr()), ct.c_void_p(wire_out.data_ptr()),
                 ct.c_void_p((m_aux if has_aux else m_sub).data_ptr()),
                 ct.c_uint64(n), ct.c_uint64(blocks_per_lane),
                 ct.c_int(aligned4), ct.c_int(int(has_aux)),
                 ct.c_void_p(primes_u32.data_ptr()), ct.c_void_p(params4.data_ptr())],
                src.device, context,
            )
    return m_sub, m_aux


def crt_comp_mul_prep(c_in: torch.Tensor, xs_in: torch.Tensor, xh_in: torch.Tensor,
                      keys: tuple[torch.Tensor, ...], nonces_all: torch.Tensor,
                      party: int, primes: tuple[int, ...], logical_shape: tuple[int, ...],
                      w_send: torch.Tensor, *,
                      src_lane_stride: int | None = None,
                      src_slice_off: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused 3-mask BLAKE3 XOF + 4-joint complementary CRT product preparation (zero mask VRAM)."""
    primes = validate_primes(primes)
    n_lanes = len(primes)
    n = math.prod(logical_shape)
    dev = c_in.device
    if src_lane_stride is None:
        src_lane_stride = n
    if not (c_in.is_contiguous() and xs_in.is_contiguous() and xh_in.is_contiguous()):
        raise ValueError('crt_comp_mul_prep inputs must be contiguous uint8 CUDA tensors')
    if w_send.dtype != torch.uint8 or w_send.device != dev or w_send.numel() != n_lanes * n or not w_send.is_contiguous():
        raise ValueError('w_send must be a contiguous uint8 CUDA buffer matching output size')
    out0 = torch.empty((n_lanes, *logical_shape), dtype=torch.uint8, device=dev)
    out1 = torch.empty((n_lanes, *logical_shape), dtype=torch.uint8, device=dev)
    out2 = torch.empty((n_lanes, *logical_shape), dtype=torch.uint8, device=dev)
    w_hash = torch.empty((n_lanes, *logical_shape), dtype=torch.uint8, device=dev)
    if n:
        blocks_per_lane = (n + 31) // 32
        if n_lanes * blocks_per_lane >= 2**63:
            raise ValueError('CRT comp_mul mask draw exceeds BLAKE3 XOF counter limit')
        aligned4 = int(
            n % 4 == 0 and src_lane_stride % 4 == 0 and src_slice_off % 4 == 0
            and all(t.data_ptr() % 4 == 0 for t in (c_in, xs_in, xh_in, out0, out1, out2, w_send, w_hash))
        )
        with torch.cuda.device(dev):
            dev_idx = dev.index
            driver, _, functions, context = _b3_prg_module(dev_idx)
            primes_u32, params4 = _prime_barrett_tables(dev_idx, primes)
            _launch_b3_prg(
                driver, functions['b3_comp_mul_prep'], (blocks_per_lane + 127) // 128, n_lanes, 128,
                [ct.c_void_p(c_in.data_ptr()), ct.c_void_p(xs_in.data_ptr()), ct.c_void_p(xh_in.data_ptr()),
                 ct.c_void_p(keys[0].data_ptr()), ct.c_void_p(keys[1].data_ptr()), ct.c_void_p(keys[2].data_ptr()),
                 ct.c_void_p(nonces_all.data_ptr()),
                 ct.c_void_p(out0.data_ptr()), ct.c_void_p(out1.data_ptr()), ct.c_void_p(out2.data_ptr()),
                 ct.c_void_p(w_send.data_ptr()), ct.c_void_p(w_hash.data_ptr()),
                 ct.c_uint64(n), ct.c_uint64(src_lane_stride), ct.c_uint64(src_slice_off),
                 ct.c_uint64(blocks_per_lane), ct.c_int(aligned4), ct.c_int(party),
                 ct.c_void_p(primes_u32.data_ptr()), ct.c_void_p(params4.data_ptr())],
                dev, context,
            )
    return out0, out1, out2, w_hash


def crt_input_correction(value: torch.Tensor, m0: torch.Tensor, m2: torch.Tensor,
                         primes: tuple[int, ...]) -> torch.Tensor:
    """Compute (value - m0 - m2) mod p_l directly into uint8 (L_primes, *shape)."""
    primes = validate_primes(primes)
    p0, p1, p2, p3 = _pad_primes(primes)
    n_lanes = len(primes)
    n = value.numel()
    out = torch.empty((n_lanes, *value.shape), dtype=torch.uint8, device=value.device)
    if n:
        v = value.contiguous().reshape(-1)
        a = m0.contiguous().reshape(-1)
        b = m2.contiguous().reshape(-1)
        with torch.cuda.device(value.device):
            if v.dtype == torch.int8 and n % 4 == 0:
                n4 = n // 4
                _input_corr_i8_vec4_kernel[(triton.cdiv(n4, 256), n_lanes)](
                    v.view(torch.uint32), a.view(torch.uint32), b.view(torch.uint32),
                    out.reshape(-1).view(torch.uint32), n4, p0, p1, p2, p3, 256, num_warps=4)
            else:
                _input_corr_kernel[(triton.cdiv(n, 256), n_lanes)](
                    v, a, b, out.reshape(-1), n, p0, p1, p2, p3, 256, num_warps=4)
    return out


def crt_poly_decode_check(ys: tuple[torch.Tensor, ...], slot0: torch.Tensor,
                          primes: tuple[int, ...], *, extra_tail_bytes: int = 0,
                          c_out_buf: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify Y(3)-3Y(2)+3Y(1)-Y(0)==0 mod p, decode C=(Y(2)-2Y(1)+Y(0))/2, add C to slot0."""
    primes = validate_primes(primes)
    p0, p1, p2, p3 = _pad_primes(primes)
    n_lanes = len(primes)
    n = slot0.numel() // n_lanes
    total_c = slot0.numel() + extra_tail_bytes
    if c_out_buf is None:
        c_out = torch.empty(total_c, dtype=torch.uint8, device=slot0.device)
    else:
        if c_out_buf.dtype != torch.uint8 or c_out_buf.device != slot0.device or c_out_buf.numel() != total_c or not c_out_buf.is_contiguous():
            raise ValueError('c_out_buf must be a contiguous uint8 CUDA buffer matching decoded C size')
        c_out = c_out_buf.reshape(-1)
    err_flag = torch.zeros(1, dtype=torch.int32, device=slot0.device)
    if n:
        y_flat = tuple(y.contiguous().reshape(-1) for y in ys)
        with torch.cuda.device(slot0.device):
            if n % 4 == 0 and all(y.data_ptr() % 4 == 0 for y in y_flat):
                n4 = n // 4
                _decode_check_vec4_kernel[(triton.cdiv(n4, 256), n_lanes)](
                    *(y.view(torch.uint32) for y in y_flat),
                    slot0.reshape(-1).view(torch.uint32),
                    c_out[:slot0.numel()].view(torch.uint32),
                    err_flag, n4, p0, p1, p2, p3, 256, num_warps=4)
            else:
                _decode_check_kernel[(triton.cdiv(n, 256), n_lanes)](
                    *y_flat, slot0.reshape(-1), c_out, err_flag, n, p0, p1, p2, p3, 256, num_warps=4)
    return c_out, (err_flag == 0).squeeze(0)


def crt_add_mod(a: torch.Tensor, b: torch.Tensor, primes: tuple[int, ...], *, inplace: bool = False) -> torch.Tensor:
    """Add b (full shape or trailing broadcast dimension) to a modulo p_l."""
    primes = validate_primes(primes)
    p0, p1, p2, p3 = _pad_primes(primes)
    n_lanes = len(primes)
    if a.dtype != torch.uint8 or b.dtype != torch.uint8 or a.shape[0] != n_lanes or b.shape[0] != n_lanes:
        raise ValueError('crt_add_mod requires matching uint8 tensors with leading prime dimension')
    n = a.numel() // n_lanes
    b_stride = b.numel() // n_lanes
    if b_stride <= 0 or n % b_stride != 0:
        raise ValueError('broadcast operand size must evenly divide target size')
    out = a if inplace else torch.empty_like(a)
    if n:
        with torch.cuda.device(a.device):
            if b_stride == n and n % 4 == 0 and a.data_ptr() % 4 == 0 and b.data_ptr() % 4 == 0:
                n4 = n // 4
                _add_mod_vec4_kernel[(triton.cdiv(n4, 256), n_lanes)](
                    a.contiguous().reshape(-1).view(torch.uint32),
                    b.contiguous().reshape(-1).view(torch.uint32),
                    out.reshape(-1).view(torch.uint32), n4, p0, p1, p2, p3, 256, num_warps=4)
            else:
                _add_mod_kernel[(triton.cdiv(n, 256), n_lanes)](
                    a.contiguous().reshape(-1), b.contiguous().reshape(-1),
                    out.reshape(-1), n, b_stride, p0, p1, p2, p3, 256, num_warps=4)
    return out
