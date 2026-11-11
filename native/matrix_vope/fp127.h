// SPDX-License-Identifier: Apache-2.0
// Adapted from vole-labs/vole FP107 at e266595c1811692a6b2df87a4b4d13d3a86ef80f.
// Changed the field width, reduction limits, inversion, sampling, and wire checks.
// Upstream: https://github.com/vole-labs/vole; see VOLE-LICENSE.
#ifndef SIV_MATRIX_VOPE_FP127_H
#define SIV_MATRIX_VOPE_FP127_H

// Arithmetic substrate for p = 2^127 - 1. This field does not establish a
// statistical security parameter for the protocols that consume it.
// Canonical values use 127 bits in an unsigned 128-bit word. Products use four
// 64-bit partial products, followed by reduction at bit 127.

#include "emp-tool/emp-tool.h"
#include "vole/fields/utils.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <array>

using namespace emp;

class FP127 {
 public:
    using u128 = unsigned __int128;
    using i128 = __int128;

    // 2^127 - 1.
    static constexpr uint64_t PR_bit_len       = 127;
    static constexpr uint64_t slot_stride_bits = 127;            // = PR_bit_len
    static constexpr u128     PR               = (((u128)1) << 127) - 1;
    static constexpr u128     HALF_PR    = ((((u128)1) << 127) - 1) / 2;
    static constexpr u128     PR_mask    = (((u128)1) << 127) - 1;
    static constexpr uint64_t PR_num_pack = 1;
    static constexpr uint64_t DS_byte_len = 16;

    u128 val;

    FP127() : val(0) {}
    FP127(u128 v, bool reduce = true) : val(reduce ? mod(v) : v) {}
    // Non-explicit so int / uint64_t literals convert implicitly (matching
    // FP61's surface). Always reduces.
    FP127(uint64_t v)                : val((u128)v) {}
    FP127(int v)                     : val(v < 0 ? PR - ((u128)(-(int64_t)v) % PR) : (u128)v) {}
    // Explicit (lo, hi) split-decoder factory — explicit so it doesn't
    // collide with FP127(uint64_t, bool=true) overload resolution.
    static FP127 from_lo_hi(uint64_t lo, uint64_t hi) {
        FP127 r; r.val = mod(((u128)hi << 64) | (u128)lo); return r;
    }

    void operator=(const FP127& rhs) { val = rhs.val; }
    void operator=(uint64_t rhs)      { val = mod((u128)rhs); }

    void setZero() { val = 0; }

    void assign_no_mod(const u128 rhs) { val = rhs; }

    u128 value() const { return val; }

    // SPDZ MAC layout helpers: a single FP127 holds either the value or the
    // MAC -- the packed (MAC, value) pair lives in FP127x2. setHigh/getHigh exist
    // for symmetry with FP61 so the same templates compile.
    void setHigh(const u128 rhs) { val = rhs; }
    void setHigh(const FP127 &rhs) { val = rhs.val; }

    FP127 getHigh() {
        FP127 r; r.val = val; return r;
    }

    void setLow(const FP127 &rhs) { val = rhs.val; }

    FP127 getLow() {
        FP127 r; r.val = val; return r;
    }

    bool operator==(const FP127& rhs) const { return val == rhs.val; }
    bool operator!=(const FP127& rhs) const { return val != rhs.val; }
    bool operator==(const u128& rhs) const { return val == mod(rhs); }
    bool operator!=(const u128& rhs) const { return val != mod(rhs); }

    FP127 operator+(const FP127& rhs) const {
        u128 s = val + rhs.val;
        if (s >= PR) s -= PR;
        FP127 r; r.val = s; return r;
    }
    FP127 operator-(const FP127& rhs) const {
        FP127 r;
        r.val = (val >= rhs.val) ? (val - rhs.val) : (val + PR - rhs.val);
        return r;
    }
    FP127 operator-() const {
        FP127 r;
        r.val = (val == 0) ? 0 : (PR - val);
        return r;
    }
    FP127 operator*(const FP127& rhs) const {
        FP127 r; r.val = mult_mod(val, rhs.val); return r;
    }

    // Scalar overloads (uint64_t / u128) — let cope/spfss code that builds
    // values from raw uint64 bit operations work without explicit casts.
    FP127 operator+(uint64_t rhs) const { return *this + FP127(rhs); }
    FP127 operator-(uint64_t rhs) const { return *this - FP127(rhs); }
    FP127 operator*(uint64_t rhs) const { FP127 r; r.val = mult_mod(val, (u128)rhs); return r; }
    FP127 operator+(u128 rhs) const { FP127 r; r.val = add_mod(val, mod(rhs)); return r; }
    FP127 operator-(u128 rhs) const { FP127 r; r.val = sub_mod(val, mod(rhs)); return r; }
    FP127 operator*(u128 rhs) const { FP127 r; r.val = mult_mod(val, mod(rhs)); return r; }

    FP127 operator<<(const int n_pos) const {
        if (n_pos < 0) throw std::invalid_argument("negative field shift");
        FP127 r; r.val = mult_mod(val, ((u128)1) << (n_pos % 127)); return r;
    }

    // Comparison with raw scalars (matches FP61).
    bool operator==(uint64_t rhs) const { return val == mod((u128)rhs); }
    bool operator!=(uint64_t rhs) const { return val != mod((u128)rhs); }

    FP127& operator+=(const FP127& rhs) { *this = *this + rhs; return *this; }
    FP127& operator-=(const FP127& rhs) { *this = *this - rhs; return *this; }
    FP127& operator*=(const FP127& rhs) { *this = *this * rhs; return *this; }

    // Only one raw addition fits before reduction at this field width.
    static constexpr unsigned lazy_adds = 1;
    void add_raw(const FP127 &rhs) { val += rhs.val; }
    void reduce() { val = mod(val); }

    FP127 negate() const {
        FP127 r;
        r.val = (val == 0) ? 0 : (PR - val);
        return r;
    }

    // Multiplicative inverse; zero maps to zero for backend compatibility.
    FP127 inv() const {
        FP127 r; r.val = mod_inv(val); return r;
    }

    template<typename PRNG>
    void rand(PRNG& prg) {
        // Rejection keeps the distribution exactly uniform under uniform PRG
        // bytes. The high input bit is discarded; p itself is rejected.
        do {
            prg.random_data_unaligned(&val, sizeof(val));
            val &= PR;
        } while (val == PR);
    }

    std::array<unsigned char, 16> encode() const {
        if (val >= PR) throw std::invalid_argument("noncanonical field value");
        std::array<unsigned char, 16> out{};
        for (unsigned i = 0; i < 16; ++i) out[i] = (unsigned char)(val >> (8*i));
        return out;
    }

    static FP127 decode(const unsigned char *in) {
        u128 value = 0;
        for (unsigned i = 0; i < 16; ++i) value |= (u128)in[i] << (8*i);
        if (value >= PR) throw std::invalid_argument("noncanonical field encoding");
        return FP127(value, false);
    }

    template<typename IO>
    void send(IO *netio) {
        auto bytes = encode();
        netio->send_data(bytes.data(), bytes.size());
    }

    template<typename IO>
    void recv(IO *netio) {
        std::array<unsigned char, 16> bytes;
        netio->recv_data(bytes.data(), bytes.size());
        *this = decode(bytes.data());
    }

    block hash() {
        return Hash::hash_for_block(&val, sizeof(u128));
    }

    void from_block(block rhs) {
        uint64_t lo = _mm_extract_epi64(rhs, 0);
        uint64_t hi = _mm_extract_epi64(rhs, 1);
        u128 x = ((u128)hi << 64) | (u128)lo;
        val = mod(x);
    }

    static u128 copy_compose(u128 basis) { return basis; }
    static uint64_t copy_compose(uint64_t basis) { return basis; }

    // Helpers: (lo, hi) decomposition for serialization. lo is 64 bits;
    // hi is the upper 63 bits, padded into a uint64_t.
    uint64_t lo64() const { return (uint64_t)(val & 0xFFFFFFFFFFFFFFFFULL); }
    uint64_t hi64() const { return (uint64_t)(val >> 64); }

    // Mersenne reduction: x mod (2^127 - 1) for x < 2^128 (a single u128).
    static u128 mod(u128 x) {
        u128 r = (x & PR) + (x >> 127);
        if (r >= PR) r -= PR;
        return r;
    }

    static u128 add_mod(u128 a, u128 b) {
        u128 s = a + b;
        return (s >= PR) ? (s - PR) : s;
    }
    static u128 sub_mod(u128 a, u128 b) {
        return (a >= b) ? (a - b) : (a + PR - b);
    }

    // Multiply two values in [0, 2^127) and reduce mod p.
    static u128 mult_mod(u128 a, u128 b) {
        uint64_t a_lo = (uint64_t)a;
        uint64_t a_hi = (uint64_t)(a >> 64);
        uint64_t b_lo = (uint64_t)b;
        uint64_t b_hi = (uint64_t)(b >> 64);

        u128 ll = (u128)a_lo * (u128)b_lo;       // bits   0..127
        u128 lh = (u128)a_lo * (u128)b_hi;       // bits  64..190
        u128 hl = (u128)a_hi * (u128)b_lo;       // bits  64..190
        u128 hh = (u128)a_hi * (u128)b_hi;       // bits 128..253

        u128 mid     = lh + hl;
        u128 mid_carry = (mid < lh) ? ((u128)1 << 64) : 0;

        u128 low_add = mid << 64;
        u128 low     = ll + low_add;
        u128 carry   = (low < ll) ? 1 : 0;

        u128 high = hh + (mid >> 64) + mid_carry + carry;

        u128 c0 = low & PR;
        u128 c1 = (low >> 127) | (high << 1);

        u128 s = c0 + c1;
        if (s >= PR) s -= PR;
        if (s >= PR) s -= PR;
        return s;
    }

    // Fermat inversion avoids signed coefficient overflow near 2^127.
    // Zero retains the backend's existing zero-inverse convention.
    static u128 mod_inv(u128 a) {
        if (a == 0) return 0;
        u128 result = 1, exponent = PR - 2;
        while (exponent) {
            if (exponent & 1) result = mult_mod(result, a);
            a = mult_mod(a, a);
            exponent >>= 1;
        }
        return result;
    }


    static std::size_t size() { return sizeof(u128); }
};

#endif  // SIV_MATRIX_VOPE_FP127_H
