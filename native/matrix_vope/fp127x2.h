// SPDX-License-Identifier: Apache-2.0
// Adapted from vole-labs/vole FP107 at e266595c1811692a6b2df87a4b4d13d3a86ef80f.
// Changed the field width, reduction limits, inversion, sampling, and wire checks.
// Upstream: https://github.com/vole-labs/vole; see VOLE-LICENSE.
#ifndef SIV_MATRIX_VOPE_FP127X2_H
#define SIV_MATRIX_VOPE_FP127X2_H

#include "emp-tool/emp-tool.h"
#include "vole/fields/utils.h"
#include "fp127.h"
#include <cstdint>

// FP127x2: packed (MAC, value) pair over F_p, p = 2^127 - 1.
//
// Layout: two unsigned 128-bit slots. The low slot holds the MAC and the
// high slot holds the value, matching the backend getLow/getHigh convention.
//
// Cannot reuse `block` (16 bytes) since 2 * 127 = 254 bits doesn't fit in a
// __m128i, so all ops are scalar (no SSE/AVX).
class FP127x2 {
public:
  using u128 = unsigned __int128;
  using i128 = __int128;

  static constexpr uint64_t PR_bit_len       = 127;
  static constexpr uint64_t slot_stride_bits = 127;             // = PR_bit_len
  static constexpr u128     PR               = (((u128)1) << 127) - 1;
  static constexpr uint64_t DS_byte_len = 32;
  static constexpr uint64_t PR_num_pack = 1;

  // Identity broadcast: large-field MAC has only one slot.
  static u128 copy_compose(u128 basis) { return basis; }
  static uint64_t copy_compose(uint64_t basis) { return basis; }

  // [0] = low slot (MAC), [1] = high slot (value).
  u128 val[2];

  FP127x2() { val[0] = 0; val[1] = 0; }

  FP127x2(const FP127x2 &inp) { val[0] = inp.val[0]; val[1] = inp.val[1]; }

  FP127x2(u128 lo, u128 hi, bool mod_it = false) {
    if(mod_it) {
      val[0] = FP127::mod(lo);
      val[1] = FP127::mod(hi);
    } else {
      val[0] = lo; val[1] = hi;
    }
  }

  FP127x2(FP127 lo, FP127 hi) {
    val[0] = lo.val; val[1] = hi.val;
  }

  FP127 operator[](bool idx) {
    FP127 r; r.val = val[idx ? 1 : 0]; return r;
  }

  FP127 getLow() const {
    FP127 r; r.val = val[0]; return r;
  }

  void setLow(FP127 lo) { val[0] = lo.val; }

  FP127 getHigh() const {
    FP127 r; r.val = val[1]; return r;
  }

  void setHigh(FP127 hi) { val[1] = hi.val; }

  void setZero() { val[0] = 0; val[1] = 0; }

  void setLowHigh(const FP127 lo, const FP127 hi) {
    val[0] = lo.val; val[1] = hi.val;
  }

  void from_block(block rhs) {
    uint64_t lo = _mm_extract_epi64(rhs, 0);
    uint64_t hi = _mm_extract_epi64(rhs, 1);
    u128 x = ((u128)hi << 64) | (u128)lo;
    val[0] = FP127::mod(x);
    val[1] = 0;
  }

  void getVal(FP127 *data) {
    data[0].val = val[0];
    data[1].val = val[1];
  }

  void operator=(const FP127x2 rhs) { val[0] = rhs.val[0]; val[1] = rhs.val[1]; }

  bool operator==(const FP127x2 rhs) const {
    return val[0] == rhs.val[0] && val[1] == rhs.val[1];
  }

  bool operator!=(const FP127x2 rhs) const {
    return !(*this == rhs);
  }

  // c = (a0 * b || a1 * b) mod PR (broadcast multiply by FP127)
  FP127x2 operator*(const FP127 b) const {
    FP127x2 res;
    res.val[0] = FP127::mult_mod(val[0], b.val);
    res.val[1] = FP127::mult_mod(val[1], b.val);
    return res;
  }

  // c = (a0 * b0 || a1 * b1) mod PR (lane-wise)
  FP127x2 operator*(const FP127x2 b) const {
    FP127x2 res;
    res.val[0] = FP127::mult_mod(val[0], b.val[0]);
    res.val[1] = FP127::mult_mod(val[1], b.val[1]);
    return res;
  }

  FP127x2 operator+(const FP127 b) const {
    FP127x2 res;
    res.val[0] = FP127::add_mod(val[0], b.val);
    res.val[1] = FP127::add_mod(val[1], b.val);
    return res;
  }

  FP127x2 operator+(const FP127x2 b) const {
    FP127x2 res;
    res.val[0] = FP127::add_mod(val[0], b.val[0]);
    res.val[1] = FP127::add_mod(val[1], b.val[1]);
    return res;
  }

  FP127x2 operator-(const FP127 b) const {
    FP127x2 res;
    res.val[0] = FP127::sub_mod(val[0], b.val);
    res.val[1] = FP127::sub_mod(val[1], b.val);
    return res;
  }

  FP127x2 operator-(const FP127x2 b) const {
    FP127x2 res;
    res.val[0] = FP127::sub_mod(val[0], b.val[0]);
    res.val[1] = FP127::sub_mod(val[1], b.val[1]);
    return res;
  }

  static constexpr unsigned lazy_adds = 1;
  void add_raw(const FP127x2 &rhs) { val[0] += rhs.val[0]; val[1] += rhs.val[1]; }
  void reduce() { val[0] = FP127::mod(val[0]); val[1] = FP127::mod(val[1]); }
  void set_low_from_block(block b) {
    uint64_t lo = _mm_extract_epi64(b, 0), hi = _mm_extract_epi64(b, 1);
    val[0] = FP127::mod(((u128)hi << 64) | (u128)lo);
    val[1] = 0;
  }

  FP127x2 negate() const {
    FP127x2 res;
    res.val[0] = (val[0] == 0) ? 0 : (PR - val[0]);
    res.val[1] = (val[1] == 0) ? 0 : (PR - val[1]);
    return res;
  }

  static std::size_t size() { return sizeof(u128) * 2; }

  template<typename IO>
  void send(IO *netio) {
    netio->send_data(&val[0], sizeof(u128) * 2);
  }

  template<typename IO>
  void recv(IO *netio) {
    netio->recv_data(&val[0], sizeof(u128) * 2);
  }

  block hash() {
    return Hash::hash_for_block(&val[0], sizeof(u128) * 2);
  }

};

#endif // SIV_MATRIX_VOPE_FP127X2_H
