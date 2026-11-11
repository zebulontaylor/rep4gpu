// SPDX-License-Identifier: Apache-2.0
#ifndef SIV_MATRIX_VOPE_SPAKE2_EQ_H
#define SIV_MATRIX_VOPE_SPAKE2_EQ_H

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace siv::spake2 {
using Bytes=std::vector<unsigned char>;
using Binding=std::array<unsigned char,32>;
enum class Role { A, B };
struct Context {
    Bytes sid;
    Bytes configuration;
    Bytes identity_a;
    Bytes identity_b;
};
class EqualityFailure : public std::runtime_error {
 public:
    EqualityFailure() : std::runtime_error("private equality failed") {}
};
struct Callbacks {
    // Consume or copy this borrowed message before returning.
    std::function<void(std::span<const unsigned char>)> send;
    std::function<Bytes(std::size_t)> receive;
    // Authenticate the common public binding and both final acceptance states.
    // Return only after the existing role/session-bound output barrier passes.
    std::function<void(const Binding&)> accepted_output_barrier;
    // Retire pending correlations/epoch and close or interrupt peer transport
    // after every failed attempt, so a blocked peer also aborts.
    std::function<void()> poison;
};

// Fixes the canonical local residual before any transport operation. The role
// order is A:pA, B:pB, A:cA, B:cB, followed by the caller's accepted-output
// barrier. Returns no keys, residual digest, difference, or comparison trace.
// Any failure invokes poison and throws EqualityFailure. EMP_TEST_MODE is
// refused; production private ephemerals always come from OpenSSL's CSPRNG.
void check_private_equal(Role role,const Context& context,
                         std::span<const unsigned char> canonical_residual,
                         const Callbacks& callbacks);

#ifdef SIV_SPAKE2_EQ_TESTING
struct DeterministicTestOnly {};
struct TestTrace {
    Bytes public_message;
    Bytes shared_point;
    Bytes transcript;
    Bytes hash_tt;
    Bytes ke;
    Bytes ka;
    Bytes kc_a;
    Bytes kc_b;
    Bytes confirm_a;
    Bytes confirm_b;
};
// This interface is absent from production builds and additionally requires
// SIV_SPAKE2_EQ_TEST_MODE=1 at runtime. Values are public RFC fixtures only.
class TestSession {
 public:
    TestSession(DeterministicTestOnly,Role role,Bytes identity_a,Bytes identity_b,
                Bytes w_big_endian_32,Bytes ephemeral_big_endian_32,Bytes aad={});
    ~TestSession();
    TestSession(const TestSession&)=delete;
    TestSession& operator=(const TestSession&)=delete;
    Bytes message1();
    Bytes message2(std::span<const unsigned char> peer_point);
    Bytes message3(std::span<const unsigned char> peer_point);
    Bytes message4(std::span<const unsigned char> peer_confirmation);
    void finish(std::span<const unsigned char> peer_confirmation);
    bool locally_confirmed() const;
    bool poisoned() const;
    TestTrace trace() const;
 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::array<unsigned char,32> test_reduce512(DeterministicTestOnly,const std::array<unsigned char,64>& input);
#endif
}  // namespace siv::spake2
#endif
