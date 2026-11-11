// SPDX-License-Identifier: Apache-2.0
#ifndef SIV_MATRIX_VOPE_SPAKE2_EQ_TLS_H
#define SIV_MATRIX_VOPE_SPAKE2_EQ_TLS_H

#include "spake2_eq.h"
#include "tls_io.h"
#include <cstdint>
#include <functional>
#include <memory>

namespace siv::spake2 {
struct TLSAdapterStats {
    std::uint64_t sent_bytes=0,received_bytes=0,rounds=0,flushes=0,tls_write_calls=0;
    std::uint64_t completed_checks=0;
    bool poisoned=false;
};

// Public context codec shared by the adapter and independently implemented
// peers. Identities must be the authenticated party<role> names in A/B order;
// the smaller certificate role is A. SID equals the TLS session binding.
Context bind_tls_context(const matrix_vope::TLSOptions& options,const Context& context,
                         std::uint64_t sequence);

// Owns one authenticated TLSIO. Poison retires the supplied epoch and destroys
// the TLS channel immediately, without flushing or waiting for close_notify.
// A channel, its counter state, and its borrowed transport have one owner thread.
class TLSAdapter {
 public:
    TLSAdapter(const char* address,int port,const matrix_vope::TLSOptions& options,
               std::function<void()> retire_epoch);
    ~TLSAdapter();
    TLSAdapter(const TLSAdapter&)=delete;
    TLSAdapter& operator=(const TLSAdapter&)=delete;
    TLSAdapter(TLSAdapter&&)=delete;
    TLSAdapter& operator=(TLSAdapter&&)=delete;
    void check(const Context& context,std::span<const unsigned char> canonical_residual);
    // Borrow for the rest of the same native protocol. IOChannel already counts
    // bytes/rounds; a forwarding counting view must not outlive this owner.
    emp::IOChannel& transport();
    void poison() noexcept;
    TLSAdapterStats statistics() const;
 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace siv::spake2
#endif
