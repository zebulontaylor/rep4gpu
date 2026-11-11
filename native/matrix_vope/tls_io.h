#ifndef SIV_MATRIX_VOPE_TLS_IO_H
#define SIV_MATRIX_VOPE_TLS_IO_H

#include "emp-tool/runtime/io/io_channel.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace matrix_vope {

struct TLSOptions {
    std::string ca_file, cert_file, key_file;
    // Certificates use exact DNS SANs party0 through party3. No wildcard or
    // common-name fallback is accepted. Empty peer_name selects party<peer_role>.
    std::string peer_name;
    uint32_t self_role = 0, peer_role = 1;
    std::array<unsigned char, 32> session{};
    std::array<unsigned char, 32> channel_id{};
    int timeout_ms = 30000;
    std::string bind_address = "127.0.0.1";
    std::size_t buffer_bytes = 65536;
};

// One thread owns each channel. Application sends coalesce into buffer_bytes;
// recv_data flushes pending sends before waiting. Call flush at the end of a
// send-only protocol step. The destructor closes without flushing or blocking:
// explicit flush/close is required when delivery errors must reach the caller.
// Address is a numeric IPv4/IPv6 literal, or nullptr to listen. Every connect,
// handshake, flush, or receive has a total monotonic deadline, not a per-chunk
// timeout. TLS 1.3 mutual authentication precedes the role/session/channel bind.
class TLSIO final : public emp::IOChannel {
public:
    TLSIO(const char* address, int port, const TLSOptions& options);
    ~TLSIO() override;
    TLSIO(const TLSIO&) = delete;
    TLSIO& operator=(const TLSIO&) = delete;
    TLSIO(TLSIO&&) = delete;
    TLSIO& operator=(TLSIO&&) = delete;
    void send_data_internal(const void* source, int64_t bytes) override;
    void recv_data_internal(void* destination, int64_t bytes) override;
    void flush() override;
    void sync() override;
    // Flush and issue TLS close_notify; closes immediately after sending it.
    void close();
    uint64_t tls_write_calls() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace matrix_vope
#endif
