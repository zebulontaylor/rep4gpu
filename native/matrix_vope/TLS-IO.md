# Authenticated channels for native OLE

`matrix_vope::TLSIO` implements the pinned EMP `IOChannel` interface over TLS 1.3. It inherits EMP's block, point, byte-counter, and transcript helpers. Producers select this type explicitly; adding this module does not change an existing `emp::NetIO` producer into an authenticated protocol.

Every endpoint loads its own certificate and private key plus the public CA certificate. Certificates must contain the exact DNS SAN `party0`, `party1`, `party2`, or `party3` matching the assigned role. Both sides verify the peer's chain and role name. Common-name fallback and wildcard names are disabled. After mutual TLS authentication, endpoints exchange and check a fixed binding containing the local role, expected peer role, public 32-byte session, and 32-byte channel ID before accepting application traffic.

```cpp
#include "tls_io.h"

matrix_vope::TLSOptions options;
options.self_role = 0;
options.peer_role = 1;
options.ca_file = "public/ca.pem";
options.cert_file = "private/party0.pem";
options.key_file = "private/party0.key";
options.session = session_bytes;       // Fresh, nonzero public run identifier.
options.channel_id = channel_bytes;   // Public protocol/subchannel identifier.
options.timeout_ms = 30000;
matrix_vope::TLSIO io(nullptr, port, options); // Listen on 127.0.0.1.
```

The peer passes a numeric IP address instead of `nullptr`, reverses the two role IDs, supplies its own certificate and key, and uses the same session and channel ID. Set `bind_address` explicitly when the listener must accept remote connections. Numeric addresses avoid an unbounded DNS-resolution step in the connection deadline. Provision only the assigned role's private key to a deployed worker; the other roles' private keys are unnecessary. Local test fixtures share one OS account and do not isolate against malicious code under that account.

Each channel has one owning thread. Small application sends accumulate in a 64 KiB buffer by default. `recv_data` flushes pending sends before waiting, while an explicit `flush()` completes a send-only protocol step. This preserves EMP's buffered transport contract without one `SSL_write` per field element. Receive staging also avoids one TLS read per small field. Buffers are bounded and configurable from 1 KiB to 16 MiB.

Connect, handshake, flush, and receive operations use total monotonic deadlines. Partial progress does not restart an operation's timeout. Certificate, binding, EOF, truncation, or network failures throw an exception and close the channel. The caller must abort its cryptographic session and retire affected preprocessing; it must not retry a failed protocol step on a replacement connection. The destructor closes without flushing or waiting. Call `flush()` before a normal send-only return, and use `close()` when sending TLS shutdown explicitly. The socket BIO uses `MSG_NOSIGNAL` without changing the process's signal handlers. EMP elliptic-curve helpers can leave unrelated entries in OpenSSL's thread-local error queue, so the adapter clears that queue immediately before each SSL operation; `SSL_get_error` then classifies that operation's result.

From the repository root, build and run the transport regression:

```sh
.venv/bin/python native/matrix_vope/build_tls_io.py
.venv/bin/python -m pytest tests/test_native_tls_io.py -q
```

The build verifies the pinned EMP dependency revisions and source hashes, then writes the driver, compiler log, source fingerprints, and binary hash under `work/matrix_vope_tls/`. Tests generate temporary CA and role identities. Private keys are not included in the build manifest or public result artifacts. Test cases cover buffered transfer, all four role identities, certificate and binding rejection, truncation, abrupt EOF, and connection/handshake/receive deadlines.
