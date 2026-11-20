"""Shared mTLS and loopback helpers for four-process local examples and tests."""
from __future__ import annotations

from pathlib import Path
import socket
import ssl
import subprocess

from rep4gpu.transports import SocketTransport


class MeteredSocketTransport(SocketTransport):
    """SocketTransport subclass that also counts communication rounds."""

    def __init__(self, *args, **kwargs):
        self.rounds = 0
        super().__init__(*args, **kwargs)

    def exchange(self, outgoing):
        self.rounds += 1
        return super().exchange(outgoing)


def mtls_certificates(directory: Path) -> tuple[str, list[dict[str, str]]]:
    """Generate one ephemeral ECDSA CA and four distinct party TLS identities."""
    directory = Path(directory)

    def openssl(*args):
        subprocess.run(['openssl', *map(str, args)], check=True, capture_output=True, text=True)

    ca, ca_key = directory / 'ca.pem', directory / 'ca.key'
    openssl(
        'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1',
        '-nodes', '-keyout', ca_key, '-out', ca, '-days', '1', '-subj', '/CN=Rep4 Local CA',
    )
    identities = []
    for p in range(4):
        key, csr, cert, ext = [directory / f'party{p}.{s}' for s in ('key', 'csr', 'pem', 'ext')]
        ext.write_text(f'subjectAltName=DNS:party{p}\nextendedKeyUsage=serverAuth,clientAuth\n')
        openssl(
            'req', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1',
            '-nodes', '-keyout', key, '-out', csr, '-subj', f'/CN=party{p}',
        )
        openssl(
            'x509', '-req', '-in', csr, '-CA', ca, '-CAkey', ca_key, '-CAcreateserial',
            '-out', cert, '-days', '1', '-extfile', ext,
        )
        identities.append({'certificate': str(cert), 'key': str(key)})
    return str(ca), identities


def reserve_addresses() -> list[list]:
    """Bind four ephemeral loopback TCP ports and return [host, port] pairs."""
    sockets = []
    try:
        for _ in range(4):
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            sockets.append(sock)
        return [list(sock.getsockname()) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def ssl_contexts(ca: str, identity: dict[str, str]) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """Build mutually authenticated server and client SSLContext objects for one party."""
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.verify_mode = ssl.CERT_REQUIRED
    server.load_cert_chain(identity['certificate'], identity['key'])
    server.load_verify_locations(ca)

    client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca)
    client.load_cert_chain(identity['certificate'], identity['key'])
    return server, client
