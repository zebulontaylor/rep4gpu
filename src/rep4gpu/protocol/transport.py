"""Round transports. Socket payloads stage tensors through host memory."""
import socket
import math
import ssl
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class NetworkAbortedError(ConnectionError):
    """A peer failed or an ideal-network rendezvous exceeded its deadline."""


class IdealNetwork:
    """In-process rendezvous for protocol testing; not a network benchmark.

    A failed party must call ``abort(error)`` to release its waiting peers.
    A round timeout also aborts the network and every later exchange.
    """
    def __init__(self, timeout=120):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('ideal-network timeout must be positive')
        self.timeout = timeout
        self.condition = threading.Condition()
        self.rounds = {}
        self.failure = None

    def abort(self, error=None):
        with self.condition:
            if self.failure is None:
                self.failure = error if error is not None else RuntimeError('peer aborted')
            self.rounds.clear()
            self.condition.notify_all()

    def endpoint(self, party):
        if party not in range(4):
            raise ValueError('party must be in [0, 3]')
        return IdealTransport(self, party)


class IdealTransport:
    def __init__(self, network, party):
        self.network, self.party, self.round = network, party, 0
        self.bytes_sent = self.messages_sent = 0

    def exchange(self, outgoing):
        self.bytes_sent += sum(map(len, outgoing.values()))
        self.messages_sent += len(outgoing)
        net = self.network
        with net.condition:
            if net.failure is not None:
                raise NetworkAbortedError('ideal network aborted') from net.failure
            state = net.rounds.setdefault(self.round, [{}, 0])
            state[0][self.party] = outgoing
            net.condition.notify_all()
            ready = net.condition.wait_for(lambda: len(state[0]) == 4 or net.failure is not None,
                                           timeout=net.timeout)
            if not ready:
                net.abort(TimeoutError(f'ideal network round {self.round} timed out'))
            if net.failure is not None:
                raise NetworkAbortedError('ideal network aborted') from net.failure
            incoming = {p: data[self.party] for p, data in state[0].items() if self.party in data}
            state[1] += 1
            if state[1] == 4:
                del net.rounds[self.round]
            self.round += 1
            return incoming


def _read(sock, count, deadline=None):
    """Read exactly count bytes within one absolute monotonic deadline."""
    result = bytearray()
    previous_timeout = sock.gettimeout()
    try:
        while len(result) < count:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('protocol frame deadline exceeded')
                sock.settimeout(remaining)
            block = sock.recv(min(count - len(result), 1 << 20))
            if not block:
                raise ConnectionError('peer closed during protocol exchange')
            result.extend(block)
        return bytes(result)
    finally:
        sock.settimeout(previous_timeout)


def _receive_frame(sock, max_frame_bytes, deadline):
    size = struct.unpack('!Q', _read(sock, 8, deadline))[0]
    if size > max_frame_bytes:
        raise ValueError(f'protocol frame length {size} exceeds limit {max_frame_bytes}')
    return _read(sock, size, deadline)


class SocketTransport:
    """Four-party mesh with one stream per unordered pair.

    Plain TCP is for trusted local validation. Supply mutually authenticated
    SSL contexts and certificate hostnames to protect distributed connections.
    Each frame has one timeout budget and a configurable size limit, defaulting
    to one GiB. A failed exchange closes all peer sockets.
    """
    def __init__(self, party, addresses, timeout=120, *, server_context=None,
                 client_context=None, peer_hostnames=None, max_frame_bytes=1 << 30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('socket timeout must be positive')
        if not isinstance(max_frame_bytes, int) or max_frame_bytes < 0:
            raise ValueError('max_frame_bytes must be a nonnegative integer')
        self.timeout = timeout
        self.max_frame_bytes = max_frame_bytes
        self.closed = False
        tls = server_context is not None or client_context is not None
        if tls:
            if server_context is None or client_context is None or peer_hostnames is None:
                raise ValueError('TLS requires both SSL contexts and all peer hostnames')
            if server_context.verify_mode != ssl.CERT_REQUIRED or client_context.verify_mode != ssl.CERT_REQUIRED:
                raise ValueError('TLS contexts must require authenticated peer certificates')
            if not client_context.check_hostname:
                raise ValueError('TLS client context must check peer hostnames')
        self.party = party
        self.authenticated = tls
        self.peers = {}
        self.bytes_sent = self.messages_sent = 0
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(addresses[party])
        listener.listen(4)
        listener.settimeout(timeout)
        for peer in range(party):
            deadline = time.monotonic() + timeout
            while True:
                sock = socket.socket()
                # Short protocol frames must not wait for a delayed TCP ACK.
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(timeout)
                try:
                    sock.connect(addresses[peer])
                    if tls:
                        sock = client_context.wrap_socket(sock, server_hostname=peer_hostnames[peer])
                    sock.sendall(bytes([party]))
                    self.peers[peer] = sock
                    break
                except (ConnectionRefusedError, TimeoutError):
                    sock.close()
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        for _ in range(party + 1, 4):
            sock, _ = listener.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(timeout)
            if tls:
                sock = server_context.wrap_socket(sock, server_side=True)
            peer = _read(sock, 1)[0]
            if peer not in range(party + 1, 4) or peer in self.peers:
                sock.close()
                raise ValueError('unexpected or duplicate socket peer identity')
            if tls:
                ssl.match_hostname(sock.getpeercert(), peer_hostnames[peer])
            self.peers[peer] = sock
        listener.close()
        self.pool = ThreadPoolExecutor(max_workers=6)

    def exchange(self, outgoing):
        if self.closed:
            raise ConnectionError('socket transport is closed')
        if any(len(data) > self.max_frame_bytes for data in outgoing.values()):
            self._close_peers()
            raise ValueError('outgoing protocol frame exceeds configured limit')
        self.bytes_sent += sum(map(len, outgoing.values()))
        self.messages_sent += len(outgoing)
        def send(peer, sock):
            data = outgoing.get(peer, b'')
            deadline = time.monotonic() + self.timeout
            try:
                for chunk in (struct.pack('!Q', len(data)), data):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('protocol frame deadline exceeded')
                    sock.settimeout(remaining)
                    sock.sendall(chunk)
            finally:
                sock.settimeout(self.timeout)
        def receive(peer, sock):
            return peer, _receive_frame(sock, self.max_frame_bytes, time.monotonic() + self.timeout)
        def transact(peer, sock):
            # One worker owns each SSL object; concurrent SSL_read/SSL_write
            # on the same connection can corrupt its record state.
            if self.party < peer:
                send(peer, sock)
                return receive(peer, sock)
            result = receive(peer, sock)
            send(peer, sock)
            return result
        transactions = [self.pool.submit(transact, p, s) for p, s in self.peers.items()]
        try:
            result = dict(f.result() for f in as_completed(transactions))
        except BaseException:
            self._close_peers()
            raise
        return {p: value for p, value in result.items() if value}

    def _close_peers(self):
        self.closed = True
        for sock in self.peers.values():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def close(self):
        self._close_peers()
        self.pool.shutdown()


class RecordingTransport:
    """Record exact byte exchanges for a deterministic benchmark fixture."""
    def __init__(self, underlying):
        self.underlying = underlying
        self.transcript = []
        self.bytes_sent = self.messages_sent = 0

    def exchange(self, outgoing):
        outgoing = {p: bytes(data) for p, data in outgoing.items()}
        incoming = self.underlying.exchange(outgoing)
        incoming = {p: bytes(data) for p, data in incoming.items()}
        self.transcript.append((dict(outgoing), dict(incoming)))
        self.bytes_sent += sum(map(len, outgoing.values()))
        self.messages_sent += len(outgoing)
        return incoming


class ReplayTransport:
    """Replay one party's fixed transcript for isolated timing.

    This is not a live protocol transport: peers do not execute and fixture
    randomness must never be reused for live inputs. The caller must recreate
    the same party state and inputs when outgoing checks are enabled.
    """
    def __init__(self, transcript, check_outgoing=True):
        self.transcript = [(dict(sent), dict(received)) for sent, received in transcript]
        self.check_outgoing = check_outgoing
        self.round = 0
        self.bytes_sent = self.messages_sent = 0
        self.phase_times = []
        self._phase_start = None
        self._clock = None
        self._synchronize = None

    def start(self, clock=None, synchronize=None):
        """Start phase timing; synchronize may wait for the party's GPU stream."""
        import time
        self._clock = time.perf_counter if clock is None else clock
        self._synchronize = synchronize
        self.phase_times = []
        if synchronize is not None:
            synchronize()
        self._phase_start = self._clock()

    def _mark_phase(self):
        if self._phase_start is not None:
            if self._synchronize is not None:
                self._synchronize()
            self.phase_times.append(self._clock() - self._phase_start)

    def finish(self):
        """Record local work after the final exchange and return phase seconds."""
        self._mark_phase()
        self._phase_start = None
        return list(self.phase_times)

    def exchange(self, outgoing):
        self._mark_phase()
        if self.round >= len(self.transcript):
            raise ValueError('replay transcript exhausted')
        expected, incoming = self.transcript[self.round]
        if self.check_outgoing and outgoing != expected:
            raise ValueError(f'replay outgoing mismatch at round {self.round}')
        self.round += 1
        self.bytes_sent += sum(map(len, outgoing.values()))
        self.messages_sent += len(outgoing)
        result = dict(incoming)
        if self._phase_start is not None:
            self._phase_start = self._clock()
        return result

    def assert_consumed(self):
        if self.round != len(self.transcript):
            raise ValueError(f'replay consumed {self.round} of {len(self.transcript)} rounds')
