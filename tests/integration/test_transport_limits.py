"""Resource bounds and abort propagation for protocol-test transports."""
from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import threading
import time

import pytest

from rep4gpu.transports.byte import IdealNetwork, NetworkAbortedError, _receive_frame


def test_oversized_frame_is_rejected_before_reading_payload():
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(struct.pack('!Q', 1025))
        with pytest.raises(ValueError, match='exceeds limit'):
            _receive_frame(receiver, 1024, time.monotonic() + .2)
    finally:
        receiver.close()
        sender.close()


def test_empty_frame_is_valid_at_zero_size_limit():
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(struct.pack('!Q', 0))
        assert _receive_frame(receiver, 0, time.monotonic() + .2) == b''
    finally:
        receiver.close()
        sender.close()


def test_partial_frame_cannot_restart_the_timeout_budget():
    receiver, sender = socket.socketpair()
    stop = threading.Event()
    def trickle():
        sender.sendall(struct.pack('!Q', 20))
        while not stop.wait(.025):
            try:
                sender.sendall(b'x')
            except OSError:
                return
    thread = threading.Thread(target=trickle)
    thread.start()
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            _receive_frame(receiver, 1024, start + .12)
        assert time.monotonic() - start < .5
    finally:
        stop.set()
        receiver.close()
        thread.join(timeout=1)
        sender.close()


def test_ideal_abort_releases_waiting_and_future_peers():
    network = IdealNetwork(timeout=1)
    with ThreadPoolExecutor(3) as pool:
        pending = [pool.submit(network.endpoint(p).exchange, {}) for p in range(3)]
        network.abort(ValueError('peer failed before entering round'))
        for future in pending:
            with pytest.raises(NetworkAbortedError):
                future.result(timeout=.5)
    with pytest.raises(NetworkAbortedError):
        network.endpoint(3).exchange({})
    assert network.rounds == {}


def test_ideal_round_deadline_aborts_the_network():
    network = IdealNetwork(timeout=.03)
    with pytest.raises(NetworkAbortedError) as error:
        network.endpoint(0).exchange({})
    assert isinstance(error.value.__cause__, TimeoutError)
    with pytest.raises(NetworkAbortedError):
        network.endpoint(1).exchange({})


def test_ideal_empty_exchange_remains_a_valid_round():
    network = IdealNetwork(timeout=1)
    with ThreadPoolExecutor(4) as pool:
        endpoints = [network.endpoint(p) for p in range(4)]
        assert list(pool.map(lambda p: p.exchange({}), endpoints)) == [{}, {}, {}, {}]
    assert network.rounds == {}


def test_socket_exchange_aborts_other_workers_on_oversized_frame():
    from rep4gpu.transports.byte import SocketTransport
    first, malicious = socket.socketpair()
    second, stalled = socket.socketpair()
    transport = SocketTransport.__new__(SocketTransport)
    transport.party = 0
    transport.timeout = 1
    transport.max_frame_bytes = 8
    transport.closed = False
    transport.bytes_sent = transport.messages_sent = 0
    transport.peers = {1: first, 2: second}
    transport.pool = ThreadPoolExecutor(2)
    def attack():
        assert malicious.recv(8) == struct.pack('!Q', 0)
        malicious.sendall(struct.pack('!Q', 9))
    attacker = threading.Thread(target=attack)
    attacker.start()
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match='exceeds limit'):
            transport.exchange({})
        assert transport.closed
        assert time.monotonic() - started < .5
    finally:
        transport.close()
        attacker.join(timeout=1)
        malicious.close()
        stalled.close()


def test_socket_exchange_rejects_oversized_outgoing_frame():
    from rep4gpu.transports.byte import SocketTransport
    transport = SocketTransport.__new__(SocketTransport)
    transport.closed = False
    transport.max_frame_bytes = 3
    transport.peers = {}
    with pytest.raises(ValueError, match='outgoing'):
        transport.exchange({1: b'1234'})
    assert transport.closed
