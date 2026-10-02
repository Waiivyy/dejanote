"""The network guard stops outbound IP traffic from this process, and only that."""

from __future__ import annotations

import os
import socket
import tempfile

import pytest

from dejanote.privacy import NetworkBlockedError, block_network


@pytest.fixture
def loopback_server():
    """A listening TCP socket on 127.0.0.1, so no test traffic leaves the machine."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        yield server.getsockname()


def _tcp_connect(addr):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(addr)


def _tcp_connect_ex(addr):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect_ex(addr)


def _tcp6_connect(addr):
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.connect(("::1", addr[1]))


def _udp_sendto(addr):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"ping", addr)


def _udp_sendmsg(addr):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendmsg([b"ping"], [], 0, addr)


def _create_connection(addr):
    socket.create_connection(addr, timeout=1).close()


def _dns_lookup(addr):
    socket.getaddrinfo("example.com", 443)


def _gethostbyname(addr):
    socket.gethostbyname("example.com")


def _reverse_dns_lookup(addr):
    socket.gethostbyaddr("93.184.215.14")


OPERATIONS = {
    "connect": _tcp_connect,
    "connect_ex": _tcp_connect_ex,
    "connect_ipv6": _tcp6_connect,
    "udp_sendto": _udp_sendto,
    "udp_sendmsg": _udp_sendmsg,
    "create_connection": _create_connection,
    "getaddrinfo": _dns_lookup,
    "gethostbyname": _gethostbyname,
    "gethostbyaddr": _reverse_dns_lookup,
}


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_outbound_operations_are_blocked_and_recorded(operation, loopback_server):
    with block_network() as guard:
        with pytest.raises(NetworkBlockedError):
            operation(loopback_server)
    assert len(guard.attempts) == 1


def test_report_names_the_destination():
    with block_network() as guard:
        with pytest.raises(NetworkBlockedError):
            socket.getaddrinfo("huggingface.co", 443)
    assert "huggingface.co" in guard.attempts[0]


def test_networking_works_again_after_the_guard_exits(loopback_server):
    with block_network():
        pass
    with socket.create_connection(loopback_server, timeout=1):
        pass


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="platform has no Unix domain sockets")
def test_unix_domain_sockets_are_not_blocked():
    # Local IPC between processes on the same machine is not network traffic.
    path = os.path.join(tempfile.mkdtemp(dir="/tmp"), "s")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen()
        with block_network() as guard:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(path)
    assert guard.attempts == []
    os.remove(path)
    os.rmdir(os.path.dirname(path))
