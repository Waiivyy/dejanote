"""Keep dejanote off the network.

Only `dejanote setup`, the one-time model download, may use the network.
Every other command runs with two layers of protection:

1. Hugging Face offline mode, so the ML libraries never try to reach the Hub.
2. A socket guard that refuses outbound IP traffic and DNS lookups made by
   this process and records each attempt, so anything unexpected is blocked
   and reported instead of quietly succeeding.

Local IPC (Unix domain sockets) is not network traffic and stays allowed.
"""

from __future__ import annotations

import contextlib
import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NoReturn

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

_IP_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_RESOLVERS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")


class NetworkBlockedError(ConnectionError):
    """Raised in place of an outbound network operation.

    It is a ConnectionError so libraries treat it like being offline.
    """


@dataclass
class NetworkGuard:
    """The network operations refused while the guard was active."""

    attempts: list[str] = field(default_factory=list)

    def refuse(self, description: str) -> NoReturn:
        self.attempts.append(description)
        raise NetworkBlockedError(f"dejanote blocked a network operation: {description}")


def enable_offline_mode() -> None:
    """Tell the Hugging Face libraries not to contact the Hub.

    Call this before they are imported, since they read these variables at import time.
    """
    os.environ.update(_OFFLINE_ENV)


@contextlib.contextmanager
def block_network() -> Iterator[NetworkGuard]:
    """Refuse outbound IP traffic and DNS lookups until the block exits."""
    guard = NetworkGuard()
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_sendmsg = socket.socket.sendmsg
    real_resolvers = {name: getattr(socket, name) for name in _RESOLVERS}

    def connect(sock, address):
        if sock.family in _IP_FAMILIES:
            guard.refuse(f"connection to {_describe(address)}")
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in _IP_FAMILIES:
            guard.refuse(f"connection to {_describe(address)}")
        return real_connect_ex(sock, address)

    def sendto(sock, data, *flags_and_address):
        if sock.family in _IP_FAMILIES:
            guard.refuse(f"datagram to {_describe(flags_and_address[-1])}")
        return real_sendto(sock, data, *flags_and_address)

    def sendmsg(sock, buffers, *args):
        # The destination is the optional fourth argument: (buffers, ancdata, flags, address).
        if sock.family in _IP_FAMILIES and len(args) >= 3:
            guard.refuse(f"datagram to {_describe(args[2])}")
        return real_sendmsg(sock, buffers, *args)

    def refuse_lookup(host, *args, **kwargs):
        guard.refuse(f"DNS lookup of {_describe(host)}")

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.socket.sendmsg = sendmsg
    for name in _RESOLVERS:
        setattr(socket, name, refuse_lookup)
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.socket.sendto = real_sendto
        socket.socket.sendmsg = real_sendmsg
        for name, real in real_resolvers.items():
            setattr(socket, name, real)


def _describe(address: object) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)
