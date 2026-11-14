"""Length-prefixed messages over the Unix socket between the harness and a task's kernel (``ws.kernel``), with file
descriptors sent along with the first bytes of a message."""

import socket
from collections.abc import Sequence

HEADER = 4  # bytes of the length before each message, big-endian
MESSAGE_MAX = 16 * 1024 * 1024  # bytes: a longer length is a broken channel, never a real cell


class ChannelClosed(EOFError):
    def __init__(self):
        super().__init__("the other end closed the channel or sent a broken message")


def send(sock: socket.socket, data: bytes, fds: Sequence[int] = ()):
    framed = len(data).to_bytes(HEADER, "big") + data
    sent = socket.send_fds(sock, [framed], list(fds)) if fds else 0
    sock.sendall(framed[sent:])


def receive(sock: socket.socket, fds: int = 0):
    """One message and the file descriptors that came with it (at most ``fds``). Raises ``ChannelClosed`` at the end
    of the stream and ``TimeoutError`` after the socket's timeout."""
    head, received = read_exact(sock, HEADER, fds)
    size = int.from_bytes(head, "big")
    if size > MESSAGE_MAX:
        raise ChannelClosed
    body, _ = read_exact(sock, size, 0)
    return body, received


def read_exact(sock: socket.socket, size: int, fds: int):
    data = b""
    received: list[int] = []
    while len(data) < size:
        if fds and not received:
            got = socket.recv_fds(sock, size - len(data), fds)
            chunk, received = got[0], got[1]
        else:
            chunk = sock.recv(size - len(data))
        if not chunk:
            raise ChannelClosed
        data += chunk
    return data, received
