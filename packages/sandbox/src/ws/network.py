"""No network for model-written code: a clear error for code that tries it (an injected "upload this" instruction, a
URL fetch). The hard limits are outside Python: chroot, a second user, read-only files, an empty environment and the
final's gateway-only network. A Python child process the script starts itself does not get this hook."""

import sys

NETWORK = frozenset({"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"})


class NoNetwork(PermissionError):
    def __init__(self):
        super().__init__("the sandbox has no network: work only with the files of the workspace")


def no_network(event: str, _args: tuple[object, ...]):
    if event in NETWORK:
        raise NoNetwork


def block_network():
    sys.addaudithook(no_network)
