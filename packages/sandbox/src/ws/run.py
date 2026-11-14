"""Runs one model-written script: ``python3 -I -m ws.run`` with the code on stdin (``src/mcnulty/agent/sandbox.py``).

Explicit instead of a start-up hook: it blocks the network, quiets pypdf, and runs the code as ``<script>`` (traceback
lines match the code) in a namespace that holds the helpers, so no script needs an import for them.
"""

import logging
import sys

from ws.files import cite, grep, text
from ws.network import block_network
from ws.shell import sh
from ws.submit import submit

HELPERS = {helper.__name__: helper for helper in (submit, text, cite, grep, sh)}


def main():
    block_network()
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a warning per unmapped font buries the real output
    code = compile(sys.stdin.read(), "<script>", "exec")
    exec(code, {"__name__": "__main__", **HELPERS})  # noqa: S102 - running the model's script is the purpose


if __name__ == "__main__":
    main()
