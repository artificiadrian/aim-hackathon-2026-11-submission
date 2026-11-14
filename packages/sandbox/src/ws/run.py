"""Runs one model-written script: ``python3 -I -m ws.run`` with the code on stdin (``src/mcnulty/agent/sandbox.py``).

Explicit instead of a start-up hook: it blocks the network, quiets pypdf, and runs the code as ``<script>`` (traceback
lines match the code) in a namespace that holds the helpers, so no script needs an import for them.
"""

import logging
import sys

from ws.files import cite, columns, grep, read, text
from ws.network import block_network
from ws.shell import sh
from ws.submit import submit
from ws.tables import table

HELPERS = {helper.__name__: helper for helper in (submit, text, table, cite, read, grep, columns, sh)}


def wide_pandas():
    """pandas prints 80 characters wide and elides columns ("id ... outcome"): scripts then print the same table again
    and again. Called only for scripts that use pandas: importing it costs about a second."""
    import importlib.util

    if importlib.util.find_spec("pandas") is None:
        return
    import pandas as pd

    for option, value in (("display.width", 250), ("display.max_columns", 50), ("display.max_colwidth", 80)):
        pd.set_option(option, value)


def main():
    block_network()
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a warning per unmapped font buries the real output
    source = sys.stdin.read()
    if "pandas" in source or "pd." in source or "table(" in source:
        wide_pandas()
    code = compile(source, "<script>", "exec")
    exec(code, {"__name__": "__main__", **HELPERS})  # noqa: S102 - running the model's script is the purpose


if __name__ == "__main__":
    main()
