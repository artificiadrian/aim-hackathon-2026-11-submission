"""One-line hints the kernel adds where the model goes wrong: after an error, what to do instead; before a cell that
greps through the shell, the helper that cites. A hint never changes a result (docs/ideas/next-round.md)."""

import difflib
import re
import sys
from pathlib import Path
from typing import cast

from ws.files import files_under

WORKSPACE = Path("/")
SCRATCH = Path("/tmp")  # noqa: S108 - the sandbox's scratch folder, the only writable one
CLOSE_MAX = 3
SHELL_GREP = re.compile(r"\b(?:subprocess|os\.system|os\.popen|sh)\b[^\n]*\bgrep\b")
WRITE_HINT = (
    'hint: the workspace is read-only. To create a file the task asks for, put {"tool": "write", "path": ..., '
    '"content": ...} in submit(actions=[...]); scratch files go under /tmp.'
)
GREP_TIP = (
    "tip: grep(pattern, *paths) searches every format (pdf, xlsx, docx, json) and returns hits you can cite; "
    "the shell's grep sees only plain text."
)
TZ_HINT = "hint: make both sides aware: pd.to_datetime(values, utc=True), or .dt.tz_convert('UTC') when already aware."


def in_scratch(path: Path):
    return path == SCRATCH or SCRATCH in path.parents


def frame_columns(namespace: dict[str, object]):
    """Variable name -> columns of each pandas DataFrame in the session (none when pandas was never imported)."""
    if "pandas" not in sys.modules:
        return dict[str, list[str]]()
    import pandas as pd

    frames = {name: value for name, value in namespace.items() if isinstance(value, pd.DataFrame)}
    return {name: [str(c) for c in frame.columns] for name, frame in frames.items()}


def missing_column(key: str, namespace: dict[str, object]):
    found = frame_columns(namespace)
    if not found:
        return None
    names = sorted({c for columns in found.values() for c in columns})
    close = difflib.get_close_matches(key, names, n=CLOSE_MAX, cutoff=0.5)
    holders = [f"{name}: {', '.join(columns)}"[:200] for name, columns in found.items() if key not in columns][:3]
    guess = f"closest: {', '.join(close)}. " if close else ""
    return f"hint: no column {key!r}. {guess}Columns: " + " | ".join(holders)


def missing_file(path: str):
    wanted = Path("/", path)
    try:
        files = [str(f) for f in files_under(WORKSPACE)]
    except OSError:  # an unreadable folder: no list, the error stays as it is
        files = list[str]()
    close = difflib.get_close_matches(str(wanted), files, n=CLOSE_MAX, cutoff=0.5)
    close += [f for f in files if Path(f).name == wanted.name and f not in close][:CLOSE_MAX]
    if not close:
        return "hint: no such file; list the files with sh('ls -R /') or search with grep()"
    return f"hint: closest workspace paths: {', '.join(close)}"


def error_hint(error: BaseException, namespace: dict[str, object]):
    """What to do instead, for the errors that cost steps in the runs; None for the others."""
    filename = cast("object", error.filename) if isinstance(error, OSError) else None  # the boundary: Any in typeshed
    if isinstance(error, PermissionError) and filename and not in_scratch(Path("/", str(filename))):
        return WRITE_HINT
    if isinstance(error, FileNotFoundError) and filename:
        path = Path("/", str(filename))
        new_folder = not in_scratch(path) and not path.parent.exists()  # a write into a folder the workspace lacks
        return missing_file(str(filename)) + (f"\n{WRITE_HINT}" if new_folder else "")
    if isinstance(error, KeyError) and len(error.args) == 1 and isinstance(error.args[0], str):
        return missing_column(error.args[0], namespace)
    if isinstance(error, TypeError) and "tz-naive and tz-aware" in str(error):
        return TZ_HINT
    return None


def code_tip(code: str):
    """A tip before a cell that greps through the shell; None otherwise."""
    return GREP_TIP if SHELL_GREP.search(code) else None
