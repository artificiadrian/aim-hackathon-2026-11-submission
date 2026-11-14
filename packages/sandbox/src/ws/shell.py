"""A shell for the model's scripts: grep, sed, sort and the like, on the read-only workspace."""

import subprocess

from ws.models import Shell

TIMEOUT = 20.0  # seconds


def sh(command: str, timeout: float = TIMEOUT):
    """Run a shell command with cwd /: a ``Shell`` with stdout, stderr and the exit code (None when it timed out)."""
    try:
        done = subprocess.run(  # noqa: S602 - the model's own shell inside its sandbox
            command, check=False, shell=True, capture_output=True, text=True, timeout=timeout, cwd="/"
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout or ""
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr or ""
        return Shell(stdout=out, stderr=err + f"\n[timed out after {timeout}s]", exit_code=None)
    return Shell(stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode)
