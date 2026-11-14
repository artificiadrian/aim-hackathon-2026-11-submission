"""A shell for the model's scripts: grep, sed, sort and the like, on the read-only workspace."""

import subprocess

TIMEOUT = 20  # seconds


def sh(command: str):
    """Run a shell command with cwd /; returns stdout and stderr together."""
    done = subprocess.run(  # noqa: S602 - the model's own shell inside its sandbox
        command, check=False, shell=True, capture_output=True, text=True, timeout=TIMEOUT, cwd="/"
    )
    return done.stdout + done.stderr
