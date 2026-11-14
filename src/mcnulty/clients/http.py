"""Helpers for HTTP responses that both clients use."""

import httpx


def header(response: httpx.Response, name: str):
    return response.headers[name] if name in response.headers else ""  # noqa: SIM401 - Headers.get is typed Any


def retry_after(value: str):
    """Seconds from a Retry-After header; an HTTP date or junk counts as 1 s."""
    try:
        return max(float(value or 0), 0.0)
    except ValueError:
        return 1.0
