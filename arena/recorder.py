"""Run recorder: one JSONL event stream per run, plus a record/replay httpx transport."""

from __future__ import annotations

import contextvars
import gzip
import hashlib
import itertools
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx

# --- context: which task / step an event belongs to (set inside each worker thread) ---------------
CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "trace_ctx", default=None
)


def set_ctx(**kw: Any) -> None:
    CTX.set({**(CTX.get() or {}), **kw})


# --- redaction --------------------------------------------------------------------------------------
_SECRET_ENV = ("ARENA_API_KEY", "OPENROUTER_API_KEY")
_SECRET_RX = re.compile(
    r"(arena_[A-Za-z0-9_-]{8,}|sk-or-[A-Za-z0-9-]{16,}|Bearer\s+[A-Za-z0-9._-]{8,})"
)
_KEEP_HEADERS = {
    "content-type",
    "retry-after",
    "x-ratelimit-remaining",
    "x-generation-id",
    "content-length",
}


def redact(text: str) -> str:
    for name in _SECRET_ENV:
        v = os.environ.get(name)
        if v and len(v) >= 8:
            text = text.replace(v, f"[REDACTED:{name}]")
    return _SECRET_RX.sub("[REDACTED]", text)


def keep_headers(h: httpx.Headers) -> dict:
    return {
        k: v for k, v in h.items() if k.lower() in _KEEP_HEADERS
    }  # Authorization is never kept


# --- recorder ---------------------------------------------------------------------------------------
class Recorder:
    INLINE_MAX = 64_000  # bytes; larger or binary bodies go to blobs/<sha256>.gz

    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        (self.dir / "blobs").mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "events.jsonl", "a", encoding="utf-8")  # noqa: SIM115  (open for the whole run)
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self.t0 = time.time()

    def emit(self, kind: str, **data: Any) -> None:
        ctx = CTX.get() or {}
        ev = {
            "seq": next(self._seq),
            "ts": round(time.time(), 3),
            "t": round(time.time() - self.t0, 3),
            "kind": kind,
            **ctx,
            **data,
        }
        line = redact(json.dumps(ev, ensure_ascii=False, default=str))
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def body(self, raw: bytes, content_type: str = "") -> dict:
        """Inline JSON/text bodies; store big or binary ones by hash."""
        if len(raw) <= self.INLINE_MAX and (
            "json" in content_type or "text" in content_type or not content_type
        ):
            try:
                txt = raw.decode("utf-8")
                try:
                    return {"json": json.loads(txt)} if txt else {"text": ""}
                except ValueError:
                    return {"text": txt}
            except UnicodeDecodeError:
                pass
        sha = hashlib.sha256(raw).hexdigest()
        p = self.dir / "blobs" / f"{sha}.gz"
        if not p.exists():
            p.write_bytes(gzip.compress(raw))
        return {"blob": sha, "size": len(raw)}

    def close(self) -> None:
        with self._lock:
            self._f.close()


def request_key(req: httpx.Request) -> str:
    """Stable hash of method + path + canonical JSON body (the replay match key)."""
    raw = req.content or b""
    try:
        canon = json.dumps(
            json.loads(raw), sort_keys=True, separators=(",", ":")
        ).encode()
    except ValueError:
        canon = raw
    return hashlib.sha256(
        req.method.encode() + b" " + req.url.raw_path + b"\n" + canon
    ).hexdigest()[:24]


# --- transport: record (live + log) or replay (from events.jsonl, no network) ----------------------
class RecordingTransport(httpx.BaseTransport):
    def __init__(
        self,
        rec: Recorder,
        inner: httpx.BaseTransport | None = None,
        replay_from: str | Path | None = None,
        live_hosts: set[str] = frozenset(),
    ):
        self.rec, self.inner = rec, inner or httpx.HTTPTransport()
        self.live_hosts = (
            live_hosts  # in replay mode: hosts still sent live (e.g. the LLM host)
        )
        self.tape: dict[str, deque] = defaultdict(deque)
        if replay_from:
            with (Path(replay_from) / "events.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    if ev["kind"] == "http":
                        self.tape[ev["key"]].append(
                            ev
                        )  # FIFO per key: repeated identical requests replay in order
            self.src = Path(replay_from)
        self.replay = replay_from is not None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key, t = request_key(request), time.time()
        req_body = self.rec.body(
            request.content or b"", request.headers.get("content-type", "")
        )
        if self.replay and request.url.host not in self.live_hosts:
            if not self.tape[key]:
                self.rec.emit(
                    "replay_miss",
                    key=key,
                    method=request.method,
                    url=str(request.url),
                    req=req_body,
                )
                raise httpx.ConnectError(
                    f"replay miss {key} {request.method} {request.url.path}",
                    request=request,
                )
            ev = self.tape[key].popleft()
            raw = self._raw(ev["resp"])
            self.rec.emit(
                "http",
                key=key,
                replayed=True,
                method=request.method,
                url=str(request.url),
                status=ev["status"],
                req=req_body,
                resp=ev["resp"],
            )
            return httpx.Response(
                ev["status"],
                headers=ev.get("resp_headers") or {},
                content=raw,
                request=request,
            )
        try:
            resp = self.inner.handle_request(request)
            raw = (
                resp.read()
            )  # load the body so it can be logged; the caller still gets it
        except Exception as e:
            self.rec.emit(
                "http_error",
                key=key,
                method=request.method,
                url=str(request.url),
                req=req_body,
                error=f"{type(e).__name__}: {e}",
                ms=int((time.time() - t) * 1000),
            )
            raise
        self.rec.emit(
            "http",
            key=key,
            method=request.method,
            url=str(request.url),
            status=resp.status_code,
            ms=int((time.time() - t) * 1000),
            req=req_body,
            resp_headers=keep_headers(resp.headers),
            resp=self.rec.body(raw, resp.headers.get("content-type", "")),
        )
        # raw is already decoded: drop the encoding headers, or httpx decodes it a second time
        hdrs = [
            (k, v)
            for k, v in resp.headers.multi_items()
            if k.lower()
            not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(
            resp.status_code, headers=hdrs, content=raw, request=request
        )

    def _raw(self, body: dict) -> bytes:
        if "json" in body:
            return json.dumps(body["json"]).encode()
        if "text" in body:
            return body["text"].encode()
        return gzip.decompress((self.src / "blobs" / f"{body['blob']}.gz").read_bytes())

    def close(self) -> None:
        self.inner.close()


# --- process-wide switch: every httpx client in arena/ takes its transport from here -----------------
_active: dict[str, Any] = {}


def start(
    out_dir: str | Path,
    replay_from: str | Path | None = None,
    live_hosts: frozenset[str] = frozenset(),
) -> Recorder:
    """Record all traffic of this process to ``out_dir``. With ``replay_from``, serve responses from that recording
    instead of the network, except for ``live_hosts`` (e.g. the LLM host, to test a prompt change on recorded tools)."""
    rec = Recorder(out_dir)
    _active["rec"] = rec
    _active["transport"] = RecordingTransport(
        rec, replay_from=replay_from, live_hosts=live_hosts
    )
    return rec


def active() -> Recorder | None:
    return _active.get("rec")


def transport() -> httpx.BaseTransport:
    """The recording transport when a recorder is started, else a plain one."""
    return _active.get("transport") or httpx.HTTPTransport()


def emit(kind: str, **data: Any) -> None:
    """Log an agent-level event (LLM turn, guard decision, parse result); a no-op without a recorder."""
    rec = active()
    if rec:
        rec.emit(kind, **data)
