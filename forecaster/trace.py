"""Per-call trace log: thread-safe JSONL appender and logging wrapper client."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


def prompt_hash(messages) -> str:
    """First 12 hex chars of sha1 of the system message content."""
    content = ""
    if isinstance(messages, (list, tuple)):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content", "")
                break
    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]


class CallLog:
    """Thread-safe JSONL appender."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            parent = self.path.parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


_STATS_KEYS = ("retries", "status_codes", "prompt_tokens", "completion_tokens")


class LoggingClient:
    """Wraps an inner client, logging one trace record per call.

    Never logs message content or credentials — only the prompt hash.
    """

    accepts_meta = True

    def __init__(self, inner, log: CallLog, run_id: str):
        self.inner = inner
        self.log = log
        self.run_id = run_id

    def complete(self, messages, model, temperature,
                 reasoning_effort=None, meta=None, response_format=None):
        start = time.monotonic()
        raw = ""
        error = None
        try:
            if reasoning_effort is None and response_format is None:
                out = self.inner.complete(messages, model, temperature)
            else:
                kwargs: dict = {}
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                if response_format is not None:
                    kwargs["response_format"] = response_format
                try:
                    out = self.inner.complete(messages, model, temperature,
                                              **kwargs)
                except TypeError:
                    if reasoning_effort is None:
                        out = self.inner.complete(messages, model, temperature)
                    else:
                        out = self.inner.complete(
                            messages, model, temperature,
                            reasoning_effort=reasoning_effort)
            raw = out if isinstance(out, str) else ("" if out is None else str(out))
            return out
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            record: dict = {
                "run_id": self.run_id,
                "ts": time.time(),
                "model": model,
                "effort": reasoning_effort,
                "prompt_hash": prompt_hash(messages),
                "meta": meta or {},
                "latency_s": time.monotonic() - start,
                "raw": raw,
                "error": error,
                "structured": response_format is not None,
            }
            stats = getattr(self.inner, "last_stats", None)
            if isinstance(stats, dict):
                for key in _STATS_KEYS:
                    if key in stats:
                        record[key] = stats[key]
            try:
                self.log.write(record)
            except OSError:
                pass
