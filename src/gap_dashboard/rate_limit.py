"""Throttle Alpaca HTTP usage (~90% of account limit by default) and backoff on 429."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def alpaca_requests_per_minute() -> float:
    """Default 180 (~90% of typical 200/min data API limit). Override with ALPACA_MAX_REQUESTS_PER_MINUTE."""
    return _env_float("ALPACA_MAX_REQUESTS_PER_MINUTE", 180.0)


class RateLimiter:
    """Minimum spacing between acquire() calls (thread-safe)."""

    def __init__(self, requests_per_minute: float) -> None:
        rpm = max(float(requests_per_minute), 1.0)
        self._min_interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_alpaca: RateLimiter | None = None


def get_alpaca_limiter() -> RateLimiter:
    global _alpaca
    if _alpaca is None:
        _alpaca = RateLimiter(alpaca_requests_per_minute())
    return _alpaca


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return True
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    # alpaca-py may nest response
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    return False


def call_with_alpaca_throttle(fn: Callable[[], T], *, max_retries: int = 8) -> T:
    """Run fn() after rate-limit acquire; on 429, exponential backoff and retry."""
    limiter = get_alpaca_limiter()
    last: BaseException | None = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            return fn()
        except BaseException as e:
            last = e
            if _is_rate_limit_error(e):
                sleep_s = min(120.0, 4.0 * (2**attempt))
                time.sleep(sleep_s)
                continue
            raise
    assert last is not None
    raise last
