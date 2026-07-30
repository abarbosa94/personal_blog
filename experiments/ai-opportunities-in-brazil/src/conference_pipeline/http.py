from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HttpClient:
    """Small injectable HTTP client used only by online commands."""

    def __init__(
        self,
        user_agent: str = "ai-opportunities-in-brazil/0.1",
        *,
        cache_dir: Path | None = None,
        attempts: int = 3,
        backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 3600.0,
        jitter_ratio: float = 0.1,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.user_agent = user_agent
        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_ratio = jitter_ratio
        self.timeout_seconds = timeout_seconds
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_backoff_seconds < 0:
            raise ValueError("max_backoff_seconds must not be negative")
        if not 0 <= jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between zero and one")

    def get_text(self, url: str) -> str:
        # Official proceedings pages occasionally contain a few invalid UTF-8
        # bytes in otherwise UTF-8 HTML. Preserve the ASCII markup and visible
        # text instead of failing the entire discovery step.
        value = self.get_bytes(url)
        if value.startswith(b"\x1f\x8b"):
            value = gzip.decompress(value)
        return value.decode("utf-8", errors="replace")

    def get_bytes(self, url: str) -> bytes:
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()
        request = Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(1, self.attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    value = response.read()
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_bytes(value)
                    temporary.replace(cache_path)
                return value
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                if attempt == self.attempts:
                    raise
                delay = self._retry_delay(attempt, error)
            except (URLError, TimeoutError):
                if attempt == self.attempts:
                    raise
                delay = self._retry_delay(attempt)
            time.sleep(delay)
        raise RuntimeError("unreachable")

    def get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self.get_text(url))

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.bin"

    def _retry_delay(self, attempt: int, error: HTTPError | None = None) -> float:
        exponential = self.backoff_seconds * (2 ** (attempt - 1))
        retry_after = 0.0
        if error is not None:
            try:
                retry_after = float(error.headers.get("Retry-After", 0))
            except (AttributeError, TypeError, ValueError):
                retry_after = 0.0
        base = min(self.max_backoff_seconds, max(exponential, retry_after))
        if not base or not self.jitter_ratio:
            return base
        return min(
            self.max_backoff_seconds,
            base + random.uniform(0, base * self.jitter_ratio),
        )
