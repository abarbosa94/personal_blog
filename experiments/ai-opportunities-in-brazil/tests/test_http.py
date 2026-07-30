from __future__ import annotations

import gzip
from io import BytesIO
from urllib.error import HTTPError

import conference_pipeline.http as http_module
from conference_pipeline.http import HttpClient


class Response(BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_http_client_retries_and_caches(monkeypatch, tmp_path) -> None:
    calls = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError("https://example.test", 503, "busy", {}, None)
        return Response(b"payload")

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
    client = HttpClient(cache_dir=tmp_path, backoff_seconds=0)

    assert client.get_bytes("https://example.test/resource") == b"payload"
    assert client.get_bytes("https://example.test/resource") == b"payload"
    assert calls == 2


def test_http_client_does_not_retry_not_found(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise HTTPError("https://example.test", 404, "missing", {}, None)

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)
    client = HttpClient(backoff_seconds=0)

    try:
        client.get_bytes("https://example.test/missing")
    except HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("Expected HTTPError")
    assert calls == 1


def test_http_client_decodes_gzipped_text_from_cache(tmp_path) -> None:
    client = HttpClient(cache_dir=tmp_path)
    url = "https://example.test/page"
    cache_path = client._cache_path(url)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(gzip.compress("AAAI article".encode()))

    assert client.get_text(url) == "AAAI article"
def test_http_timeout_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="timeout_seconds"):
        HttpClient(timeout_seconds=0)


def test_http_client_prefers_retry_after_over_exponential_backoff(
    monkeypatch,
) -> None:
    sleeps: list[float] = []
    calls = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                "https://example.test",
                429,
                "rate limited",
                {"Retry-After": "120"},
                None,
            )
        return Response(b"payload")

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(http_module.time, "sleep", sleeps.append)
    client = HttpClient(backoff_seconds=5, jitter_ratio=0)

    assert client.get_bytes("https://example.test/rate-limited") == b"payload"
    assert sleeps == [120]


def test_http_client_caps_retry_after() -> None:
    error = HTTPError(
        "https://example.test",
        429,
        "rate limited",
        {"Retry-After": "7200"},
        None,
    )
    client = HttpClient(max_backoff_seconds=300, jitter_ratio=0)

    assert client._retry_delay(1, error) == 300
