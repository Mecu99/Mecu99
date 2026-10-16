from __future__ import annotations

import asyncio
import email.utils
import gzip
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from pricehunter import fetch as fetch_mod
from pricehunter.fetch import PLAYWRIGHT_MISSING, Fetcher, RateLimiter, RobotsCache, parse_retry_after
from pricehunter.models import Competitor, Settings

UA = Settings().user_agent
HTML = "<html><body><span class='price'>19.99</span></body></html>"


class FakeTime:
    """Clock + sleep pair: sleeping advances the clock instantly and records the duration."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Recorder:
    """Sleep stand-in that records durations without waiting."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def comp(**kw: object) -> Competitor:
    return Competitor(name="shop", base_url="https://shop.test", **{"requests_per_minute": 0, **kw})


def settings(**kw: object) -> Settings:
    return Settings(**{"respect_robots": False, **kw})


def make_fetcher(
    handler: Callable[[httpx.Request], object], *, sleep: Recorder | None = None, **settings_kw: object
) -> Fetcher:
    return Fetcher(
        settings(**settings_kw),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        backoff_base=0.5,
        sleep=sleep or Recorder(),
    )


# ---------------------------------------------------------------------- RateLimiter


async def test_rate_limiter_spaces_concurrent_starts_per_host() -> None:
    ft = FakeTime()
    limiter = RateLimiter(clock=ft, sleep=ft.sleep)
    starts: dict[str, list[float]] = {"a": [], "b": []}

    async def hit(host: str) -> None:
        await limiter.wait(host, 30)  # 2 s interval
        starts[host].append(ft.now)

    await asyncio.gather(*(hit("a") for _ in range(3)), hit("b"))
    assert starts["a"] == [0.0, 2.0, 4.0]
    assert ft.sleeps == [2.0, 2.0]  # host "b" never waited on host "a"


async def test_rate_limiter_unlimited_and_elapsed_interval() -> None:
    ft = FakeTime()
    limiter = RateLimiter(clock=ft, sleep=ft.sleep)
    for _ in range(5):
        await limiter.wait("a", 0)
    assert ft.sleeps == []
    await limiter.wait("a", 60)
    ft.now += 5  # caller was busy for longer than the interval
    await limiter.wait("a", 60)
    assert ft.sleeps == []


async def test_rate_limiter_real_clock() -> None:
    limiter = RateLimiter()
    t0 = time.monotonic()
    await asyncio.gather(*(limiter.wait("h", 1200) for _ in range(4)))  # 50 ms interval
    assert time.monotonic() - t0 >= 0.14


# ---------------------------------------------------------------------- RobotsCache


def robots_transport(
    text: str | None, *, status: int = 200, delay: float = 0.0
) -> tuple[httpx.MockTransport, list[str]]:
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        await asyncio.sleep(delay)
        if text is None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(status, text=text)

    return httpx.MockTransport(handler), hits


async def test_robots_single_fetch_under_concurrency() -> None:
    transport, hits = robots_transport("User-agent: *\nDisallow: /private\n", delay=0.05)
    cache = RobotsCache()
    async with httpx.AsyncClient(transport=transport) as client:
        results = await asyncio.gather(
            *(
                cache.allowed(client, f"https://shop.test/{p}/{i}", UA)
                for i in range(10)
                for p in ("ok", "private")
            )
        )
        assert results == [True, False] * 10
        # Other origins (scheme or port differ) get their own file.
        await cache.allowed(client, "http://shop.test/x", UA)
        await cache.allowed(client, "https://shop.test:8443/x", UA)
    assert hits == [
        "https://shop.test/robots.txt",
        "http://shop.test/robots.txt",
        "https://shop.test:8443/robots.txt",
    ]


async def test_robots_user_agent_specific_rules_and_crawl_delay() -> None:
    transport, _ = robots_transport(
        "User-agent: PriceHunter\nDisallow: /\n\nUser-agent: *\nAllow: /\nCrawl-delay: 3\n"
    )
    cache = RobotsCache()
    async with httpx.AsyncClient(transport=transport) as client:
        assert not await cache.allowed(client, "https://shop.test/p/1", UA)
        assert await cache.allowed(client, "https://shop.test/p/1", "OtherBot/1.0")
    assert cache.crawl_delay("https://shop.test/p/1", "OtherBot/1.0") == 3.0
    assert cache.crawl_delay("https://unknown.test/", UA) is None


@pytest.mark.parametrize("status", [401, 403, 404, 410])
async def test_robots_4xx_allows_all(status: int) -> None:
    transport, hits = robots_transport("User-agent: *\nDisallow: /", status=status)
    cache = RobotsCache()
    async with httpx.AsyncClient(transport=transport) as client:
        assert await cache.allowed(client, "https://shop.test/a", UA)
        assert await cache.allowed(client, "https://shop.test/b", UA)
    assert len(hits) == 1


@pytest.mark.parametrize(
    ("text", "status", "reason"),
    [
        (None, 200, "ConnectError: connection refused"),
        ("oops", 500, "HTTP 500"),
        ("oops", 503, "HTTP 503"),
        ("slow down", 429, "HTTP 429"),
    ],
)
async def test_robots_unreachable_disallows_all_and_is_rechecked_soon(
    text: str | None, status: int, reason: str
) -> None:
    # RFC 9309 2.3.1.4: server or network errors mean "assume complete disallow".
    ft = FakeTime()
    transport, hits = robots_transport(text, status=status)
    cache = RobotsCache(error_ttl=60, clock=ft)
    unavailable = f"robots.txt unavailable ({reason}); retrying later"
    async with httpx.AsyncClient(transport=transport) as client:
        assert await cache.refusal(client, "https://shop.test/a", UA) == unavailable
        ft.now += 30
        assert not await cache.allowed(client, "https://shop.test/b", UA)
        assert len(hits) == 1
        ft.now += 31
        assert await cache.refusal(client, "https://shop.test/a", UA) == unavailable
    assert len(hits) == 2
    assert cache.crawl_delay("https://shop.test/a", UA) is None


async def test_robots_timeout_disallows_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    cache = RobotsCache()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        refusal = await cache.refusal(client, "https://shop.test/a", UA)
    assert refusal == "robots.txt unavailable (ReadTimeout: timed out); retrying later"


async def test_robots_unreachable_keeps_using_the_copy_read_before() -> None:
    ft = FakeTime()
    answers = [httpx.Response(200, text="User-agent: *\nDisallow: /x\nCrawl-delay: 2\n")]
    answers += [httpx.Response(503), httpx.Response(503)]

    def handler(request: httpx.Request) -> httpx.Response:
        return answers.pop(0)

    cache = RobotsCache(ttl=100, error_ttl=10, clock=ft)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await cache.refusal(client, "https://shop.test/x", UA) == "blocked by robots.txt"
        ft.now += 101  # expired; the shop now answers 503, so the old rules stay in force
        assert await cache.refusal(client, "https://shop.test/x", UA) == "blocked by robots.txt"
        assert await cache.allowed(client, "https://shop.test/y", UA)
        assert cache.crawl_delay("https://shop.test/y", UA) == 2.0
        ft.now += 11  # re-checked after error_ttl, still with the previous copy
        assert await cache.allowed(client, "https://shop.test/y", UA)
    assert answers == []


async def test_robots_success_is_cached_until_ttl() -> None:
    ft = FakeTime()
    transport, hits = robots_transport("User-agent: *\nDisallow: /x\n")
    cache = RobotsCache(ttl=100, clock=ft)
    async with httpx.AsyncClient(transport=transport) as client:
        assert not await cache.allowed(client, "https://shop.test/x", UA)
        ft.now += 99
        assert not await cache.allowed(client, "https://shop.test/x", UA)
        ft.now += 2
        assert not await cache.allowed(client, "https://shop.test/x", UA)
    assert len(hits) == 2


# ---------------------------------------------------------------------- Fetcher: robots + rate limit


def site(
    routes: dict[str, httpx.Response | Callable[[httpx.Request], httpx.Response]],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        route = routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, text="not found")
        return route(request) if callable(route) else route

    return handler, seen


def html(body: str = HTML, status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"Content-Type": "text/html; charset=utf-8", **headers})


async def test_fetcher_respects_robots() -> None:
    handler, seen = site(
        {
            "/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /private\n"),
            "/p/1": html(),
            "/private/2": html(),
            "/feed.xml.gz": httpx.Response(200, content=b"x"),
        }
    )
    async with make_fetcher(handler, respect_robots=True) as f:
        ok, blocked = await asyncio.gather(
            f.get("https://shop.test/p/1", comp()), f.get("https://shop.test/private/2", comp())
        )
        res, data = await f.get_bytes("https://shop.test/private/feed.xml.gz", comp())
    assert ok.ok and "19.99" in ok.html
    assert (blocked.status, blocked.error, blocked.final_url) == (0, "blocked by robots.txt", blocked.url)
    assert res.error == "blocked by robots.txt" and data == b""
    assert [r.url.path for r in seen] == ["/robots.txt", "/p/1"]
    assert f.pages_fetched == 1


@pytest.mark.parametrize("status", [429, 503])
async def test_fetcher_fetches_nothing_while_robots_txt_is_unreachable(status: int) -> None:
    handler, seen = site({"/robots.txt": httpx.Response(status), "/checkout/private-area": html()})
    async with make_fetcher(handler, respect_robots=True, max_retries=2) as f:
        page = await f.get("https://shop.test/checkout/private-area", comp())
        _, data = await f.get_bytes("https://shop.test/feed.xml.gz", comp())
    assert (page.ok, page.status, page.html, data) == (False, 0, "", b"")
    assert page.error == f"robots.txt unavailable (HTTP {status}); retrying later"
    assert [r.url.path for r in seen] == ["/robots.txt"]
    assert f.pages_fetched == 0


async def test_fetcher_ignores_robots_when_disabled() -> None:
    handler, seen = site(
        {"/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /\n"), "/p": html()}
    )
    async with make_fetcher(handler, respect_robots=False) as f:
        assert (await f.get("https://shop.test/p", comp())).ok
    assert [r.url.path for r in seen] == ["/p"]


async def test_fetcher_rate_limits_per_host() -> None:
    handler, _ = site({"/p": html()})
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec) as f:
        for _ in range(3):
            await f.get("https://shop.test/p", comp(requests_per_minute=1))
        await f.get("https://other.test/p", comp(requests_per_minute=1))
    # The recorder does not advance time, so the third start is two full intervals out.
    assert rec.sleeps == [pytest.approx(60, abs=1), pytest.approx(120, abs=1)]


async def test_default_rate_without_competitor_is_60_rpm() -> None:
    handler, _ = site({"/p": html()})
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec) as f:
        await f.get("https://shop.test/p")
        await f.get("https://shop.test/p")
    assert rec.sleeps == [pytest.approx(1.0, abs=0.2)]


async def test_crawl_delay_slows_down_competitor_rate() -> None:
    handler, _ = site(
        {"/robots.txt": httpx.Response(200, text="User-agent: *\nCrawl-delay: 5\n"), "/p": html()}
    )
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec, respect_robots=True) as f:
        await f.get("https://shop.test/p", comp(requests_per_minute=600))
        await f.get("https://shop.test/p", comp(requests_per_minute=600))
    assert rec.sleeps == [pytest.approx(5.0, abs=0.2)]


async def test_global_concurrency_limit() -> None:
    inflight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.02)
        inflight -= 1
        return html()

    async with make_fetcher(handler, concurrency=2) as f:
        results = await asyncio.gather(*(f.get(f"https://h{i}.test/p", comp()) for i in range(6)))
    assert all(r.ok for r in results)
    assert peak == 2
    assert f.pages_fetched == 6


# ---------------------------------------------------------------------- Fetcher: retries


def sequence(
    *responses: httpx.Response | Exception,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise type(item)(str(item), request=request)  # type: ignore[call-arg]
        return item

    return handler, seen


async def test_retries_503_then_success() -> None:
    handler, seen = sequence(html("busy", 503), html("busy", 503), html())
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec, max_retries=3) as f:
        res = await f.get("https://shop.test/p", comp())
    assert res.ok and res.status == 200 and "19.99" in res.html
    assert len(seen) == 3
    assert f.pages_fetched == 3
    assert len(rec.sleeps) == 2
    assert 0.5 <= rec.sleeps[0] <= 0.55 and 1.0 <= rec.sleeps[1] <= 1.1


@pytest.mark.parametrize("status", [429, 500, 502, 504])
async def test_other_retryable_statuses(status: int) -> None:
    handler, seen = sequence(html("x", status), html())
    async with make_fetcher(handler) as f:
        assert (await f.get("https://shop.test/p", comp())).ok
    assert len(seen) == 2


async def test_retries_exhausted_returns_last_status() -> None:
    handler, seen = sequence(html("busy", 503))
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec, max_retries=2) as f:
        res = await f.get("https://shop.test/p", comp())
    assert (res.status, res.error, res.ok, res.html) == (503, "HTTP 503", False, "")
    assert len(seen) == 3 and len(rec.sleeps) == 2


def _http_date_in_30s() -> str:
    # Built when the test runs: collection can happen many seconds earlier in a full suite.
    return email.utils.format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)


@pytest.mark.parametrize(
    "make_header,expected",
    [
        (lambda: "7", 7.0),
        (lambda: "3600", 60.0),  # capped
        (_http_date_in_30s, 30.0),
    ],
    ids=["seconds", "capped", "http-date"],
)
async def test_retry_after_is_honoured(make_header: Callable[[], str], expected: float) -> None:
    handler, _ = sequence(html("slow down", 429, **{"Retry-After": make_header()}), html())
    rec = Recorder()
    fetcher = Fetcher(settings(), transport=httpx.MockTransport(handler), backoff_base=0.0, sleep=rec)
    async with fetcher as f:
        assert (await f.get("https://shop.test/p", comp())).ok
    assert rec.sleeps == [pytest.approx(expected, abs=2)]


def test_parse_retry_after() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after(" 12 ") == 12.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after("999", cap=10) == 10
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


async def test_404_is_not_retried() -> None:
    handler, seen = sequence(html("gone", 404))
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec) as f:
        res = await f.get("https://shop.test/p", comp())
    assert (res.status, res.error) == (404, "HTTP 404")
    assert len(seen) == 1 and rec.sleeps == []
    assert f.pages_fetched == 1


async def test_network_error_is_returned_not_raised() -> None:
    handler, seen = sequence(httpx.ConnectError("connection refused"))
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec, max_retries=2) as f:
        res = await f.get("https://shop.test/p", comp())
    assert res.status == 0 and not res.ok
    assert res.error == "ConnectError: connection refused"
    assert len(seen) == 3 and len(rec.sleeps) == 2
    assert f.pages_fetched == 0


async def test_network_error_then_success() -> None:
    handler, seen = sequence(httpx.ReadTimeout("timed out"), html())
    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/p", comp())
    assert res.ok and len(seen) == 2


async def test_redirect_loop_is_not_retried() -> None:
    handler, _ = site({"/loop": httpx.Response(302, headers={"Location": "/loop"})})
    rec = Recorder()
    async with make_fetcher(handler, sleep=rec) as f:
        res = await f.get("https://shop.test/loop", comp())
    assert res.status == 0 and res.error is not None and res.error.startswith("TooManyRedirects")
    assert rec.sleeps == []


async def test_unexpected_exception_becomes_error_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("bug in transport")

    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/p", comp())
    assert res.status == 0 and res.error is not None and "bug in transport" in res.error


@pytest.mark.parametrize(
    "url", ["ftp://shop.test/file", "not a url", "http://shop.test:port/", "https:///nohost"]
)
async def test_unsupported_urls(url: str) -> None:
    handler, seen = site({})
    async with make_fetcher(handler) as f:
        res = await f.get(url)
    assert res.status == 0 and res.error is not None and res.error.startswith("unsupported URL")
    assert seen == []


async def test_fetcher_not_open_returns_error() -> None:
    handler, _ = site({"/p": html()})
    res = await make_fetcher(handler).get("https://shop.test/p")
    assert res.error is not None and "not open" in res.error


# ---------------------------------------------------------------------- Fetcher: responses


async def test_success_follows_redirects() -> None:
    handler, _ = site({"/old": httpx.Response(301, headers={"Location": "/new?x=1"}), "/new": html()})
    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/old", comp())
    assert res.ok and not res.rendered
    assert res.url == "https://shop.test/old"
    assert res.final_url == "https://shop.test/new?x=1"
    assert f.pages_fetched == 1


async def test_headers_merge_with_competitor_overrides() -> None:
    handler, seen = site({"/p": html()})
    async with make_fetcher(handler) as f:
        await f.get("https://shop.test/p")
        await f.get(
            "https://shop.test/p",
            comp(headers={"accept-language": "ro-RO,ro;q=0.9", "user-agent": "Custom/1", "X-Api": "k"}),
        )
    default, custom = seen
    assert default.headers["User-Agent"] == UA
    assert default.headers["Accept-Language"] == "en-US,en;q=0.9"
    assert "text/html" in default.headers["Accept"]
    assert custom.headers.get_list("Accept-Language") == ["ro-RO,ro;q=0.9"]
    assert custom.headers.get_list("User-Agent") == ["Custom/1"]
    assert custom.headers["X-Api"] == "k"
    assert "text/html" in custom.headers["Accept"]


@pytest.mark.parametrize("ctype", ["image/png", "application/pdf", "application/octet-stream", "video/mp4"])
async def test_binary_content_type_rejected(ctype: str) -> None:
    handler, _ = site(
        {"/p": httpx.Response(200, content=b"\x89PNG", headers={"Content-Type": f"{ctype}; q=1"})}
    )
    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/p", comp())
    assert (res.status, res.error, res.html) == (200, f"unsupported content-type: {ctype}", "")


@pytest.mark.parametrize(
    "ctype",
    [
        "text/html",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
        "application/json",
        "application/rss+xml",
        "",
    ],
)
async def test_textual_content_types_accepted(ctype: str) -> None:
    headers = {"Content-Type": ctype} if ctype else {}
    handler, _ = site({"/p": httpx.Response(200, content=b"<x>1</x>", headers=headers)})
    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/p", comp())
    assert res.ok and res.html == "<x>1</x>"


async def test_body_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_mod, "MAX_BODY_BYTES", 100)

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"x" * 20

    handler, _ = site(
        {
            "/declared": html("x" * 200),
            "/chunked": lambda r: httpx.Response(
                200, content=chunks(), headers={"Content-Type": "text/html"}
            ),
            "/small": html("x" * 100),
        }
    )
    async with make_fetcher(handler) as f:
        declared = await f.get("https://shop.test/declared", comp())
        chunked = await f.get("https://shop.test/chunked", comp())
        small = await f.get("https://shop.test/small", comp())
    assert declared.error is not None and declared.error.startswith("response too large")
    assert chunked.error is not None and chunked.error.startswith("response too large")
    assert small.ok


async def test_charset_detection() -> None:
    text = "<p>Cena: 129,00 zł</p>"
    handler, _ = site(
        {
            "/meta": httpx.Response(
                200,
                content=f'<meta charset="windows-1250">{text}'.encode("cp1250"),
                headers={"Content-Type": "text/html"},
            ),
            "/header": httpx.Response(
                200,
                content=text.encode("iso-8859-2"),
                headers={"Content-Type": "text/html; charset=ISO-8859-2"},
            ),
            "/bom": httpx.Response(
                200, content=b"\xef\xbb\xbf" + text.encode(), headers={"Content-Type": "text/html"}
            ),
            "/bogus": httpx.Response(
                200, content=f'<meta charset="nope-42">{text}'.encode(), headers={"Content-Type": "text/html"}
            ),
        }
    )
    async with make_fetcher(handler) as f:
        for path in ("/meta", "/header", "/bom", "/bogus"):
            res = await f.get(f"https://shop.test{path}", comp())
            assert "129,00 zł" in res.html, path
            assert not res.html.startswith("﻿")


async def test_get_bytes_returns_raw_body_for_any_content_type() -> None:
    payload = gzip.compress(b"<urlset><url><loc>https://shop.test/p/1</loc></url></urlset>")
    handler, _ = site(
        {
            "/sitemap.xml.gz": httpx.Response(
                200, content=payload, headers={"Content-Type": "application/x-gzip"}
            )
        }
    )
    async with make_fetcher(handler) as f:
        res, data = await f.get_bytes("https://shop.test/sitemap.xml.gz", comp())
        as_text = await f.get("https://shop.test/sitemap.xml.gz", comp())
        missing, nothing = await f.get_bytes("https://shop.test/nope.gz", comp())
    assert res.ok and res.html == "" and data == payload
    assert as_text.error == "unsupported content-type: application/x-gzip"
    assert (missing.status, missing.error, nothing) == (404, "HTTP 404", b"")
    assert f.pages_fetched == 3


async def test_get_bytes_retries_like_get() -> None:
    handler, seen = sequence(httpx.Response(502), httpx.Response(200, content=b"\x1f\x8b"))
    async with make_fetcher(handler) as f:
        res, data = await f.get_bytes("https://shop.test/s.gz", comp())
    assert res.ok and data == b"\x1f\x8b" and len(seen) == 2


# ---------------------------------------------------------------------- real local servers


class _ServerState:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]

    def headers_for(self, path: str) -> dict[str, str]:
        return next(h for p, h in self.requests if p == path)


PRODUCT_PAGE = b"""<!doctype html><html><head><title>Widget</title></head><body>
<h1>Widget</h1><img src="/img.png"><div id="price">loading</div>
<script>
fetch('/api/price').then(r => r.json()).then(d => {
  document.getElementById('price').textContent = d.price + ' EUR';
});
</script></body></html>"""


@contextmanager
def serve(handler_cls: type[BaseHTTPRequestHandler]) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def shop() -> Iterator[tuple[str, _ServerState]]:
    state = _ServerState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            state.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
            routes: dict[str, tuple[int, str, bytes]] = {
                "/robots.txt": (200, "text/plain", b"User-agent: *\nDisallow: /private\n"),
                "/product": (200, "text/html; charset=utf-8", PRODUCT_PAGE),
                "/api/price": (200, "application/json", b'{"price": "149.99"}'),
                "/img.png": (200, "image/png", b"\x89PNG"),
                "/private": (200, "text/html", b"<p>secret</p>"),
            }
            status, ctype, body = routes.get(self.path, (404, "text/html", b"<p>missing</p>"))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with serve(Handler) as server:
        yield f"http://127.0.0.1:{server.server_port}", state


async def test_render_executes_javascript(shop: tuple[str, _ServerState]) -> None:
    base, state = shop
    competitor = Competitor(
        name="js-shop", base_url=base, render_js=True, requests_per_minute=0, headers={"X-Probe": "yes"}
    )
    async with Fetcher(Settings(respect_robots=True, timeout=10)) as f:
        rendered = await f.get(f"{base}/product", competitor)
        plain = await f.get(f"{base}/product", competitor, render=False)
        blocked = await f.get(f"{base}/private", competitor)
        missing = await f.get(f"{base}/missing", competitor)

    assert rendered.ok and rendered.rendered and rendered.status == 200
    assert "149.99 EUR" in rendered.html
    assert rendered.final_url == f"{base}/product"
    assert not plain.rendered and "149.99" not in plain.html and "loading" in plain.html
    assert blocked.error == "blocked by robots.txt"
    assert (missing.status, missing.error, missing.rendered) == (404, "HTTP 404", False)
    assert f.pages_fetched == 3  # rendered product, plain product, missing page

    page_headers = state.headers_for("/product")
    assert page_headers["user-agent"] == UA
    assert page_headers["x-probe"] == "yes"
    assert page_headers["accept-language"] == "en-US,en;q=0.9"
    api_headers = state.headers_for("/api/price")  # the page's own XHR
    assert "text/html" not in api_headers["accept"]  # browser picks Accept per resource
    assert api_headers["x-probe"] == "yes"
    assert "/img.png" not in state.paths()  # heavy resources are skipped
    assert "/private" not in state.paths()
    assert state.paths().count("/robots.txt") == 1


async def test_render_without_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)
    handler, seen = site({"/p": html()})
    async with make_fetcher(handler) as f:
        res = await f.get("https://shop.test/p", comp(render_js=True))
    assert (res.status, res.error, res.rendered) == (0, PLAYWRIGHT_MISSING, False)
    assert seen == []


async def test_proxies_rotate_round_robin() -> None:
    hits: list[tuple[int, str]] = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            port: int = self.connection.getsockname()[1]
            hits.append((port, self.path))  # forward proxies receive the absolute URL
            body = f"<p>via {port}</p>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with serve(Proxy) as p1, serve(Proxy) as p2:
        ports = [p1.server_port, p2.server_port]
        cfg = settings(proxies=[f"http://127.0.0.1:{port}" for port in ports])
        async with Fetcher(cfg) as f:
            results = [await f.get("http://shop.invalid/p", comp()) for _ in range(3)]
    assert all(r.ok for r in results)
    assert [port for port, _ in hits] == [ports[0], ports[1], ports[0]]
    assert {path for _, path in hits} == {"http://shop.invalid/p"}
