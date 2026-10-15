"""Polite page fetching: robots.txt, per-host rate limits, retries, proxies and JS rendering.

Every public fetch method returns errors as data (``FetchResult.error``) instead of raising.
"""

from __future__ import annotations

import asyncio
import codecs
import email.utils
import itertools
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .models import Competitor, FetchResult, Settings

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright, ProxySettings

log = logging.getLogger(__name__)

SleepFn = Callable[[float], Awaitable[object]]
Clock = Callable[[], float]

MAX_BODY_BYTES = 8 * 1024 * 1024
ROBOTS_MAX_BYTES = 512 * 1024  # RFC 9309 asks parsers to read at least 500 KiB
MAX_RETRY_AFTER = 60.0
MAX_CRAWL_DELAY = 60.0
DEFAULT_RPM = 60.0
NETWORKIDLE_TIMEOUT_MS = 8_000
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
PLAYWRIGHT_MISSING = "playwright not installed: pip install pricehunter[js]"
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
# The browser picks its own Accept per resource; forcing text/html would break the page's XHRs.
_BROWSER_MANAGED_HEADERS = frozenset({"user-agent", "accept"})
_HEAVY_RESOURCES = frozenset({"image", "media", "font"})
_TEXT_MIME_TYPES = frozenset({"application/xhtml+xml", "application/xml", "application/json"})
_META_CHARSET = re.compile(rb"""<meta[^>]+?charset\s*=\s*["']?\s*([a-zA-Z0-9_.:-]+)""", re.IGNORECASE)
_DELAY_SECONDS = re.compile(r"\d+(?:\.\d+)?")


class RateLimiter:
    """Minimum interval between request starts to the same host.

    Each caller reserves the next free slot for its host synchronously (no ``await`` between
    reading and writing the schedule), so concurrent callers are spaced out without a lock.
    """

    def __init__(self, *, clock: Clock = time.monotonic, sleep: SleepFn = asyncio.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._next_start: dict[str, float] = {}

    async def wait(self, host: str, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            return
        now = self._clock()
        start = max(now, self._next_start.get(host, now))
        self._next_start[host] = start + 60.0 / requests_per_minute
        if start > now:
            await self._sleep(start - now)


class RobotsCache:
    """Fetches and caches ``/robots.txt`` per origin (scheme + host + port).

    Status handling follows RFC 9309 for 4xx: any 4xx, *including 401/403*, means "no usable
    robots.txt" and allows everything (``urllib.robotparser.RobotFileParser.read`` would
    disallow all on 401/403 instead). 5xx and network errors also allow, but are only cached
    for ``error_ttl`` seconds so the file is re-checked soon.
    """

    def __init__(
        self, *, ttl: float = 86_400.0, error_ttl: float = 300.0, clock: Clock = time.monotonic
    ) -> None:
        self._ttl = ttl
        self._error_ttl = error_ttl
        self._clock = clock
        # origin -> (parser, expires_at); a ``None`` parser allows everything.
        self._entries: dict[str, tuple[RobotFileParser | None, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str, user_agent: str) -> bool:
        origin = _origin(url)
        if origin is None:
            return True
        parser = await self._parser(client, origin, user_agent)
        return parser is None or parser.can_fetch(user_agent, url)

    def crawl_delay(self, url: str, user_agent: str) -> float | None:
        """``Crawl-delay`` for ``user_agent`` from the cached file (call after :meth:`allowed`)."""
        origin = _origin(url)
        entry = self._entries.get(origin) if origin else None
        if entry is None or entry[0] is None:
            return None
        delay = entry[0].crawl_delay(user_agent)
        return float(delay) if delay else None

    def _fresh(self, origin: str) -> tuple[RobotFileParser | None, float] | None:
        entry = self._entries.get(origin)
        return entry if entry is not None and entry[1] > self._clock() else None

    async def _parser(
        self, client: httpx.AsyncClient, origin: str, user_agent: str
    ) -> RobotFileParser | None:
        entry = self._fresh(origin)
        if entry is None:
            async with self._locks.setdefault(origin, asyncio.Lock()):
                entry = self._fresh(origin)
                if entry is None:
                    entry = await self._load(client, origin, user_agent)
                    self._entries[origin] = entry
        return entry[0]

    async def _load(
        self, client: httpx.AsyncClient, origin: str, user_agent: str
    ) -> tuple[RobotFileParser | None, float]:
        robots_url = f"{origin}/robots.txt"
        now = self._clock()
        try:
            async with client.stream("GET", robots_url, headers={"User-Agent": user_agent}) as resp:
                if resp.status_code >= 500:
                    log.debug(
                        "robots.txt %s returned HTTP %s; allowing for now", robots_url, resp.status_code
                    )
                    return None, now + self._error_ttl
                if resp.status_code >= 400:
                    return None, now + self._ttl
                body, _ = await _read_capped(resp, ROBOTS_MAX_BYTES)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            log.debug("robots.txt %s unavailable (%s); allowing for now", robots_url, _describe(exc))
            return None, now + self._error_ttl
        parser = RobotFileParser(robots_url)
        parser.parse(body.decode("utf-8", errors="replace").splitlines())
        return parser, now + self._ttl


@dataclass(slots=True)
class _Outcome:
    result: FetchResult
    body: bytes = b""
    retryable: bool = False
    retry_after: float | None = None


class Fetcher:
    """Shared HTTP / headless-browser fetcher. Use as ``async with Fetcher(settings) as f:``.

    ``pages_fetched`` counts HTTP responses received (retries included) plus pages rendered.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_base: float = 0.5,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.pages_fetched = 0
        self.rate_limiter = RateLimiter(sleep=sleep)
        self.robots = RobotsCache()
        self._transport = transport
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max(1, settings.concurrency))
        self._clients: list[httpx.AsyncClient] = []
        self._client_cycle: Iterator[httpx.AsyncClient] = iter(())
        self._proxy_cycle: Iterator[str] = itertools.cycle(settings.proxies)
        self._browser_lock = asyncio.Lock()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> Fetcher:
        # A custom transport (tests) would be bypassed by httpx's proxy mounts, so proxies are ignored.
        proxies = self.settings.proxies if self._transport is None else []
        self._clients = [self._make_client(p) for p in proxies] or [self._make_client(None)]
        self._client_cycle = itertools.cycle(self._clients)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        clients, self._clients = self._clients, []
        for client in clients:
            await client.aclose()
        await self._close_browser()

    async def get(
        self, url: str, competitor: Competitor | None = None, *, render: bool | None = None
    ) -> FetchResult:
        """Fetch ``url`` as text (rendered with a headless browser when ``render``)."""
        if render is None:
            render = competitor.render_js if competitor else False
        return (await self._safe_fetch(url, competitor, render=render, raw=False)).result

    async def get_bytes(self, url: str, competitor: Competitor | None = None) -> tuple[FetchResult, bytes]:
        """Fetch raw bytes (e.g. gzipped sitemaps); ``html`` stays empty, any content type is accepted."""
        outcome = await self._safe_fetch(url, competitor, render=False, raw=True)
        return outcome.result, outcome.body

    # ------------------------------------------------------------------ pipeline

    async def _safe_fetch(
        self, url: str, competitor: Competitor | None, *, render: bool, raw: bool
    ) -> _Outcome:
        try:
            return await self._fetch(url, competitor, render=render, raw=raw)
        except Exception as exc:  # last line of defence: fetching must never raise
            log.exception("unexpected error fetching %s", url)
            return _Outcome(_error(url, f"unexpected error: {exc!r}"))

    async def _fetch(self, url: str, competitor: Competitor | None, *, render: bool, raw: bool) -> _Outcome:
        if not self._clients:
            return _Outcome(_error(url, "fetcher is not open; use 'async with Fetcher(...)'"))
        host = _host_key(url)
        if host is None:
            return _Outcome(_error(url, f"unsupported URL: {url!r}"))
        user_agent = self.settings.user_agent
        if self.settings.respect_robots and not await self.robots.allowed(
            next(self._client_cycle), url, user_agent
        ):
            return _Outcome(_error(url, "blocked by robots.txt"))

        headers = httpx.Headers({"User-Agent": user_agent, **DEFAULT_HEADERS})
        if competitor:
            headers.update(competitor.headers)
        if render:
            browser = await self._ensure_browser()
            if isinstance(browser, str):
                return _Outcome(_error(url, browser))
            attempt = partial(self._render_once, browser, url, headers)
        else:
            attempt = partial(self._http_once, url, headers, raw=raw)
        return await self._with_retries(url, host, self._requests_per_minute(url, competitor), attempt)

    def _requests_per_minute(self, url: str, competitor: Competitor | None) -> float:
        rpm = competitor.requests_per_minute if competitor else DEFAULT_RPM
        if self.settings.respect_robots and (delay := self.robots.crawl_delay(url, self.settings.user_agent)):
            robots_rpm = 60.0 / min(delay, MAX_CRAWL_DELAY)
            rpm = robots_rpm if rpm <= 0 else min(rpm, robots_rpm)
        return rpm

    async def _with_retries(
        self, url: str, host: str, rpm: float, attempt: Callable[[], Awaitable[_Outcome]]
    ) -> _Outcome:
        retries = max(0, self.settings.max_retries)
        for n in range(retries + 1):
            await self.rate_limiter.wait(host, rpm)
            async with self._semaphore:
                outcome = await attempt()
            if not outcome.retryable or n == retries:
                break
            delay = self._backoff_base * 2**n
            delay += random.uniform(0, delay * 0.1)
            if outcome.retry_after is not None:
                delay = max(delay, outcome.retry_after)
            log.debug("retrying %s in %.2fs after: %s", url, delay, outcome.result.error)
            await self._sleep(delay)
        return outcome

    # ------------------------------------------------------------------ plain HTTP

    def _make_client(self, proxy: str | None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport,
            proxy=proxy,
            follow_redirects=True,
            timeout=self.settings.timeout,
            http2=False,
        )

    async def _http_once(self, url: str, headers: httpx.Headers, *, raw: bool) -> _Outcome:
        try:
            async with next(self._client_cycle).stream("GET", url, headers=headers) as resp:
                self.pages_fetched += 1
                return await self._read_response(url, resp, raw=raw)
        except httpx.TransportError as exc:
            return _Outcome(
                _error(url, _describe(exc)), retryable=not isinstance(exc, httpx.UnsupportedProtocol)
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:  # e.g. TooManyRedirects, DecodingError
            return _Outcome(_error(url, _describe(exc)))

    @staticmethod
    async def _read_response(url: str, resp: httpx.Response, *, raw: bool) -> _Outcome:
        final_url = str(resp.url)
        status = resp.status_code
        if not 200 <= status < 300:
            return _status_outcome(url, final_url, status, resp.headers.get("Retry-After"))
        content_type = resp.headers.get("Content-Type", "")
        if not raw and not _is_textual(content_type):
            return _Outcome(
                _error(url, f"unsupported content-type: {_mime(content_type)}", final_url, status)
            )
        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return _Outcome(_error(url, _too_large(), final_url, status))
        body, truncated = await _read_capped(resp, MAX_BODY_BYTES)
        if truncated:
            return _Outcome(_error(url, _too_large(), final_url, status))
        html = "" if raw else _decode(body, resp.charset_encoding)
        return _Outcome(FetchResult(url=url, final_url=final_url, status=status, html=html), body=body)

    # ------------------------------------------------------------------ JS rendering

    async def _ensure_browser(self) -> Browser | str:
        """Start the shared browser on first use; returns an error message on failure."""
        async with self._browser_lock:
            if self._browser is not None:
                return self._browser
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                return PLAYWRIGHT_MISSING
            try:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch()
            except Exception as exc:  # missing browser binary, driver failure, ...
                await self._close_browser()
                return f"browser launch failed: {_first_line(exc)}"
            return self._browser

    async def _render_once(self, browser: Browser, url: str, headers: httpx.Headers) -> _Outcome:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import Route
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        async def skip_heavy(route: Route) -> None:
            # Images, media and fonts never carry prices; skipping them spares the competitor's servers.
            with suppress(PlaywrightError):
                if route.request.resource_type in _HEAVY_RESOURCES:
                    await route.abort()
                else:
                    await route.continue_()

        proxy = _playwright_proxy(next(self._proxy_cycle)) if self.settings.proxies else None
        extra = {k: v for k, v in headers.items() if k not in _BROWSER_MANAGED_HEADERS}
        try:
            context = await browser.new_context(
                user_agent=headers["user-agent"], extra_http_headers=extra, proxy=proxy
            )
        except PlaywrightError as exc:
            return _Outcome(_error(url, _first_line(exc)), retryable=True)
        try:
            await context.route("**/*", skip_heavy)
            page = await context.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=self.settings.timeout * 1000)
            self.pages_fetched += 1
            # ``None`` means no network response was involved (e.g. same-document navigation).
            status = resp.status if resp is not None else 200
            if not 200 <= status < 300:
                retry_after = resp.headers.get("retry-after") if resp is not None else None
                return _status_outcome(url, page.url, status, retry_after)
            content_type = resp.headers.get("content-type", "") if resp is not None else ""
            if not _is_textual(content_type):
                return _Outcome(
                    _error(url, f"unsupported content-type: {_mime(content_type)}", page.url, status)
                )
            with suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT_MS)
            html = await page.content()
            final_url = page.url
        except PlaywrightError as exc:
            return _Outcome(_error(url, _first_line(exc)), retryable=True)
        finally:
            with suppress(PlaywrightError):
                await context.close()
        if len(html.encode("utf-8", errors="replace")) > MAX_BODY_BYTES:
            return _Outcome(_error(url, _too_large(), final_url, status))
        return _Outcome(FetchResult(url=url, final_url=final_url, status=status, html=html, rendered=True))

    async def _close_browser(self) -> None:
        browser, pw = self._browser, self._playwright
        self._browser = self._playwright = None
        if browser is not None:
            with suppress(Exception):
                await browser.close()
        if pw is not None:
            with suppress(Exception):
                await pw.stop()


# ---------------------------------------------------------------------- helpers


def _error(url: str, message: str, final_url: str | None = None, status: int = 0) -> FetchResult:
    return FetchResult(url=url, final_url=final_url or url, status=status, error=message)


def _status_outcome(url: str, final_url: str, status: int, retry_after: str | None) -> _Outcome:
    result = _error(url, f"HTTP {status}", final_url, status)
    if status in RETRY_STATUSES:
        return _Outcome(result, retryable=True, retry_after=parse_retry_after(retry_after))
    return _Outcome(result)


def parse_retry_after(value: str | None, *, cap: float = MAX_RETRY_AFTER) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delay-seconds or HTTP-date), capped."""
    if not value:
        return None
    value = value.strip()
    if _DELAY_SECONDS.fullmatch(value):
        seconds = float(value)
    else:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return min(max(seconds, 0.0), cap)


def _origin(url: str) -> str | None:
    host = _host_key(url)
    return f"{urlsplit(url).scheme.lower()}://{host}" if host else None


def _host_key(url: str) -> str | None:
    """``host[:port]`` for http(s) URLs, ``None`` for anything we cannot fetch."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{host}:{port}" if port else host


def _mime(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_textual(content_type: str) -> bool:
    mime = _mime(content_type)
    return (
        not mime or mime.startswith("text/") or mime in _TEXT_MIME_TYPES or mime.endswith(("+xml", "+json"))
    )


def _too_large() -> str:
    return f"response too large (> {MAX_BODY_BYTES // (1024 * 1024)} MB)"


async def _read_capped(resp: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` (decoded) bytes; the flag says whether the body was longer."""
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf += chunk
        if len(buf) > limit:
            return bytes(buf[:limit]), True
    return bytes(buf), False


def _decode(body: bytes, declared: str | None) -> str:
    """Decode by BOM, then the HTTP charset, then a ``<meta>`` charset, falling back to UTF-8."""
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    candidates = [declared]
    if m := _META_CHARSET.search(body[:4096]):
        candidates.append(m.group(1).decode("ascii"))
    for encoding in candidates:
        if encoding:
            with suppress(LookupError):
                return body.decode(encoding, errors="replace")
    return body.decode("utf-8", errors="replace")


def _describe(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _first_line(exc: BaseException) -> str:
    """Playwright errors carry multi-line call logs; keep the headline."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__


def _playwright_proxy(proxy_url: str) -> ProxySettings:
    parts = urlsplit(proxy_url)
    server = f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    proxy: ProxySettings = {"server": server}
    if parts.username:
        proxy["username"] = unquote(parts.username)
    if parts.password:
        proxy["password"] = unquote(parts.password)
    return proxy
