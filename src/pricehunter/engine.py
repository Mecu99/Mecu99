"""Scan orchestration: discovery, extraction, matching, storage, market analysis and alerts.

:func:`run_scan` is the heart of PriceHunter. For every selected product x enabled competitor
it registers pinned URLs, (re)discovers candidate pages when needed, reads every listing, and
records one :class:`Observation` per listing. It then rebuilds each product's market snapshot,
suggests a price and raises alerts. Failures are recorded as data; one broken competitor or
product never stops the rest of the scan.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import unquote, urlsplit

from .alerts import dispatch, evaluate_alerts
from .analyze import market_snapshot, missing_rates, product_currency, reprice
from .discover import discover_candidates
from .extract import VERDICTS, extract_offer
from .fetch import Fetcher
from .match import score_match
from .models import (
    Alert,
    AppConfig,
    Availability,
    Competitor,
    ExtractedOffer,
    FetchResult,
    Listing,
    MarketSnapshot,
    MatchResult,
    Observation,
    PriceSuggestion,
    Product,
    ScanResult,
    ScanStats,
    Settings,
    utcnow,
)
from .money import convert
from .storage import Storage

log = logging.getLogger(__name__)

Progress = Callable[[str], None]

PINNED = MatchResult(score=1.0, reason="pinned")
MAX_MATCHES_PER_PAIR = 2  # allows a second variant listing without letting noise in
IDENTIFIED = frozenset({"gtin", "mpn"})  # match reasons that pin down the exact item
# A new title-only match priced under this share of ours is likelier a part than a bargain.
MIN_PLAUSIBLE_PRICE_RATIO = Decimal("0.4")
# A listing's last good reading stands in for failed fetches this long, then stops counting.
FALLBACK_MAX_AGE = timedelta(days=7)
MATCH_LOST = "match lost"
NO_PRICE = "no price found (check the competitor's selectors)"
_GONE = frozenset({404, 410})

# --------------------------------------------------------------------------- selection


def _select_products(config: AppConfig, skus: list[str] | None) -> tuple[list[Product], list[str]]:
    """Products to work on (all when ``skus`` is None) and an error per unknown SKU."""
    if skus is None:
        return list(config.products), []
    wanted = set(skus)
    errors = [f"unknown SKU {sku!r}" for sku in dict.fromkeys(skus) if config.product(sku) is None]
    return [p for p in config.products if p.sku in wanted], errors


def _select_competitors(
    config: AppConfig, names: list[str] | None
) -> tuple[list[Competitor], list[str]]:
    if names is None:
        return [c for c in config.competitors if c.enabled], []
    errors: list[str] = []
    for name in dict.fromkeys(names):
        found = config.competitor(name)
        if found is None or not found.enabled:
            errors.append(f"{'unknown' if found is None else 'disabled'} competitor {name!r}")
    wanted = set(names)
    return [c for c in config.competitors if c.enabled and c.name in wanted], errors


# --------------------------------------------------------------------------- one run


@dataclass
class _Run:
    """State shared by every (product, competitor) task of one scan."""

    config: AppConfig
    storage: Storage
    run_id: int
    discover: bool
    progress: Progress | None
    stats: ScanStats
    fetcher: Fetcher = field(init=False)
    pages: dict[tuple[str, bool], asyncio.Task[FetchResult]] = field(default_factory=dict)
    sitemaps: dict[str, list[str]] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    pairs: dict[str, list[tuple[Observation, Observation | None]]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def settings(self) -> Settings:
        return self.config.settings

    def say(self, message: str) -> None:
        log.info(message)
        if self.progress is not None:
            self.progress(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.say(message)

    async def get(
        self, url: str, competitor: Competitor | None = None, *, render: bool | None = None
    ) -> FetchResult:
        """Fetch each page once per run, also under its final URL after redirects.

        The run is the :class:`~pricehunter.discover.PageFetcher` handed to discovery, so a
        search that redirects to a product page and the later read of that page share one
        request, and so do candidates and listings.
        """
        if render is None:
            render = competitor.render_js if competitor else False
        task = self.pages.get((url, render))
        if task is None:
            task = asyncio.ensure_future(self.fetcher.get(url, competitor, render=render))
            self.pages[(url, render)] = task
        result = await task
        self.pages.setdefault((result.final_url, render), task)
        return result

    async def get_bytes(
        self, url: str, competitor: Competitor | None = None
    ) -> tuple[FetchResult, bytes]:
        return await self.fetcher.get_bytes(url, competitor)

    async def guard(self, label: str, work: Coroutine[Any, Any, None]) -> None:
        """Run one unit of work; an unexpected exception is recorded instead of ending the scan."""
        try:
            await work
        except Exception as exc:
            log.exception("%s failed", label)
            self.fail(f"{label}: {type(exc).__name__}: {exc}")


def _extract(
    result: FetchResult, competitor: Competitor, product: Product | None = None
) -> ExtractedOffer:
    return extract_offer(
        result.html, result.final_url, competitor.selectors, competitor.currency, product
    )


def _needs_discovery(
    run: _Run, product: Product, competitor: Competitor, listings: list[Listing]
) -> bool:
    """Discover when the pair has no discovered listing, or its last discovery is stale."""
    if not any(not listing.pinned for listing in listings):
        return True
    last = run.storage.last_discovered(product.sku, competitor.name)
    max_age = timedelta(hours=run.settings.rediscover_after_hours)
    return last is None or utcnow() - last > max_age


def _implausibly_cheap(
    settings: Settings, product: Product, offer: ExtractedOffer, match: MatchResult
) -> bool:
    """A title-only match far below our price: an accessory or spare part the matcher missed
    would otherwise set the market low and raise false undercut alerts."""
    if match.reason != "title" or offer.price is None or not product.our_price:
        return False
    rates, base = settings.currency_rates, settings.base_currency
    price = convert(offer.price, offer.currency, product_currency(product), rates, base)
    return price is not None and price < product.our_price * MIN_PLAUSIBLE_PRICE_RATIO


def _moved(url: str, final_url: str) -> bool:
    """Whether a request ended on another page: a different path, not merely another scheme,
    host spelling, letter case, percent-encoding or trailing slash."""

    def page(address: str) -> str:
        return unquote(urlsplit(address).path).rstrip("/").lower()

    return page(url) != page(final_url)


async def _discover(
    run: _Run, product: Product, competitor: Competitor, listings: list[Listing]
) -> None:
    """Find candidate pages, keep the best matches as listings and remember when we looked.

    The pair's current listings compete with the new matches (and win ties), so a better page
    replaces a weak listing, which is deactivated. Once a GTIN or MPN pins the exact item down,
    title-only look-alikes (a used unit, another variant) are not kept next to it.
    """
    settings = run.settings
    listed = {listing.url for listing in listings}
    candidates = await discover_candidates(
        run, competitor, product, settings.max_candidates_per_competitor, run.sitemaps
    )
    run.say(f"{competitor.name}: {len(candidates)} candidate(s) for {product.sku}")

    async def check(url: str) -> tuple[MatchResult, str, ExtractedOffer] | None:
        result = await run.get(url, competitor)
        run.stats.candidates_checked += 1
        if not result.ok or result.final_url in listed:
            return None
        offer = _extract(result, competitor, product)
        if offer.method in VERDICTS:  # a category grid, or variants we cannot tell apart
            run.say(f"{competitor.name}: skipped {result.final_url}: {VERDICTS[offer.method]}")
            return None
        return score_match(product, offer), result.final_url, offer

    checked = await asyncio.gather(*(check(url) for url in candidates if url not in listed))
    found: dict[str, MatchResult] = {}
    for match, url, offer in sorted(filter(None, checked), key=lambda c: -c[0].score):
        if match.score < settings.match_threshold or url in found:
            continue
        if _implausibly_cheap(settings, product, offer, match):
            price = f"{offer.price} {offer.currency or ''}".strip()
            run.say(f"{competitor.name}: skipped {url} for {product.sku}: {price} is too cheap")
            continue
        found[url] = match
    current = {
        listing.url: MatchResult(score=listing.match_score, reason=listing.match_reason)
        for listing in listings
        if not listing.pinned
    }
    ranked = sorted({**current, **found}.items(), key=lambda item: -item[1].score)
    if any(match.reason in IDENTIFIED for _, match in ranked):
        ranked = [(url, match) for url, match in ranked if match.reason in IDENTIFIED]
    keep = dict(ranked[:MAX_MATCHES_PER_PAIR])
    for url in (url for url in current if url not in keep):
        run.storage.deactivate_listing(product.sku, competitor.name, url)
        run.say(f"{competitor.name}: replaced {url} for {product.sku} with a better match")
    for url, match in keep.items():
        if url in current:
            continue
        run.storage.upsert_listing(
            Listing(
                sku=product.sku,
                competitor=competitor.name,
                url=url,
                match_score=match.score,
                match_reason=match.reason,
            )
        )
        run.stats.listings_matched += 1
        run.say(
            f"{competitor.name}: matched {product.sku} -> {url} ({match.score:.2f}, {match.reason})"
        )
    run.storage.mark_discovered(product.sku, competitor.name)


def _pinned_page_lost(
    settings: Settings, product: Product, result: FetchResult, offer: ExtractedOffer
) -> str | None:
    """Why a pinned URL that now redirects elsewhere no longer shows ``product``, if it does not.

    Shops send a delisted product's URL to its category or home page. Pinned pages are trusted
    without matching, so a redirected one has to pass the matcher like a discovered page.
    """
    match = score_match(product, offer)
    if offer.method not in VERDICTS and match.score >= settings.match_threshold:
        return None
    why = VERDICTS.get(offer.method) or f"match {match.score:.2f}"
    return f"pinned URL now redirects to {result.final_url} ({why}); pin the product's new URL"


async def _observe(run: _Run, product: Product, competitor: Competitor, listing: Listing) -> None:
    """Read one listing and store the observation (errors included)."""
    result = await run.get(listing.url, competitor)
    obs = Observation(
        run_id=run.run_id,
        sku=product.sku,
        competitor=competitor.name,
        url=listing.url,
        match_score=listing.match_score,
        match_reason=listing.match_reason,
    )
    if not result.ok:
        error = result.error or f"HTTP {result.status}"
        if result.status in _GONE and not listing.pinned:
            run.storage.deactivate_listing(product.sku, competitor.name, listing.url)
            error = f"{error} (listing deactivated)"
        obs.error = error
        run.fail(f"{competitor.name}: {product.sku}: {listing.url}: {error}")
    else:
        offer = _extract(result, competitor, product)
        match = PINNED if listing.pinned else score_match(product, offer)
        error = None
        if listing.pinned and _moved(listing.url, result.final_url):
            error = _pinned_page_lost(run.settings, product, result, offer)
        lost = error is not None  # the page no longer shows this listing's product
        if lost:
            run.fail(f"{competitor.name}: {product.sku}: {listing.url}: {error}")
        elif match.score < run.settings.match_threshold:
            run.storage.deactivate_listing(product.sku, competitor.name, listing.url)
            lost, error = True, MATCH_LOST
            run.say(
                f"{competitor.name}: {listing.url} no longer matches {product.sku}"
                f" ({match.score:.2f})"
            )
        elif offer.price is None and offer.availability is not Availability.OUT_OF_STOCK:
            verdict = VERDICTS.get(offer.method)
            error = f"no price: {verdict}" if verdict else NO_PRICE
            run.fail(f"{competitor.name}: {product.sku}: {listing.url}: {error}")
        obs = obs.model_copy(
            update={
                "title": offer.title,
                "price": offer.price,
                "currency": offer.currency,
                "availability": offer.availability,
                "original_price": offer.original_price,
                "shipping": offer.shipping,
                "match_score": match.score,
                "match_reason": match.reason,
                "method": offer.method,
                "error": error,
            }
        )
        if not lost:
            run.storage.upsert_listing(
                listing.model_copy(
                    update={
                        "match_score": match.score,
                        "match_reason": match.reason,
                        "last_seen_at": obs.observed_at,
                    }
                )
            )
    obs_id = run.storage.add_observation(obs)
    obs = obs.model_copy(update={"id": obs_id})
    run.observations.append(obs)
    run.stats.observations += 1
    if obs.error is None:
        prev = run.storage.previous_observation(obs.sku, obs.competitor, obs.url, obs_id)
        run.pairs.setdefault(product.sku, []).append((obs, prev))
        price = f"{obs.price} {obs.currency or ''}".strip() if obs.price is not None else "no price"
        run.say(f"{competitor.name}: {product.sku} {price} ({obs.availability.value})")


async def _scan_pair(run: _Run, product: Product, competitor: Competitor) -> None:
    storage = run.storage
    pinned = list(dict.fromkeys(product.competitor_urls.get(competitor.name, [])))
    for url in pinned:
        storage.upsert_listing(
            Listing(
                sku=product.sku,
                competitor=competitor.name,
                url=url,
                match_score=PINNED.score,
                match_reason=PINNED.reason,
                pinned=True,
            )
        )
    listings: list[Listing] = []
    for listing in storage.listings(product.sku, competitor.name):
        # Unpinned in the config since, or a discovered page that pinned URLs now replace.
        if listing.url not in pinned and (listing.pinned or bool(pinned)):
            storage.deactivate_listing(listing.sku, listing.competitor, listing.url)
        else:
            listings.append(listing)
    # Pinned URLs replace discovery and its listings for this pair (see Product.competitor_urls).
    if run.discover and not pinned and _needs_discovery(run, product, competitor, listings):
        await _discover(run, product, competitor, listings)
    await asyncio.gather(
        *(
            run.guard(
                f"{competitor.name}: {product.sku}: {listing.url}",
                _observe(run, product, competitor, listing),
            )
            for listing in storage.listings(product.sku, competitor.name)
        )
    )


def _readings(
    config: AppConfig, storage: Storage, product: Product, before_run: int | None = None
) -> list[Observation]:
    """The latest good reading of each of ``product``'s listings that is still part of its market.

    Competitors that are disabled or no longer configured drop out, and so do discovered pages
    where the product now pins URLs for that competitor. A last good reading stands in for
    failed fetches for at most :data:`FALLBACK_MAX_AGE`.
    """
    enabled = {c.name for c in config.competitors if c.enabled}
    pins = product.competitor_urls
    latest = storage.latest_observations(
        product.sku, before_run=before_run, max_fallback_age=FALLBACK_MAX_AGE
    )
    return [
        obs
        for obs in latest
        if obs.competitor in enabled
        and (not pins.get(obs.competitor) or obs.url in pins[obs.competitor])
    ]


def _snapshot(
    config: AppConfig, product: Product, observations: list[Observation]
) -> MarketSnapshot:
    settings = config.settings
    return market_snapshot(
        product, observations, settings, min_match_score=settings.match_threshold
    )


def _analyze(
    config: AppConfig, product: Product, observations: list[Observation]
) -> tuple[MarketSnapshot, PriceSuggestion]:
    """Market and repricing suggestion for ``product`` from its :func:`_readings`."""
    settings = config.settings
    suggestion = reprice(
        product, observations, settings, config.repricing, settings.match_threshold
    )
    return _snapshot(config, product, observations), suggestion


def _rate_errors(
    config: AppConfig, markets: Iterable[tuple[Product, list[Observation]]]
) -> list[str]:
    """One error per currency without a rate, naming the competitors it leaves out."""
    missing: dict[str, set[str]] = {}
    for product, readings in markets:
        for code, names in missing_rates(product, readings, config.settings).items():
            missing.setdefault(code, set()).update(names)
    return [
        f"no currency rate for {code} in settings.currency_rates:"
        f" prices from {', '.join(sorted(names))} are left out of the market"
        for code, names in sorted(missing.items())
    ]


async def _collect(
    run: _Run, products: list[Product], competitors: list[Competitor], fetcher: Fetcher | None
) -> None:
    """Scan every (product, competitor) pair concurrently; the Fetcher paces the requests."""
    try:
        async with AsyncExitStack() as stack:
            if fetcher is None:
                fetcher = await stack.enter_async_context(Fetcher(run.settings))
            run.fetcher = fetcher
            pages_before = fetcher.pages_fetched
            await asyncio.gather(
                *(
                    run.guard(f"{c.name}: {p.sku}", _scan_pair(run, p, c))
                    for p in products
                    for c in competitors
                )
            )
            run.stats.pages_fetched = fetcher.pages_fetched - pages_before
    except Exception as exc:  # e.g. an invalid proxy URL; the stored data stays usable
        log.exception("scan %s aborted", run.run_id)
        run.fail(f"scan aborted: {type(exc).__name__}: {exc}")


def _evaluate(
    run: _Run, products: list[Product]
) -> tuple[list[MarketSnapshot], list[PriceSuggestion], list[Alert]]:
    """Snapshot, suggestion and alerts per product, comparing with the market before this run."""
    config, storage = run.config, run.storage
    snapshots: list[MarketSnapshot] = []
    suggestions: list[PriceSuggestion] = []
    alerts: list[Alert] = []
    markets: list[tuple[Product, list[Observation]]] = []
    for product in products:
        try:
            current = _readings(config, storage, product)
            snapshot, suggestion = _analyze(config, product, current)
            previous = _readings(config, storage, product, before_run=run.run_id)
            prev_snapshot = _snapshot(config, product, previous) if previous else None
            pairs = run.pairs.get(product.sku, [])
            found = evaluate_alerts(product, pairs, snapshot, prev_snapshot, config.alerts)
        except Exception as exc:
            log.exception("analysis of %s failed", product.sku)
            run.fail(f"{product.sku}: analysis failed: {type(exc).__name__}: {exc}")
            continue
        markets.append((product, current))
        snapshots.append(snapshot)
        suggestions.append(suggestion)
        alerts.extend(found)
    for error in _rate_errors(config, markets):
        run.fail(error)
    return snapshots, suggestions, alerts


async def run_scan(
    config: AppConfig,
    storage: Storage,
    *,
    skus: list[str] | None = None,
    competitors: list[str] | None = None,
    discover: bool = True,
    notify: bool = True,
    fetcher: Fetcher | None = None,
    progress: Progress | None = None,
) -> ScanResult:
    """Scan the selected products on the selected (enabled) competitors and store the results.

    A ``fetcher`` passed in must already be open (``async with``); otherwise one is created
    from ``config.settings``. ``progress`` receives short human-readable status lines.
    Problems are reported in ``ScanResult.errors``; the run is always finished in storage.
    """
    products, errors = _select_products(config, skus)
    targets, competitor_errors = _select_competitors(config, competitors)
    run_id = storage.start_run()
    stats = ScanStats(run_id=run_id, products=len(products))
    run = _Run(
        config, storage, run_id, discover, progress, stats, errors=errors + competitor_errors
    )
    snapshots: list[MarketSnapshot] = []
    suggestions: list[PriceSuggestion] = []
    alerts: list[Alert] = []
    try:
        await _collect(run, products, targets, fetcher)
        snapshots, suggestions, alerts = _evaluate(run, products)
        storage.add_alerts(run_id, alerts)
        stats.alerts = len(alerts)
        if notify and alerts:
            run.errors.extend(await dispatch(alerts, config.notifiers))
    finally:
        stats.errors = len(run.errors)
        stats.finished_at = utcnow()
        storage.finish_run(run_id, stats)
    return ScanResult(
        stats=stats,
        observations=run.observations,
        snapshots=snapshots,
        suggestions=suggestions,
        alerts=alerts,
        errors=run.errors,
    )


# --------------------------------------------------------------------------- reporting helpers


def build_snapshots(
    config: AppConfig, storage: Storage, skus: list[str] | None = None
) -> tuple[list[MarketSnapshot], list[PriceSuggestion]]:
    """Market snapshots and repricing suggestions from stored observations (no network)."""
    analyzed = [
        _analyze(config, p, _readings(config, storage, p))
        for p in _select_products(config, skus)[0]
    ]
    return [s for s, _ in analyzed], [g for _, g in analyzed]


def market_warnings(
    config: AppConfig, storage: Storage, skus: list[str] | None = None
) -> list[str]:
    """Problems that silently keep stored prices out of :func:`build_snapshots` (no network)."""
    products = _select_products(config, skus)[0]
    return _rate_errors(config, ((p, _readings(config, storage, p)) for p in products))


def load_history(
    config: AppConfig, storage: Storage, skus: list[str] | None = None, days: float | None = 30
) -> dict[str, list[Observation]]:
    """Successful observations per SKU (oldest first), limited to the last ``days`` days."""
    since = utcnow() - timedelta(days=days) if days else None
    return {p.sku: storage.history(p.sku, since=since) for p in _select_products(config, skus)[0]}


async def probe(
    url: str,
    competitor: Competitor | None,
    settings: Settings,
    product: Product | None = None,
    *,
    render: bool | None = None,
) -> tuple[FetchResult, ExtractedOffer, MatchResult | None]:
    """Fetch and extract one page as a scan would; score it against ``product`` when given.

    Handy for configuring selectors. ``render`` overrides ``competitor.render_js``.
    The match is ``None`` without a product or when the fetch failed.
    """
    async with Fetcher(settings) as fetcher:
        result = await fetcher.get(url, competitor, render=render)
    if not result.ok:
        return result, ExtractedOffer(url=result.final_url), None
    offer = (
        _extract(result, competitor, product)
        if competitor is not None
        else extract_offer(result.html, result.final_url, product=product)
    )
    return result, offer, score_match(product, offer) if product is not None else None
