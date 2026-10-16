"""End-to-end tests of the scan engine against the local demo marketplace (and, for the
discovery guards, a mocked shop)."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

from pricehunter.demo import DemoMarket
from pricehunter.engine import (
    FALLBACK_MAX_AGE,
    MATCH_LOST,
    NO_PRICE,
    build_snapshots,
    load_history,
    market_warnings,
    probe,
    run_scan,
)
from pricehunter.fetch import Fetcher
from pricehunter.models import (
    AlertRule,
    AppConfig,
    Availability,
    Competitor,
    FetchResult,
    Listing,
    NotifierConfig,
    Observation,
    Product,
    ScanResult,
    Settings,
    utcnow,
)
from pricehunter.storage import Storage

D = Decimal

# (sku, shop, demo item id) of every listing a first scan must end up with.
EXPECTED = {
    ("SONY-XM5", "MegaShop", "1001"),
    ("SONY-XM5", "PretMic", "2001"),
    ("SONY-XM5", "BudgetBay", "3001"),
    ("IPHONE15-128", "MegaShop", "1003"),
    ("IPHONE15-128", "PretMic", "2003"),
    ("IPHONE15-128", "BudgetBay", "3002"),
    ("SAMSUNG-Q80C-55", "MegaShop", "1006"),
    ("SAMSUNG-Q80C-55", "PretMic", "2005"),
    ("SAMSUNG-Q80C-55", "BudgetBay", "3005"),
    ("DELONGHI-ECAM22", "MegaShop", "1007"),
    ("DELONGHI-ECAM22", "BudgetBay", "3006"),
    ("DYSON-V15", "PretMic", "2006"),  # pinned in the demo config
    ("DYSON-V15", "BudgetBay", "3007"),
}
# (shop, item id, the product it imitates): previous models, other storage sizes, accessories.
DECOYS = [
    ("MegaShop", "1002", "SONY-XM5"),  # WH-1000XM4
    ("MegaShop", "1004", "IPHONE15-128"),  # 256GB
    ("MegaShop", "1005", "IPHONE15-128"),  # silicone case
    ("MegaShop", "1008", "DYSON-V15"),  # V12
    ("PretMic", "2002", "SONY-XM5"),
    ("PretMic", "2004", "IPHONE15-128"),  # "Husa ..."
    ("BudgetBay", "3003", "IPHONE15-128"),
    ("BudgetBay", "3004", "IPHONE15-128"),
]


@pytest.fixture
def market() -> Iterator[DemoMarket]:
    with DemoMarket() as m:
        yield m


@pytest.fixture
def config(market: DemoMarket, tmp_path: Path) -> AppConfig:
    cfg = market.config(tmp_path / "demo.db", requests_per_minute=0)
    cfg.notifiers = [NotifierConfig(type="file", path=str(tmp_path / "alerts.jsonl"))]
    return cfg


@pytest.fixture
def storage(config: AppConfig) -> Iterator[Storage]:
    with Storage(config.settings.database) as s:
        yield s


@pytest.fixture
async def fast_fetcher(config: AppConfig) -> AsyncIterator[Fetcher]:
    """Retries without backoff sleeps, for tests where a shop fails on purpose."""
    async with Fetcher(config.settings, backoff_base=0.0) as fetcher:
        yield fetcher


def listing_keys(storage: Storage, market: DemoMarket) -> set[tuple[str, str, str]]:
    """(sku, shop, item id) of every active listing."""
    items = {(shop, item) for _, shop, item in EXPECTED} | {
        (shop, item) for shop, item, _ in DECOYS
    }
    by_url = {market.product_url(shop, item): (shop, item) for shop, item in items}
    return {(listing.sku, *by_url[listing.url]) for listing in storage.listings()}


def by_listing(result: ScanResult) -> dict[tuple[str, str], list[Observation]]:
    """This run's observations grouped by (sku, competitor)."""
    out: dict[tuple[str, str], list[Observation]] = {}
    for obs in result.observations:
        out.setdefault((obs.sku, obs.competitor), []).append(obs)
    return out


# --------------------------------------------------------------------------- first scan


async def test_first_scan_discovers_and_matches_the_right_listing_on_every_shop(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    result = await run_scan(config, storage)

    assert result.errors == []
    assert listing_keys(storage, market) == EXPECTED
    stats = result.stats
    assert (stats.products, stats.listings_matched, stats.observations) == (5, 12, 13)
    assert (stats.errors, stats.alerts) == (0, 0)  # nobody undercuts us by 2% on day 1
    assert stats.candidates_checked >= 12 and stats.pages_fetched >= stats.candidates_checked
    assert result.alerts == [] and storage.recent_alerts() == []
    run = storage.runs()[0]
    assert run.run_id == stats.run_id and run.finished_at is not None and run.observations == 13


async def test_decoys_and_wrong_variants_are_rejected(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    listed = {listing.url for listing in storage.listings(active_only=False)}
    threshold = config.settings.match_threshold
    for shop, item, sku in DECOYS:
        url = market.product_url(shop, item)
        assert url not in listed, (shop, item)
        if shop != "BudgetBay":  # search shops: the decoy page was fetched and scored...
            assert market.requests[url] >= 1, (shop, item)
        competitor, product = config.competitor(shop), config.product(sku)
        _, offer, match = await probe(url, competitor, config.settings, product)
        assert offer.price is not None and match is not None
        assert match.score < threshold, (shop, item, match)  # ...and rejected by the matcher


async def test_observations_are_stored_and_ron_is_converted(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    result = await run_scan(config, storage, notify=False)
    obs = {key: values[0] for key, values in by_listing(result).items()}

    pretmic = obs[("SONY-XM5", "PretMic")]
    assert (pretmic.price, pretmic.currency, pretmic.method) == (D("1749.99"), "RON", "selectors")
    assert (pretmic.availability, pretmic.shipping) == (Availability.IN_STOCK, D("19.99"))
    assert pretmic.match_reason == "title" and pretmic.match_score >= 0.8
    assert obs[("SONY-XM5", "MegaShop")].method == "jsonld"
    assert obs[("SONY-XM5", "MegaShop")].match_reason == "gtin"
    assert obs[("SONY-XM5", "BudgetBay")].method == "microdata"
    assert obs[("SAMSUNG-Q80C-55", "BudgetBay")].method == "opengraph"
    assert obs[("DELONGHI-ECAM22", "MegaShop")].match_reason == "mpn"

    assert len(storage.latest_observations()) == 13
    snapshot = next(s for s in result.snapshots if s.sku == "SONY-XM5")
    converted = next(o for o in snapshot.offers if o.competitor == "PretMic")
    assert (converted.price, converted.original_currency) == (D("351.75"), "RON")  # 1749.99 * 0.201
    assert (snapshot.lowest, snapshot.lowest_competitor, snapshot.our_rank) == (
        D("345.00"),
        "MegaShop",
        2,
    )
    suggestion = next(s for s in result.suggestions if s.sku == "SONY-XM5")
    assert suggestion.suggested_price == D("343.99")  # beat 345.00 by 1.00, ending .99


async def test_each_page_is_fetched_once_per_scan(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    product_pages = {u: n for u, n in market.requests.items() if re.search(r"/(p|produs|item)/", u)}
    assert product_pages
    # Search redirects, candidates and listings share one request per page within a run.
    assert set(product_pages.values()) == {1}


# --------------------------------------------------------------------------- following days


async def test_second_day_raises_drop_undercut_and_stock_alerts(
    market: DemoMarket, config: AppConfig, storage: Storage, tmp_path: Path
) -> None:
    await run_scan(config, storage)
    market.set_day(2)
    result = await run_scan(config, storage)

    found = {(a.type, a.sku, a.competitor) for a in result.alerts}
    assert {
        ("price_drop", "SONY-XM5", "MegaShop"),
        ("undercut", "SONY-XM5", "MegaShop"),
        ("new_lowest", "SONY-XM5", "MegaShop"),
        ("out_of_stock", "IPHONE15-128", "PretMic"),
        ("undercut", "IPHONE15-128", "BudgetBay"),
        ("price_drop", "SAMSUNG-Q80C-55", "MegaShop"),
        ("price_rise", "DELONGHI-ECAM22", "MegaShop"),
        ("undercut", "DYSON-V15", "BudgetBay"),
    } <= found
    assert result.stats.alerts == len(result.alerts) == len(storage.recent_alerts())
    jsonl = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == len(result.alerts)  # the file notifier received them
    assert result.stats.listings_matched == 0  # known pairs are not rediscovered

    iphone = next(s for s in result.snapshots if s.sku == "IPHONE15-128")
    out = next(o for o in iphone.offers if o.competitor == "PretMic")
    assert out.availability is Availability.OUT_OF_STOCK and iphone.in_stock_offers == 2
    sony = next(s for s in result.suggestions if s.sku == "SONY-XM5")
    assert sony.suggested_price == D("298.99")

    market.set_day(3)
    day3 = {(a.type, a.sku, a.competitor) for a in (await run_scan(config, storage)).alerts}
    assert {
        ("back_in_stock", "IPHONE15-128", "PretMic"),
        ("price_rise", "SAMSUNG-Q80C-55", "BudgetBay"),
    } <= day3
    assert ("undercut", "SONY-XM5", "MegaShop") not in day3  # still cheaper, not news


async def test_known_pairs_are_rediscovered_only_when_stale(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    first = storage.last_discovered("SONY-XM5", "MegaShop")
    await run_scan(config, storage, notify=False)
    assert storage.last_discovered("SONY-XM5", "MegaShop") == first

    config.settings.rediscover_after_hours = 0
    await run_scan(config, storage, notify=False)
    second = storage.last_discovered("SONY-XM5", "MegaShop")
    assert first is not None and second is not None and second > first
    assert listing_keys(storage, market) == EXPECTED  # rediscovery adds no wrong listing


async def test_no_discover_only_reads_known_listings(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    result = await run_scan(config, storage, discover=False, notify=False)
    # Only the pinned vacuum page is known before any discovery.
    assert [(o.sku, o.competitor) for o in result.observations] == [("DYSON-V15", "PretMic")]
    assert result.stats.candidates_checked == 0
    assert not any("/search" in u or "/cauta" in u or "sitemap" in u for u in market.requests)


# --------------------------------------------------------------------------- pinned URLs


async def test_pinned_urls_skip_discovery_and_matching(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    variant = market.product_url("BudgetBay", "3003")  # 256GB, which matching would reject
    iphone = config.product("IPHONE15-128")
    assert iphone is not None
    iphone.competitor_urls = {"BudgetBay": [variant]}
    result = await run_scan(config, storage, notify=False)

    pinned = storage.listings("IPHONE15-128", "BudgetBay")
    assert [(lst.url, lst.pinned, lst.match_score, lst.match_reason) for lst in pinned] == [
        (variant, True, 1.0, "pinned")
    ]
    obs = by_listing(result)[("IPHONE15-128", "BudgetBay")]
    assert [(o.price, o.match_reason) for o in obs] == [(D("979.00"), "pinned")]
    vacuum = storage.listings("DYSON-V15", "PretMic")
    assert [(lst.pinned, lst.match_reason) for lst in vacuum] == [(True, "pinned")]
    assert not any("/cauta" in u and "dyson" in u.lower() for u in market.requests)

    # Unpinning hands the pair back to discovery, which finds the right page.
    iphone.competitor_urls = {}
    await run_scan(config, storage, notify=False)
    assert [lst.url for lst in storage.listings("IPHONE15-128", "BudgetBay")] == [
        market.product_url("BudgetBay", "3002")
    ]


async def test_unpinned_page_is_rediscovered_in_the_same_scan(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    right = market.product_url("BudgetBay", "3002")
    iphone = config.product("IPHONE15-128")
    assert iphone is not None
    iphone.competitor_urls = {"BudgetBay": [right]}
    await run_scan(config, storage, skus=["IPHONE15-128"], competitors=["BudgetBay"], notify=False)
    iphone.competitor_urls = {}
    result = await run_scan(
        config, storage, skus=["IPHONE15-128"], competitors=["BudgetBay"], notify=False
    )
    listings = storage.listings("IPHONE15-128", "BudgetBay")
    assert [(lst.url, lst.pinned, lst.match_reason) for lst in listings] == [(right, False, "gtin")]
    assert [o.match_reason for o in result.observations] == ["gtin"]


async def test_pinned_page_without_a_price_is_reported(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    sony = config.product("SONY-XM5")
    assert sony is not None
    sony.competitor_urls = {"BudgetBay": [market.base_url("BudgetBay") + "/about-us"]}
    result = await run_scan(
        config, storage, competitors=["BudgetBay"], skus=["SONY-XM5"], notify=False
    )
    [obs] = result.observations
    assert obs.error == NO_PRICE and obs.price is None
    assert result.stats.errors == 1 and "about-us" in result.errors[0]


# --------------------------------------------------------------------------- listing upkeep


async def test_listing_that_no_longer_matches_is_deactivated(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    decoy = market.product_url("BudgetBay", "3003")
    storage.upsert_listing(
        Listing(
            sku="IPHONE15-128",
            competitor="BudgetBay",
            url=decoy,
            match_score=0.9,
            match_reason="title",
        )
    )
    result = await run_scan(config, storage, discover=False, notify=False)

    lost = next(o for o in result.observations if o.url == decoy)
    assert lost.error == MATCH_LOST and lost.match_score < config.settings.match_threshold
    assert decoy not in {lst.url for lst in storage.listings()}
    assert all(o.url != decoy for o in storage.latest_observations("IPHONE15-128"))
    assert result.stats.errors == 0  # an expected event, not a failure


async def test_listing_that_disappeared_is_deactivated(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    gone = market.base_url("MegaShop") + "/p/discontinued-thing-9999"
    storage.upsert_listing(
        Listing(
            sku="SONY-XM5", competitor="MegaShop", url=gone, match_score=0.95, match_reason="title"
        )
    )
    result = await run_scan(
        config, storage, skus=["SONY-XM5"], competitors=["MegaShop"], discover=False
    )
    [obs] = result.observations
    assert obs.error is not None and obs.error.startswith("HTTP 404")
    assert storage.listings("SONY-XM5", "MegaShop") == []
    assert result.stats.errors == 1


# --------------------------------------------------------------------------- failure isolation


async def test_a_shop_that_is_down_does_not_stop_the_others(
    market: DemoMarket, config: AppConfig, storage: Storage, fast_fetcher: Fetcher
) -> None:
    await run_scan(config, storage, notify=False)
    market.set_offline("PretMic")
    result = await run_scan(config, storage, notify=False, fetcher=fast_fetcher)

    failed = [o for o in result.observations if o.competitor == "PretMic"]
    assert len(failed) == 4 and all(o.error == "HTTP 503" for o in failed)
    healthy = [o for o in result.observations if o.competitor != "PretMic"]
    assert len(healthy) == 9 and all(o.error is None and o.price for o in healthy)
    assert result.stats.errors == 4 and len(result.errors) == 4
    assert all(e.startswith("PretMic: ") and "HTTP 503" in e for e in result.errors)
    # The market keeps PretMic's last good price until it answers again.
    sony = next(s for s in result.snapshots if s.sku == "SONY-XM5")
    assert {o.competitor for o in sony.offers} == {"MegaShop", "PretMic", "BudgetBay"}


class _BrokenFor(Fetcher):
    """A fetcher that crashes (instead of returning an error) for one competitor."""

    def __init__(self, *args: object, broken: str, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.broken = broken

    async def get(
        self, url: str, competitor: Competitor | None = None, *, render: bool | None = None
    ) -> FetchResult:
        if competitor is not None and competitor.name == self.broken:
            raise RuntimeError("boom")
        return await super().get(url, competitor, render=render)


async def test_an_unexpected_crash_is_recorded_per_listing(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    async with _BrokenFor(config.settings, broken="BudgetBay") as fetcher:
        result = await run_scan(config, storage, discover=False, notify=False, fetcher=fetcher)
    assert len(result.errors) == result.stats.errors == 5
    assert all(e.startswith("BudgetBay: ") and "RuntimeError: boom" in e for e in result.errors)
    assert {o.competitor for o in result.observations} == {"MegaShop", "PretMic"}
    assert len(result.snapshots) == 5 and storage.runs()[0].finished_at is not None


async def test_scan_is_finished_and_reported_when_the_fetcher_cannot_start(
    config: AppConfig, storage: Storage
) -> None:
    config.settings.proxies = ["not-a-proxy"]
    result = await run_scan(config, storage, notify=False)
    assert result.errors and result.errors[0].startswith("scan aborted:")
    assert result.observations == [] and len(result.snapshots) == 5
    assert storage.runs()[0].finished_at is not None


async def test_selection_filters_and_reports_unknown_names(
    config: AppConfig, storage: Storage
) -> None:
    budget = config.competitor("BudgetBay")
    assert budget is not None
    budget.enabled = False
    result = await run_scan(
        config,
        storage,
        skus=["SONY-XM5", "NOPE"],
        competitors=["MegaShop", "BudgetBay", "Ghost"],
        notify=False,
    )
    assert {(o.sku, o.competitor) for o in result.observations} == {("SONY-XM5", "MegaShop")}
    assert result.errors == [
        "unknown SKU 'NOPE'",
        "disabled competitor 'BudgetBay'",
        "unknown competitor 'Ghost'",
    ]
    assert [s.sku for s in result.snapshots] == ["SONY-XM5"] and result.stats.products == 1


# --------------------------------------------------------------------------- reporting helpers


async def test_build_snapshots_and_history_read_stored_data(
    market: DemoMarket, config: AppConfig, storage: Storage
) -> None:
    await run_scan(config, storage, notify=False)
    market.set_day(2)
    result = await run_scan(config, storage, notify=False)

    snapshots, suggestions = build_snapshots(config, storage)
    assert snapshots == result.snapshots and suggestions == result.suggestions
    only, _ = build_snapshots(config, storage, skus=["DYSON-V15"])
    assert [s.sku for s in only] == ["DYSON-V15"]

    history = load_history(config, storage)
    assert set(history) == {p.sku for p in config.products}
    sony = history["SONY-XM5"]
    assert len(sony) == 6 and sony == sorted(sony, key=lambda o: o.observed_at)
    megashop = [o.price for o in sony if o.competitor == "MegaShop"]
    assert megashop == [D("345.00"), D("299.99")]
    assert load_history(config, storage, days=None)["SONY-XM5"] == sony


async def test_probe_explains_a_single_page(market: DemoMarket, config: AppConfig) -> None:
    url = market.product_url("PretMic", "2001")
    pretmic, sony = config.competitor("PretMic"), config.product("SONY-XM5")

    result, offer, match = await probe(url, pretmic, config.settings, sony)
    assert result.ok and urlsplit(result.final_url).path == urlsplit(url).path
    assert (offer.price, offer.currency, offer.method) == (D("1749.99"), "RON", "selectors")
    assert match is not None and match.score >= config.settings.match_threshold

    _, generic, no_match = await probe(url, None, config.settings)
    assert (
        generic.price is None and generic.method == "none" and no_match is None
    )  # needs selectors

    missing, empty, nothing = await probe(url + "-missing", pretmic, config.settings, sony)
    assert (missing.status, missing.error, empty.price, nothing) == (404, "HTTP 404", None, None)


# --------------------------------------------------------------------------- discovery guards


def _product_page(
    title: str,
    price: str | None,
    currency: str = "EUR",
    gtin: str | None = None,
    *,
    brand: str | None = None,
    availability: str | None = None,
) -> str:
    offer: dict[str, object] = {"@type": "Offer"}
    if price is not None:
        offer |= {"price": price, "priceCurrency": currency}
    if availability:
        offer["availability"] = f"https://schema.org/{availability}"
    data: dict[str, object] = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": title,
        "offers": offer,
    }
    if gtin:
        data["gtin13"] = gtin
    if brand:
        data["brand"] = {"@type": "Brand", "name": brand}
    return (
        f'<html><head><script type="application/ld+json">{json.dumps(data)}</script></head>'
        f"<body><h1>{title}</h1></body></html>"
    )


async def _scan_search_shop(
    product: Product, pages: dict[str, str]
) -> tuple[ScanResult, list[Listing]]:
    """One scan of a shop whose site search lists every page in ``pages`` (path -> HTML)."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            cards = "".join(f'<div class="product-card"><a href="{p}">x</a></div>' for p in pages)
            return httpx.Response(200, html=f"<html><body><main>{cards}</main></body></html>")
        if request.url.path in pages:
            return httpx.Response(200, html=pages[request.url.path])
        return httpx.Response(404)

    config = AppConfig(
        settings=Settings(database=":memory:", respect_robots=False, max_retries=0),
        competitors=[
            Competitor(
                name="Shop",
                base_url="https://shop.example",
                search_url="/search?q={query}",
                requests_per_minute=0,
            )
        ],
        products=[product],
        alerts=[AlertRule(type="undercut")],
    )
    with Storage(":memory:") as storage:
        async with Fetcher(config.settings, transport=httpx.MockTransport(handle)) as fetcher:
            result = await run_scan(config, storage, fetcher=fetcher, notify=False)
        return result, storage.listings()


async def test_identifier_match_keeps_title_only_lookalikes_out() -> None:
    gtin = "8806095299853"
    s24 = Product(
        sku="S24", name="Samsung Galaxy S24 128GB Onyx Black", brand="Samsung", gtin=gtin,
        our_price=D("819"), currency="EUR",
    )  # fmt: skip
    pages = {
        "/samsung-galaxy-s24-128gb-onyx-black-p-1001.html": _product_page(
            "Samsung Galaxy S24 128GB Onyx Black", "799.00", "EUR", gtin=gtin
        ),
        # Accepted on its title alone, but the GTIN page already is the exact item.
        "/samsung-galaxy-s24-128gb-onyx-black-dual-sim-p-1002.html": _product_page(
            "Samsung Galaxy S24 128GB Onyx Black Dual SIM", "549.00", "EUR"
        ),
    }
    result, listings = await _scan_search_shop(s24, pages)

    assert [(urlsplit(x.url).path, x.match_reason) for x in listings] == [
        ("/samsung-galaxy-s24-128gb-onyx-black-p-1001.html", "gtin")
    ]
    assert result.snapshots[0].lowest == D("799.00")


async def test_implausibly_cheap_title_match_is_not_listed() -> None:
    xm5 = Product(
        sku="XM5", name="Sony WH-1000XM5", brand="Sony", our_price=D("1549.99"), currency="RON"
    )
    pages = {
        "/casti-sony-wh-1000xm5-negru-p-1001.html": _product_page(
            "Casti wireless Sony WH-1000XM5, Noise Cancelling, Negru", "1499.99", "RON"
        ),
        # Czech ear pads: no keyword the matcher knows, but 94% under our price.
        "/nausniky-sony-wh-1000xm5-p-1002.html": _product_page(
            "Náušníky Sony WH-1000XM5", "89.99", "RON"
        ),
    }
    result, listings = await _scan_search_shop(xm5, pages)

    assert [urlsplit(x.url).path for x in listings] == [
        "/casti-sony-wh-1000xm5-negru-p-1001.html"
    ]
    assert result.snapshots[0].lowest == D("1499.99")
    assert [a.new_price for a in result.alerts] == [D("1499.99")]


def _variants_page(title: str, offers: object) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": title,
        "brand": "Apple",
        "offers": offers,
    }
    return (
        f'<html><head><script type="application/ld+json">{json.dumps(data)}</script></head>'
        f'<body><main><h1>{title}</h1><p class="price">899,00 €</p></main></body></html>'
    )


PRICE_RANGE = {
    "@type": "AggregateOffer",
    "lowPrice": "899",
    "highPrice": "1279",
    "priceCurrency": "EUR",
}
IPHONE_256 = Product(
    sku="IP15-256", name="Apple iPhone 15 256GB", brand="Apple", our_price=D("1029"), currency="EUR"
)


async def test_multi_variant_page_is_read_at_our_variants_price() -> None:
    capacities = [("128GB", "899.00"), ("256GB", "1029.00"), ("512GB", "1279.00")]
    offers = [
        {
            "@type": "Offer",
            "name": capacity,
            "price": price,
            "priceCurrency": "EUR",
            "availability": "https://schema.org/InStock",
            "url": f"https://shop.example/apple-iphone-15-p-1001.html?variant={i}",
        }
        for i, (capacity, price) in enumerate(capacities)
    ]
    pages = {"/apple-iphone-15-p-1001.html": _variants_page("Apple iPhone 15", offers)}
    result, listings = await _scan_search_shop(IPHONE_256, pages)

    assert [urlsplit(x.url).path for x in listings] == ["/apple-iphone-15-p-1001.html"]
    assert [o.price for o in result.observations] == [D("1029.00")]
    assert result.alerts == []  # the 128GB price is no undercut of our 256GB


async def test_listing_and_ambiguous_variant_pages_are_not_listed() -> None:
    capacities = [("128gb", "899,00"), ("256gb", "1.029,00"), ("512gb", "1.279,00")]
    tiles = "".join(
        f'<div class="product-card"><a href="/p/{c}-p-{700 + i}.html">Apple iPhone 15 {c}</a>'
        f'<span class="price">{p} €</span></div>'
        for i, (c, p) in enumerate(capacities)
    )
    pages = {
        "/apple-iphone-15-256gb-p-1001.html": _product_page(
            "Apple iPhone 15 256GB", "1029.00", "EUR"
        ),
        "/telefoane/apple-iphone-15-p-1002.html": (
            f"<html><body><main><h1>Apple iPhone 15</h1>{tiles}</main></body></html>"
        ),
        "/apple-iphone-15-p-1003.html": _variants_page("Apple iPhone 15", PRICE_RANGE),
    }
    result, listings = await _scan_search_shop(IPHONE_256, pages)

    assert [urlsplit(x.url).path for x in listings] == ["/apple-iphone-15-256gb-p-1001.html"]
    assert result.snapshots[0].lowest == D("1029.00")


async def test_pinned_page_with_indistinguishable_variants_reports_why() -> None:
    url = "https://shop.example/apple-iphone-15-p-1003.html"
    pages = {
        "/apple-iphone-15-p-1003.html": _variants_page("Apple iPhone 15", PRICE_RANGE)
    }
    pinned = IPHONE_256.model_copy(update={"competitor_urls": {"Shop": [url]}})
    result, _ = await _scan_search_shop(pinned, pages)

    [obs] = result.observations
    assert obs.price is None and obs.method == "ambiguous"
    assert obs.error is not None and obs.error.startswith("no price: variants at different prices")
    assert result.alerts == []


# --------------------------------------------------------------------------- market upkeep

SONY = Product(sku="XM5", name="Sony WH-1000XM5", brand="Sony", our_price=D("350"), cost=D("200"))
SONY_URL = "https://shop.example/p/sony-wh-1000xm5"
SONY_PAGE = _product_page("Sony WH-1000XM5", "329.00", brand="Sony")
ALERT_TYPES = ("undercut", "price_drop", "out_of_stock", "back_in_stock", "new_lowest")


def _shops(product: Product, *names: str, **settings: Any) -> AppConfig:
    """Mocked shops at ``https://<name>.example`` selling a copy of ``product``."""
    return AppConfig(
        settings=Settings(database=":memory:", respect_robots=False, max_retries=0, **settings),
        competitors=[
            Competitor(
                name=name,
                base_url=f"https://{name.lower()}.example",
                search_url="/search?q={query}",
                requests_per_minute=0,
            )
            for name in names
        ],
        products=[product.model_copy(deep=True)],
        alerts=[AlertRule(type=t) for t in ALERT_TYPES],
    )


def _pinned(url: str, competitor: str = "Shop") -> Product:
    return SONY.model_copy(update={"competitor_urls": {competitor: [url]}})


async def _scan_pages(
    config: AppConfig,
    storage: Storage,
    pages: dict[str, str | int],
    redirects: dict[str, str] | None = None,
    **options: Any,
) -> ScanResult:
    """One scan in which ``pages`` maps a URL to its HTML or an HTTP status (others are 404)
    and ``redirects`` maps a URL to where it now permanently redirects."""
    moved = redirects or {}

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in moved:
            return httpx.Response(301, headers={"Location": moved[url]})
        page = pages.get(url, 404)
        return httpx.Response(page) if isinstance(page, int) else httpx.Response(200, html=page)

    async with Fetcher(config.settings, transport=httpx.MockTransport(handle)) as fetcher:
        return await run_scan(config, storage, fetcher=fetcher, **{"notify": False, **options})


async def test_sold_out_page_hiding_its_price_leaves_the_market_and_alerts_once() -> None:
    config = _shops(_pinned(SONY_URL), "Shop")
    days = [
        ("299.00", "InStock"),
        (None, "OutOfStock"),
        (None, "OutOfStock"),
        ("299.00", "InStock"),
    ]
    seen = []
    with Storage(":memory:") as storage:
        for price, stock in days:
            page = _product_page("Sony WH-1000XM5", price, brand="Sony", availability=stock)
            result = await _scan_pages(config, storage, {SONY_URL: page})
            offers = [o.price for o in result.snapshots[0].offers]
            seen.append(
                (offers, result.suggestions[0].suggested_price, {a.type for a in result.alerts})
            )
    assert seen == [
        ([D("299.00")], D("298.99"), {"undercut"}),
        ([], D("350"), {"out_of_stock"}),  # no repricing against an item nobody can buy
        ([], D("350"), set()),  # not repeated
        ([D("299.00")], D("298.99"), {"back_in_stock", "undercut"}),
    ]


async def test_disabled_or_removed_competitor_leaves_the_market() -> None:
    a, b = "https://a.example/p/xm5", "https://b.example/p/xm5"
    config = _shops(SONY.model_copy(update={"competitor_urls": {"A": [a], "B": [b]}}), "A", "B")
    pages: dict[str, str | int] = {
        a: _product_page("Sony WH-1000XM5", "330.00", brand="Sony"),
        b: _product_page("Sony WH-1000XM5", "249.00", brand="Sony"),
    }
    with Storage(":memory:") as storage:
        first = await _scan_pages(config, storage, pages)
        assert [o.competitor for o in first.snapshots[0].offers] == ["B", "A"]

        config.competitors[1].enabled = False
        second = await _scan_pages(config, storage, pages)
        assert [o.competitor for o in second.snapshots[0].offers] == ["A"]
        assert second.suggestions[0].reason.startswith("beat lowest (A @ 330.00)")
        assert second.alerts == []

        config.competitors[1].enabled = True  # back: its stored prices count again
        assert [o.competitor for o in build_snapshots(config, storage)[0][0].offers] == ["B", "A"]
        config.competitors.pop()  # removed (or renamed) in the config
        snapshots, suggestions = build_snapshots(config, storage)
    assert [o.competitor for o in snapshots[0].offers] == ["A"]
    assert suggestions[0].suggested_price == D("329.99")


CATEGORY_PAGE = """<html><head><title>Headphones | Shop</title></head><body><h1>Headphones</h1>
<div class="product-card"><a href="/p/jbl-tune-510">JBL Tune 510BT</a>
<span class="price">39,99 &euro;</span></div>
<div class="product-card"><a href="/p/sony-wh-ch520">Sony WH-CH520</a>
<span class="price">49,99 &euro;</span></div></body></html>"""


async def test_pinned_url_redirecting_elsewhere_must_still_show_the_product() -> None:
    config = _shops(_pinned(SONY_URL), "Shop")
    category = "https://shop.example/c/headphones"
    moved = "https://shop.example/p/sony-wh-1000xm5-black-123"
    with Storage(":memory:") as storage:
        await _scan_pages(config, storage, {SONY_URL: SONY_PAGE})

        # Delisted: the shop sends the product URL to its category page.
        result = await _scan_pages(config, storage, {category: CATEGORY_PAGE}, {SONY_URL: category})
        [obs] = result.observations
        assert obs.error is not None
        assert obs.error.startswith(f"pinned URL now redirects to {category} (match 0.")
        assert result.errors == [f"Shop: XM5: {SONY_URL}: {obs.error}"]
        assert result.stats.errors == 1 and result.alerts == []
        assert [o.price for o in result.snapshots[0].offers] == [D("329.00")]  # last good one
        assert [(x.url, x.pinned, x.active) for x in storage.listings(active_only=False)] == [
            (SONY_URL, True, True)
        ]

        # Moved to a new address that still is the product: read as usual.
        page = _product_page("Sony WH-1000XM5", "319.00", brand="Sony")
        result = await _scan_pages(config, storage, {moved: page}, {SONY_URL: moved})
    assert [(o.price, o.error, o.match_reason) for o in result.observations] == [
        (D("319.00"), None, "pinned")
    ]


async def test_pinned_url_redirecting_to_its_own_canonical_spelling_stays_trusted() -> None:
    pinned = "http://shop.example/P/Căști-Sony-XM5"
    canonical = "https://shop.example/p/c%C4%83%C8%99ti-sony-xm5/"
    config = _shops(_pinned(pinned), "Shop")
    page = _product_page("Casti over-ear, negru", "329.00")  # the matcher alone would reject it
    with Storage(":memory:") as storage:
        result = await _scan_pages(
            config, storage, {canonical: page}, {str(httpx.URL(pinned)): canonical}
        )
    assert [(o.price, o.error, o.match_reason) for o in result.observations] == [
        (D("329.00"), None, "pinned")
    ]


async def test_pinned_url_replaces_the_listings_discovery_found() -> None:
    wrong = "https://shop.example/p/sony-wh-1000xm5-refurb-123"
    right = "https://shop.example/p/sony-wh-1000xm5-new-456"
    pages: dict[str, str | int] = {
        wrong: _product_page("Sony WH-1000XM5 Wireless Headphones", "199.00", brand="Sony"),
        right: SONY_PAGE,
    }
    config = _shops(SONY, "Shop")
    with Storage(":memory:") as storage:
        storage.upsert_listing(
            Listing(sku="XM5", competitor="Shop", url=wrong, match_score=0.9, match_reason="title")
        )
        storage.mark_discovered("XM5", "Shop")
        await _scan_pages(config, storage, pages)  # reads the wrong page at 199.00

        config.products[0].competitor_urls = {"Shop": [right]}
        assert build_snapshots(config, storage)[0][0].offers == []  # only pins count already
        result = await _scan_pages(config, storage, pages)
        assert [(x.url, x.pinned) for x in storage.listings()] == [(right, True)]
    assert [o.url for o in result.observations] == [right]
    assert [o.price for o in result.snapshots[0].offers] == [D("329.00")]


S24_GTIN = "8806095299853"
S24 = Product(
    sku="S24", name="Samsung Galaxy S24 128GB Onyx Black", brand="Samsung", gtin=S24_GTIN,
    our_price=D("819"),
)  # fmt: skip
S24_WEAK = {  # title-only listings a past discovery kept: no storage size on either page
    "https://shop.example/p/galaxy-s24-5g-dual-sim-1": (
        "Samsung Galaxy S24 5G Dual SIM Onyx Black",
        "699.00",
        0.845,
    ),
    "https://shop.example/p/galaxy-s24-onyx-2": ("Samsung Galaxy S24 Onyx Black", "719.00", 0.93),
}


@pytest.mark.parametrize(
    ("gtin", "kept"),
    [
        (S24_GTIN, ["better-3"]),  # an identifier match leaves no room for look-alikes
        (None, ["better-3", "galaxy-s24-onyx-2"]),  # a better title match ousts the weakest
    ],
)
async def test_rediscovery_replaces_weaker_listings(gtin: str | None, kept: list[str]) -> None:
    better = "https://shop.example/p/better-3"
    pages: dict[str, str | int] = {
        url: _product_page(title, price, brand="Samsung")
        for url, (title, price, _) in S24_WEAK.items()
    }
    pages[better] = _product_page(
        "Samsung Galaxy S24 128GB Black", "799.00", gtin=gtin, brand="Samsung"
    )
    pages[f"https://shop.example/search?q={S24_GTIN}"] = f'<a href="{better}">S24</a>'
    config = _shops(S24, "Shop", rediscover_after_hours=1)
    with Storage(":memory:") as storage:
        for url, (_, _, score) in S24_WEAK.items():
            storage.upsert_listing(
                Listing(
                    sku="S24", competitor="Shop", url=url, match_score=score, match_reason="title"
                )
            )
        storage.mark_discovered("S24", "Shop", utcnow() - timedelta(hours=5))
        result = await _scan_pages(config, storage, pages)
        listed = [urlsplit(x.url).path.removeprefix("/p/") for x in storage.listings()]
    assert listed == kept and result.stats.listings_matched == 1
    assert sorted(urlsplit(o.url).path.removeprefix("/p/") for o in result.observations) == sorted(
        kept
    )
    assert result.snapshots[0].lowest == (D("799.00") if gtin else D("719.00"))


@pytest.mark.parametrize(
    ("age", "offers"),
    [
        (FALLBACK_MAX_AGE - timedelta(days=1), [D("249.00")]),
        (FALLBACK_MAX_AGE + timedelta(days=1), []),
    ],
)
async def test_last_good_price_stands_in_for_a_failing_page_for_a_limited_time(
    age: timedelta, offers: list[Decimal]
) -> None:
    config = _shops(_pinned(SONY_URL), "Shop")
    with Storage(":memory:") as storage:
        storage.add_observation(
            Observation(
                sku="XM5", competitor="Shop", url=SONY_URL, price=D("249.00"), currency="EUR",
                match_score=1.0, match_reason="pinned", observed_at=utcnow() - age,
            )
        )  # fmt: skip
        result = await _scan_pages(config, storage, {SONY_URL: 403})
    assert [o.error for o in result.observations] == ["HTTP 403"]
    assert [o.price for o in result.snapshots[0].offers] == offers


async def test_prices_in_a_currency_without_a_rate_are_reported() -> None:
    url = "https://pl.example/p/xm5"
    config = _shops(_pinned(url, "PL"), "PL", currency_rates={"RON": D("0.201")})
    page = _product_page("Sony WH-1000XM5", "999.00", "PLN", brand="Sony")
    with Storage(":memory:") as storage:
        result = await _scan_pages(config, storage, {url: page})
        warnings = market_warnings(config, storage)
    expected = (
        "no currency rate for PLN in settings.currency_rates:"
        " prices from PL are left out of the market"
    )
    assert result.errors == warnings == [expected]
    assert result.stats.errors == 1 and result.snapshots[0].offers == []


async def test_run_stats_count_every_reported_error(tmp_path: Path) -> None:
    config = _shops(_pinned(SONY_URL), "Shop")
    config.notifiers = [NotifierConfig(type="file", path=str(tmp_path))]  # a directory: fails
    with Storage(":memory:") as storage:
        result = await _scan_pages(
            config, storage, {SONY_URL: SONY_PAGE}, skus=["XM5", "TYPO"], notify=True
        )
        stored = storage.runs()[0]
    assert result.alerts and len(result.errors) == 2
    assert result.errors[0] == "unknown SKU 'TYPO'"
    assert result.errors[1].startswith("file notifier failed: IsADirectoryError")
    assert result.stats.errors == stored.errors == 2
