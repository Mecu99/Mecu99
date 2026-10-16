"""End-to-end tests of the scan engine against the local demo marketplace (and, for the
discovery guards, a mocked shop)."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterator
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from pricehunter.demo import DemoMarket
from pricehunter.engine import MATCH_LOST, NO_PRICE, build_snapshots, load_history, probe, run_scan
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


def _product_page(title: str, price: str, currency: str, gtin: str | None = None) -> str:
    data: dict[str, object] = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": title,
        "offers": {"@type": "Offer", "price": price, "priceCurrency": currency},
    }
    if gtin:
        data["gtin13"] = gtin
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
