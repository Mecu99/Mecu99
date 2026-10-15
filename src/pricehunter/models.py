"""Shared data contracts. Every module imports its types from here."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Availability(str, Enum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    PREORDER = "preorder"
    LIMITED = "limited"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- config


class SelectorConfig(BaseModel):
    """Optional CSS selectors for sites where structured data is missing or wrong.

    Every field is a CSS selector. ``*_attr`` fields name an attribute to read
    instead of the element's text (e.g. ``content`` or ``data-price``).
    """

    model_config = ConfigDict(extra="forbid")

    price: str | None = None
    price_attr: str | None = None
    original_price: str | None = None
    title: str | None = None
    currency: str | None = None
    availability: str | None = None
    gtin: str | None = None
    mpn: str | None = None
    brand: str | None = None
    shipping: str | None = None
    # Discovery: links to product pages on a search/listing page.
    search_result_link: str | None = None


class Competitor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    # Site search URL template; ``{query}`` is replaced by the URL-encoded query.
    search_url: str | None = None
    sitemap_url: str | None = None
    # Regex a URL must match to be considered a product page during discovery.
    product_url_pattern: str | None = None
    selectors: SelectorConfig = Field(default_factory=SelectorConfig)
    render_js: bool = False
    currency: str | None = None
    requests_per_minute: float = 30.0
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class Product(BaseModel):
    """An item in *your* catalogue that you want to price against the market."""

    model_config = ConfigDict(extra="forbid")

    sku: str
    name: str
    brand: str | None = None
    gtin: str | None = None  # EAN-13 / UPC-A / GTIN-14
    mpn: str | None = None
    our_price: Decimal | None = None
    cost: Decimal | None = None
    currency: str = "EUR"
    keywords: list[str] = Field(default_factory=list)
    # Known competitor URLs, keyed by competitor name. Skips discovery & matching.
    competitor_urls: dict[str, list[str]] = Field(default_factory=dict)
    # Per-product repricing override.
    min_price: Decimal | None = None
    max_price: Decimal | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database: str = "pricehunter.db"
    concurrency: int = 8
    user_agent: str = "PriceHunter/0.1 (+https://github.com/mecu99/mecu99)"
    respect_robots: bool = True
    timeout: float = 20.0
    max_retries: int = 3
    proxies: list[str] = Field(default_factory=list)
    match_threshold: float = 0.80
    max_candidates_per_competitor: int = 8
    base_currency: str = "EUR"
    # Units of base_currency per 1 unit of the key currency, e.g. {"RON": "0.201"}.
    currency_rates: dict[str, Decimal] = Field(default_factory=dict)
    # Re-run discovery for a (product, competitor) pair after this many hours.
    rediscover_after_hours: float = 168.0
    include_shipping: bool = False


class RepricingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["beat_lowest", "match_lowest", "median", "hold"] = "beat_lowest"
    undercut_amount: Decimal = Decimal("0.01")
    undercut_percent: Decimal | None = None  # takes precedence over amount if set
    min_margin_percent: Decimal = Decimal("10")  # floor = cost * (1 + pct/100)
    ignore_out_of_stock: bool = True
    min_match_score: float = 0.85
    # Round final price to end with this fractional part, e.g. 0.99. None = 2dp.
    price_ending: Decimal | None = None


AlertType = Literal[
    "undercut",  # a competitor is now cheaper than our_price
    "price_drop",  # competitor price fell by >= threshold_percent
    "price_rise",  # competitor price rose by >= threshold_percent
    "out_of_stock",  # competitor went out of stock
    "back_in_stock",
    "new_lowest",  # a new market-lowest price appeared
]


class AlertRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AlertType
    threshold_percent: float = 0.0
    competitors: list[str] | None = None  # None = all
    skus: list[str] | None = None  # None = all


class NotifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["console", "webhook", "slack", "discord", "email", "file"]
    url: str | None = None  # webhook / slack / discord
    path: str | None = None  # file (JSON lines)
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password_env: str | None = None  # name of env var holding the password
    smtp_starttls: bool = True
    email_from: str | None = None
    email_to: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: Settings = Field(default_factory=Settings)
    competitors: list[Competitor] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    repricing: RepricingRule = Field(default_factory=RepricingRule)
    alerts: list[AlertRule] = Field(default_factory=list)
    notifiers: list[NotifierConfig] = Field(default_factory=list)

    @field_validator("competitors")
    @classmethod
    def _unique_competitors(cls, v: list[Competitor]) -> list[Competitor]:
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError("competitor names must be unique")
        return v

    @field_validator("products")
    @classmethod
    def _unique_skus(cls, v: list[Product]) -> list[Product]:
        skus = [p.sku for p in v]
        if len(skus) != len(set(skus)):
            raise ValueError("product SKUs must be unique")
        return v

    def competitor(self, name: str) -> Competitor | None:
        return next((c for c in self.competitors if c.name == name), None)

    def product(self, sku: str) -> Product | None:
        return next((p for p in self.products if p.sku == sku), None)


# --------------------------------------------------------------------------- runtime


class FetchResult(BaseModel):
    url: str
    final_url: str
    status: int  # 0 when the request never got a response
    html: str = ""
    fetched_at: datetime = Field(default_factory=utcnow)
    rendered: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


class ExtractedOffer(BaseModel):
    url: str
    title: str | None = None
    price: Decimal | None = None
    currency: str | None = None  # ISO-4217
    availability: Availability = Availability.UNKNOWN
    original_price: Decimal | None = None  # list / strike-through price
    shipping: Decimal | None = None
    gtin: str | None = None
    mpn: str | None = None
    brand: str | None = None
    sku: str | None = None
    image: str | None = None
    method: str = "none"  # jsonld | microdata | rdfa | opengraph | selectors | heuristic
    confidence: float = 0.0  # 0..1, how sure the extractor is about the price


class MatchResult(BaseModel):
    score: float  # 0..1
    reason: str  # gtin | mpn | title | pinned | gtin_conflict | ...
    details: dict[str, float] = Field(default_factory=dict)


class Observation(BaseModel):
    """One price reading of one competitor listing at one point in time."""

    id: int | None = None
    run_id: int | None = None
    sku: str
    competitor: str
    url: str
    title: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    availability: Availability = Availability.UNKNOWN
    original_price: Decimal | None = None
    shipping: Decimal | None = None
    match_score: float = 0.0
    match_reason: str = ""
    method: str = ""
    observed_at: datetime = Field(default_factory=utcnow)
    error: str | None = None


class Listing(BaseModel):
    """A competitor URL associated with one of our products."""

    sku: str
    competitor: str
    url: str
    match_score: float
    match_reason: str
    pinned: bool = False
    active: bool = True
    discovered_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime | None = None


class CompetitorPrice(BaseModel):
    competitor: str
    url: str
    price: Decimal  # converted to base currency (+shipping if configured)
    original_currency: str | None
    availability: Availability
    match_score: float


class MarketSnapshot(BaseModel):
    sku: str
    currency: str
    our_price: Decimal | None
    offers: list[CompetitorPrice] = Field(default_factory=list)  # sorted ascending by price
    lowest: Decimal | None = None
    lowest_competitor: str | None = None
    median: Decimal | None = None
    average: Decimal | None = None
    highest: Decimal | None = None
    # our_price / median * 100. <100 = we are cheaper than the typical market.
    price_index: float | None = None
    # 1 = cheapest. Counted among in-stock offers + ours.
    our_rank: int | None = None
    competitors_cheaper: int = 0
    in_stock_offers: int = 0


class PriceSuggestion(BaseModel):
    sku: str
    current_price: Decimal | None
    suggested_price: Decimal | None
    floor: Decimal | None
    ceiling: Decimal | None
    strategy: str
    reason: str
    change_percent: float | None = None


class Alert(BaseModel):
    type: str
    sku: str
    competitor: str | None = None
    url: str | None = None
    message: str
    old_price: Decimal | None = None
    new_price: Decimal | None = None
    change_percent: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class ScanStats(BaseModel):
    run_id: int | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    products: int = 0
    pages_fetched: int = 0
    candidates_checked: int = 0
    listings_matched: int = 0
    observations: int = 0
    errors: int = 0
    alerts: int = 0


class ScanResult(BaseModel):
    stats: ScanStats
    observations: list[Observation] = Field(default_factory=list)
    snapshots: list[MarketSnapshot] = Field(default_factory=list)
    suggestions: list[PriceSuggestion] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
