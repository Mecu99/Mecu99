"""Extract a competitor offer (price, currency, stock, identifiers) from a product page.

Layers, most trusted first: configured CSS selectors, JSON-LD, microdata, RDFa,
OpenGraph/meta tags, and finally class-name heuristics. The first layer with a
price decides price, currency and method; identity fields (title, GTIN, ...)
are filled from whichever layer has them, in the same order.

A layer may instead return a *verdict* (:data:`LISTING`, :data:`AMBIGUOUS`): the
page is a category grid, or offers variants at different prices none of which is
recognisably ours. A verdict stops lower layers from guessing a price.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
from itertools import chain, islice, takewhile
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urljoin, urlsplit

import extruct
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PreformattedString
from soupsieve import SelectorSyntaxError

from .match import normalize_text, score_match, spec_tokens
from .models import Availability, ExtractedOffer, Product, SelectorConfig
from .pricing import currency_code, detect_currency, parse_decimal, parse_price

log = logging.getLogger(__name__)

A = Availability
T = TypeVar("T")

CONFIDENCE: dict[str, float] = {
    "selectors": 0.95,
    "jsonld": 0.9,
    "microdata": 0.85,
    "rdfa": 0.85,
    "opengraph": 0.75,
    "meta": 0.75,
    "heuristic": 0.4,
}
LISTING = "listing"
AMBIGUOUS = "ambiguous"
# Methods that mean "no price on purpose", with the reason shown to the user.
VERDICTS: dict[str, str] = {
    LISTING: "a category or search page, not a product page",
    AMBIGUOUS: "variants at different prices and none is identifiably ours; pin the variant's URL",
}
_IDENTITY_FIELDS = ("title", "gtin", "mpn", "brand", "sku", "image")
_BUYABLE = (A.IN_STOCK, A.LIMITED)


@dataclass(frozen=True)
class _Page:
    """What every layer needs to know about the page besides its HTML."""

    url: str
    urls: tuple[str, ...]  # the URL plus its canonical / og:url aliases
    ccy: str | None  # the competitor's currency: what "$" or "lei" means there
    product: Product | None  # our catalogue item, to tell the page's variants apart


# --------------------------------------------------------------------------- values


def _first(value: Any) -> Any:
    while isinstance(value, list):
        value = value[0] if value else None
    return value


def _str(value: Any) -> str | None:
    """Plain text of a scalar, a one-element list, or a ``{"name"|"@value": ...}`` node."""
    value = _first(value)
    if isinstance(value, dict):
        return _str(value.get("@value", value.get("name")))
    if value is None or isinstance(value, bool):
        return None
    return " ".join(html_lib.unescape(str(value)).split()) or None


def _abs(value: Any, base: str) -> str | None:
    s = _str(value)
    return urljoin(base, s) if s else None


def _currency(value: Any, preferred: str | None = None) -> str | None:
    return currency_code(_str(value), preferred)


def _gtin(*values: Any) -> str | None:
    """First value that has at least 8 digits, reduced to digits only."""
    for value in values:
        digits = re.sub(r"\D", "", _str(value) or "")
        if len(digits) >= 8:
            return digits
    return None


def _image(value: Any, base: str) -> str | None:
    value = _first(value)
    if isinstance(value, dict):
        value = value.get("url") or value.get("contentUrl") or value.get("@id")
    return _abs(value, base)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@lru_cache(maxsize=4096)
def _fold_char(c: str) -> str:
    folded = _fold(c)
    return folded if len(folded) == 1 else c


def _fold_same_length(text: str) -> str:
    """:func:`_fold` one character at a time, so match offsets in the result fit ``text``."""
    return "".join(map(_fold_char, text))


# --------------------------------------------------------------------------- availability

_SCHEMA_AVAILABILITY: dict[str, Availability] = {
    "instock": A.IN_STOCK,
    "onlineonly": A.IN_STOCK,
    "available": A.IN_STOCK,
    "limitedavailability": A.LIMITED,
    "instoreonly": A.LIMITED,
    "outofstock": A.OUT_OF_STOCK,
    "soldout": A.OUT_OF_STOCK,
    "discontinued": A.OUT_OF_STOCK,
    "oos": A.OUT_OF_STOCK,
    "preorder": A.PREORDER,
    "presale": A.PREORDER,
    "backorder": A.PREORDER,
    "madetoorder": A.PREORDER,
    "availablefororder": A.PREORDER,
}
# Checked in order on accent-folded lowercase text with hyphens as spaces; negatives first
# because "indisponibil" contains "disponibil", "not in stock" contains "in stock", and a
# back-in-stock sign-up ("Notify me when available") names the very state it lacks.
_AVAILABILITY_TEXT: tuple[tuple[Availability, re.Pattern[str]], ...] = (
    (
        A.OUT_OF_STOCK,
        re.compile(
            r"\b(?:out of stock|sold out|not (?:in stock|available)|unavailable|stoc epuizat"
            r"|epuizat|indisponibil|nu (?:mai )?(?:este|e) (?:in stoc|disponibil)|ausverkauft"
            r"|nicht (?:auf lager|verfugbar|lieferbar|vorratig)|epuise|rupture de stock"
            r"|agotado|esaurito|niedostepny|brak w magazynie|discontinued"
            r"|notify me|e ?mail me when|alert me when|let me know when|(?:back )?in stock soon"
            r"|anunta ma|anuntati ma|avertizeaza ma|notifica ma|revine in stoc"
            r"|benachrichtig|bald wieder|prevenez moi|me prevenir|m avertir|alertez moi"
            r"|bientot (?:disponible|en stock|de retour)|avisame|notificarme|avvisami"
            r"|presto disponibile|disponibile a breve|powiadom|wkrotce dostepn)"
        ),
    ),
    (
        A.PREORDER,
        re.compile(r"\b(?:pre ?order|precomand|vorbestell|pre ?commande|back ?order)"),
    ),
    (
        A.LIMITED,
        re.compile(
            r"\b(?:limited (?:stock|availability|quantity)|only \d+ left|low stock|few left"
            r"|stoc limitat|ultimul produs|ultimele \d+|nur noch \d+)"
        ),
    ),
    (
        A.IN_STOCK,
        re.compile(
            r"\b(?:in stock|in stoc|disponibil|available|auf lager|lieferbar|verfugbar|vorratig"
            r"|en stock|disponible|dostepny|na stanie|in magazzino|en existencia)"
        ),
    ),
)


def availability_from_text(text: str | None) -> Availability:
    """Map a stock label such as "În stoc" or "Ausverkauft" to an :class:`Availability`."""
    if not text:
        return A.UNKNOWN
    folded = _fold(text).replace("-", " ")
    return next((a for a, pattern in _AVAILABILITY_TEXT if pattern.search(folded)), A.UNKNOWN)


def _availability(value: Any) -> Availability:
    """Schema.org availability (URL, short name, OG value) or, failing that, label text."""
    value = _first(value)
    if isinstance(value, dict):
        value = value.get("@id") or value.get("name") or value.get("@value")
    s = _str(value)
    if not s:
        return A.UNKNOWN
    for key in (s, _local(s)):
        if hit := _SCHEMA_AVAILABILITY.get(re.sub(r"[^a-z]", "", key.lower())):
            return hit
    return availability_from_text(s)


# --------------------------------------------------------------------------- URLs


def _url_key(url: str) -> tuple[str, str, frozenset[tuple[str, str]]]:
    p = urlsplit(url)
    host = p.netloc.lower().removeprefix("www.")
    return host, p.path.rstrip("/"), frozenset(parse_qsl(p.query))


def _url_matches(candidate: str, page: str) -> bool:
    """Same page; a candidate with a query (``?variant=1``) may match a page with extra params."""
    c_host, c_path, c_query = _url_key(candidate)
    p_host, p_path, p_query = _url_key(page)
    if (c_host, c_path) != (p_host, p_path):
        return False
    return c_query == p_query or (bool(c_query) and c_query <= p_query)


def _address(url: str) -> tuple[str, str]:
    """Host and path: the page an URL shows, whatever variant or tracking query it carries."""
    return _url_key(url)[:2]


def _sku_in_url(sku: str | None, url: str) -> bool:
    if not sku:
        return False
    parts = urlsplit(url)
    values = {v.lower() for _, v in parse_qsl(parts.query)}
    values |= {seg.lower().removesuffix(".html") for seg in parts.path.split("/") if seg}
    return sku.lower() in values


def _page_urls(soup: BeautifulSoup, url: str) -> tuple[str, ...]:
    """``url`` plus the canonical and ``og:url`` addresses the page gives itself."""
    aliases = [
        tag.get(attr)
        for tag, attr in (
            (soup.find("link", rel="canonical"), "href"),
            (soup.find("meta", property="og:url"), "content"),
        )
        if tag is not None
    ]
    return (url, *(urljoin(url, a.strip()) for a in aliases if isinstance(a, str) and a.strip()))


# --------------------------------------------------------------------------- schema.org graph

_PRODUCT_TYPES = frozenset(
    {"Product", "ProductGroup", "IndividualProduct", "ProductModel", "SomeProducts", "Vehicle", "Car", "Book"}
)
_OFFER_TYPES = frozenset({"Offer", "AggregateOffer"})
_LIST_PAGE_TYPES = frozenset({"CollectionPage", "SearchResultsPage"})
_ITEM_LIST_TYPES = frozenset({"ItemList", "OfferCatalog"})
_GTIN_KEYS = ("gtin", "gtin13", "gtin12", "gtin14", "gtin8", "isbn")
_LIST_PRICE_RE = re.compile(r"strikethrough|listprice|msrp", re.IGNORECASE)
# Price specifications that are a part of the price (an instalment, a subscription fee).
_COMPONENT_KEYS = ("billingDuration", "billingIncrement", "billingStart", "priceComponentType")
_TIME_UNIT_CODES = frozenset({"DAY", "WEE", "MON", "ANN"})
_CONDITIONS = frozenset({"new", "used", "refurbished", "damaged"})
# Added to a non-new offer's title so the matcher's used/refurbished guard sees the condition.
_CONDITION_LABELS = {"used": "Used", "refurbished": "Refurbished", "damaged": "Used, damaged"}
_REF_KEYS = frozenset({"@id", "@type"})
_MAX_DEPTH = 12
_IDENTIFIED = frozenset({"gtin", "mpn"})
_CONFLICTS = frozenset({"gtin_conflict", "mpn_conflict", "model_conflict", "spec_conflict"})


def _local(name: str) -> str:
    """``"http://schema.org/Product"`` / ``"schema:Product"`` -> ``"Product"``."""
    return re.split(r"[/#:]", name.rstrip("/"))[-1]


def _types(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    return {_local(t) for t in (raw if isinstance(raw, list) else [raw]) if isinstance(t, str)}


def _node_gtin(node: dict[str, Any]) -> str | None:
    return _gtin(*(node.get(k) for k in _GTIN_KEYS))


def _condition(value: Any) -> str | None:
    """``"https://schema.org/UsedCondition"`` or ``"used"`` -> ``"used"``."""
    value = _first(value)
    if isinstance(value, dict):
        value = value.get("@id") or value.get("name")
    s = _str(value)
    name = _local(s).lower().removesuffix("condition") if s else None
    return name if name in _CONDITIONS else None


def _with_condition(title: str | None, condition: str | None) -> str | None:
    label = _CONDITION_LABELS.get(condition or "")
    if not title or not label or label.split(",")[0].lower() in title.lower():
        return title
    return f"{title} ({label})"


def _is_component(spec: dict[str, Any]) -> bool:
    return any(k in spec for k in _COMPONENT_KEYS) or (
        (_str(spec.get("unitCode")) or "").upper() in _TIME_UNIT_CODES
    )


@dataclass(frozen=True)
class _Offer:
    price: Decimal | None = None
    currency: str | None = None
    availability: Availability = A.UNKNOWN
    original_price: Decimal | None = None
    shipping: Decimal | None = None
    urls: tuple[str, ...] = ()
    sku: str | None = None
    gtin: str | None = None
    mpn: str | None = None
    condition: str | None = None  # new | used | refurbished | damaged
    label: str | None = None  # the variant's or offer's own name ("256GB", "Tee - S / White")
    high_price: Decimal | None = None  # top of an AggregateOffer's price range
    variant: dict[str, Any] | None = None  # the hasVariant node this offer came from
    ambiguous: bool = False  # a verdict: several prices, none identifiably ours


@dataclass(frozen=True)
class _Hint:
    """Our catalogue product, used to tell the page's variants apart."""

    product: Product
    title: str  # the name of the page's product or product group
    brand: str | None

    def _fit(self, offer: _Offer) -> tuple[int, float]:
        """Identified (GTIN/MPN) > plausible > conflicting (another capacity or model), then the
        share of the offer's own name that our product's name also says ("256GB Black")."""
        label = offer.label or ""
        title = label if _fold(self.title) in _fold(label) else f"{self.title} {label}".strip()
        seen = ExtractedOffer(url="", title=title or None, brand=self.brand, gtin=offer.gtin, mpn=offer.mpn)
        reason = score_match(self.product, seen).reason
        rank = 2 if reason in _IDENTIFIED else 0 if reason in _CONFLICTS else 1
        words = set(normalize_text(label).split())
        ours = set(normalize_text(self.product.name).split())
        return rank, len(words & ours) / len(words) if words else 0.0

    def narrow(self, pool: list[_Offer]) -> list[_Offer]:
        """The offers that fit our product best (all of them when none fits better)."""
        fits = [self._fit(o) for o in pool]
        best = max(fits)
        return [o for o, fit in zip(pool, fits, strict=True) if fit == best]

    def ambiguous(self, pool: list[_Offer]) -> bool:
        """Several prices remain while our product names a spec ("256GB") the page leaves open."""
        prices = {o.price for o in pool} | {o.high_price for o in pool if o.high_price}
        if len(prices) < 2:
            return False
        page = spec_tokens(self.title)
        ours = spec_tokens(self.product.name)
        return any(not values <= page.get(unit, set()) for unit, values in ours.items())


def _pick(offers: list[_Offer], page_url: str, hint: _Hint | None = None) -> _Offer | None:
    """The offer the page is about.

    An offer whose URL or SKU the page URL names decides. Otherwise new goods beat used or
    refurbished ones, ``hint`` narrows variants down to those fitting our product, and the
    lowest buyable price wins - unless the remaining variants differ in price while our product
    names a spec the page leaves open: that gives an ``ambiguous`` offer without a price.
    """
    priced = [o for o in offers if o.price]
    if not priced:
        return offers[0] if offers else None
    matched = [
        o for o in priced if any(_url_matches(u, page_url) for u in o.urls) or _sku_in_url(o.sku, page_url)
    ]
    pool = matched or priced
    pool = [o for o in pool if o.condition in (None, "new")] or pool
    if hint is not None and not matched:
        pool = hint.narrow(pool)
        if hint.ambiguous(pool):
            return _Offer(ambiguous=True)
    pool = (
        [o for o in pool if o.availability in _BUYABLE]
        or [o for o in pool if o.availability is not A.OUT_OF_STOCK]
        or pool
    )
    return min(pool, key=lambda o: o.price or Decimal(0))


class _Graph:
    """Schema.org nodes from one syntax, with bare ``{"@id": ...}`` references resolvable."""

    def __init__(self, docs: Iterable[Any], ccy: str | None = None) -> None:
        self.ccy = ccy
        self.roots: list[dict[str, Any]] = []
        self.index: dict[str, dict[str, Any]] = {}
        for doc in docs:
            self._collect_roots(doc, 0)
            self._index(doc, 0)

    def _collect_roots(self, obj: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(obj, list):
            for item in obj:
                self._collect_roots(item, depth + 1)
        elif isinstance(obj, dict):
            self.roots.append(obj)
            for key in ("@graph", "mainEntity"):
                self._collect_roots(obj.get(key), depth + 1)

    def _index(self, obj: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(obj, dict):
            if isinstance(obj.get("@id"), str) and not obj.keys() <= _REF_KEYS:
                self.index.setdefault(obj["@id"], obj)
            for value in obj.values():
                self._index(value, depth + 1)
        elif isinstance(obj, list):
            for value in obj:
                self._index(value, depth + 1)

    def resolve(self, value: Any) -> list[Any]:
        """``value`` as a list, with reference-only nodes replaced by their full node."""
        out = []
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and item.keys() <= _REF_KEYS:
                item = self.index.get(item.get("@id") or "", item)
            if item is not None:
                out.append(item)
        return out

    def nodes(self, value: Any) -> list[dict[str, Any]]:
        return [v for v in self.resolve(value) if isinstance(v, dict)]

    def is_listing(self, page_urls: Iterable[str]) -> bool:
        """A category or search page: a CollectionPage, SearchResultsPage or list of several
        items without a product of its own, or several products that all live elsewhere."""
        products = [n for n in self.roots if _types(n) & _PRODUCT_TYPES]
        if not products:
            return any(
                _types(n) & _LIST_PAGE_TYPES
                or (_types(n) & _ITEM_LIST_TYPES and len(self.nodes(n.get("itemListElement"))) >= 2)
                for n in self.roots
            )
        base = next(iter(page_urls), "")
        homes = [_address(url) if (url := _abs(n.get("url"), base)) else None for n in products]
        here = {_address(u) for u in page_urls}
        return None not in homes and len(set(homes)) >= 2 and not here & set(homes)

    def main_product(self, page_url: str) -> dict[str, Any] | None:
        candidates = [n for n in self.roots if _types(n) & _PRODUCT_TYPES] or [
            n for n in self.roots if n.get("offers") and not _types(n) & _OFFER_TYPES
        ]
        if not candidates:
            return self._offer_only_product()

        def rank(node: dict[str, Any]) -> tuple[bool, bool, bool]:
            url = _abs(node.get("url"), page_url)
            return (
                bool(node.get("hasVariant")),
                bool(url and _url_matches(url, page_url)),
                bool(node.get("offers")),
            )

        return max(candidates, key=rank)

    def _root_offers(self, item_id: str | None = None) -> list[dict[str, Any]]:
        """Top-level ``Offer`` nodes, optionally only those whose ``itemOffered`` is ``item_id``."""
        return [
            n
            for n in self.roots
            if _types(n) & _OFFER_TYPES
            and (item_id is None or any(i.get("@id") == item_id for i in self.nodes(n.get("itemOffered"))))
        ]

    def _offer_only_product(self) -> dict[str, Any] | None:
        """Stand-in product for pages marking up bare offers; ``itemOffered`` gives its identity."""
        offers = self._root_offers()
        if not offers:
            return None
        item = next(iter(self.nodes(offers[0].get("itemOffered"))), {})
        return {**item, "offers": offers}

    def shipping(self, offer: dict[str, Any]) -> Decimal | None:
        for detail in self.nodes(offer.get("shippingDetails")):
            for rate in self.nodes(detail.get("shippingRate")):
                currency = _currency(rate.get("currency"), self.ccy)
                value = parse_decimal(_first(rate.get("value", rate.get("price"))), currency)
                if value is not None:
                    return value
        return None

    def offer(self, node: dict[str, Any], base: str) -> _Offer:
        specs = [s for s in self.nodes(node.get("priceSpecification")) if not _is_component(s)]
        list_specs: list[dict[str, Any]] = []
        sale_specs: list[dict[str, Any]] = []
        for spec in specs:
            is_list = _LIST_PRICE_RE.search(_str(spec.get("priceType")) or "")
            (list_specs if is_list else sale_specs).append(spec)
        sale_specs.sort(key=lambda s: "referenceQuantity" in s)  # per-unit prices last
        aggregate = "AggregateOffer" in _types(node)
        keys = ("lowPrice", "price") if aggregate else ("price", "lowPrice")
        currency = _currency(node.get("priceCurrency"), self.ccy) or next(
            (c for s in [*sale_specs, *list_specs] if (c := _currency(s.get("priceCurrency"), self.ccy))),
            None,
        )

        def amount(value: Any) -> Decimal | None:
            return parse_decimal(_first(value), currency)

        price = next((p for k in keys if (p := amount(node.get(k))) is not None), None)
        if not price:
            price = next((p for s in sale_specs if (p := amount(s.get("price")))), None)
        currency = currency or next(
            (c for k in keys if (c := detect_currency(_str(node.get(k)), self.ccy))), None
        )
        original = next((p for s in list_specs if (p := amount(s.get("price"))) is not None), None)
        url = _abs(node.get("url"), base)
        return _Offer(
            price=price,
            currency=currency,
            availability=_availability(node.get("availability")),
            original_price=original,
            shipping=self.shipping(node),
            urls=(url,) if url else (),
            sku=_str(node.get("sku")),
            gtin=_node_gtin(node),
            mpn=_str(node.get("mpn")),
            condition=_condition(node.get("itemCondition")),
            label=_str(node.get("name")),
            high_price=amount(node.get("highPrice")) if aggregate else None,
        )

    def offers_of(self, node: dict[str, Any], base: str) -> list[_Offer]:
        """Offers nested in ``node``, else top-level offers whose ``itemOffered`` points at it."""
        offers = self.nodes(node.get("offers"))
        if not offers and isinstance(item_id := node.get("@id"), str):
            offers = self._root_offers(item_id)
        return [self.offer(o, base) for o in offers]


def _structured_layer(docs: list[Any], page: _Page, method: str) -> ExtractedOffer | None:
    """Shared by JSON-LD, microdata and RDFa: all reduce to schema.org Product nodes."""
    url = page.url
    graph = _Graph(docs, page.ccy)
    if graph.is_listing(page.urls):
        return ExtractedOffer(url=url, method=LISTING)
    product = graph.main_product(url)
    if product is None:
        return None
    name = _str(product.get("name"))
    brand = _str(graph.resolve(product.get("brand")))
    hint = _Hint(page.product, name or "", brand) if page.product is not None else None
    offer: _Offer | None = None
    if variants := graph.nodes(product.get("hasVariant")):
        choices = []
        for variant in variants:
            best = _pick(graph.offers_of(variant, url), url) or _Offer()
            own_url = _abs(variant.get("url"), url)
            choices.append(
                replace(
                    best,
                    urls=(*best.urls, own_url) if own_url else best.urls,
                    sku=_str(variant.get("sku")) or best.sku,
                    gtin=_node_gtin(variant) or best.gtin,
                    mpn=_str(variant.get("mpn")) or best.mpn,
                    condition=best.condition or _condition(variant.get("itemCondition")),
                    label=_str(variant.get("name")) or best.label,
                    variant=variant,
                )
            )
        offer = _pick(choices, url, hint)
    if offer is None or not (offer.price or offer.ambiguous):
        offer = _pick(graph.offers_of(product, url), url, hint) or offer or _Offer()
    variant = offer.variant or {}
    condition = offer.condition or _condition(product.get("itemCondition"))
    return ExtractedOffer(
        url=url,
        method=AMBIGUOUS if offer.ambiguous else method,
        title=_with_condition(_str(variant.get("name")) or name, condition),
        price=offer.price,
        currency=offer.currency,
        availability=offer.availability,
        original_price=offer.original_price,
        shipping=offer.shipping,
        gtin=offer.gtin or _node_gtin(product),
        mpn=offer.mpn or _str(product.get("mpn")),
        sku=offer.sku or _str(product.get("sku")),
        brand=_str(graph.resolve(variant.get("brand"))) or brand,
        image=_image(variant.get("image"), url) or _image(product.get("image"), url),
    )


# --------------------------------------------------------------------------- JSON-LD

_JSONLD_WRAPPER_RE = re.compile(r"<!--|-->|<!\[CDATA\[|\]\]>|^\s*//.*$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
# A string value up to the first quote that really closes it (one followed by , } or ]).
_STRING_VALUE_RE = re.compile(r'(:\s*")(.{0,2000}?)(?="\s*[,}\]])', re.DOTALL)
_BARE_QUOTE_RE = re.compile(r'(?<!\\)"')
_JSON_START_RE = re.compile(r"[^\s,;]")
_DECODER = json.JSONDecoder(strict=False)  # strict=False allows raw control chars in strings


def _decode_all(text: str) -> tuple[list[Any], bool]:
    """Decode one or more concatenated JSON documents; the flag says all text was consumed."""
    docs: list[Any] = []
    pos = 0
    while (start := _JSON_START_RE.search(text, pos)) is not None:
        try:
            doc, pos = _DECODER.raw_decode(text, start.start())
        except json.JSONDecodeError:
            return docs, False
        docs.append(doc)
    return docs, True


def _repair(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text.replace("\\'", "'"))


def _escape_inner_quotes(text: str) -> str:
    """``"name": "TV 55" 4K",`` -> ``"name": "TV 55\\" 4K",``."""
    return _STRING_VALUE_RE.sub(lambda m: m.group(1) + _BARE_QUOTE_RE.sub(r'\\"', m.group(2)), text)


def _repairs(text: str) -> Iterator[str]:
    """``text`` and progressively more invasive fixes of it, computed only when needed."""
    yield text
    repaired = _repair(text)
    yield repaired
    yield _repair(html_lib.unescape(text))
    yield _escape_inner_quotes(repaired)


def load_jsonld(raw: str) -> list[Any]:
    """Leniently parse a JSON-LD script body: comment/CDATA wrappers, trailing commas,
    raw newlines in strings, unescaped quotes in values, HTML-escaped JSON and several
    concatenated objects."""
    text = _JSONLD_WRAPPER_RE.sub("", raw).strip()
    best: list[Any] = []
    for candidate in _repairs(text):
        docs, complete = _decode_all(candidate)
        if complete:
            return docs
        if len(docs) > len(best):
            best = docs
    return best


def _jsonld_layer(soup: BeautifulSoup, page: _Page) -> ExtractedOffer | None:
    docs: list[Any] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.IGNORECASE)}):
        docs.extend(load_jsonld(script.get_text()))
    return _structured_layer(docs, page, "jsonld") if docs else None


# --------------------------------------------------------------------------- microdata / RDFa

_PRICE_PROPS = frozenset({"price", "lowPrice", "highPrice"})


def _compact_rdfa(node: dict[str, Any]) -> dict[str, Any]:
    """Expanded RDFa node -> JSON-LD-like dict with schema.org local names as keys."""
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "@id":
            out[key] = value
            continue
        values = [
            v.get("@value", v) if isinstance(v, dict) else v
            for v in (value if isinstance(value, list) else [value])
        ]
        out[key if key == "@type" else _local(key)] = values[0] if len(values) == 1 else values
    return out


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_shown_price_prop(tag: Tag) -> bool:
    """A microdata/RDFa price property whose value is the element's text (no ``content``)."""
    if tag.has_attr("content") or tag.has_attr("itemscope"):
        return False
    props = [*str(tag.get("itemprop") or "").split(), *map(_local, str(tag.get("property") or "").split())]
    return not _PRICE_PROPS.isdisjoint(props)


def _dom_prices(soup: BeautifulSoup, ccy: str | None) -> dict[str, str]:
    """Shown microdata/RDFa prices re-read from the DOM, keyed by their text without spaces.

    extruct flattens ``2.599<sup>99</sup> Lei`` to ``"2.599 99 Lei"``, which reads as 99 Lei.
    """
    prices: dict[str, str] = {}
    for el in soup.find_all(_is_shown_price_prop):
        amount, currency = _element_price(el, ccy, current=True)
        if amount is not None:
            prices[_squash(_text(el))] = f"{amount} {currency or ''}".strip()
    return prices


def _with_dom_prices(obj: Any, prices: dict[str, str], is_price: bool = False) -> Any:
    if isinstance(obj, str):
        return prices.get(_squash(obj), obj) if is_price else obj
    if isinstance(obj, list):
        return [_with_dom_prices(v, prices, is_price) for v in obj]
    if isinstance(obj, dict):
        return {k: _with_dom_prices(v, prices, k in _PRICE_PROPS) for k, v in obj.items()}
    return obj


def _extruct_layers(soup: BeautifulSoup, html: str, page: _Page) -> list[ExtractedOffer]:
    syntaxes = [
        syntax
        for syntax, marker in (("microdata", r"itemscope"), ("rdfa", r"\btypeof\s*="))
        if re.search(marker, html, re.IGNORECASE)
    ]
    if not syntaxes:
        return []
    data = extruct.extract(html, base_url=page.url, syntaxes=syntaxes, uniform=True, errors="ignore")
    prices = _dom_prices(soup, page.ccy)
    rdfa = [_compact_rdfa(n) for n in data.get("rdfa") or [] if isinstance(n, dict)]
    layers = [
        _structured_layer(_with_dom_prices(data.get("microdata") or [], prices), page, "microdata"),
        _structured_layer(_with_dom_prices(rdfa, prices), page, "rdfa"),
    ]
    return [layer for layer in layers if layer is not None]


# --------------------------------------------------------------------------- DOM helpers

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_PAGE_ROOTS = frozenset({"body", "html", "[document]"})
_STRUCK_TAGS = ("del", "s", "strike")
_SUP_DECIMALS_RE = re.compile(r"[.,]?([0-9]{1,2})")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


def _token_re(words: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z])(?:{words})(?![a-z])")


_OLD_RE = _token_re(
    r"old|was|strike[a-z]*|streich[a-z]*|crossed|list-?price|compare(?:-at)?|msrp|rrp|uvp|prp|pvp"
    r"|before|original|full-price|vechi|initial|anterior|intreg|statt|barre|ancien|przekreslona"
)
_PER_UNIT_RE = _token_re(r"per-?unit|unit-?price|price-?per[a-z-]*|base-?price|grundpreis")
# Amounts shown around the price that are not what the item sells for: instalments, shipping,
# savings, member-only and net (ex-VAT) prices, and the EU "lowest price in 30 days" notice.
_NOT_SELLING_RE = _token_re(
    r"install?ments?|rates?|rata|monthly|per-?month|financ[a-z]*|leasing"
    r"|shipping|delivery|livrare|transport|versand[a-z]*"
    r"|save|savings?|you-?save|economi[a-z]*|ersparnis"
    r"|club|members?|loyalty|omnibus"
    r"|excl|excluding[a-z-]*|ex-?vat|ex-?tax|tax-?excl[a-z]*|without-?(?:tax|vat)|net|netto|fara-tva"
)
# Text only screen readers get; it repeats the visible prices ("Original price was: ...").
_SCREEN_READER_RE = _token_re(
    r"screen-reader[a-z-]*|visually-?hidden|sr-only|a11y-hidden|assistive-text|hidden-visually"
)
_NOT_SELLING_DEPTH = 3

# One amount with an optional currency marker on either side ("$1,199.99", "1.499,99 lei").
_AMOUNT = (
    r"(?:(?:[$€£¥₹₺₴]|[a-z]{1,3}\$)\s?)?"
    r"(?:\d{1,3}(?:[ ']\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*)"
    r"(?:\s?(?:[$€£¥₹₺₴]|[^\W\d_]{1,4}\.?)(?![^\W\d_]))?"
)
_NET_WORDS = (
    r"fara tva|excl\.? (?:tva|vat|tax|mwst|btw|iva)|ex\.? vat|exkl\.? mwst|zzgl\.? mwst"
    r"|netto|net|sin iva|senza iva|iva esclusa|bez vat"
)
# An amount labelled as not the current price, matched on accent-folded lowercase text.
_LABELLED_AMOUNT_RE = re.compile(
    r"\b(?:"
    # old / list / reference prices
    r"was|prp|pvp|rrp|msrp|uvp|statt|instead of|au lieu de|invece di|zamiast|antes|bisher"
    r"|(?:regular|original|old|list|reference|retail) price(?: was)?|compare at"
    r"|pret(?:ul)? (?:vechi|intreg|initial|anterior|de lista|recomandat)"
    r"|alter preis|ursprunglicher preis|prix (?:barre|initial|de reference|conseille)|ancien prix"
    r"|precio (?:anterior|original|de referencia)|prezzo (?:di listino|consigliato|precedente|pieno)"
    r"|cena (?:regularna|katalogowa|poprzednia)|(?:stara|poprzednia) cena"
    r"|(?:lowest price|cel mai mic pret|niedrigster preis|najnizsza cena|prix le plus bas"
    r"|precio mas bajo|prezzo piu basso)[^:\d]{0,40}(?:\d+[^:\d]{0,20})?:"
    # savings
    r"|(?:you )?save[ds]?|savings?|economisesti|economisiti|economie|reducere|(?:sie )?sparen"
    r"|ersparnis|economisez|ahorras?|ahorro|risparmi[ao]?|oszczedzasz|oszczednosc|rabat"
    # shipping
    r"|shipping|delivery|livrare|transport|versand(?:kosten)?|lieferung|livraison|frais de port"
    r"|envio|gastos de envio|spedizione|dostawa|wysylka"
    # member and loyalty prices
    r"|club|member(?:s|ship)?(?: price)?|loyalty"
    # instalments: "12 rate de", "4 payments of", "24 x"
    r"|(?:\d+ )?(?:rate|rata|raty|installments?|instalments?|payments?)(?: (?:de|of|x|a|po))?|\d+ ?[x×]"
    rf"|{_NET_WORDS}"
    r")\W{0,3}" + _AMOUNT + r"|" + _AMOUNT + r"\s?(?:"
    # per month: "108,33 lei/luna", "$41.58/mo", "99 € pro Monat", "x 24 rate"
    r"/\s?(?:luna|lun|month|mo|mth|monat|mtl|mois|mes|mese|mies)"
    r"|(?:pe|per|a|al|pro|par) (?:luna|month|monat|mois|mes|mese|miesiac)"
    r"|x \d+|lunar|monthly|monatlich|mensuel|mensual|mensile|miesiecznie"
    rf"|\+ ?(?:tva|vat|mwst)|ht|hors taxes?|{_NET_WORDS}"
    r"|livrare|shipping|delivery|versand"
    r")\b"
)


def _text(el: Tag) -> str:
    return " ".join(el.get_text(" ").split())


def _value(el: Tag) -> str:
    content = el.get("content")
    return content.strip() if isinstance(content, str) else _text(el)


def _attrs(tag: Tag) -> str:
    """Class names and id, camelCase split and lowercased: ``"oldPrice"`` -> ``"old-price"``."""
    raw = " ".join([*tag.get_attribute_list("class"), str(tag.get("id") or "")])
    return _CAMEL_RE.sub(r"\1-\2", raw).lower()


def _ancestry(tag: Tag) -> Iterator[Tag]:
    """``tag`` and its ancestors below ``<body>`` (themes put layout classes on body)."""
    return takewhile(lambda t: t.name not in _PAGE_ROOTS, chain([tag], tag.parents))


def _lineage(tag: Tag, depth: int) -> Iterator[Tag]:
    return islice(_ancestry(tag), depth)


def _not_current(tag: Tag) -> bool:
    """Descendants to ignore when reading a current price: struck, old, per-unit, instalment,
    shipping, savings, member and net prices, and screen-reader-only text."""
    attrs = _attrs(tag)
    return tag.name in _STRUCK_TAGS or any(
        r.search(attrs) for r in (_OLD_RE, _PER_UNIT_RE, _NOT_SELLING_RE, _SCREEN_READER_RE)
    )


def _without_labelled_amounts(text: str) -> str:
    """``"Was $1,199.99 Now $999.99"`` -> ``" Now $999.99"``: drops amounts labelled as old,
    saved, shipping, instalment, member or net prices."""
    folded = _fold_same_length(text)
    kept, pos = [], 0
    for m in _LABELLED_AMOUNT_RE.finditer(folded):
        kept.append(text[pos : m.start()])
        pos = m.end()
    return " ".join(kept) + " " + text[pos:] if kept else text


def _element_availability(el: Tag) -> Availability:
    """From a ``<link href="…/InStock">``, a ``content`` attribute or the label text."""
    for raw in (el.get("href"), _value(el)):
        if (availability := _availability(raw)) is not A.UNKNOWN:
            return availability
    return A.UNKNOWN


def _element_price(
    el: Tag, ccy: str | None = None, *, current: bool = False
) -> tuple[Decimal | None, str | None]:
    """Price shown by an element; ``ccy`` says what shared markers ("$", "lei") mean.

    Split prices like ``1.299<sup>99</sup> Lei`` become 1299.99: digits in the
    first ``<sup>`` are decimals when the main number has none. With ``current``,
    struck, old and other non-selling amounts inside the element are ignored.
    """
    parts: list[str] = []
    decimals: str | None = None

    def walk(node: Tag) -> None:
        nonlocal decimals
        for child in node.children:
            if isinstance(child, Tag):
                if child.name in _SKIP_TAGS or (current and _not_current(child)):
                    continue
                m = _SUP_DECIMALS_RE.fullmatch(child.get_text().strip()) if child.name == "sup" else None
                if m and decimals is None:
                    decimals = m.group(1)
                    parts.append(" ")
                else:
                    walk(child)
            elif isinstance(child, NavigableString) and not isinstance(child, PreformattedString):
                parts.append(str(child))

    walk(el)
    text = " ".join("".join(parts).split())
    amount, currency = parse_price(_without_labelled_amounts(text) if current else text, preferred=ccy)
    if amount is not None and decimals and amount.as_tuple().exponent == 0:
        amount = Decimal(f"{amount}.{decimals}")
    return amount, currency


def _select(soup: BeautifulSoup, css: str | None) -> Tag | None:
    if not css:
        return None
    try:
        return soup.select_one(css)
    except (SelectorSyntaxError, NotImplementedError):
        log.warning("ignoring invalid CSS selector %r", css)
        return None


# --------------------------------------------------------------------------- selectors layer


def _selector_layer(soup: BeautifulSoup, sel: SelectorConfig, page: _Page) -> ExtractedOffer:
    def price_of(el: Tag, *, current: bool) -> tuple[Decimal | None, str | None]:
        raw = el.get(sel.price_attr) if sel.price_attr else None
        if isinstance(raw, str):
            currency = detect_currency(raw, page.ccy)
            return parse_decimal(raw, currency or page.ccy), currency
        # A selected price block may also hold the struck old price or a savings badge.
        shown = _element_price(el, page.ccy, current=current)
        return shown if shown[0] is not None else _element_price(el, page.ccy)

    price = currency = original = shipping = None
    if (el := _select(soup, sel.price)) is not None:
        price, currency = price_of(el, current=True)
    if (el := _select(soup, sel.original_price)) is not None:
        original = price_of(el, current=False)[0]
    if (el := _select(soup, sel.currency)) is not None:
        currency = _currency(_value(el), page.ccy) or currency
    if (el := _select(soup, sel.shipping)) is not None:
        shipping = parse_price(_value(el), preferred=page.ccy)[0]
    avail_el = _select(soup, sel.availability)

    def text_of(css: str | None) -> str | None:
        el = _select(soup, css)
        return (_value(el) or None) if el is not None else None

    return ExtractedOffer(
        url=page.url,
        method="selectors",
        price=price,
        currency=currency,
        original_price=original,
        shipping=shipping,
        availability=_element_availability(avail_el) if avail_el is not None else A.UNKNOWN,
        title=text_of(sel.title),
        gtin=_gtin(text_of(sel.gtin)),
        mpn=text_of(sel.mpn),
        brand=text_of(sel.brand),
    )


# --------------------------------------------------------------------------- meta layer

_PRICE_LABEL_RE = re.compile(r"price|pre[tț]|preis|prix|precio|prezzo|cena", re.IGNORECASE)
_STOCK_LABEL_RE = re.compile(r"availab|stock|stoc|disponib|verf[uü]gbar|lager", re.IGNORECASE)


def _meta_layer(soup: BeautifulSoup, page: _Page) -> ExtractedOffer:
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name") or tag.get("itemprop")
        content = tag.get("content")
        if isinstance(key, str) and isinstance(content, str):
            meta.setdefault(key.strip().lower(), content.strip())

    method = "opengraph"
    price: Decimal | None
    original: Decimal | None = None
    currency = _currency(
        meta.get("product:sale_price:currency")
        or meta.get("product:price:currency")
        or meta.get("og:price:currency"),
        page.ccy,
    )
    if sale := parse_decimal(meta.get("product:sale_price:amount"), currency):
        price, original = sale, parse_decimal(meta.get("product:price:amount"), currency)
    else:
        price = parse_decimal(meta.get("product:price:amount") or meta.get("og:price:amount"), currency)
    availability = _availability(meta.get("product:availability") or meta.get("og:availability"))

    if not price and (el := soup.find(itemprop="price")) is not None:
        method = "meta"
        ccy_el = soup.find(itemprop="priceCurrency")
        declared = _currency(_value(ccy_el), page.ccy) if ccy_el is not None else None
        content = el.get("content")
        if isinstance(content, str):
            price, shown = parse_decimal(content, declared or currency), None
        else:
            price, shown = _element_price(el, page.ccy)
        currency = declared or shown or currency
    if availability is A.UNKNOWN and (el := soup.find(itemprop="availability")) is not None:
        availability = _element_availability(el)

    for i in range(1, 5):
        label, data = meta.get(f"twitter:label{i}"), meta.get(f"twitter:data{i}")
        if not (label and data):
            continue
        if not price and _PRICE_LABEL_RE.search(label):
            method = "meta"
            price, shown = parse_price(data, preferred=page.ccy)
            currency = shown or currency
        elif availability is A.UNKNOWN and _STOCK_LABEL_RE.search(label):
            availability = _availability(data)

    return ExtractedOffer(
        url=page.url,
        method=method,
        price=price,
        currency=currency,
        original_price=original,
        availability=availability,
        title=meta.get("og:title") or meta.get("twitter:title") or None,
        image=_abs(
            meta.get("og:image:secure_url") or meta.get("og:image") or meta.get("twitter:image"), page.url
        ),
        brand=meta.get("product:brand") or meta.get("og:brand") or None,
        gtin=_gtin(*(meta.get(k) for k in ("product:ean", "product:gtin", "product:upc", "product:isbn"))),
        mpn=meta.get("product:mfr_part_no") or None,
        sku=meta.get("product:retailer_item_id") or None,
    )


# --------------------------------------------------------------------------- heuristic layer

_REGULAR_RE = _token_re(r"regular")
_SALE_RE = _token_re(r"sale|special|promo[a-z]*|discount[a-z]*|final|current|new|now|actual|reduced|deal")
_EXCLUDED_RE = _token_re(
    r"related|recommend[a-z]*|similar|upsells?|up-sell|cross-?sells?|carousel|swiper|slick"
    r"|mini-?cart|cart-drawer|basket|wishlist|sidebar|filters?|newsletter|breadcrumbs?|bundles?"
    r"|recently|also-(?:bought|viewed|like)|navbar|menu|megamenu|site-header|site-footer|cookie"
)
_PRODUCT_AREA_RE = _token_re(r"pdp|product|entry-summary")
_STOCK_ATTR_RE = re.compile(r"stock|availab|disponib")
# Stock state classes, read on a stock element before its label text.
_STOCK_STATE_CLASSES: tuple[tuple[Availability, re.Pattern[str]], ...] = (
    (A.OUT_OF_STOCK, _token_re(r"out-of-stock|outofstock|sold-?out|unavailable|not-available|indisponibil")),
    (A.PREORDER, _token_re(r"pre-?order|back-?order")),
    (A.LIMITED, _token_re(r"low-stock|limited-stock")),
    (A.IN_STOCK, _token_re(r"in-stock|instock|available")),
)
_NON_PRICE_TAGS = _SKIP_TAGS | {"meta", "link", "input", "option", "select", "button", "textarea"}
_LINEAGE_DEPTH = 5
_MAX_PRICE_TEXT = 100
_MAX_STOCK_TEXT = 120
# A category grid: at least this many look-alike sibling tiles, each with a price and a link.
_MIN_TILES = 3
_TILE_DEPTH = 6


def _excluded(tag: Tag) -> bool:
    """Inside navigation, page chrome or another product's tile (related, carousel, cart...)."""
    for node in _ancestry(tag):
        if node.name in ("nav", "aside"):
            return True
        if node.name in ("header", "footer") and node.find_parent(["main", "article"]) is None:
            return True
        if _EXCLUDED_RE.search(_attrs(node)):
            return True
    return False


def _in_product_area(tag: Tag) -> bool:
    return any(
        node.name == "main"
        or "product" in str(node.get("itemtype") or "").lower()
        or _PRODUCT_AREA_RE.search(_attrs(node))
        for node in _ancestry(tag)
    )


@dataclass(frozen=True)
class _Candidate:
    amount: Decimal
    currency: str | None
    in_product_area: bool
    sale: bool = False
    regular: bool = False
    tag: Tag | None = None

    @property
    def score(self) -> int:
        return 2 * self.in_product_area + self.sale


def _is_price_tag(tag: Tag) -> bool:
    return tag.name not in _NON_PRICE_TAGS and "price" in _attrs(tag)


def _tile_link(tile: Tag, base: str) -> tuple[str, str] | None:
    link = tile if tile.name == "a" and tile.has_attr("href") else tile.find("a", href=True)
    href = str(link["href"]).strip() if link is not None else ""
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    return _address(urljoin(base, href))


class _TileGrid:
    """Finds category/search grids: look-alike sibling tiles that each show a price and link to
    another page. A price inside such a grid is no single page's price."""

    def __init__(self, page: _Page) -> None:
        self.page = page
        self.here = {_address(u) for u in page.urls}
        self.seen: dict[tuple[int, str, str | None], bool] = {}

    def contains(self, tag: Tag) -> bool:
        for node in _lineage(tag, _TILE_DEPTH):
            parent = node.parent
            if parent is None or parent.name in _PAGE_ROOTS:
                return False
            kind = next(iter(node.get_attribute_list("class")), None)
            key = (id(parent), node.name, kind)
            if key not in self.seen:
                self.seen[key] = self._is_grid(parent, node.name, kind)
            if self.seen[key]:
                return True
        return False

    def _is_grid(self, parent: Tag, name: str, kind: str | None) -> bool:
        tiles = [
            tile
            for tile in parent.find_all(name, recursive=False)
            if next(iter(tile.get_attribute_list("class")), None) == kind
        ]
        if len(tiles) < _MIN_TILES:
            return False
        links = {
            link
            for tile in tiles
            if tile.find(_is_price_tag) is not None
            and (link := _tile_link(tile, self.page.url)) is not None
        }
        return len(links - self.here) >= _MIN_TILES


def _heuristic_layer(soup: BeautifulSoup, page: _Page) -> ExtractedOffer:
    current: list[_Candidate] = []
    old: list[_Candidate] = []
    for tag in soup.find_all(_is_price_tag):
        if len(_text(tag)) > _MAX_PRICE_TEXT or _excluded(tag):
            continue
        lineage = [_attrs(t) for t in _lineage(tag, _LINEAGE_DEPTH)]
        if any(_PER_UNIT_RE.search(a) or _SCREEN_READER_RE.search(a) for a in lineage) or any(
            _NOT_SELLING_RE.search(a) for a in lineage[:_NOT_SELLING_DEPTH]
        ):
            continue
        area = _in_product_area(tag)
        struck = tag.name in _STRUCK_TAGS or tag.find_parent(_STRUCK_TAGS) is not None
        if struck or any(_OLD_RE.search(a) for a in lineage):
            amount, currency = _element_price(tag, page.ccy)
            if amount:
                old.append(_Candidate(amount, currency, area))
            continue
        amount, currency = _element_price(tag, page.ccy, current=True)
        if amount:
            sale = any(_SALE_RE.search(a) for a in lineage)
            regular = any(_REGULAR_RE.search(a) for a in lineage)
            current.append(_Candidate(amount, currency, area, sale, regular, tag))
    for tag in soup.find_all(_STRUCK_TAGS):
        amount, currency = _element_price(tag, page.ccy)
        if amount and currency and not _excluded(tag):
            old.append(_Candidate(amount, currency, _in_product_area(tag)))

    # A price block holding complete inner prices (regular, sale) is read through them.
    containers = {id(p) for c in current if c.currency and c.tag is not None for p in c.tag.parents}
    current = [c for c in current if id(c.tag) not in containers]
    if any(c.sale for c in current):  # a sale price exists, so "regular" is the old one
        current = [c for c in current if not c.regular]
    ranked = sorted(current, key=lambda c: -c.score)  # stable: document order breaks ties
    grid = _TileGrid(page)
    best = next((c for c in ranked if c.tag is None or not grid.contains(c.tag)), None)
    original = None
    if best is not None:
        higher = [c for c in old if c.amount > best.amount]
        original = max(higher, key=lambda c: c.score).amount if higher else None

    headings = (_text(el) for name in ("h1", "title") if (el := soup.find(name)) is not None)
    return ExtractedOffer(
        url=page.url,
        method=LISTING if ranked and best is None else "heuristic",
        price=best.amount if best else None,
        currency=best.currency if best else None,
        original_price=original,
        availability=_page_availability(soup),
        title=next((h for h in headings if h), None),
    )


def _is_stock_tag(tag: Tag) -> bool:
    return tag.name not in _SKIP_TAGS and _STOCK_ATTR_RE.search(_attrs(tag)) is not None


def _page_availability(soup: BeautifulSoup) -> Availability:
    """Stock state from elements whose class/id mention stock/availability: their state class
    (``out-of-stock``, ``in-stock``) first, else their label text."""
    for tag in soup.find_all(_is_stock_tag):
        if _excluded(tag):
            continue
        attrs = _attrs(tag)
        if state := next((a for a, pattern in _STOCK_STATE_CLASSES if pattern.search(attrs)), None):
            return state
        text = _text(tag)
        if len(text) <= _MAX_STOCK_TEXT and (availability := availability_from_text(text)) is not A.UNKNOWN:
            return availability
    return A.UNKNOWN


# --------------------------------------------------------------------------- public API


def _guard(fn: Callable[..., T], *args: Any) -> T | None:
    """Run one extraction layer; a broken layer must not take the others down."""
    try:
        return fn(*args)
    except Exception:  # parsing arbitrary third-party HTML
        log.debug("%s failed", fn.__name__, exc_info=True)
        return None


def _merge(url: str, layers: list[ExtractedOffer], default_currency: str | None) -> ExtractedOffer:
    def first(name: str, pool: Iterable[ExtractedOffer]) -> Any:
        return next((v for layer in pool if (v := getattr(layer, name)) is not None), None)

    # Zero is treated as "no price": shops use 0 for "price on request". A verdict (listing
    # page, ambiguous variants) decides like a price does, so lower layers cannot guess one.
    primary = next((layer for layer in layers if layer.price or layer.method in VERDICTS), None)
    fields: dict[str, Any] = {name: first(name, layers) for name in _IDENTITY_FIELDS}
    if primary is not None and primary.method in VERDICTS:
        return ExtractedOffer(url=url, method=primary.method, **fields)
    by_offer = [primary, *(x for x in layers if x is not primary)] if primary else layers
    fields["availability"] = next(
        (x.availability for x in by_offer if x.availability is not A.UNKNOWN), A.UNKNOWN
    )
    if primary is not None and primary.price is not None:
        price = primary.price
        fields |= {
            "price": price,
            "method": primary.method,
            "confidence": CONFIDENCE[primary.method],
            "currency": first("currency", by_offer) or currency_code(default_currency),
            "shipping": first("shipping", by_offer),
            # Only trust a strike price from a layer that saw the same selling price.
            "original_price": next(
                (
                    x.original_price
                    for x in by_offer
                    if x.price == price and x.original_price and x.original_price > price
                ),
                None,
            ),
        }
    return ExtractedOffer(url=url, **fields)


def extract_offer(
    html: str,
    url: str,
    selectors: SelectorConfig | None = None,
    default_currency: str | None = None,
    product: Product | None = None,
) -> ExtractedOffer:
    """Extract the offer on a product page. Never raises.

    Returns an :class:`ExtractedOffer` with ``price=None`` and ``method="none"``
    when no price is found (identity fields such as the title may still be set).
    ``default_currency`` (the competitor's currency) applies when a price is found
    but no currency is shown, and says what a marker several currencies share
    means: "$" in a CAD shop is CAD, "lei" in a Moldovan one MDL. ``product``
    (our catalogue item) picks its variant on pages that offer several.

    Category/search pages give ``method=LISTING`` and variants at different prices
    that cannot be told apart give ``method=AMBIGUOUS``, both without a price.
    """
    try:
        soup = BeautifulSoup(html or "", "lxml")
        page = _Page(url, _page_urls(soup, url), currency_code(default_currency), product)
        layers = [
            _guard(_selector_layer, soup, selectors, page) if selectors is not None else None,
            _guard(_jsonld_layer, soup, page),
            *(_guard(_extruct_layers, soup, html, page) or []),
            _guard(_meta_layer, soup, page),
            _guard(_heuristic_layer, soup, page),
        ]
        return _merge(url, [x for x in layers if x is not None], default_currency)
    except Exception:  # the contract is "never raises"
        log.warning("extraction failed for %s", url, exc_info=True)
        return ExtractedOffer(url=url)
