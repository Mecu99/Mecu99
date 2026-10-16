"""Extract a competitor offer (price, currency, stock, identifiers) from a product page.

Layers, most trusted first: configured CSS selectors, JSON-LD, microdata, RDFa,
OpenGraph/meta tags, and finally class-name heuristics. The first layer with a
price decides price, currency and method; identity fields (title, GTIN, ...)
are filled from whichever layer has them, in the same order.
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
from itertools import chain, islice, takewhile
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urljoin, urlsplit

import extruct
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PreformattedString
from soupsieve import SelectorSyntaxError

from .models import Availability, ExtractedOffer, SelectorConfig
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
_IDENTITY_FIELDS = ("title", "gtin", "mpn", "brand", "sku", "image")
_BUYABLE = (A.IN_STOCK, A.LIMITED)

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


def _currency(value: Any) -> str | None:
    return currency_code(_str(value))


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
# Checked in order on accent-folded lowercase text; negatives first because
# "indisponibil" contains "disponibil" and "not in stock" contains "in stock".
_AVAILABILITY_TEXT: tuple[tuple[Availability, re.Pattern[str]], ...] = (
    (
        A.OUT_OF_STOCK,
        re.compile(
            r"\b(?:out of stock|sold out|not (?:in stock|available)|unavailable|stoc epuizat"
            r"|epuizat|indisponibil|nu (?:mai )?(?:este|e) (?:in stoc|disponibil)|ausverkauft"
            r"|nicht (?:auf lager|verfugbar|lieferbar|vorratig)|epuise|rupture de stock"
            r"|agotado|esaurito|niedostepny|brak w magazynie|discontinued)"
        ),
    ),
    (
        A.PREORDER,
        re.compile(r"\b(?:pre-?order|precomand|vorbestell|pre-?commande|back-?order)"),
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


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def availability_from_text(text: str | None) -> Availability:
    """Map a stock label such as "În stoc" or "Ausverkauft" to an :class:`Availability`."""
    if not text:
        return A.UNKNOWN
    folded = _fold(text)
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


def _sku_in_url(sku: str | None, url: str) -> bool:
    if not sku:
        return False
    parts = urlsplit(url)
    values = {v.lower() for _, v in parse_qsl(parts.query)}
    values |= {seg.lower().removesuffix(".html") for seg in parts.path.split("/") if seg}
    return sku.lower() in values


# --------------------------------------------------------------------------- schema.org graph

_PRODUCT_TYPES = frozenset(
    {"Product", "ProductGroup", "IndividualProduct", "ProductModel", "SomeProducts", "Vehicle", "Car", "Book"}
)
_OFFER_TYPES = frozenset({"Offer", "AggregateOffer"})
_GTIN_KEYS = ("gtin", "gtin13", "gtin12", "gtin14", "gtin8", "isbn")
_LIST_PRICE_RE = re.compile(r"strikethrough|listprice|msrp", re.IGNORECASE)
_REF_KEYS = frozenset({"@id", "@type"})
_MAX_DEPTH = 12


def _local(name: str) -> str:
    """``"http://schema.org/Product"`` / ``"schema:Product"`` -> ``"Product"``."""
    return re.split(r"[/#:]", name.rstrip("/"))[-1]


def _types(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    return {_local(t) for t in (raw if isinstance(raw, list) else [raw]) if isinstance(t, str)}


def _node_gtin(node: dict[str, Any]) -> str | None:
    return _gtin(*(node.get(k) for k in _GTIN_KEYS))


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
    variant: dict[str, Any] | None = None  # the hasVariant node this offer came from


def _pick(offers: list[_Offer], page_url: str) -> _Offer | None:
    """The offer the page is about: URL/SKU match first, else the lowest buyable price."""
    priced = [o for o in offers if o.price]
    if not priced:
        return offers[0] if offers else None
    matched = [
        o for o in priced if any(_url_matches(u, page_url) for u in o.urls) or _sku_in_url(o.sku, page_url)
    ]
    pool = matched or priced
    pool = (
        [o for o in pool if o.availability in _BUYABLE]
        or [o for o in pool if o.availability is not A.OUT_OF_STOCK]
        or pool
    )
    return min(pool, key=lambda o: o.price or Decimal(0))


class _Graph:
    """Schema.org nodes from one syntax, with bare ``{"@id": ...}`` references resolvable."""

    def __init__(self, docs: Iterable[Any]) -> None:
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
                value = parse_decimal(_first(rate.get("value", rate.get("price"))))
                if value is not None:
                    return value
        return None

    def offer(self, node: dict[str, Any], base: str) -> _Offer:
        specs = self.nodes(node.get("priceSpecification"))
        list_specs: list[dict[str, Any]] = []
        sale_specs: list[dict[str, Any]] = []
        for spec in specs:
            is_list = _LIST_PRICE_RE.search(_str(spec.get("priceType")) or "")
            (list_specs if is_list else sale_specs).append(spec)
        sale_specs.sort(key=lambda s: "referenceQuantity" in s)  # per-unit prices last
        keys = ("lowPrice", "price") if "AggregateOffer" in _types(node) else ("price", "lowPrice")
        price = next((p for k in keys if (p := parse_decimal(_first(node.get(k)))) is not None), None)
        currency = _currency(node.get("priceCurrency"))
        if not price:
            for spec in sale_specs:
                if spec_price := parse_decimal(_first(spec.get("price"))):
                    price, currency = spec_price, currency or _currency(spec.get("priceCurrency"))
                    break
        currency = (
            currency
            or next((c for s in specs if (c := _currency(s.get("priceCurrency")))), None)
            or next((c for k in keys if (c := detect_currency(_str(node.get(k))))), None)
        )
        original = next(
            (p for s in list_specs if (p := parse_decimal(_first(s.get("price")))) is not None),
            None,
        )
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
        )

    def offers_of(self, node: dict[str, Any], base: str) -> list[_Offer]:
        """Offers nested in ``node``, else top-level offers whose ``itemOffered`` points at it."""
        offers = self.nodes(node.get("offers"))
        if not offers and isinstance(item_id := node.get("@id"), str):
            offers = self._root_offers(item_id)
        return [self.offer(o, base) for o in offers]


def _structured_layer(docs: list[Any], url: str, method: str) -> ExtractedOffer | None:
    """Shared by JSON-LD, microdata and RDFa: all reduce to schema.org Product nodes."""
    graph = _Graph(docs)
    product = graph.main_product(url)
    if product is None:
        return None
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
                    variant=variant,
                )
            )
        offer = _pick(choices, url)
    if offer is None or not offer.price:
        offer = _pick(graph.offers_of(product, url), url) or offer or _Offer()
    variant = offer.variant or {}
    return ExtractedOffer(
        url=url,
        method=method,
        title=_str(variant.get("name")) or _str(product.get("name")),
        price=offer.price,
        currency=offer.currency,
        availability=offer.availability,
        original_price=offer.original_price,
        shipping=offer.shipping,
        gtin=_node_gtin(variant) or offer.gtin or _node_gtin(product),
        mpn=_str(variant.get("mpn")) or offer.mpn or _str(product.get("mpn")),
        sku=offer.sku or _str(product.get("sku")),
        brand=_str(graph.resolve(variant.get("brand"))) or _str(graph.resolve(product.get("brand"))),
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


def _jsonld_layer(soup: BeautifulSoup, url: str) -> ExtractedOffer | None:
    docs: list[Any] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.IGNORECASE)}):
        docs.extend(load_jsonld(script.get_text()))
    return _structured_layer(docs, url, "jsonld") if docs else None


# --------------------------------------------------------------------------- microdata / RDFa


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


def _extruct_layers(html: str, url: str) -> list[ExtractedOffer]:
    syntaxes = [
        syntax
        for syntax, marker in (("microdata", r"itemscope"), ("rdfa", r"\btypeof\s*="))
        if re.search(marker, html, re.IGNORECASE)
    ]
    if not syntaxes:
        return []
    data = extruct.extract(html, base_url=url, syntaxes=syntaxes, uniform=True, errors="ignore")
    layers = [
        _structured_layer(data.get("microdata") or [], url, "microdata"),
        _structured_layer(
            [_compact_rdfa(n) for n in data.get("rdfa") or [] if isinstance(n, dict)], url, "rdfa"
        ),
    ]
    return [layer for layer in layers if layer is not None]


# --------------------------------------------------------------------------- DOM helpers

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_PAGE_ROOTS = frozenset({"body", "html", "[document]"})
_STRUCK_TAGS = ("del", "s", "strike")
_SUP_DECIMALS_RE = re.compile(r"[.,]?([0-9]{1,2})")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


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


def _element_availability(el: Tag) -> Availability:
    """From a ``<link href="…/InStock">``, a ``content`` attribute or the label text."""
    for raw in (el.get("href"), _value(el)):
        if (availability := _availability(raw)) is not A.UNKNOWN:
            return availability
    return A.UNKNOWN


def _element_price(el: Tag, skip: Callable[[Tag], bool] | None = None) -> tuple[Decimal | None, str | None]:
    """Price shown by an element, ignoring ``skip``-ped descendants.

    Split prices like ``1.299<sup>99</sup> Lei`` become 1299.99: digits in the
    first ``<sup>`` are decimals when the main number has none.
    """
    parts: list[str] = []
    decimals: str | None = None

    def walk(node: Tag) -> None:
        nonlocal decimals
        for child in node.children:
            if isinstance(child, Tag):
                if child.name in _SKIP_TAGS or (skip and skip(child)):
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
    amount, currency = parse_price("".join(parts))
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


def _selector_layer(soup: BeautifulSoup, sel: SelectorConfig, url: str) -> ExtractedOffer:
    def price_of(el: Tag) -> tuple[Decimal | None, str | None]:
        raw = el.get(sel.price_attr) if sel.price_attr else None
        if isinstance(raw, str):
            return parse_decimal(raw), detect_currency(raw)
        return _element_price(el)

    price = currency = original = shipping = None
    if (el := _select(soup, sel.price)) is not None:
        price, currency = price_of(el)
    if (el := _select(soup, sel.original_price)) is not None:
        original = price_of(el)[0]
    if (el := _select(soup, sel.currency)) is not None:
        currency = _currency(_value(el)) or currency
    if (el := _select(soup, sel.shipping)) is not None:
        shipping = parse_price(_value(el))[0]
    avail_el = _select(soup, sel.availability)

    def text_of(css: str | None) -> str | None:
        el = _select(soup, css)
        return (_value(el) or None) if el is not None else None

    return ExtractedOffer(
        url=url,
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


def _meta_layer(soup: BeautifulSoup, url: str) -> ExtractedOffer:
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name") or tag.get("itemprop")
        content = tag.get("content")
        if isinstance(key, str) and isinstance(content, str):
            meta.setdefault(key.strip().lower(), content.strip())

    method = "opengraph"
    price: Decimal | None
    original: Decimal | None = None
    if sale := parse_decimal(meta.get("product:sale_price:amount")):
        price, original = sale, parse_decimal(meta.get("product:price:amount"))
    else:
        price = parse_decimal(meta.get("product:price:amount") or meta.get("og:price:amount"))
    currency = _currency(
        meta.get("product:sale_price:currency")
        or meta.get("product:price:currency")
        or meta.get("og:price:currency")
    )
    availability = _availability(meta.get("product:availability") or meta.get("og:availability"))

    if not price and (el := soup.find(itemprop="price")) is not None:
        method = "meta"
        content = el.get("content")
        price, shown = (parse_decimal(content), None) if isinstance(content, str) else _element_price(el)
        currency = shown or currency
        if (ccy_el := soup.find(itemprop="priceCurrency")) is not None:
            currency = _currency(_value(ccy_el)) or currency
    if availability is A.UNKNOWN and (el := soup.find(itemprop="availability")) is not None:
        availability = _element_availability(el)

    for i in range(1, 5):
        label, data = meta.get(f"twitter:label{i}"), meta.get(f"twitter:data{i}")
        if not (label and data):
            continue
        if not price and _PRICE_LABEL_RE.search(label):
            method = "meta"
            price, shown = parse_price(data)
            currency = shown or currency
        elif availability is A.UNKNOWN and _STOCK_LABEL_RE.search(label):
            availability = _availability(data)

    return ExtractedOffer(
        url=url,
        method=method,
        price=price,
        currency=currency,
        original_price=original,
        availability=availability,
        title=meta.get("og:title") or meta.get("twitter:title") or None,
        image=_abs(meta.get("og:image:secure_url") or meta.get("og:image") or meta.get("twitter:image"), url),
        brand=meta.get("product:brand") or meta.get("og:brand") or None,
        gtin=_gtin(*(meta.get(k) for k in ("product:ean", "product:gtin", "product:upc", "product:isbn"))),
        mpn=meta.get("product:mfr_part_no") or None,
        sku=meta.get("product:retailer_item_id") or None,
    )


# --------------------------------------------------------------------------- heuristic layer


def _token_re(words: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z])(?:{words})(?![a-z])")


_OLD_RE = _token_re(
    r"old|was|strike[a-z]*|crossed|list-?price|compare(?:-at)?|msrp|rrp|uvp|before|original|full-price"
)
_PER_UNIT_RE = _token_re(r"per-?unit|unit-?price|price-?per[a-z-]*|base-?price|grundpreis")
_REGULAR_RE = _token_re(r"regular")
_SALE_RE = _token_re(r"sale|special|promo[a-z]*|discount[a-z]*|final|current|new|now|actual|reduced|deal")
_EXCLUDED_RE = _token_re(
    r"related|recommend[a-z]*|similar|upsells?|up-sell|cross-?sells?|carousel|swiper|slick"
    r"|mini-?cart|cart-drawer|basket|wishlist|sidebar|filters?|newsletter|breadcrumbs?|bundles?"
    r"|recently|also-(?:bought|viewed|like)|navbar|menu|megamenu|site-header|site-footer|cookie"
)
_PRODUCT_AREA_RE = _token_re(r"pdp|product|entry-summary")
_STOCK_ATTR_RE = re.compile(r"stock|availab|disponib")
_NON_PRICE_TAGS = _SKIP_TAGS | {"meta", "link", "input", "option", "select", "button", "textarea"}
_LINEAGE_DEPTH = 5
_MAX_PRICE_TEXT = 100
_MAX_STOCK_TEXT = 120


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


def _not_current(tag: Tag) -> bool:
    """Descendants to ignore when reading a current price: struck, old and per-unit prices."""
    attrs = _attrs(tag)
    return tag.name in _STRUCK_TAGS or bool(_OLD_RE.search(attrs) or _PER_UNIT_RE.search(attrs))


@dataclass(frozen=True)
class _Candidate:
    amount: Decimal
    currency: str | None
    in_product_area: bool
    sale: bool = False
    regular: bool = False

    @property
    def score(self) -> int:
        return 2 * self.in_product_area + self.sale


def _is_price_tag(tag: Tag) -> bool:
    return tag.name not in _NON_PRICE_TAGS and "price" in _attrs(tag)


def _heuristic_layer(soup: BeautifulSoup, url: str) -> ExtractedOffer:
    current: list[_Candidate] = []
    old: list[_Candidate] = []
    for tag in soup.find_all(_is_price_tag):
        if len(_text(tag)) > _MAX_PRICE_TEXT or _excluded(tag):
            continue
        lineage = [_attrs(t) for t in _lineage(tag, _LINEAGE_DEPTH)]
        if any(_PER_UNIT_RE.search(a) for a in lineage):
            continue
        area = _in_product_area(tag)
        struck = tag.name in _STRUCK_TAGS or tag.find_parent(_STRUCK_TAGS) is not None
        if struck or any(_OLD_RE.search(a) for a in lineage):
            amount, currency = _element_price(tag)
            if amount:
                old.append(_Candidate(amount, currency, area))
            continue
        amount, currency = _element_price(tag, skip=_not_current)
        if amount:
            sale = any(_SALE_RE.search(a) for a in lineage)
            regular = any(_REGULAR_RE.search(a) for a in lineage)
            current.append(_Candidate(amount, currency, area, sale, regular))
    for tag in soup.find_all(_STRUCK_TAGS):
        amount, currency = _element_price(tag)
        if amount and currency and not _excluded(tag):
            old.append(_Candidate(amount, currency, _in_product_area(tag)))

    if any(c.sale for c in current):  # a sale price exists, so "regular" is the old one
        current = [c for c in current if not c.regular]
    best = max(current, key=lambda c: c.score, default=None)
    original = None
    if best is not None:
        higher = [c for c in old if c.amount > best.amount]
        original = max(higher, key=lambda c: c.score).amount if higher else None

    headings = (_text(el) for name in ("h1", "title") if (el := soup.find(name)) is not None)
    return ExtractedOffer(
        url=url,
        method="heuristic",
        price=best.amount if best else None,
        currency=best.currency if best else None,
        original_price=original,
        availability=_page_availability(soup),
        title=next((h for h in headings if h), None),
    )


def _is_stock_tag(tag: Tag) -> bool:
    return tag.name not in _SKIP_TAGS and _STOCK_ATTR_RE.search(_attrs(tag)) is not None


def _page_availability(soup: BeautifulSoup) -> Availability:
    """Stock label text, read only from elements whose class/id mention stock/availability."""
    for tag in soup.find_all(_is_stock_tag):
        text = _text(tag)
        if len(text) > _MAX_STOCK_TEXT or _excluded(tag):
            continue
        if (availability := availability_from_text(text)) is not A.UNKNOWN:
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

    # Zero is treated as "no price": shops use 0 for "price on request".
    primary = next((layer for layer in layers if layer.price), None)
    by_offer = [primary, *(x for x in layers if x is not primary)] if primary else layers
    fields: dict[str, Any] = {name: first(name, layers) for name in _IDENTITY_FIELDS}
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
) -> ExtractedOffer:
    """Extract the offer on a product page. Never raises.

    Returns an :class:`ExtractedOffer` with ``price=None`` and ``method="none"``
    when no price is found (identity fields such as the title may still be set).
    ``default_currency`` applies when a price is found but no currency is shown.
    """
    try:
        soup = BeautifulSoup(html or "", "lxml")
        layers = [
            _guard(_selector_layer, soup, selectors, url) if selectors is not None else None,
            _guard(_jsonld_layer, soup, url),
            *(_guard(_extruct_layers, html, url) or []),
            _guard(_meta_layer, soup, url),
            _guard(_heuristic_layer, soup, url),
        ]
        return _merge(url, [x for x in layers if x is not None], default_currency)
    except Exception:  # the contract is "never raises"
        log.warning("extraction failed for %s", url, exc_info=True)
        return ExtractedOffer(url=url)
