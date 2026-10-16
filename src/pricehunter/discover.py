"""Find candidate product pages on a competitor site, via its own search or its sitemaps.

Discovery only proposes URLs; :mod:`pricehunter.match` decides later whether a page really is
our product. Everything here degrades to "fewer candidates" instead of raising.
"""

from __future__ import annotations

import asyncio
import gzip
import html as html_lib
import io
import logging
import re
import zlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
from urllib.parse import (
    parse_qsl,
    quote_plus,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

from bs4 import BeautifulSoup, SoupStrainer, Tag
from lxml import etree
from soupsieve import SelectorSyntaxError

from .match import model_tokens, normalize_gtin, normalize_text, score_match
from .models import Competitor, ExtractedOffer, FetchResult, Product

__all__ = [
    "PageFetcher",
    "SitemapCache",
    "build_queries",
    "discover_candidates",
    "extract_links",
    "page_key",
    "parse_sitemap",
    "rank_sitemap_urls",
    "search_competitor",
    "sitemap_urls",
]

log = logging.getLogger(__name__)

# The sitemap protocol caps an uncompressed file at 50 MB; anything bigger is refused.
MAX_SITEMAP_BYTES = 50 * 1024 * 1024


class PageFetcher(Protocol):
    """The slice of :class:`pricehunter.fetch.Fetcher` discovery needs. Must never raise."""

    async def get(
        self, url: str, competitor: Competitor | None = None, *, render: bool | None = None
    ) -> FetchResult: ...

    async def get_bytes(
        self, url: str, competitor: Competitor | None = None
    ) -> tuple[FetchResult, bytes]: ...


# --------------------------------------------------------------------------- queries


def build_queries(product: Product) -> list[str]:
    """Search queries from most to least specific: GTIN, "brand MPN", name, keywords."""
    queries: list[str] = []
    if product.gtin and normalize_gtin(product.gtin):
        queries.append(re.sub(r"\D", "", product.gtin))
    if product.mpn:
        queries.append(f"{product.brand} {product.mpn}" if product.brand else product.mpn)
    name = product.name
    brand = normalize_text(product.brand or "")
    if brand and f" {brand} " not in f" {normalize_text(name)} ":
        name = f"{product.brand} {name}"
    queries.append(name)
    queries.extend(product.keywords)

    out: list[str] = []
    seen: set[str] = set()
    for query in (" ".join(q.split()) for q in queries):
        if query and (key := query.casefold()) not in seen:
            seen.add(key)
            out.append(query)
    return out


# --------------------------------------------------------------------------- URL hygiene

_SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "sms:", "whatsapp:", "viber:")
# Query parameters that only record how a visitor arrived (ads, mailings, the shop's own search
# or recommendation slots) or carry a session; they never select another page. Pages reached
# under params missing here are still told apart by their canonical URL (see page_key).
_TRACKING_PARAMS = frozenset({
    "gclid", "gclsrc", "gbraid", "wbraid", "dclid", "fbclid", "msclkid", "yclid", "ttclid",
    "twclid", "igshid", "srsltid", "mc_cid", "mc_eid", "mkt_tok", "_ga", "_gl", "_hsenc", "_hsmi",
    "_kx", "sc_cid", "spm", "scm", "trk", "ref_", "ref_src",
    "_pos", "_sid", "_ss", "_psq", "_fid", "_v",  # Shopify search results
    "pr_prod_strat", "pr_rec_id", "pr_rec_pid", "pr_ref_pid", "pr_seq",  # Shopify recommendations
    "phpsessid", "jsessionid", "sessionid", "oscsid", "zenid",
})  # fmt: skip
# "x-": non-standard names such as eMAG's X-Search-Id, X-Product-Id and X-Section.
_TRACKING_PREFIXES = ("utm_", "pf_rd_", "pd_rd_", "x-")
# "?ref=search" tags a referral, but "?ref=AB123" may be the product reference itself.
_REFERRAL_PARAMS = frozenset({"ref", "referrer", "referer"})
_ASSET_EXT = re.compile(
    r"\.(?:jpe?g|png|gif|webp|avif|svg|ico|bmp|css|js|mjs|map|json|xml|txt|pdf|zip|gz|rar"
    r"|mp4|webm|mp3|woff2?|ttf|otf|eot)$"
)
# A path segment equal to one of these never leads to a product page.
_JUNK_SEGMENTS = frozenset({
    "cart", "basket", "bag", "cos", "cos-cumparaturi", "checkout", "comanda", "login", "logout",
    "signin", "sign-in", "signup", "sign-up", "register", "inregistrare", "autentificare",
    "account", "my-account", "myaccount", "contul-meu", "cont", "customer", "wishlist",
    "favorite", "favorites", "favourites", "compare", "comparare", "search", "cautare", "blog",
    "news", "stiri", "help", "ajutor", "support", "contact", "terms", "termeni", "privacy",
    "cookies", "gdpr", "faq", "about", "despre", "careers", "cariere", "page", "pagina", "tag",
    "tags", "author", "feed", "rss", "cdn-cgi", "wp-admin", "wp-login.php", "newsletter",
})  # fmt: skip
# Info pages with a descriptive slug ("contact-us", "termeni-si-conditii").
_INFO_PREFIX = re.compile(
    r"^(?:contact|terms|termeni|privacy|politica|help|ajutor|about|despre|faq|cookie|livrare"
    r"|shipping|delivery|returns?|retur|garantie|warranty|careers)[-_]"
)
# Listing roots: dropped unless the URL also carries a product signal (Shopify
# "/collections/x/products/y" is a product, "/collections/x" is not).
_LISTING_SEGMENTS = frozenset({
    "c", "cat", "category", "categories", "categorie", "categorii", "collections", "collection",
    "department", "departments", "brand", "brands", "branduri", "producator", "producatori",
})  # fmt: skip
_LISTING_PARAMS = frozenset({
    "page", "pg", "pagina", "p_page", "sort", "sort_by", "sortby", "order", "orderby", "dir",
    "filter", "filters", "limit", "per_page", "perpage", "view", "mode", "price_min", "price_max",
})  # fmt: skip
_PRODUCT_SEGMENTS = frozenset({
    "p", "pd", "dp", "product", "products", "produs", "produse", "item", "items", "itm", "ip",
    "sku", "prod",
})  # fmt: skip
_PRODUCT_PATH = re.compile(r"-p-\d|[-_]pid[-_]?\d|\.html?$|\.php$")
_PRODUCT_PARAMS = frozenset({"id", "pid", "product_id", "productid", "products_id", "item_id", "sku"})
_SLUG_WITH_ID = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+){2,}$")
_ID_TAIL = re.compile(r"[a-z][-_]\d{4,}$")


def _host_key(host: str | None) -> str:
    host = (host or "").lower().rstrip(".")
    return host.removeprefix("www.")


@lru_cache(maxsize=256)
def _compile(pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        log.warning("ignoring invalid product_url_pattern %r: %s", pattern, exc)
        return None


def _is_tracking(key: str, value: str) -> bool:
    key = key.lower()
    if key in _REFERRAL_PARAMS:
        return not any(c.isdigit() for c in value)
    return key in _TRACKING_PARAMS or key.startswith(_TRACKING_PREFIXES)


def _clean_url(href: str, base: str, site: str, pattern: re.Pattern[str] | None) -> str | None:
    """Absolute same-site http(s) URL without fragment or tracking params, or ``None``."""
    href = href.strip()
    if not href or href.startswith("#") or href.lower().startswith(_SKIP_SCHEMES):
        return None
    try:
        parts = urlsplit(urljoin(base, href))
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or _host_key(host) != site:
        return None
    params = parse_qsl(parts.query, keep_blank_values=True)
    kept = [(k, v) for k, v in params if not _is_tracking(k, v)]
    query = parts.query if len(kept) == len(params) else urlencode(kept)
    url = urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))
    if pattern is not None and not pattern.search(url):
        return None
    return url


def _segments(url: str) -> tuple[list[str], set[str]]:
    parts = urlsplit(url)
    segments = [s for s in unquote(parts.path).lower().split("/") if s]
    keys = {k.lower() for k, _ in parse_qsl(parts.query, keep_blank_values=True)}
    return segments, keys


def _looks_like_product(url: str) -> bool:
    segments, keys = _segments(url)
    if any(seg in _PRODUCT_SEGMENTS for seg in segments[:-1]):
        return True
    path = "/" + "/".join(segments)
    if _PRODUCT_PATH.search(path) or keys & _PRODUCT_PARAMS:
        return True
    if any(seg.isdigit() and len(seg) >= 5 for seg in segments):
        return True
    last = re.sub(r"\.[a-z0-9]{2,5}$", "", segments[-1]) if segments else ""
    if _ID_TAIL.search(last):
        return True  # "item-123456"
    return bool(_SLUG_WITH_ID.match(last)) and any(c.isdigit() for c in last)


def _is_junk(url: str) -> bool:
    segments, keys = _segments(url)
    if not segments:
        return True  # home page
    if _ASSET_EXT.search(segments[-1]) or keys & _LISTING_PARAMS:
        return True
    if any(seg in _JUNK_SEGMENTS for seg in segments):
        return True
    if _looks_like_product(url):
        return False
    return any(seg in _LISTING_SEGMENTS or _INFO_PREFIX.match(seg) for seg in segments)


def _dedupe(urls: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(urls))


_SELF_REFERENCES = SoupStrainer(["link", "meta"])


def _declared_address(html: str, url: str, site: str) -> str | None:
    """The same-site ``<link rel=canonical>`` (else ``og:url``) of a page, if it names a page
    that could be a product (a canonical pointing at the home page or a category is ignored)."""
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml", parse_only=_SELF_REFERENCES)
    for tag, attr in (
        (soup.find("link", rel="canonical"), "href"),
        (soup.find("meta", property="og:url"), "content"),
    ):
        value = tag.get(attr) if isinstance(tag, Tag) else None
        address = _clean_url(value, url, site, None) if isinstance(value, str) else None
        if address is not None and not _is_junk(address):
            return address
    return None


def page_key(html: str, url: str, competitor: Competitor) -> str:
    """Which page the ``html`` fetched from ``url`` is, to spot one page reached under several
    addresses (``?ref=search``, a category path, a session id).

    The page's own canonical address decides when it declares one on the competitor's site,
    otherwise ``url`` does. Tracking and session parameters, parameter order, the scheme,
    ``www.``, percent-encoding and a trailing slash do not count. A key, not a URL to fetch.
    """
    site = _host_key(urlsplit(competitor.base_url).hostname)
    parts = urlsplit(_declared_address(html, url, site) or url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    query = urlencode(sorted((k, v) for k, v in params if not _is_tracking(k, v)))
    key = _host_key(parts.hostname) + unquote(parts.path).rstrip("/")
    return f"{key}?{query}" if query else key


# --------------------------------------------------------------------------- search pages


def _href_of(el: Tag) -> str | None:
    """The link an element stands for: itself, a descendant/ancestor ``<a>``, or a data attribute."""
    anchor = el if el.name == "a" and el.has_attr("href") else None
    anchor = anchor or el.find("a", href=True) or el.find_parent("a", href=True)
    for node, attr in ((anchor, "href"), (el, "data-href"), (el, "data-url")):
        value = node.get(attr) if isinstance(node, Tag) else None
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_links(html: str, page_url: str, competitor: Competitor) -> list[str]:
    """Candidate product URLs on a search/listing page, product-looking ones first.

    With ``selectors.search_result_link`` only the selected elements are used; otherwise every
    ``<a href>`` is considered and navigation, account, cart, category, pagination and asset
    links are dropped. Only links on the competitor's own site survive (``www.`` ignored), and
    ``product_url_pattern`` must match when configured.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    base = page_url
    if isinstance(base_tag := soup.find("base", href=True), Tag):
        base = urljoin(page_url, str(base_tag["href"]))
    site = _host_key(urlsplit(competitor.base_url).hostname)
    pattern = _compile(competitor.product_url_pattern)

    selector = competitor.selectors.search_result_link
    if selector:
        try:
            elements = soup.select(selector)
        except SelectorSyntaxError as exc:
            log.warning("%s: bad search_result_link selector %r: %s", competitor.name, selector, exc)
        else:
            hrefs = (h for el in elements if (h := _href_of(el)))
            return _dedupe(u for h in hrefs if (u := _clean_url(h, base, site, pattern)))

    current = _clean_url(page_url, page_url, site, None)
    products: list[str] = []
    others: list[str] = []
    for a in soup.find_all("a", href=True):
        url = _clean_url(str(a["href"]), base, site, pattern)
        if url is None or url == current or _is_junk(url):
            continue
        (products if _looks_like_product(url) else others).append(url)
    return _dedupe(products + others)


def _search_url(competitor: Competitor, query: str) -> str:
    template = competitor.search_url or ""
    return urljoin(competitor.base_url, template.replace("{query}", quote_plus(query)))


def _redirected(requested: str, final: str) -> bool:
    return urlsplit(requested).path.rstrip("/") != urlsplit(final).path.rstrip("/")


async def search_competitor(
    fetcher: PageFetcher, competitor: Competitor, product: Product, limit: int
) -> list[str]:
    """Run :func:`build_queries` through the competitor's site search until ``limit`` URLs.

    A search that redirects straight to a product page (common for exact GTIN/MPN hits) is the
    strongest signal there is, so such landing pages come before every result-list link.
    Failed fetches are skipped.
    """
    if not competitor.search_url or limit <= 0:
        return []
    site = _host_key(urlsplit(competitor.base_url).hostname)
    pattern = _compile(competitor.product_url_pattern)
    landings: dict[str, None] = {}
    found: dict[str, None] = {}
    searched: set[str] = set()
    for query in build_queries(product):
        if len(landings.keys() | found.keys()) >= limit:
            break
        url = _search_url(competitor, query)
        if url in searched:
            continue
        searched.add(url)
        try:
            result = await fetcher.get(url, competitor)
            if not result.ok:
                log.debug("%s: search %r failed: %s", competitor.name, query, result.error)
                continue
            final = result.final_url or url
            if _redirected(url, final):
                landing = _clean_url(final, final, site, pattern)
                if landing and not _is_junk(landing):
                    landings.setdefault(landing)
            for link in extract_links(result.html, final, competitor):
                found.setdefault(link)
        except Exception:
            log.exception("%s: search %r crashed", competitor.name, query)
    return _dedupe([*landings, *found])[:limit]


# --------------------------------------------------------------------------- sitemaps

_LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE | re.DOTALL)
_BLOCK = re.compile(r"<(url|sitemap)\b[^>]*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_XML_PARSER = etree.XMLParser(
    recover=True, resolve_entities=False, no_network=True, encoding="utf-8", remove_comments=True
)


def _local(tag: object) -> str:
    return etree.QName(tag).localname.lower() if isinstance(tag, str) else ""


def _parse_tree(text: str) -> tuple[list[str], list[str]]:
    try:
        root = etree.fromstring(text.encode("utf-8"), _XML_PARSER)
    except (etree.XMLSyntaxError, ValueError):
        return [], []
    pages: list[str] = []
    children: list[str] = []
    if root is None:
        return pages, children
    for loc in root.iter():
        if _local(loc.tag) != "loc" or not (url := (loc.text or "").strip()):
            continue
        parent = loc.getparent()
        kind = _local(parent.tag) if parent is not None else ""
        if kind == "url":
            pages.append(url)
        elif kind == "sitemap":
            children.append(url)
        # <image:loc>, <video:loc>, ... describe media, not pages.
    return pages, children


def _parse_regex(text: str) -> tuple[list[str], list[str]]:
    def loc(block: str) -> str | None:
        m = _LOC.search(block)
        return html_lib.unescape(_CDATA.sub(r"\1", m[1])).strip() if m else None

    pages: list[str] = []
    children: list[str] = []
    for kind, body in _BLOCK.findall(text):
        if url := loc(body):
            (children if kind.lower() == "sitemap" else pages).append(url)
    if not pages and not children and "<" not in text:
        # Plain-text sitemap: one URL per line.
        pages = [ln.strip() for ln in text.splitlines() if ln.strip().lower().startswith("http")]
    return pages, children


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Split a sitemap into (page URLs, child sitemap URLs).

    Handles ``<urlset>`` and ``<sitemapindex>`` with or without namespaces, odd whitespace,
    CDATA and broken markup (lxml in recover mode, then a regex fallback), and plain-text
    sitemaps. Image/video ``<loc>`` entries are ignored.
    """
    text = xml_text.lstrip("﻿ \t\r\n")
    if not text:
        return [], []
    pages, children = _parse_tree(text)
    if pages or children:
        return pages, children
    return _parse_regex(text)


def _gunzip(body: bytes) -> bytes | None:
    """Decompress a ``.gz`` sitemap, refusing more than ``MAX_SITEMAP_BYTES`` (decompression bombs)."""
    if body[:2] != b"\x1f\x8b":
        return body  # served with Content-Encoding: gzip and already decoded in transit
    cap = MAX_SITEMAP_BYTES
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
            data = gz.read(cap + 1)
    except (OSError, EOFError, zlib.error) as exc:
        log.warning("corrupt gzip sitemap: %s", exc)
        return None
    if len(data) > cap:
        log.warning("gzip sitemap expands beyond %d bytes; skipped", cap)
        return None
    return data


async def _fetch_sitemap(fetcher: PageFetcher, competitor: Competitor, url: str) -> str | None:
    if urlsplit(url).path.lower().endswith(".gz"):
        result, body = await fetcher.get_bytes(url, competitor)
        data = _gunzip(body) if result.ok else None
        return data.decode("utf-8", errors="replace") if data is not None else None
    result = await fetcher.get(url, competitor, render=False)
    return result.html if result.ok else None


async def _sitemap_roots(fetcher: PageFetcher, competitor: Competitor) -> list[str]:
    """Configured sitemap, else the ones robots.txt declares, else ``/sitemap.xml``."""
    if competitor.sitemap_url:
        return [urljoin(competitor.base_url, competitor.sitemap_url)]
    robots = await fetcher.get(urljoin(competitor.base_url, "/robots.txt"), competitor, render=False)
    declared = re.findall(r"(?im)^\s*sitemap\s*:\s*(\S+)", robots.html) if robots.ok else []
    return _dedupe(declared) or [urljoin(competitor.base_url, "/sitemap.xml")]


# Anchored on a word start: "sitemap" itself contains "item".
_PRODUCT_SITEMAP = re.compile(r"(?<![a-z])(?:product|produs|item|catalog|shop)")
_OTHER_SITEMAP = re.compile(
    r"(?<![a-z])(?:blog|post|news|article|categor|tag|author|brand|page|image|video|static)"
)


def _sitemap_priority(url: str) -> int:
    """Visit product sitemaps first and editorial/category ones last (judged by the path only)."""
    name = urlsplit(url).path.lower()
    return 0 if _PRODUCT_SITEMAP.search(name) else 2 if _OTHER_SITEMAP.search(name) else 1


# Safety valves of one sitemap crawl: URLs kept, sitemap files read and index levels followed.
MAX_SITEMAP_URLS = 200_000
MAX_SITEMAP_FILES = 500
MAX_SITEMAP_DEPTH = 3
_NARROW = "pages listed further on were not considered (set search_url or product_url_pattern)"
_DIGIT_RUN = re.compile(r"\d{8,14}")


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _url_path(url: str) -> str:
    return unquote(urlsplit(url).path).lower()


def _after_host(url: str) -> str:
    """Lower-cased path and query of an absolute or root-relative URL, without parsing it."""
    return unquote(url.split("://", 1)[-1].partition("/")[2]).lower()


def _product_tokens(product: Product) -> set[str]:
    return set(normalize_text(" ".join(filter(None, [product.brand, product.name, product.mpn]))).split())


@dataclass(frozen=True)
class _Terms:
    """What a sitemap URL must mention to be worth scoring against one of some products: a word
    (3+ characters, not a number) of a brand, name or MPN, an MPN, or a GTIN."""

    words: frozenset[str]
    mpns: frozenset[str]
    gtins: frozenset[str]

    @classmethod
    def of(cls, products: Iterable[Product]) -> _Terms:
        words: set[str] = set()
        mpns: set[str] = set()
        gtins: set[str] = set()
        for product in products:
            words.update(w for w in _product_tokens(product) if len(w) >= 3 and not w.isdigit())
            if mpn := _compact(product.mpn or ""):
                mpns.add(mpn)
            if gtin := normalize_gtin(product.gtin):
                gtins.add(gtin)
        return cls(frozenset(words), frozenset(mpns), frozenset(gtins))

    def has_gtin(self, path: str) -> bool:
        return bool(self.gtins) and any(normalize_gtin(r) in self.gtins for r in _DIGIT_RUN.findall(path))

    def mentioned_in(self, path: str) -> bool:
        """Whether a lower-cased URL path (and query) names one of the words, MPNs or GTINs."""
        if any(word in path for word in self.words) or self.has_gtin(path):
            return True
        compact = _compact(path)
        return any(mpn in compact for mpn in self.mpns)


async def _crawl(
    fetcher: PageFetcher,
    competitor: Competitor,
    terms: _Terms | None,
    max_urls: int,
    max_depth: int,
) -> tuple[list[str], str | None]:
    """Page URLs from the competitor's sitemaps and, when a safety valve stopped the crawl, why."""
    site = _host_key(urlsplit(competitor.base_url).hostname)
    pattern = _compile(competitor.product_url_pattern)
    queue = deque((url, 0) for url in await _sitemap_roots(fetcher, competitor))
    visited: set[str] = set()
    found: dict[str, None] = {}
    while queue:
        url, depth = queue.popleft()
        if url in visited:
            continue
        if len(visited) >= MAX_SITEMAP_FILES:
            stop = f"stopped after {MAX_SITEMAP_FILES} sitemap files"
            return list(found), f"{competitor.name}: sitemap discovery {stop}; {_NARROW}"
        visited.add(url)
        text = await _fetch_sitemap(fetcher, competitor, url)
        if text is None:
            continue
        pages, children = parse_sitemap(text)
        for page in pages:
            if terms is not None and not terms.mentioned_in(_after_host(page)):
                continue  # checked first: most URLs of a big shop are about other things
            clean = _clean_url(page, url, site, pattern)
            if clean is None or clean in found or _is_junk(clean):
                continue
            if len(found) >= max_urls:
                stop = f"stopped at {max_urls} candidate URLs"
                return list(found), f"{competitor.name}: sitemap discovery {stop}; {_NARROW}"
            found[clean] = None
        if depth < max_depth:
            queue.extend((urljoin(url, c), depth + 1) for c in sorted(children, key=_sitemap_priority))
    return list(found), None


async def sitemap_urls(
    fetcher: PageFetcher,
    competitor: Competitor,
    max_urls: int = MAX_SITEMAP_URLS,
    max_depth: int = MAX_SITEMAP_DEPTH,
    products: Iterable[Product] | None = None,
) -> list[str]:
    """Page URLs from the competitor's sitemaps, following index files up to ``max_depth``.

    ``.gz`` files are fetched as bytes and decompressed (capped at 50 MB). Only same-site,
    non-junk URLs matching ``product_url_pattern`` (when set) are kept and, given ``products``,
    only those that mention one of them, so a shop of any size is covered without holding all
    its URLs. ``max_urls`` and :data:`MAX_SITEMAP_FILES` are safety valves; hitting one is logged.
    """
    terms = None if products is None else _Terms.of(products)
    urls, problem = await _crawl(fetcher, competitor, terms, max_urls, max_depth)
    if problem:
        log.warning(problem)
    return urls


# --------------------------------------------------------------------------- sitemap ranking

_EXTENSION = re.compile(r"\.(?:html?|php|aspx?|jsp)$")
# Matcher signs that a URL names something other than the item itself.
_OTHER_ITEM = ("accessory_penalty", "bundle_penalty", "condition_penalty")
_ID_SUFFIX = re.compile(r"[-_](?:p|pd|pid|id|i|sku|prod)[-_]?\d+$")
_LONG_ID = re.compile(r"^\d{5,}$")


def _url_slug(url: str, keep: set[str]) -> str:
    """Readable words from the last meaningful path segments ("/apple-iphone-15-128gb-p-123.html").

    Marker segments (``/p/``, ``/product/``), id suffixes (``-p-123``), long numeric ids and
    extensions are dropped; a long number survives when it also appears in ``keep`` (e.g. a
    LEGO set number in our product name).
    """
    segments = [
        _ID_SUFFIX.sub("", _EXTENSION.sub("", s))
        for s in _url_path(url).split("/")
        if s and s not in _PRODUCT_SEGMENTS
    ]
    words: list[str] = []
    for segment in segments[-2:]:
        tokens = re.split(r"[-_+.\s]+", segment)
        words.extend(t for t in tokens if t and (not _LONG_ID.match(t) or t in keep))
    return " ".join(words)


def rank_sitemap_urls(
    urls: list[str], product: Product, limit: int, min_score: float = 0.5
) -> list[str]:
    """The ``limit`` sitemap URLs whose slugs best match ``product``, scoring >= ``min_score``.

    Slugs are scored with :func:`pricehunter.match.score_match` (so variant and spec guards
    apply, and slugs naming an accessory, bundle or used item are dropped). A URL carrying our
    GTIN wins outright, one carrying our MPN ranks just below it, and a shared model token adds
    0.1.
    """
    if limit <= 0 or not urls:
        return []
    terms = _Terms.of([product])
    keep = _product_tokens(product)
    mpn = _compact(product.mpn or "")
    models = model_tokens(f"{product.name} {product.mpn or ''}")

    scored: list[tuple[float, int, str]] = []
    for index, url in enumerate(urls):
        path = _url_path(url)
        if terms.has_gtin(path):
            scored.append((1.0, index, url))
            continue
        if not terms.mentioned_in(path):
            continue  # cheap prefilter: no word of ours appears in the URL at all
        slug = _url_slug(url, keep)
        if not slug:
            continue
        result = score_match(product, ExtractedOffer(url=url, title=slug))
        if any(key in result.details for key in _OTHER_ITEM):
            continue  # "husa-...", "...-refurbished": our identifiers in it must not lift it
        score = result.score
        if result.reason == "title":  # no conflict found, so identifiers in the URL can lift it
            compact = _compact(path)
            if len(mpn) >= 5 and mpn in compact:
                score = max(score, 0.97)
            elif any(m in compact for m in models):
                score += 0.1
        if score >= min_score:
            scored.append((min(score, 1.0), index, url))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, _, url in scored[:limit]]


# --------------------------------------------------------------------------- entry point


class SitemapCache:
    """Sitemap crawls shared by the products of one run.

    Each competitor's sitemaps are read once, keeping only the URLs that mention one of
    ``products`` (the prefilter of :func:`rank_sitemap_urls`), so even a shop with millions of
    pages is covered without holding them all. ``problems`` gets a message for each crawl that
    a safety valve stopped early.
    """

    def __init__(self, products: Iterable[Product]) -> None:
        products = list(products)
        self._skus = frozenset(p.sku for p in products)
        self._terms = _Terms.of(products)
        self._crawls: dict[str, asyncio.Task[list[str]]] = {}
        self.problems: list[str] = []

    def covers(self, product: Product) -> bool:
        return product.sku in self._skus

    async def urls(self, fetcher: PageFetcher, competitor: Competitor) -> list[str]:
        """The competitor's sitemap URLs that mention our products; callers share one crawl."""
        task = self._crawls.get(competitor.name)
        if task is None:
            task = asyncio.ensure_future(self._load(fetcher, competitor))
            self._crawls[competitor.name] = task
        return await asyncio.shield(task)

    async def _load(self, fetcher: PageFetcher, competitor: Competitor) -> list[str]:
        urls, problem = await _crawl(
            fetcher, competitor, self._terms, MAX_SITEMAP_URLS, MAX_SITEMAP_DEPTH
        )
        if problem:
            self.problems.append(problem)
        return urls


async def discover_candidates(
    fetcher: PageFetcher,
    competitor: Competitor,
    product: Product,
    limit: int,
    sitemap_cache: SitemapCache | None = None,
) -> list[str]:
    """Up to ``limit`` candidate product URLs for ``product`` on ``competitor``. Never raises.

    Uses the site search when ``search_url`` is configured, falling back to the sitemap only
    if search found nothing and a ``sitemap_url`` is explicitly configured. Without a search
    URL the sitemap (configured, from robots.txt, or ``/sitemap.xml``) is crawled, once per
    competitor when ``sitemap_cache`` covers the product, and ranked against the product.
    """
    found: list[str] = []
    try:
        if competitor.search_url:
            found = await search_competitor(fetcher, competitor, product, limit)
            if found or not competitor.sitemap_url:
                return found
        if sitemap_cache is not None and sitemap_cache.covers(product):
            urls = await sitemap_cache.urls(fetcher, competitor)
        else:
            urls = await sitemap_urls(fetcher, competitor, products=[product])
        return rank_sitemap_urls(urls, product, limit)
    except Exception:
        log.exception("%s: discovery failed for %s", competitor.name, product.sku)
        return found
