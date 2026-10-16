"""A local, offline demo marketplace: three fake competitor shops on 127.0.0.1.

Each shop runs its own :class:`~http.server.ThreadingHTTPServer` (a distinct host:port with an
allow-all ``robots.txt``) and exercises a different integration path:

* **MegaShop**: JSON-LD ``@graph`` product pages and a site search at ``/search?q=``. A GTIN
  query redirects straight to the product page, and related-product links add noise.
* **PretMic**: a Romanian shop with prices like ``1.749<sup>99</sup> Lei`` and no structured
  data, so it needs CSS selectors. Search is at ``/cauta?q=``.
* **BudgetBay**: microdata and OpenGraph pages, discoverable only through its sitemap index.
  It also lists an accessory ("Husa ...") and a 256GB variant that the matcher must reject.

Prices depend on a mutable *day*, so consecutive scans see drops, rises and stock changes.
"""

from __future__ import annotations

import html
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .models import (
    AlertRule,
    AppConfig,
    Competitor,
    NotifierConfig,
    Product,
    RepricingRule,
    SelectorConfig,
    Settings,
)

SHOPS = ("MegaShop", "PretMic", "BudgetBay")
RON_TO_EUR = Decimal("0.201")

# --------------------------------------------------------------------------- catalogue


@dataclass(frozen=True)
class DemoItem:
    """One listing in a demo shop; ``prices``/``in_stock`` hold one value per day (last repeats)."""

    id: str
    slug: str
    title: str
    brand: str
    prices: tuple[str, ...]
    in_stock: tuple[bool, ...] = (True,)
    gtin: str | None = None
    mpn: str | None = None
    markup: str = "microdata"  # BudgetBay only: "microdata" | "opengraph"

    def price(self, day: int) -> Decimal:
        return Decimal(self.prices[min(day, len(self.prices)) - 1])

    def available(self, day: int) -> bool:
        return self.in_stock[min(day, len(self.in_stock)) - 1]

    def was(self, day: int) -> Decimal | None:
        """The day-1 price, shown struck through once the price has dropped below it."""
        first = Decimal(self.prices[0])
        return first if self.price(day) < first else None


# Our own catalogue. Identifiers vary on purpose: some products carry a GTIN, some only an
# MPN, the vacuum neither (its PretMic page is pinned in the config instead).
PRODUCTS: tuple[Product, ...] = (
    Product(
        sku="SONY-XM5",
        name="Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        brand="Sony",
        gtin="4548736132610",
        mpn="WH1000XM5B",
        our_price=Decimal("349.00"),
        cost=Decimal("255.00"),
    ),
    Product(
        sku="IPHONE15-128",
        name="Apple iPhone 15 128GB Black",
        brand="Apple",
        gtin="0195949035500",
        mpn="MTP03",
        our_price=Decimal("899.00"),
        cost=Decimal("745.00"),
    ),
    Product(
        sku="SAMSUNG-Q80C-55",
        name='Samsung 55" QE55Q80C QLED 4K Smart TV',
        brand="Samsung",
        gtin="8806094812343",
        mpn="QE55Q80C",
        our_price=Decimal("999.00"),
        cost=Decimal("760.00"),
        max_price=Decimal("1099.00"),
    ),
    Product(
        sku="DELONGHI-ECAM22",
        name="De'Longhi Magnifica S ECAM22.110.B Coffee Machine",
        brand="De'Longhi",
        mpn="ECAM22.110.B",
        our_price=Decimal("379.00"),
        cost=Decimal("265.00"),
    ),
    Product(
        sku="DYSON-V15",
        name="Dyson V15 Detect Absolute Cordless Vacuum",
        brand="Dyson",
        our_price=Decimal("749.00"),
        cost=Decimal("560.00"),
    ),
)

_MEGASHOP = (
    DemoItem("1001", "sony-wh-1000xm5-wireless-headphones-black",
             "Sony WH-1000XM5 Wireless Noise Cancelling Headphones, Black", "Sony",
             ("345.00", "299.99", "309.99"), gtin="4548736132610", mpn="WH1000XM5B"),
    DemoItem("1002", "sony-wh-1000xm4-wireless-headphones",
             "Sony WH-1000XM4 Wireless Noise Cancelling Headphones", "Sony",
             ("249.99",), gtin="4548736000018", mpn="WH1000XM4B"),
    DemoItem("1003", "apple-iphone-15-128gb-black",
             "Apple iPhone 15 128GB Black", "Apple",
             ("899.00", "899.00", "869.00"), gtin="0195949035500", mpn="MTP03"),
    DemoItem("1004", "apple-iphone-15-256gb-black",
             "Apple iPhone 15 256GB Black", "Apple",
             ("999.00",), gtin="0195949035517", mpn="MTP63"),
    DemoItem("1005", "apple-iphone-15-silicone-case-magsafe-black",
             "Apple iPhone 15 Silicone Case with MagSafe - Black", "Apple",
             ("49.00",)),
    DemoItem("1006", "samsung-qe55q80c-qled-4k-smart-tv",
             'Samsung 55" QE55Q80C QLED 4K Smart TV', "Samsung",
             ("1019.00", "949.00"), gtin="8806094812343", mpn="QE55Q80C"),
    DemoItem("1007", "delonghi-magnifica-s-ecam-22-110-b",
             "De'Longhi Magnifica S ECAM 22.110.B Automatic Coffee Machine", "De'Longhi",
             ("389.00", "409.00"), gtin="8056400012343", mpn="ECAM22.110.B"),
    DemoItem("1008", "dyson-v12-detect-slim-cordless-vacuum",
             "Dyson V12 Detect Slim Cordless Vacuum", "Dyson",
             ("549.00",), gtin="5041123000015"),
)  # fmt: skip

_PRETMIC = (
    DemoItem("2001", "casti-sony-wh-1000xm5-negru",
             "Casti audio over-ear Sony WH-1000XM5, Wireless, Noise Cancelling, Negru", "Sony",
             ("1749.99", "1749.99", "1649.99")),
    DemoItem("2002", "casti-sony-wh-1000xm4-negru",
             "Casti audio over-ear Sony WH-1000XM4, Wireless, Negru", "Sony",
             ("1199.99",)),
    DemoItem("2003", "telefon-apple-iphone-15-128gb-black",
             "Telefon mobil Apple iPhone 15, 128GB, 5G, Black", "Apple",
             ("4499.99", "4499.99", "4399.99"), in_stock=(True, False, True)),
    DemoItem("2004", "husa-apple-iphone-15-silicon-negru",
             "Husa de protectie Apple iPhone 15, silicon, negru", "Apple",
             ("99.99",)),
    DemoItem("2005", "televizor-samsung-qled-qe55q80c",
             "Televizor Samsung QLED QE55Q80C, 138 cm, Smart, 4K Ultra HD", "Samsung",
             ("4999.00", "4999.00", "4799.00")),
    DemoItem("2006", "aspirator-dyson-v15-detect-absolute",
             "Aspirator vertical Dyson V15 Detect Absolute, 660W, 0.76 l", "Dyson",
             ("3699.99",)),
)  # fmt: skip

_BUDGETBAY = (
    DemoItem("3001", "sony-wh-1000xm5-headphones-black",
             "Sony WH-1000XM5 Headphones - Black", "Sony",
             ("354.90",)),
    DemoItem("3002", "apple-iphone-15-128gb-black",
             "Apple iPhone 15 128GB - Black", "Apple",
             ("889.00", "879.00"), gtin="0195949035500"),
    DemoItem("3003", "apple-iphone-15-256gb-black",
             "Apple iPhone 15 256GB - Black", "Apple",
             ("979.00",), gtin="0195949035517"),
    DemoItem("3004", "husa-silicon-apple-iphone-15-clear-case",
             "Husa Silicon Apple iPhone 15 Clear Case", "Generic",
             ("19.90",)),
    DemoItem("3005", "samsung-qe55q80c-55-inch-qled-tv",
             "Samsung QE55Q80C 55 inch QLED TV (2023)", "Samsung",
             ("989.00", "989.00", "1049.00"), markup="opengraph"),
    DemoItem("3006", "delonghi-magnifica-s-ecam22-110-b",
             "DeLonghi Magnifica S ECAM22.110.B Bean to Cup Coffee Machine", "De'Longhi",
             ("375.00", "375.00", "339.00"), markup="opengraph"),
    DemoItem("3007", "dyson-v15-detect-absolute-cordless-vacuum",
             "Dyson V15 Detect Absolute Cordless Vacuum Cleaner", "Dyson",
             ("759.00", "729.00")),
)  # fmt: skip

# --------------------------------------------------------------------------- rendering helpers


@dataclass(frozen=True)
class _Response:
    status: int
    body: str = ""
    content_type: str = "text/html; charset=utf-8"
    location: str | None = None


_NOT_FOUND = _Response(404, "<!doctype html><title>Not found</title><h1>404 Not found</h1>")
_e = html.escape


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _search(items: tuple[DemoItem, ...], query: str) -> list[DemoItem]:
    """A forgiving shop search: items matching at least half of the query words, best first."""
    tokens = _words(query)
    if not tokens:
        return []
    scored: list[tuple[float, DemoItem]] = []
    for item in items:
        text = " ".join(filter(None, [item.brand, item.title, item.mpn, item.gtin]))
        words, compact = set(_words(text)), re.sub(r"[^a-z0-9]", "", text.lower())
        hits = sum(1 for t in tokens if t in words or (len(t) >= 4 and t in compact))
        if hits / len(tokens) >= 0.5:
            scored.append((hits / len(tokens), item))
    return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].id))]


def _page(title: str, body: str, *, lang: str = "en", head: str = "") -> str:
    return (
        f'<!doctype html>\n<html lang="{lang}"><head><meta charset="utf-8">'
        f"<title>{_e(title)}</title>{head}</head>\n<body>\n{body}\n</body></html>\n"
    )


def _sitemap(root: str, entries: str) -> _Response:
    namespace = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    body = f'<?xml version="1.0" encoding="UTF-8"?>\n<{root} {namespace}>{entries}</{root}>'
    return _Response(200, body, "application/xml; charset=utf-8")


def _eur(amount: Decimal) -> str:
    return f"€{amount:,.2f}"


def _lei(amount: Decimal) -> str:
    """Romanian shop style: ``1.749<sup>99</sup> Lei``."""
    whole, cents = divmod(int(amount * 100), 100)
    return f"{whole:,}".replace(",", ".") + f"<sup>{cents:02d}</sup> Lei"


class _Shop:
    """Routing shared by the demo shops; subclasses render pages in their own style."""

    name: str
    item_prefix: str

    def __init__(self, items: tuple[DemoItem, ...]) -> None:
        self.items = items
        self._by_path = {self.item_path(item): item for item in items}

    def item_path(self, item: DemoItem) -> str:
        return f"{self.item_prefix}{item.slug}-{item.id}"

    def robots(self, base: str) -> str:
        return "User-agent: *\nAllow: /\n"

    def route(self, path: str, query: dict[str, list[str]], day: int, base: str) -> _Response:
        if path == "/robots.txt":
            return _Response(200, self.robots(base), "text/plain; charset=utf-8")
        if path == "/":
            return _Response(200, _page(self.name, f"<h1>Welcome to {_e(self.name)}</h1>"))
        if (item := self._by_path.get(path)) is not None:
            return _Response(200, self.product_page(item, day, base))
        return self.extra(path, query, day, base) or _NOT_FOUND

    def extra(
        self, path: str, query: dict[str, list[str]], day: int, base: str
    ) -> _Response | None:
        return None

    def product_page(self, item: DemoItem, day: int, base: str) -> str:
        raise NotImplementedError


class _MegaShop(_Shop):
    name = "MegaShop"
    item_prefix = "/p/"
    _NAV = (
        '<header class="site-header"><a href="/">MegaShop</a><nav class="menu">'
        '<a href="/c/audio">Audio</a> <a href="/c/phones">Phones</a> <a href="/c/tv">TV</a> '
        '<a href="/account/login">Sign in</a> <a href="/cart">Cart</a></nav>'
        '<form action="/search"><input name="q" aria-label="Search"></form></header>'
    )
    _FOOTER = (
        '<footer class="site-footer"><a href="/help/contact">Contact</a> '
        '<a href="/terms">Terms</a></footer>'
    )

    def _card(self, item: DemoItem, day: int) -> str:
        return (
            f'<article class="product-card"><a href="{self.item_path(item)}">'
            f'<img src="/img/{item.id}.jpg" alt=""><h3>{_e(item.title)}</h3></a>'
            f'<span class="price">{_eur(item.price(day))}</span></article>'
        )

    def extra(
        self, path: str, query: dict[str, list[str]], day: int, base: str
    ) -> _Response | None:
        if path != "/search":
            return None
        q = query.get("q", [""])[0].strip()
        if q.isdigit() and (hit := next((i for i in self.items if i.gtin == q), None)):
            return _Response(
                302, location=self.item_path(hit)
            )  # exact GTIN: straight to the product
        results = _search(self.items, q)
        cards = "".join(self._card(i, day) for i in results) or "<p>No products found.</p>"
        body = (
            f"{self._NAV}<main><h1>Results for “{_e(q)}”</h1>"
            f'<a href="/search?q={_e(q)}&amp;sort=price">Sort by price</a>'
            f'<div class="product-grid">{cards}</div></main>{self._FOOTER}'
        )
        return _Response(200, _page(f"Search: {q} | MegaShop", body))

    def product_page(self, item: DemoItem, day: int, base: str) -> str:
        url = base + self.item_path(item)
        available = item.available(day)
        product: dict[str, Any] = {
            "@type": "Product",
            "@id": f"{url}#product",
            "name": item.title,
            "sku": f"MS-{item.id}",
            "brand": {"@type": "Brand", "name": item.brand},
            "image": f"{base}/img/{item.id}.jpg",
            "offers": {
                "@type": "Offer",
                "url": url,
                "price": str(item.price(day)),
                "priceCurrency": "EUR",
                "availability": f"https://schema.org/{'InStock' if available else 'OutOfStock'}",
                "seller": {"@id": f"{base}/#org"},
            },
        }
        product |= {k: v for k, v in (("gtin13", item.gtin), ("mpn", item.mpn)) if v}
        graph = {
            "@context": "https://schema.org",
            "@graph": [
                {"@type": "Organization", "@id": f"{base}/#org", "name": "MegaShop", "url": base},
                {
                    "@type": "BreadcrumbList",
                    "itemListElement": [
                        {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{base}/"}
                    ],
                },
                product,
            ],
        }
        # "</" must not appear inside a script element.
        ld = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
        related = [i for i in self.items if i.brand == item.brand and i is not item][:2]
        body = (
            f'{self._NAV}<main class="product-page">'
            '<nav class="breadcrumbs"><a href="/">Home</a></nav>'
            f"<h1>{_e(item.title)}</h1>"
            f'<div class="product-price">{_eur(item.price(day))}</div>'
            f'<p class="stock">{"In stock" if available else "Out of stock"}</p></main>'
            f'<section class="related-products"><h2>Customers also viewed</h2>'
            f"{''.join(self._card(i, day) for i in related)}</section>{self._FOOTER}"
        )
        head = f'<script type="application/ld+json">{ld}</script>'
        return _page(f"{item.title} | MegaShop", body, head=head)


class _PretMic(_Shop):
    name = "PretMic"
    item_prefix = "/produs/"
    _NAV = (
        '<header class="antet"><a class="logo" href="/">PretMic</a>'
        '<form action="/cauta"><input name="q" aria-label="Caută"></form>'
        '<nav><a href="/categorie/telefoane">Telefoane</a> <a href="/categorie/electrocasnice">'
        'Electrocasnice</a> <a href="/cont">Contul meu</a> <a href="/cos">Coș</a></nav></header>'
    )

    def extra(
        self, path: str, query: dict[str, list[str]], day: int, base: str
    ) -> _Response | None:
        if path != "/cauta":
            return None
        q = query.get("q", [""])[0].strip()
        rows = "".join(
            f'<li class="produs-card"><a class="titlu" href="{self.item_path(i)}">{_e(i.title)}</a>'
            f'<div class="pret">{_lei(i.price(day))}</div></li>'
            for i in _search(self.items, q)
        )
        found = (
            f'<ul class="rezultate">{rows}</ul>' if rows else "<p>Nu am găsit niciun rezultat.</p>"
        )
        body = f"{self._NAV}<main><h1>Rezultate pentru „{_e(q)}”</h1>{found}</main>"
        return _Response(200, _page(f"Caută: {q} | PretMic.ro", body, lang="ro"))

    def product_page(self, item: DemoItem, day: int, base: str) -> str:
        was = item.was(day)
        old = f'<span class="pret-vechi"><s>{_lei(was)}</s></span> ' if was else ""
        body = (
            f'{self._NAV}<main><div class="fisa-produs">'
            f'<h1 class="titlu-produs">{_e(item.title)}</h1>'
            '<table class="specificatii">'
            f'<tr><td>Brand</td><td class="brand">{_e(item.brand)}</td></tr>'
            f"<tr><td>Cod produs</td><td>PM{item.id}</td></tr></table>"
            f'<div class="pret-produs">{old}'
            f'<span class="pret-curent">{_lei(item.price(day))}</span></div>'
            f'<div class="stoc">{"În stoc" if item.available(day) else "Stoc epuizat"}</div>'
            f'<div class="livrare">Livrare: 19,99 Lei</div>'
            f'<button type="button">Adaugă în coș</button></div></main>'
            f'<footer><a href="/contact">Contact</a></footer>'
        )
        return _page(f"{item.title} | PretMic.ro", body, lang="ro")


class _BudgetBay(_Shop):
    name = "BudgetBay"
    item_prefix = "/item/"
    _PAGES = ("/about-us", "/contact", "/category/phones")
    _TOPBAR = (
        '<div class="topbar"><a href="/">BudgetBay</a> <a href="/category/phones">Phones</a> '
        '<a href="/about-us">About</a></div>'
    )

    def robots(self, base: str) -> str:
        return f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n"

    def extra(
        self, path: str, query: dict[str, list[str]], day: int, base: str
    ) -> _Response | None:
        if path == "/sitemap.xml":
            children = "".join(
                f"<sitemap><loc>{base}/sitemaps/{n}.xml</loc></sitemap>"
                for n in ("pages", "products")
            )
            return _sitemap("sitemapindex", children)
        if path in ("/sitemaps/products.xml", "/sitemaps/pages.xml"):
            paths = [self.item_path(i) for i in self.items] if "products" in path else self._PAGES
            urls = "".join(
                f"<url><loc>{base}{p}</loc><lastmod>2026-10-01</lastmod></url>" for p in paths
            )
            return _sitemap("urlset", urls)
        if path in self._PAGES:
            return _Response(
                200, _page(f"{path.strip('/')} | BudgetBay", f"{self._TOPBAR}<h1>BudgetBay</h1>")
            )
        return None

    def product_page(self, item: DemoItem, day: int, base: str) -> str:
        price, available = item.price(day), item.available(day)
        if item.markup == "opengraph":
            meta = {
                "og:type": "product",
                "og:title": item.title,
                "og:url": base + self.item_path(item),
                "product:brand": item.brand,
                "product:price:amount": str(price),
                "product:price:currency": "EUR",
                "product:availability": "in stock" if available else "out of stock",
            }
            head = "".join(f'<meta property="{k}" content="{_e(v)}">' for k, v in meta.items())
            body = f'{self._TOPBAR}<h1>{_e(item.title)}</h1><p class="deal">{_eur(price)}</p>'
            return _page(f"{item.title} | BudgetBay", body, head=head)
        gtin = f'<meta itemprop="gtin13" content="{item.gtin}">' if item.gtin else ""
        stock = "InStock" if available else "OutOfStock"
        body = (
            f'{self._TOPBAR}<div itemscope itemtype="https://schema.org/Product">'
            f'<h1 itemprop="name">{_e(item.title)}</h1>'
            f'<meta itemprop="brand" content="{_e(item.brand)}">{gtin}'
            f'<div itemprop="offers" itemscope itemtype="https://schema.org/Offer">'
            f'<span itemprop="price" content="{price}">{_eur(price)}</span>'
            f'<meta itemprop="priceCurrency" content="EUR">'
            f'<link itemprop="availability" href="https://schema.org/{stock}"></div></div>'
        )
        return _page(f"{item.title} | BudgetBay", body)


# --------------------------------------------------------------------------- HTTP plumbing


class _ShopServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, shop: _Shop, market: DemoMarket) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.shop = shop
        self.market = market
        self.base_url = f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: _ShopServer
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # header and body are separate writes; avoid 40 ms stalls

    def do_GET(self) -> None:
        shop, market = self.server.shop, self.server.market
        parts = urlsplit(self.path)
        market._record(self.server.base_url + self.path)
        if market.is_offline(shop.name):
            resp = _Response(503, "Service Unavailable", "text/plain; charset=utf-8")
        else:
            resp = shop.route(parts.path, parse_qs(parts.query), market.day, self.server.base_url)
        body = resp.body.encode("utf-8")
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.content_type)
        self.send_header("Content-Length", str(len(body)))
        if resp.location:
            self.send_header("Location", resp.location)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the demo quiet: no access log lines."""


class DemoMarket:
    """Three fake competitor shops on localhost. Use as a context manager or start()/stop()."""

    def __init__(self) -> None:
        self.day = 1
        self._shops: dict[str, _Shop] = {
            "MegaShop": _MegaShop(_MEGASHOP),
            "PretMic": _PretMic(_PRETMIC),
            "BudgetBay": _BudgetBay(_BUDGETBAY),
        }
        self._servers: dict[str, _ShopServer] = {}
        self._threads: list[threading.Thread] = []
        self._offline: set[str] = set()
        self._requests: Counter[str] = Counter()
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> DemoMarket:
        if self._servers:
            return self
        for name, shop in self._shops.items():
            server = _ShopServer(shop, self)
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.05},  # quick shutdown()
                name=f"demo-{name}",
                daemon=True,
            )
            thread.start()
            self._servers[name] = server
            self._threads.append(thread)
        return self

    def stop(self) -> None:
        for server in self._servers.values():
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=5)
        self._servers.clear()
        self._threads.clear()

    def __enter__(self) -> DemoMarket:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- state ----------------------------------------------------------------

    def set_day(self, day: int) -> None:
        """Switch every shop to the prices and stock of ``day`` (1-based; the last day repeats)."""
        if day < 1:
            raise ValueError("day must be >= 1")
        self.day = day

    def set_offline(self, shop: str, offline: bool = True) -> None:
        """Make ``shop`` answer every request with 503 Service Unavailable (or recover)."""
        self._check(shop)
        with self._lock:
            (self._offline.add if offline else self._offline.discard)(shop)

    def is_offline(self, shop: str) -> bool:
        with self._lock:
            return shop in self._offline

    def _record(self, url: str) -> None:
        with self._lock:
            self._requests[url] += 1

    @property
    def requests(self) -> Counter[str]:
        """How often each absolute URL was requested so far."""
        with self._lock:
            return Counter(self._requests)

    def _check(self, shop: str) -> None:
        if shop not in self._shops:
            raise KeyError(f"unknown demo shop {shop!r}; choose from {', '.join(SHOPS)}")

    # -- addresses ------------------------------------------------------------

    def base_url(self, shop: str) -> str:
        self._check(shop)
        if shop not in self._servers:
            raise RuntimeError("the demo market is not running; call start() first")
        return self._servers[shop].base_url

    def product_url(self, shop: str, item_id: str) -> str:
        """Absolute URL of a demo listing, e.g. ``product_url("BudgetBay", "3003")``."""
        self._check(shop)
        target = self._shops[shop]
        item = next((i for i in target.items if i.id == item_id), None)
        if item is None:
            raise KeyError(f"no item {item_id!r} in {shop!r}")
        return self.base_url(shop) + target.item_path(item)

    # -- configuration --------------------------------------------------------

    def config(self, db_path: str | Path, *, requests_per_minute: float = 600.0) -> AppConfig:
        """An :class:`AppConfig` wired to the running shops.

        ``requests_per_minute`` is generous because the shops are local; for real sites
        keep it at 10-30. Alerts are also appended to ``alerts.jsonl`` next to ``db_path``.
        """
        db = Path(db_path)
        pinned_vacuum = self.product_url("PretMic", "2006")
        # Deep copies: callers may edit their config without touching the shared catalogue.
        products = [p.model_copy(deep=True) for p in PRODUCTS]
        for product in products:
            if product.sku == "DYSON-V15":
                product.competitor_urls = {"PretMic": [pinned_vacuum]}

        def competitor(**fields: Any) -> Competitor:
            return Competitor(requests_per_minute=requests_per_minute, **fields)

        return AppConfig(
            settings=Settings(
                database=str(db),
                timeout=10.0,
                max_retries=1,
                base_currency="EUR",
                currency_rates={"RON": RON_TO_EUR},
            ),
            competitors=[
                competitor(
                    name="MegaShop",
                    base_url=self.base_url("MegaShop"),
                    search_url="/search?q={query}",
                    product_url_pattern="/p/",
                    currency="EUR",
                ),
                competitor(
                    name="PretMic",
                    base_url=self.base_url("PretMic"),
                    search_url="/cauta?q={query}",
                    currency="RON",
                    headers={"Accept-Language": "ro-RO,ro;q=0.9"},
                    selectors=SelectorConfig(
                        price=".pret-produs .pret-curent",
                        original_price=".pret-produs .pret-vechi",
                        title="h1.titlu-produs",
                        brand=".specificatii .brand",
                        availability=".stoc",
                        shipping=".livrare",
                        search_result_link="ul.rezultate a.titlu",
                    ),
                ),
                competitor(name="BudgetBay", base_url=self.base_url("BudgetBay"), currency="EUR"),
            ],
            products=products,
            repricing=RepricingRule(
                strategy="beat_lowest",
                undercut_amount=Decimal("1.00"),
                min_margin_percent=Decimal("12"),
                price_ending=Decimal("0.99"),
            ),
            alerts=[
                AlertRule(type="undercut", threshold_percent=2.0),
                AlertRule(type="price_drop", threshold_percent=5.0),
                AlertRule(type="price_rise", threshold_percent=5.0),
                AlertRule(type="out_of_stock"),
                AlertRule(type="back_in_stock"),
                AlertRule(type="new_lowest"),
            ],
            notifiers=[
                NotifierConfig(type="console"),
                NotifierConfig(type="file", path=str(db.with_name("alerts.jsonl"))),
            ],
        )
