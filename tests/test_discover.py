"""Tests for candidate discovery: search pages, link heuristics and sitemaps (all offline)."""

from __future__ import annotations

import asyncio
import gzip

import pytest

from pricehunter import discover
from pricehunter.discover import (
    build_queries,
    discover_candidates,
    extract_links,
    parse_sitemap,
    rank_sitemap_urls,
    search_competitor,
    sitemap_urls,
)
from pricehunter.models import Competitor, FetchResult, Product, SelectorConfig

BASE = "https://www.shop.example"


class FakeFetcher:
    """Serves canned pages; unknown URLs answer 404. Never raises, like the real Fetcher."""

    def __init__(
        self,
        pages: dict[str, str] | None = None,
        blobs: dict[str, bytes] | None = None,
        redirects: dict[str, str] | None = None,
    ) -> None:
        self.pages = pages or {}
        self.blobs = blobs or {}
        self.redirects = redirects or {}
        self.calls: list[tuple[str, str]] = []

    async def get(
        self, url: str, competitor: Competitor | None = None, *, render: bool | None = None
    ) -> FetchResult:
        self.calls.append(("get", url))
        await asyncio.sleep(0)  # let concurrent callers interleave
        final = self.redirects.get(url, url)
        if final not in self.pages:
            return FetchResult(url=url, final_url=final, status=404, error="HTTP 404")
        return FetchResult(url=url, final_url=final, status=200, html=self.pages[final])

    async def get_bytes(
        self, url: str, competitor: Competitor | None = None
    ) -> tuple[FetchResult, bytes]:
        self.calls.append(("get_bytes", url))
        await asyncio.sleep(0)
        if url not in self.blobs:
            return FetchResult(url=url, final_url=url, status=404, error="HTTP 404"), b""
        return FetchResult(url=url, final_url=url, status=200), self.blobs[url]

    def fetched(self, url: str) -> int:
        return sum(1 for _, u in self.calls if u == url)


class ExplodingFetcher:
    """Violates the never-raise contract, to prove discovery still never raises."""

    async def get(self, url: str, competitor: Competitor | None = None, *, render: bool | None = None) -> FetchResult:
        raise RuntimeError("boom")

    async def get_bytes(self, url: str, competitor: Competitor | None = None) -> tuple[FetchResult, bytes]:
        raise RuntimeError("boom")


def competitor(**kw: object) -> Competitor:
    return Competitor.model_validate({"name": "shop", "base_url": BASE, **kw})


IPHONE = Product(
    sku="IP15",
    name="iPhone 15 128GB Black",
    brand="Apple",
    gtin="0195949036323",
    mpn="MTP03ZD/A",
    keywords=["iphone 15 negru", "Apple iPhone 15 128GB Black"],
)


def urlset(*urls: str) -> str:
    locs = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{locs}</urlset>'


def sitemapindex(*urls: str) -> str:
    locs = "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in urls)
    return f'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{locs}</sitemapindex>'


# --------------------------------------------------------------------------- build_queries


def test_build_queries_order() -> None:
    assert build_queries(IPHONE) == [
        "0195949036323",
        "Apple MTP03ZD/A",
        "Apple iPhone 15 128GB Black",
        "iphone 15 negru",  # the duplicate keyword (case-insensitive) is dropped
    ]


def test_build_queries_without_identifiers_or_brand() -> None:
    product = Product(sku="x", name="Lapte Zuzu 1L", keywords=["lapte zuzu", "  Lapte   Zuzu 1L "])
    assert build_queries(product) == ["Lapte Zuzu 1L", "lapte zuzu"]


def test_build_queries_does_not_duplicate_brand_and_skips_invalid_gtin() -> None:
    product = Product(sku="x", name="Sony WH-1000XM5", brand="SONY", gtin="1234", mpn="WH1000XM5B")
    assert build_queries(product) == ["SONY WH1000XM5B", "Sony WH-1000XM5"]


# --------------------------------------------------------------------------- extract_links

SEARCH_PAGE = """
<html><head><title>Cautare: iphone 15</title><base href="https://www.shop.example/"></head>
<body>
<header>
  <a href="/">Acasa</a>
  <a href="/cart">Cos (0)</a>
  <a href="/account/login?next=%2Fsearch">Intra in cont</a>
  <a href="/wishlist">Favorite</a>
  <a href="/compare">Compara</a>
  <a href="/categorie/telefoane">Telefoane</a>
  <a href="/c/laptopuri">Laptopuri</a>
  <a href="/collections/apple">Apple</a>
  <a href="javascript:void(0)">Meniu</a>
  <a href="mailto:office@shop.example">Email</a>
  <a href="tel:+40700000000">Suna-ne</a>
  <a href="#main">Sari la continut</a>
</header>
<main id="main">
  <div class="filters">
    <a href="/search?q=iphone+15&amp;sort=price_asc">Pret crescator</a>
    <a href="/telefoane?brand=apple&amp;filter=1">Apple</a>
  </div>
  <div class="grid">
    <div class="product-card">
      <a href="/telefon-apple-iphone-15-128gb-negru/pd/DXK123/#reviews"><img src="/img/ip15.jpg"></a>
      <a href="/telefon-apple-iphone-15-128gb-negru/pd/DXK123/?utm_source=search&amp;utm_medium=grid">
        Telefon Apple iPhone 15, 128GB, Negru</a>
    </div>
    <div class="product-card">
      <a href="https://shop.example/apple-iphone-15-256gb-p-1002.html">Apple iPhone 15 256GB</a>
    </div>
    <div class="product-card"><a href="products/husa-silicon-iphone-15">Husa iPhone 15</a></div>
    <div class="product-card"><a href="/collections/apple/products/iphone-15-plus">iPhone 15 Plus</a></div>
  </div>
  <a href="/oferte-black-friday">Vezi ofertele</a>
  <nav class="pagination"><a href="/search?q=iphone+15&amp;page=2">2</a></nav>
</main>
<footer>
  <a href="/blog/top-telefoane-2024">Blog</a>
  <a href="/contact">Contact</a>
  <a href="/termeni-si-conditii">Termeni si conditii</a>
  <a href="/politica-de-confidentialitate">Confidentialitate</a>
  <a href="/help/livrare">Livrare</a>
  <a href="https://facebook.com/shop.example">Facebook</a>
  <a href="https://cdn.shop.example/catalog.pdf">Catalog</a>
  <a href="/static/app.css">css</a>
  <a href="/media/banner.webp">banner</a>
  <a href="ftp://www.shop.example/file">ftp</a>
</footer>
</body></html>
"""
SEARCH_URL = f"{BASE}/search?q=iphone+15"


def test_extract_links_heuristic_filters_messy_page() -> None:
    assert extract_links(SEARCH_PAGE, SEARCH_URL, competitor()) == [
        f"{BASE}/telefon-apple-iphone-15-128gb-negru/pd/DXK123/",
        "https://shop.example/apple-iphone-15-256gb-p-1002.html",  # bare domain == www
        f"{BASE}/products/husa-silicon-iphone-15",
        f"{BASE}/collections/apple/products/iphone-15-plus",
        f"{BASE}/oferte-black-friday",  # not junk, but not product-looking: last
    ]


def test_extract_links_respects_product_url_pattern() -> None:
    comp = competitor(product_url_pattern=r"/pd/|-p-\d+\.html$")
    assert extract_links(SEARCH_PAGE, SEARCH_URL, comp) == [
        f"{BASE}/telefon-apple-iphone-15-128gb-negru/pd/DXK123/",
        "https://shop.example/apple-iphone-15-256gb-p-1002.html",
    ]


def test_extract_links_bare_domain_competitor_accepts_www_links() -> None:
    comp = competitor(base_url="https://shop.example")
    links = extract_links(SEARCH_PAGE, SEARCH_URL, comp)
    assert f"{BASE}/products/husa-silicon-iphone-15" in links


def test_extract_links_without_base_tag_resolves_against_page_url() -> None:
    html = '<a href="item-123456">x</a><a href="../dp/B0CHX1W1XY">y</a>'
    assert extract_links(html, f"{BASE}/ro/search?q=x", competitor()) == [
        f"{BASE}/ro/item-123456",
        f"{BASE}/dp/B0CHX1W1XY",
    ]


def test_extract_links_keeps_meaningful_query_and_drops_tracking() -> None:
    html = '<a href="/product.php?id=42&amp;gclid=abc">x</a><a href="/product.php?id=42">dup</a>'
    assert extract_links(html, SEARCH_URL, competitor()) == [f"{BASE}/product.php?id=42"]


SELECTOR_PAGE = """
<div class="results">
  <div class="card"><h2 class="card-title"><a href="/p/iphone-15-128gb">iPhone 15</a></h2>
    <a href="/cart/add/1">Adauga in cos</a></div>
  <div class="card"><a href="/p/iphone-15-256gb"><span class="card-title">iPhone 15 256GB</span></a></div>
  <div class="card"><div class="card-title" data-href="/p/iphone-15-pro">iPhone 15 Pro</div></div>
  <div class="card"><a class="card-title" href="https://partner.example/p/iphone-15">Partner</a></div>
  <div class="card"><a class="card-title" href="/p/iphone-15-128gb#top">dup</a></div>
  <div class="card"><span class="card-title">no link at all</span></div>
</div>
<aside><a href="/p/recommended-123">Recomandat</a></aside>
"""


def test_extract_links_with_selector_uses_only_selected_elements() -> None:
    comp = competitor(selectors=SelectorConfig(search_result_link=".card-title"))
    assert extract_links(SELECTOR_PAGE, SEARCH_URL, comp) == [
        f"{BASE}/p/iphone-15-128gb",  # <a> inside the element
        f"{BASE}/p/iphone-15-256gb",  # <a> around the element
        f"{BASE}/p/iphone-15-pro",  # data-href
    ]


def test_extract_links_with_selector_still_applies_pattern() -> None:
    comp = competitor(
        selectors=SelectorConfig(search_result_link=".card-title"), product_url_pattern=r"pro$"
    )
    assert extract_links(SELECTOR_PAGE, SEARCH_URL, comp) == [f"{BASE}/p/iphone-15-pro"]


def test_extract_links_bad_selector_falls_back_to_heuristic() -> None:
    comp = competitor(selectors=SelectorConfig(search_result_link="a[href"))
    links = extract_links(SELECTOR_PAGE, SEARCH_URL, comp)
    assert f"{BASE}/p/recommended-123" in links
    assert f"{BASE}/cart/add/1" not in links


@pytest.mark.parametrize("html", ["", "<html></html>", "not html at all", "<a>no href</a>"])
def test_extract_links_degenerate_input(html: str) -> None:
    assert extract_links(html, SEARCH_URL, competitor()) == []


# --------------------------------------------------------------------------- search_competitor

SEARCH_COMP = competitor(search_url=f"{BASE}/search?q={{query}}")


def search(query: str) -> str:
    return f"{BASE}/search?q={query}"


def results_page(*hrefs: str) -> str:
    return "".join(f'<div class="product-card"><a href="{h}">item</a></div>' for h in hrefs)


async def test_search_redirect_to_product_comes_first() -> None:
    product_page = f"{BASE}/telefon-apple-iphone-15-128gb-negru/pd/DXK123/"
    fetcher = FakeFetcher(
        pages={
            product_page: "<h1>iPhone 15</h1>" + results_page("/p/related-accessory-99"),
            search("Apple+MTP03ZD%2FA"): results_page("/p/iphone-15-a", "/p/iphone-15-b"),
            search("Apple+iPhone+15+128GB+Black"): results_page("/p/iphone-15-b", "/p/iphone-15-c"),
        },
        redirects={search("0195949036323"): product_page},
    )
    urls = await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=10)
    assert urls[0] == product_page
    assert urls[1:] == [
        f"{BASE}/p/related-accessory-99",
        f"{BASE}/p/iphone-15-a",
        f"{BASE}/p/iphone-15-b",
        f"{BASE}/p/iphone-15-c",
    ]
    # Every query was tried, URL-encoded with quote_plus.
    assert fetcher.fetched(search("iphone+15+negru")) == 1


async def test_search_redirect_from_later_query_still_ranks_first() -> None:
    product_page = f"{BASE}/p/iphone-15-128gb"
    fetcher = FakeFetcher(
        pages={
            search("0195949036323"): results_page("/p/iphone-15-x"),
            product_page: "<h1>iPhone</h1>",
        },
        redirects={search("Apple+MTP03ZD%2FA"): product_page},
    )
    urls = await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=10)
    assert urls[:2] == [product_page, f"{BASE}/p/iphone-15-x"]


async def test_search_redirect_to_non_product_is_ignored() -> None:
    fetcher = FakeFetcher(
        pages={f"{BASE}/account/login": "<form></form>", f"{BASE}/cautare/iphone": results_page("/p/a-1")},
        redirects={
            search("0195949036323"): f"{BASE}/account/login",
            search("Apple+MTP03ZD%2FA"): f"{BASE}/cautare/iphone",
        },
    )
    assert await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=10) == [f"{BASE}/p/a-1"]


async def test_search_stops_at_limit() -> None:
    fetcher = FakeFetcher(pages={search("0195949036323"): results_page("/p/a-1", "/p/b-2", "/p/c-3")})
    urls = await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=2)
    assert urls == [f"{BASE}/p/a-1", f"{BASE}/p/b-2"]
    assert len(fetcher.calls) == 1  # no further queries once the limit is reached


async def test_search_skips_failed_fetches() -> None:
    fetcher = FakeFetcher(pages={search("Apple+iPhone+15+128GB+Black"): results_page("/p/a-1")})
    assert await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=5) == [f"{BASE}/p/a-1"]
    assert len(fetcher.calls) == 4


async def test_search_relative_template_and_literal_braces() -> None:
    comp = competitor(search_url="/catalogsearch/result/?q={query}&f={}")
    product = Product(sku="x", name="Ceai verde 20 plicuri")
    url = f"{BASE}/catalogsearch/result/?q=Ceai+verde+20+plicuri&f={{}}"
    fetcher = FakeFetcher(pages={url: results_page("/ceai-verde-20-plicuri-p-77.html")})
    assert await search_competitor(fetcher, comp, product, limit=5) == [
        f"{BASE}/ceai-verde-20-plicuri-p-77.html"
    ]


async def test_search_without_search_url_or_limit_does_nothing() -> None:
    fetcher = FakeFetcher()
    assert await search_competitor(fetcher, competitor(), IPHONE, limit=5) == []
    assert await search_competitor(fetcher, SEARCH_COMP, IPHONE, limit=0) == []
    assert fetcher.calls == []


async def test_search_never_raises() -> None:
    assert await search_competitor(ExplodingFetcher(), SEARCH_COMP, IPHONE, limit=5) == []


# --------------------------------------------------------------------------- parse_sitemap


def test_parse_sitemap_urlset_with_namespaces_and_whitespace() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
            xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
      <url>
        <loc>
            https://www.shop.example/p/one
        </loc>
        <lastmod>2026-10-01</lastmod>
        <image:image><image:loc>https://www.shop.example/img/one.jpg</image:loc></image:image>
      </url>
      <url><loc>https://www.shop.example/p/two?a=1&amp;b=2</loc></url>
    </urlset>"""
    assert parse_sitemap(xml) == (
        ["https://www.shop.example/p/one", "https://www.shop.example/p/two?a=1&b=2"],
        [],
    )


def test_parse_sitemap_index() -> None:
    xml = sitemapindex(f"{BASE}/sitemap-products-1.xml.gz", f"{BASE}/sitemap-blog.xml")
    assert parse_sitemap(xml) == ([], [f"{BASE}/sitemap-products-1.xml.gz", f"{BASE}/sitemap-blog.xml"])


def test_parse_sitemap_without_namespace_with_cdata_and_bom() -> None:
    xml = "﻿<urlset><url><loc><![CDATA[https://www.shop.example/p/x]]></loc></url></urlset>"
    assert parse_sitemap(xml) == (["https://www.shop.example/p/x"], [])


def test_parse_sitemap_recovers_from_broken_markup() -> None:
    xml = (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://www.shop.example/p/a</loc></url>"
        "<url><loc>https://www.shop.example/p/b & c</loc></url>"
        "<url><loc>https://www.shop.example/p/d</loc>"
    )
    pages, children = parse_sitemap(xml)
    assert "https://www.shop.example/p/a" in pages
    assert "https://www.shop.example/p/d" in pages
    assert children == []


def test_parse_sitemap_plain_text() -> None:
    text = "https://www.shop.example/p/a\n\n  https://www.shop.example/p/b  \n# comment\n"
    assert parse_sitemap(text) == (["https://www.shop.example/p/a", "https://www.shop.example/p/b"], [])


@pytest.mark.parametrize("text", ["", "   ", "<html><body>Not found</body></html>", "<<<>>>"])
def test_parse_sitemap_garbage(text: str) -> None:
    assert parse_sitemap(text) == ([], [])


# --------------------------------------------------------------------------- sitemap_urls


async def test_sitemap_index_with_gzip_child() -> None:
    comp = competitor(sitemap_url=f"{BASE}/sitemap_index.xml")
    fetcher = FakeFetcher(
        pages={
            f"{BASE}/sitemap_index.xml": sitemapindex(
                f"{BASE}/sitemap-pages.xml", f"{BASE}/sitemap-products.xml.gz"
            ),
            f"{BASE}/sitemap-pages.xml": urlset(
                f"{BASE}/",
                f"{BASE}/contact",
                f"{BASE}/categorie/telefoane",
                f"{BASE}/oferte-black-friday",
            ),
        },
        blobs={
            f"{BASE}/sitemap-products.xml.gz": gzip.compress(
                urlset(
                    f"{BASE}/p/iphone-15-128gb",
                    "https://shop.example/p/iphone-15-256gb",
                    "https://elsewhere.example/p/iphone-15",
                ).encode()
            )
        },
    )
    urls = await sitemap_urls(fetcher, comp)
    # Product sitemaps are visited before the others; junk and off-site URLs are dropped.
    assert urls == [
        f"{BASE}/p/iphone-15-128gb",
        "https://shop.example/p/iphone-15-256gb",
        f"{BASE}/oferte-black-friday",
    ]
    assert ("get_bytes", f"{BASE}/sitemap-products.xml.gz") in fetcher.calls


async def test_sitemap_urls_filters_by_product_url_pattern() -> None:
    comp = competitor(sitemap_url=f"{BASE}/sitemap.xml", product_url_pattern=r"/p/")
    fetcher = FakeFetcher(
        pages={f"{BASE}/sitemap.xml": urlset(f"{BASE}/p/a-1", f"{BASE}/oferte-black-friday", f"{BASE}/p/b-2")}
    )
    assert await sitemap_urls(fetcher, comp) == [f"{BASE}/p/a-1", f"{BASE}/p/b-2"]


async def test_sitemap_urls_respects_max_urls_and_max_depth() -> None:
    comp = competitor(sitemap_url=f"{BASE}/index.xml")
    fetcher = FakeFetcher(
        pages={
            f"{BASE}/index.xml": sitemapindex(f"{BASE}/a.xml", f"{BASE}/nested-index.xml"),
            f"{BASE}/a.xml": urlset(*(f"{BASE}/p/item-{i}" for i in range(10))),
            f"{BASE}/nested-index.xml": sitemapindex(f"{BASE}/deep.xml"),
            f"{BASE}/deep.xml": urlset(f"{BASE}/p/deep-1"),
        }
    )
    assert len(await sitemap_urls(fetcher, comp, max_urls=3)) == 3

    shallow = await sitemap_urls(fetcher, comp, max_depth=1)
    assert f"{BASE}/p/deep-1" not in shallow
    assert len(shallow) == 10
    assert f"{BASE}/p/deep-1" in await sitemap_urls(fetcher, comp, max_depth=2)


async def test_sitemap_urls_survives_cycles_and_missing_children() -> None:
    comp = competitor(sitemap_url=f"{BASE}/index.xml")
    fetcher = FakeFetcher(
        pages={
            f"{BASE}/index.xml": sitemapindex(f"{BASE}/index.xml", f"{BASE}/missing.xml", f"{BASE}/a.xml"),
            f"{BASE}/a.xml": urlset(f"{BASE}/p/a-1"),
        }
    )
    assert await sitemap_urls(fetcher, comp) == [f"{BASE}/p/a-1"]
    assert fetcher.fetched(f"{BASE}/index.xml") == 1


async def test_gzip_bomb_and_corrupt_gzip_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "MAX_SITEMAP_BYTES", 10_000)
    bomb = urlset(*(f"{BASE}/p/item-{i}" for i in range(1000)))  # ~40 KB uncompressed
    comp = competitor(sitemap_url=f"{BASE}/index.xml")
    fetcher = FakeFetcher(
        pages={f"{BASE}/index.xml": sitemapindex(f"{BASE}/bomb.xml.gz", f"{BASE}/bad.xml.gz", f"{BASE}/ok.xml.gz")},
        blobs={
            f"{BASE}/bomb.xml.gz": gzip.compress(bomb.encode()),
            f"{BASE}/bad.xml.gz": b"\x1f\x8b\x08\x00garbage",
            # Served with Content-Encoding: gzip, so the client already decoded it.
            f"{BASE}/ok.xml.gz": urlset(f"{BASE}/p/ok-1").encode(),
        },
    )
    assert await sitemap_urls(fetcher, comp) == [f"{BASE}/p/ok-1"]


async def test_sitemap_from_robots_txt_then_default_location() -> None:
    robots = f"User-agent: *\nDisallow: /cart\nSitemap: {BASE}/sitemaps/products.xml\n"
    fetcher = FakeFetcher(
        pages={f"{BASE}/robots.txt": robots, f"{BASE}/sitemaps/products.xml": urlset(f"{BASE}/p/a-1")}
    )
    assert await sitemap_urls(fetcher, competitor()) == [f"{BASE}/p/a-1"]

    fallback = FakeFetcher(pages={f"{BASE}/sitemap.xml": urlset(f"{BASE}/p/b-2")})
    assert await sitemap_urls(fallback, competitor()) == [f"{BASE}/p/b-2"]


# --------------------------------------------------------------------------- rank_sitemap_urls

SITEMAP_URLS = [
    f"{BASE}/samsung-galaxy-s24-p-1005.html",
    f"{BASE}/apple-iphone-15-256gb-black-p-1002.html",
    f"{BASE}/husa-silicon-iphone-15-p-1004.html",
    f"{BASE}/apple-iphone-15-pro-128gb-p-1003.html",
    f"{BASE}/telefoane/iphone-15-128gb-negru-p-1006.html",
    f"{BASE}/apple-iphone-14-128gb-black-p-1007.html",
    f"{BASE}/apple-iphone-15-128gb-black-p-1001.html",
    f"{BASE}/blog/review-iphone-15",
]


def test_rank_sitemap_urls_by_slug() -> None:
    product = IPHONE.model_copy(update={"gtin": None, "mpn": None})
    ranked = rank_sitemap_urls(SITEMAP_URLS, product, limit=10)
    assert ranked[:2] == [
        f"{BASE}/apple-iphone-15-128gb-black-p-1001.html",
        f"{BASE}/telefoane/iphone-15-128gb-negru-p-1006.html",
    ]
    for wrong in ("256gb", "husa", "iphone-14", "galaxy"):
        assert not any(wrong in url for url in ranked), wrong


def test_rank_sitemap_urls_limit_and_min_score() -> None:
    product = IPHONE.model_copy(update={"gtin": None, "mpn": None})
    assert rank_sitemap_urls(SITEMAP_URLS, product, limit=1) == [f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"]
    assert rank_sitemap_urls(SITEMAP_URLS, product, limit=10, min_score=0.99) == [
        f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"
    ]
    assert rank_sitemap_urls(SITEMAP_URLS, product, limit=0) == []
    assert rank_sitemap_urls([], product, limit=5) == []


def test_rank_sitemap_urls_identifiers_win() -> None:
    urls = [*SITEMAP_URLS, f"{BASE}/p/mtp03zd-a", f"{BASE}/produs/0195949036323"]
    ranked = rank_sitemap_urls(urls, IPHONE, limit=3)
    assert ranked == [
        f"{BASE}/produs/0195949036323",  # our GTIN in the URL
        f"{BASE}/p/mtp03zd-a",  # our MPN as the slug
        f"{BASE}/apple-iphone-15-128gb-black-p-1001.html",
    ]


def test_rank_sitemap_urls_model_token_boost() -> None:
    sony = Product(sku="s", name="Sony WH-1000XM5 Headphones Black", brand="Sony")
    urls = [
        f"{BASE}/sony-wh-1000xm4-headphones-black-123456",
        f"{BASE}/sony-wh-1000xm5-headphones-black-123457",
        f"{BASE}/husa-sony-wh-1000xm5-123458",
    ]
    assert rank_sitemap_urls(urls, sony, limit=5) == [f"{BASE}/sony-wh-1000xm5-headphones-black-123457"]


# --------------------------------------------------------------------------- discover_candidates


async def test_discover_uses_site_search_when_configured() -> None:
    fetcher = FakeFetcher(pages={search("0195949036323"): results_page("/p/a-1")})
    assert await discover_candidates(fetcher, SEARCH_COMP, IPHONE, limit=5) == [f"{BASE}/p/a-1"]
    assert all("sitemap" not in url and "robots" not in url for _, url in fetcher.calls)


async def test_discover_falls_back_to_configured_sitemap_when_search_finds_nothing() -> None:
    comp = competitor(search_url=f"{BASE}/search?q={{query}}", sitemap_url=f"{BASE}/sitemap.xml")
    fetcher = FakeFetcher(pages={f"{BASE}/sitemap.xml": urlset(*SITEMAP_URLS)})
    urls = await discover_candidates(fetcher, comp, IPHONE, limit=1)
    assert urls == [f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"]


def _sitemap_fetcher() -> FakeFetcher:
    return FakeFetcher(
        pages={
            f"{BASE}/sitemap.xml": sitemapindex(f"{BASE}/products.xml"),
            f"{BASE}/products.xml": urlset(*SITEMAP_URLS, f"{BASE}/sony-wh-1000xm5-headphones-black-123457"),
        }
    )


async def test_discover_sitemap_is_fetched_once_per_competitor() -> None:
    comp = competitor(sitemap_url=f"{BASE}/sitemap.xml")
    sony = Product(sku="s", name="Sony WH-1000XM5 Headphones Black", brand="Sony")
    fetcher = _sitemap_fetcher()
    cache: dict[str, list[str]] = {}

    iphone_urls = await discover_candidates(fetcher, comp, IPHONE, limit=2, sitemap_cache=cache)
    sony_urls = await discover_candidates(fetcher, comp, sony, limit=2, sitemap_cache=cache)

    assert iphone_urls[0] == f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"
    assert sony_urls == [f"{BASE}/sony-wh-1000xm5-headphones-black-123457"]
    assert fetcher.fetched(f"{BASE}/products.xml") == 1
    assert len(cache["shop"]) == len(SITEMAP_URLS)  # /blog/ was dropped, the Sony URL added


async def test_discover_concurrent_products_share_one_sitemap_crawl() -> None:
    comp = competitor(sitemap_url=f"{BASE}/sitemap.xml")
    fetcher = _sitemap_fetcher()
    cache: dict[str, list[str]] = {}
    results = await asyncio.gather(
        *(discover_candidates(fetcher, comp, IPHONE, limit=1, sitemap_cache=cache) for _ in range(5))
    )
    assert all(r == [f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"] for r in results)
    assert fetcher.fetched(f"{BASE}/sitemap.xml") == 1
    assert fetcher.fetched(f"{BASE}/products.xml") == 1


async def test_discover_without_cache_still_works() -> None:
    comp = competitor(sitemap_url=f"{BASE}/sitemap.xml")
    urls = await discover_candidates(_sitemap_fetcher(), comp, IPHONE, limit=1)
    assert urls == [f"{BASE}/apple-iphone-15-128gb-black-p-1001.html"]


async def test_discover_without_search_or_sitemap_returns_empty() -> None:
    fetcher = FakeFetcher()
    cache: dict[str, list[str]] = {}
    assert await discover_candidates(fetcher, competitor(), IPHONE, limit=5, sitemap_cache=cache) == []
    assert cache == {"shop": []}  # remembered, so the next product does not retry


@pytest.mark.parametrize(
    "comp",
    [SEARCH_COMP, competitor(sitemap_url=f"{BASE}/sitemap.xml"), competitor()],
    ids=["search", "sitemap", "robots"],
)
async def test_discover_never_raises(comp: Competitor) -> None:
    assert await discover_candidates(ExplodingFetcher(), comp, IPHONE, limit=5, sitemap_cache={}) == []
