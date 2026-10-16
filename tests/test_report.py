from __future__ import annotations

import base64
import csv
import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag

from pricehunter.models import (
    Alert,
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    Observation,
    PriceSuggestion,
    Product,
    ScanResult,
    ScanStats,
)
from pricehunter.report import BASE_COLUMNS, build_rows, render_html, write_csv, write_json

D = Decimal
A = Availability
T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
EVIL_NAME = '<script>alert("x")</script> & Co'
EVIL_COMPETITOR = '"><img src=x onerror=alert(1)>'
ACME_URL = "https://acme.example/w1?a=1&b=2"


def offer(
    competitor: str, price: str, availability: Availability = A.IN_STOCK, url: str | None = None
) -> CompetitorPrice:
    return CompetitorPrice(
        competitor=competitor,
        url=url or f"https://{competitor.lower()}.example/p",
        price=D(price),
        original_currency="EUR",
        availability=availability,
        match_score=0.95,
    )


PRODUCTS = [
    Product(sku="W1", name="Widget Pro", our_price=D("100.00")),
    Product(sku="W2", name=EVIL_NAME, our_price=D("50.00")),
    Product(sku="W3", name="Gadget", our_price=D("120.00")),
    Product(sku="W4", name="Gizmo", our_price=D("10.00")),
]
SNAPSHOTS = [
    MarketSnapshot(
        sku="W1",
        currency="EUR",
        our_price=D("100.00"),
        offers=[
            offer("Acme", "89.99", url=ACME_URL),
            offer("Bolt", "95.00", A.OUT_OF_STOCK, url="javascript:alert(1)"),
            offer("Cart", "105.00"),
        ],
        lowest=D("89.99"),
        lowest_competitor="Acme",
        median=D("97.50"),
        highest=D("105.00"),
        price_index=102.6,
        our_rank=2,
        competitors_cheaper=1,
        in_stock_offers=2,
    ),
    MarketSnapshot(
        sku="W2",
        currency="EUR",
        our_price=D("50.00"),
        offers=[offer("Acme", "60.00")],
        lowest=D("60.00"),
        lowest_competitor="Acme",
        median=D("60.00"),
        highest=D("60.00"),
        price_index=83.3,
        our_rank=1,
        in_stock_offers=1,
    ),
    MarketSnapshot(
        sku="W3",
        currency="EUR",
        our_price=D("120.00"),
        offers=[offer("Cart", "100.00")],
        lowest=D("100.00"),
        lowest_competitor="Cart",
        median=D("100.00"),
        highest=D("100.00"),
        price_index=120.0,
        our_rank=2,
        competitors_cheaper=1,
        in_stock_offers=1,
    ),
    MarketSnapshot(
        sku="W4",
        currency="EUR",
        our_price=D("10.00"),
        offers=[offer(EVIL_COMPETITOR, "12.00", A.OUT_OF_STOCK, url="https://evil.example/")],
        our_rank=1,
    ),
]
SUGGESTIONS = [
    PriceSuggestion(
        sku="W1",
        current_price=D("100.00"),
        suggested_price=D("89.98"),
        floor=None,
        ceiling=None,
        strategy="beat_lowest",
        reason="beat lowest (Acme @ 89.99) by 0.01 <b>",
        change_percent=-10.02,
    ),
    PriceSuggestion(
        sku="W2",
        current_price=D("50.00"),
        suggested_price=D("59.99"),
        floor=None,
        ceiling=None,
        strategy="beat_lowest",
        reason="beat lowest (Acme @ 60.00) by 0.01",
        change_percent=19.98,
    ),
]
ALERTS = [
    Alert(
        type="undercut",
        sku="W1",
        competitor="Acme",
        url=ACME_URL,
        message="Acme undercuts Widget Pro (W1): 89.99 EUR vs our 100.00 EUR (-10.0%)",
        created_at=T0 + timedelta(days=2),
    ),
    Alert(
        type="price_drop",
        sku="W2",
        competitor="Acme",
        url="javascript:alert(2)",
        message=f"Acme dropped the price of {EVIL_NAME}",
        created_at=T0 + timedelta(days=3),
    ),
]


def hist(
    sku: str,
    competitor: str,
    price: str,
    when: datetime,
    run_id: int | None,
    availability: Availability = A.IN_STOCK,
    currency: str = "EUR",
) -> Observation:
    return Observation(
        sku=sku,
        competitor=competitor,
        url=f"https://{competitor.lower()}.example/p",
        price=D(price),
        currency=currency,
        availability=availability,
        observed_at=when,
        run_id=run_id,
    )


HISTORY = {
    "W1": [
        hist("W1", "Acme", "95.00", T0, 1),
        hist("W1", "Bolt", "97.00", T0 + timedelta(minutes=2), 1),
        hist("W1", "Acme", "92.00", T0 + timedelta(days=1), 2),
        hist("W1", "Bolt", "80.00", T0 + timedelta(days=1), 2, A.OUT_OF_STOCK),  # not buyable
        hist("W1", "Acme", "89.99", T0 + timedelta(days=2), 3),
        hist("W1", "Cart", "50.00", T0 + timedelta(days=2), 3, currency="USD"),  # not comparable
    ],
    "W2": [  # no run ids: one point per day
        hist("W2", "Acme", "61.00", T0, None),
        hist("W2", "Acme", "60.00", T0 + timedelta(hours=3), None),
        hist("W2", "Acme", "59.00", T0 + timedelta(days=1), None),
    ],
    "W3": [],
}


def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def report(**kw: object) -> str:
    args: dict[str, object] = {
        "snapshots": SNAPSHOTS,
        "suggestions": SUGGESTIONS,
        "alerts": ALERTS,
        "products": PRODUCTS,
        "generated_at": T0,
    }
    return render_html(**(args | kw))  # type: ignore[arg-type]


def body_rows(soup: BeautifulSoup) -> dict[str, Tag]:
    """Table rows keyed by sku."""
    rows: dict[str, Tag] = {}
    for tr in soup.select("#products tbody tr"):
        sub = tr.select_one("td.product .sub")
        assert sub is not None
        rows[sub.get_text()] = tr
    return rows


def header_names(soup: BeautifulSoup) -> list[str]:
    return [th.get_text() for th in soup.select("#products thead th")]


# --------------------------------------------------------------------------- rows / csv / json


def test_build_rows_one_row_per_product_with_competitor_columns() -> None:
    rows = build_rows(SNAPSHOTS, SUGGESTIONS, PRODUCTS)
    assert [r["sku"] for r in rows] == ["W1", "W2", "W3", "W4"]
    competitors = [EVIL_COMPETITOR, "Acme", "Bolt", "Cart"]
    assert all(list(r) == [*BASE_COLUMNS, *competitors] for r in rows)
    assert rows[0] == {
        "sku": "W1",
        "name": "Widget Pro",
        "our_price": D("100.00"),
        "currency": "EUR",
        "lowest": D("89.99"),
        "lowest_competitor": "Acme",
        "median": D("97.50"),
        "highest": D("105.00"),
        "price_index": 102.6,
        "our_rank": 2,
        "competitors_cheaper": 1,
        "in_stock_offers": 2,
        "suggested_price": D("89.98"),
        "change_percent": -10.02,
        "reason": "beat lowest (Acme @ 89.99) by 0.01 <b>",
        EVIL_COMPETITOR: "",
        "Acme": D("89.99"),
        "Bolt": D("95.00"),
        "Cart": D("105.00"),
    }
    w3 = rows[2]
    assert (w3["suggested_price"], w3["change_percent"], w3["reason"]) == (None, None, None)
    assert (w3["Acme"], w3["Bolt"], w3["Cart"]) == ("", "", D("100.00"))


def test_build_rows_without_products_or_competitors() -> None:
    rows = build_rows(SNAPSHOTS[:1], [])
    assert rows[0]["name"] is None and rows[0]["suggested_price"] is None
    bare = MarketSnapshot(sku="X", currency="EUR", our_price=None)
    assert list(build_rows([bare], [])[0]) == list(BASE_COLUMNS)
    assert build_rows([], []) == []


def test_build_rows_competitor_named_like_a_base_column() -> None:
    snap = MarketSnapshot(sku="X", currency="EUR", our_price=None, offers=[offer("sku", "5")])
    [row] = build_rows([snap], [])
    assert (row["sku"], row["competitor:sku"]) == ("X", D("5"))


def test_write_csv_excel_friendly(tmp_path: Path) -> None:
    path = tmp_path / "out" / "report.csv"
    rows = build_rows(SNAPSHOTS, SUGGESTIONS, PRODUCTS)
    write_csv(path, rows)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        parsed = list(csv.DictReader(fh))
    assert list(parsed[0]) == [*BASE_COLUMNS, EVIL_COMPETITOR, "Acme", "Bolt", "Cart"]
    first, last = parsed[0], parsed[3]
    assert (first["our_price"], first["Acme"], first["price_index"]) == ("100.00", "89.99", "102.6")
    assert (first["change_percent"], first[EVIL_COMPETITOR]) == ("-10.02", "")
    assert (last["lowest"], last["price_index"], last["suggested_price"]) == ("", "", "")
    assert parsed[1]["name"] == EVIL_NAME


def test_write_csv_header_is_union_of_keys_in_first_seen_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, [{"b": 1, "a": "x"}, {"a": "y", "c": D("2.50")}, {}])
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert lines == ["b,a,c", "1,x,", ",y,2.50", ",,"]


def test_write_csv_neutralises_formulas(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, [{"name": '=HYPERLINK("http://x","y")', "note": "@SUM(A1)", "n": -5}])
    with path.open(encoding="utf-8-sig", newline="") as fh:
        [row] = list(csv.DictReader(fh))
    assert row == {"name": '\'=HYPERLINK("http://x","y")', "note": "'@SUM(A1)", "n": "-5"}


def test_write_csv_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    write_csv(path, [])
    assert path.read_bytes() == b""  # utf-8-sig writes the BOM with the first data


def test_write_json_round_trip(tmp_path: Path) -> None:
    result = ScanResult(
        stats=ScanStats(run_id=7, started_at=T0, finished_at=T0 + timedelta(minutes=3), products=4),
        observations=HISTORY["W1"],
        snapshots=SNAPSHOTS,
        suggestions=SUGGESTIONS,
        alerts=ALERTS,
        errors=["Bolt: HTTP 503"],
    )
    path = tmp_path / "nested" / "scan.json"
    write_json(path, result)
    text = path.read_text(encoding="utf-8")
    assert '"price": "89.99"' in text  # Decimals keep their exact digits
    assert ScanResult.model_validate_json(text) == result


# --------------------------------------------------------------------------- html


def test_html_escapes_untrusted_text() -> None:
    html = report(title='PH <img src=x onerror=alert(1)> "report"')
    soup = soup_of(html)
    assert len(soup.find_all("script")) == 1  # only the report's own sort script
    assert soup.find("img") is None and soup.find("svg", attrs={"onload": True}) is None
    assert not [t for t in soup.find_all(True) if any(a.startswith("on") for a in t.attrs)]
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; Co" in html
    assert soup.title is not None
    assert soup.title.get_text() == 'PH <img src=x onerror=alert(1)> "report"'
    w2 = body_rows(soup)["W2"].select_one(".name")
    assert w2 is not None and w2.get_text() == EVIL_NAME
    assert EVIL_COMPETITOR in header_names(soup)
    w1 = body_rows(soup)["W1"]
    suggested = w1.find("td", title=True)
    assert suggested is not None and suggested["title"] == "beat lowest (Acme @ 89.99) by 0.01 <b>"
    assert "by 0.01 &lt;b&gt;" in html


def test_html_links_only_http_urls() -> None:
    soup = soup_of(report())
    hrefs = [a["href"] for a in soup.find_all("a")]
    assert hrefs and all(str(h).startswith("https://") for h in hrefs)
    assert "javascript:" not in " ".join(map(str, hrefs))
    assert ACME_URL in hrefs  # "&" survives escaping intact
    assert all(a["rel"] == ["noopener", "noreferrer", "nofollow"] for a in soup.find_all("a"))
    bolt_cell = body_rows(soup)["W1"].find_all("td")[9]
    assert bolt_cell.find("a") is None and bolt_cell.find("s") is not None
    alert_items = soup.select("ul.alerts li")
    assert [li.find("a") is not None for li in alert_items] == [False, True]  # newest first


def test_html_table_cells() -> None:
    soup = soup_of(report())
    assert header_names(soup) == [
        "Product",
        "Our price",
        "Lowest",
        "Median",
        "Index",
        "Rank",
        "Suggested",
        EVIL_COMPETITOR,
        "Acme",
        "Bolt",
        "Cart",
    ]
    rows = body_rows(soup)
    w1 = rows["W1"].find_all("td")
    assert w1[1].get_text(" ", strip=True) == "100.00 EUR"
    assert w1[2].get_text(" ", strip=True) == "89.99 Acme"
    assert w1[5].get_text() == "2 / 3"
    assert w1[6].get_text(" ", strip=True) == "89.98 -10.0%"
    assert w1[6].select_one(".sub.down") is not None
    acme, bolt, cart = w1[8], w1[9], w1[10]
    assert "cheapest" in acme["class"] and acme["data-sort"] == "89.99"
    assert "oos" in bolt["class"] and "cheapest" not in bolt["class"]
    assert bolt.find("s").get_text() == "95.00"  # type: ignore[union-attr]
    assert "cheapest" not in cart["class"]
    assert w1[7].get_text() == "—" and w1[7]["data-sort"] == ""
    w4 = rows["W4"].find_all("td")
    assert [w4[i].get_text() for i in (2, 4, 6)] == ["—", "—", "—"]
    assert "oos" in w4[7]["class"]
    big = MarketSnapshot(sku="B", currency="RON", our_price=D("12345.6"), offers=[])
    cell = body_rows(soup_of(report(snapshots=[big])))["B"].find_all("td")[1]
    assert cell.get_text(" ", strip=True) == "12,345.60 RON"


@pytest.mark.parametrize(("sku", "tone"), [("W1", "warn"), ("W2", "good"), ("W3", "bad")])
def test_html_index_colour_coding(sku: str, tone: str) -> None:
    index_cell = body_rows(soup_of(report()))[sku].find_all("td")[4]
    assert index_cell.select_one(f".idx.idx-{tone}") is not None


def test_html_index_boundaries() -> None:
    def tone(index: float) -> str:
        snap = MarketSnapshot(sku="Z", currency="EUR", our_price=D("1"), price_index=index)
        pill = soup_of(report(snapshots=[snap])).select_one("#products .idx")
        assert pill is not None
        return str(pill["class"][1])

    assert [tone(i) for i in (100.0, 100.1, 105.0, 105.1)] == [
        "idx-good",
        "idx-warn",
        "idx-warn",
        "idx-bad",
    ]


def test_html_kpis() -> None:
    soup = soup_of(report())
    tiles = {
        t.select_one(".tile-label").get_text(): t.select_one(".tile-value").get_text()  # type: ignore[union-attr]
        for t in soup.select(".tile")
    }
    assert tiles == {
        "Products tracked": "4",
        "We're cheapest": "1",
        "Undercut": "2",
        "Avg price index": "102.0",
    }
    assert "of 3 with in-stock competition" in soup.get_text()


def test_html_sparklines_from_history() -> None:
    soup = soup_of(report(history=HISTORY))
    assert header_names(soup)[-1] == "Trend"
    rows = body_rows(soup)
    w1 = rows["W1"].select_one("svg.spark")
    assert w1 is not None
    assert w1["aria-label"] == "Lowest competitor price over 3 scans: 95.00 to 89.99 EUR"
    points = str(w1.find("polyline")["points"]).split()  # type: ignore[index]
    assert len(points) == 3
    ys = [float(p.split(",")[1]) for p in points]
    assert ys[0] < ys[1] < ys[2]  # falling price is drawn going down (larger y)
    w2 = rows["W2"].select_one("svg.spark")
    assert w2 is not None
    assert w2["aria-label"] == "Lowest competitor price over 2 scans: 60.00 to 59.00 EUR"
    assert rows["W3"].select_one("svg") is None and rows["W3"].find_all("td")[-1].get_text() == "—"
    assert rows["W4"].select_one("svg") is None


def test_html_single_point_sparkline() -> None:
    history = {"W1": [hist("W1", "Acme", "90.00", T0, 1)]}
    soup = soup_of(report(snapshots=SNAPSHOTS[:1], history=history))
    spark = soup.select_one("svg.spark")
    assert spark is not None and spark.find("polyline") is None and spark.find("circle") is not None


def test_html_without_history_has_no_trend_column() -> None:
    soup = soup_of(report())
    assert "Trend" not in header_names(soup) and soup.select_one("svg.spark") is None


def test_html_alerts_section() -> None:
    soup = soup_of(report())
    items = soup.select("ul.alerts li")
    assert [li.select_one(".badge").get_text() for li in items] == ["Price drop", "Undercut"]  # type: ignore[union-attr]
    time = items[1].find("time")
    assert time is not None and time["datetime"] == (T0 + timedelta(days=2)).isoformat()
    many = [
        Alert(type="price_rise", sku="W1", message=f"m{i}", created_at=T0 + timedelta(minutes=i))
        for i in range(60)
    ]
    soup = soup_of(report(alerts=many))
    messages = [li.select_one(".msg").get_text() for li in soup.select("ul.alerts li")]  # type: ignore[union-attr]
    assert messages == [f"m{i}" for i in range(59, 9, -1)]
    assert "Showing the 50 most recent of 60 alerts." in soup.get_text()


def test_html_empty_report() -> None:
    html = render_html([], [], [], [])
    soup = soup_of(html)
    assert "No products scanned yet." in soup.get_text() and "No alerts." in soup.get_text()
    assert soup.title is not None and soup.title.get_text() == "PriceHunter report"


def test_html_is_self_contained_and_responsive() -> None:
    html = report(history=HISTORY)
    soup = soup_of(html)
    assert soup.find("link") is None and soup.find_all(src=True) == []
    assert "@import" not in html and "url(" not in html
    viewport = soup.find("meta", attrs={"name": "viewport"})
    assert viewport is not None and "width=device-width" in str(viewport["content"])
    wrap = soup.select_one(".table-wrap > table#products")
    assert wrap is not None
    style = soup.style.get_text()  # type: ignore[union-attr]
    assert "prefers-color-scheme:dark" in style and "overflow-x:auto" in style and "--bg" in style
    assert "Generated 2026-10-01 08:00 UTC" in soup.get_text()


def test_html_csp_allows_only_the_report_assets() -> None:
    soup = soup_of(report())

    def digest(text: str) -> str:
        return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()

    meta = soup.find("meta", attrs={"http-equiv": "Content-Security-Policy"})
    assert meta is not None
    csp = str(meta["content"])
    assert "default-src 'none'" in csp
    assert f"'sha256-{digest(soup.script.string)}'" in csp  # type: ignore[union-attr, arg-type]
    assert f"'sha256-{digest(soup.style.string)}'" in csp  # type: ignore[union-attr, arg-type]


def test_dashboard_mode_adds_controls_and_allows_only_its_own_script() -> None:
    def digest(text: str) -> str:
        return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()

    plain = soup_of(report())
    assert plain.find(id="run-scan") is None and len(plain.find_all("script")) == 1
    plain_csp = str(plain.find("meta", attrs={"http-equiv": "Content-Security-Policy"})["content"])  # type: ignore[index]
    assert "connect-src" not in plain_csp  # a saved report never talks to a server

    soup = soup_of(report(dashboard=True))
    assert soup.find(id="run-scan") is not None and soup.find(id="refresh") is not None
    status = soup.find(id="scan-status")
    assert status is not None and status["role"] == "status"
    csp = str(soup.find("meta", attrs={"http-equiv": "Content-Security-Policy"})["content"])  # type: ignore[index]
    scripts = soup.find_all("script")
    assert len(scripts) == 2
    for script in scripts:
        assert f"'sha256-{digest(script.string)}'" in csp  # type: ignore[arg-type]
    assert "connect-src 'self'" in csp and "default-src 'none'" in csp and "form-action 'none'" in csp
    assert 'fetch("api/scan"' in scripts[1].string  # relative: works behind a path prefix  # type: ignore[operator]


async def test_html_in_browser_sorts_and_fits_phone_width() -> None:
    """The CSP-hashed script and styles really run, sorting works, and 375px has no page scroll."""
    async_api = pytest.importorskip("playwright.async_api")
    html = report(history=HISTORY)
    async with async_api.async_playwright() as pw:
        try:
            browser = await pw.chromium.launch()
        except async_api.Error as exc:  # browser not installed in this environment
            pytest.skip(f"chromium unavailable: {exc}")
        page = await browser.new_page(viewport={"width": 375, "height": 800})
        csp_errors: list[str] = []
        page.on(
            "console",
            lambda m: csp_errors.append(m.text) if "Content Security Policy" in m.text else None,
        )
        await page.set_content(html)

        async def skus() -> list[str]:
            return list(await page.locator("#products tbody td.product .sub").all_inner_texts())

        assert await skus() == ["W1", "W2", "W3", "W4"]
        header = page.locator("#products thead th", has_text="Our price")
        await header.locator("button").click()
        assert await header.get_attribute("aria-sort") == "ascending"
        assert await skus() == ["W4", "W2", "W1", "W3"]
        await header.locator("button").click()
        assert await header.get_attribute("aria-sort") == "descending"
        assert await skus() == ["W3", "W1", "W2", "W4"]
        index = page.locator("#products thead th", has_text="Index").locator("button")
        await index.click()
        assert await skus() == ["W2", "W1", "W3", "W4"]  # W4 has no index: blanks last
        await index.click()
        assert await skus() == ["W3", "W1", "W2", "W4"]  # ...in both directions
        await page.locator("#products thead th", has_text="Product").locator("button").click()
        assert await skus() == ["W2", "W3", "W4", "W1"]  # "<script>..." < Gadget < Gizmo < Widget

        layout = await page.evaluate(
            """() => {
                const wrap = document.querySelector('.table-wrap');
                return {
                    page: document.documentElement.scrollWidth,
                    wrapScroll: wrap.scrollWidth,
                    wrapClient: wrap.clientWidth,
                    overflow: getComputedStyle(wrap).overflowX,
                };
            }"""
        )
        assert layout["page"] <= 375
        assert layout["overflow"] == "auto" and layout["wrapScroll"] > layout["wrapClient"]
        assert csp_errors == []
        await browser.close()
