"""The demo marketplace itself: addresses, robots.txt, sitemaps, day state and outages."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from pricehunter.demo import PRODUCTS, SHOPS, DemoMarket


@pytest.fixture
def market() -> Iterator[DemoMarket]:
    with DemoMarket() as m:
        yield m


def test_each_shop_is_its_own_origin_and_allows_robots(market: DemoMarket) -> None:
    urls = [market.base_url(shop) for shop in SHOPS]
    assert len(set(urls)) == 3 and all(u.startswith("http://127.0.0.1:") for u in urls)
    for shop in SHOPS:
        robots = httpx.get(market.base_url(shop) + "/robots.txt")
        assert robots.status_code == 200 and "User-agent: *\nAllow: /" in robots.text
        assert "Disallow" not in robots.text
    budget = httpx.get(market.base_url("BudgetBay") + "/robots.txt").text
    assert f"Sitemap: {market.base_url('BudgetBay')}/sitemap.xml" in budget


def test_budgetbay_is_reachable_only_through_its_sitemap(market: DemoMarket) -> None:
    base = market.base_url("BudgetBay")
    assert httpx.get(base + "/search", params={"q": "iphone"}).status_code == 404
    index = httpx.get(base + "/sitemap.xml").text
    assert "<sitemapindex" in index and f"{base}/sitemaps/products.xml" in index
    products = httpx.get(base + "/sitemaps/products.xml").text
    assert market.product_url("BudgetBay", "3004") in products  # the accessory decoy is listed too


def test_megashop_search_redirects_exact_gtins(market: DemoMarket) -> None:
    base = market.base_url("MegaShop")
    hit = httpx.get(base + "/search", params={"q": "0195949035500"})
    assert hit.status_code == 302 and hit.headers["location"].endswith("-1003")
    results = httpx.get(base + "/search", params={"q": "Apple iPhone 15 128GB Black"}).text
    for item in ("1003", "1004", "1005"):  # the right phone, the 256GB variant and a case
        assert f'-{item}"' in results


def test_prices_follow_the_day_and_outages_return_503(market: DemoMarket) -> None:
    url = market.product_url("PretMic", "2003")
    assert "4.499<sup>99</sup> Lei" in httpx.get(url).text and "În stoc" in httpx.get(url).text
    market.set_day(2)
    assert "Stoc epuizat" in httpx.get(url).text
    market.set_day(9)  # the last scripted day repeats
    assert "4.399<sup>99</sup> Lei" in httpx.get(url).text and "pret-vechi" in httpx.get(url).text
    market.set_offline("PretMic")
    assert httpx.get(url).status_code == 503
    market.set_offline("PretMic", False)
    assert httpx.get(url).status_code == 200
    assert market.requests[url] == 7
    with pytest.raises(ValueError):
        market.set_day(0)
    with pytest.raises(KeyError):
        market.set_offline("Nowhere")


def test_config_is_wired_to_the_running_shops(market: DemoMarket, tmp_path: Path) -> None:
    config = market.config(tmp_path / "x.db")
    assert [c.base_url for c in config.competitors] == [market.base_url(s) for s in SHOPS]
    assert config.settings.currency_rates == {"RON": Decimal("0.201")}
    assert config.notifiers[1].path == str(tmp_path / "alerts.jsonl")
    vacuum = config.product("DYSON-V15")
    assert vacuum is not None and vacuum.competitor_urls == {
        "PretMic": [market.product_url("PretMic", "2006")]
    }
    vacuum.competitor_urls.clear()  # editing a config never leaks into the shared catalogue
    assert market.config(tmp_path / "y.db").product("DYSON-V15").competitor_urls  # type: ignore[union-attr]
    assert all(p.competitor_urls == {} for p in PRODUCTS)


def test_addresses_need_a_running_market() -> None:
    market = DemoMarket()
    with pytest.raises(RuntimeError, match="not running"):
        market.base_url("MegaShop")
    with pytest.raises(KeyError):
        market.product_url("MegaShop", "9999")
