from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pricehunter import extract as extract_mod
from pricehunter.extract import (
    CONFIDENCE,
    availability_from_text,
    extract_offer,
    load_jsonld,
)
from pricehunter.models import Availability, ExtractedOffer, SelectorConfig

A = Availability
D = Decimal
FIXTURES = Path(__file__).parent / "fixtures" / "extract"
URL = "https://shop.example/p/widget"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def page(*jsonld: Any, head: str = "", body: str = "") -> str:
    scripts = "".join(
        f'<script type="application/ld+json">{d if isinstance(d, str) else json.dumps(d)}</script>'
        for d in jsonld
    )
    return f"<html><head>{scripts}{head}</head><body>{body}</body></html>"


def product(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Widget",
        "offers": {"@type": "Offer", "price": "19.99", "priceCurrency": "EUR"},
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- fixtures


class TestShopifyVariants:
    BASE = "https://outdoor-shop.example/products/trail-runner-pro"

    def test_default_url_picks_lowest_in_stock_variant(self) -> None:
        o = extract_offer(fixture("shopify_variants.html"), self.BASE)
        assert (o.price, o.currency, o.availability) == (D("94.95"), "EUR", A.IN_STOCK)
        assert (o.sku, o.gtin) == ("TRP-42-BLK", "5901234123464")
        assert (o.method, o.confidence) == ("jsonld", 0.9)
        assert o.title == "Trail Runner Pro"
        assert o.brand == "Stridewell"
        assert o.image == (
            "https://outdoor-shop.example/cdn/shop/products/trail-runner-pro.jpg?v=1712345678&width=1920"
        )
        # Strike price comes from the visible price block, which shows the same selling price.
        assert o.original_price == D("119.95")

    def test_variant_url_selects_that_variant_even_when_sold_out(self) -> None:
        o = extract_offer(fixture("shopify_variants.html"), f"{self.BASE}?variant=40001")
        assert (o.price, o.availability, o.sku, o.gtin) == (
            D("89.95"),
            A.OUT_OF_STOCK,
            "TRP-41-BLK",
            "5901234123457",
        )
        # The page's strike price belongs to another variant's selling price.
        assert o.original_price is None

    def test_variant_url_with_tracking_params(self) -> None:
        o = extract_offer(fixture("shopify_variants.html"), f"{self.BASE}?utm_source=mail&variant=40003")
        assert (o.price, o.sku) == (D("99.95"), "TRP-43-BLK")


def test_woocommerce_graph() -> None:
    o = extract_offer(fixture("woocommerce_graph.html"), "https://casa-gadget.example/produs/espressor-x200/")
    assert o.method == "jsonld"
    assert (o.price, o.currency, o.availability) == (D("1899.00"), "RON", A.IN_STOCK)
    assert o.original_price == D("2199.00")  # StrikethroughPrice spec
    assert o.title == "Espressor automat X200, 15 bar, 1.8 L"
    assert o.brand == "De'Longhi"  # resolved through @id across scripts, entity-decoded
    assert (o.gtin, o.sku) == ("5941234567890", "X200-BLK")
    assert o.image == "https://casa-gadget.example/wp-content/uploads/2026/03/x200.jpg"


def test_magento_microdata() -> None:
    o = extract_offer(fixture("magento_microdata.html"), "https://bergwelt.example/fusion-backpack.html")
    assert (o.method, o.confidence) == ("microdata", 0.85)
    assert (o.price, o.currency, o.availability) == (D("49"), "EUR", A.IN_STOCK)
    assert o.original_price == D("59.00")  # from the .old-price block next to the same €49.00
    assert o.gtin == "4006381333931"  # "4 006381 333931" normalised to digits
    assert (o.sku, o.mpn, o.brand, o.title) == ("24-MB02", "FB-2026-BLU", "Bergwelt", "Fusion Backpack")
    assert o.image == "https://bergwelt.example/media/catalog/product/m/b/mb02-gray-0.jpg"


class TestProductGroup:
    BASE = "https://fashion.example/p/organic-tee"

    def test_lowest_in_stock_variant_with_strike_price_and_shipping(self) -> None:
        o = extract_offer(fixture("productgroup_variants.html"), self.BASE)
        assert (o.price, o.currency, o.availability) == (D("24.90"), "EUR", A.IN_STOCK)
        assert (o.original_price, o.shipping) == (D("29.90"), D("4.95"))
        assert (o.sku, o.gtin) == ("TEE-01-S-WHT", "00012345600012")
        assert o.title == "Organic Cotton Tee - S / White"
        assert o.brand == "Loomcraft"  # inherited from the group
        assert o.image == "https://fashion.example/img/tee-white.jpg"

    def test_variant_url_match_ignores_query_order(self) -> None:
        o = extract_offer(fixture("productgroup_variants.html"), f"{self.BASE}?color=white&size=M")
        assert (o.price, o.availability, o.sku, o.gtin) == (
            D("22.90"),
            A.OUT_OF_STOCK,
            "TEE-01-M-WHT",
            "00012345600029",
        )
        assert o.title == "Organic Cotton Tee - M / White"

    def test_variant_sku_in_url_path(self) -> None:
        o = extract_offer(fixture("productgroup_variants.html"), f"{self.BASE}/TEE-01-L-WHT")
        assert (o.price, o.sku, o.availability) == (D("26.90"), "TEE-01-L-WHT", A.IN_STOCK)


def test_aggregate_offer() -> None:
    o = extract_offer(fixture("aggregate_offer.html"), "https://compara.example/sony-wh-1000xm5")
    assert (o.method, o.price, o.currency) == ("jsonld", D("1349.99"), "RON")
    assert (o.gtin, o.mpn, o.brand) == ("4548736132610", "WH1000XM5B.CE7", "Sony")
    assert o.image == "https://compara.example/img/xm5-1.jpg"
    assert o.availability is A.UNKNOWN


def test_malformed_jsonld() -> None:
    o = extract_offer(fixture("malformed_jsonld.html"), "https://megashop.example/robot")
    assert o.method == "jsonld"
    assert (o.price, o.currency, o.availability) == (D("2499.90"), "RON", A.IN_STOCK)
    assert o.title == "Robot de bucătărie ChefMaster 3000"
    assert (o.gtin, o.sku, o.brand) == ("5941234567891", "CM-3000", "ChefMaster")


class TestEmagLike:
    URL = "https://www.emag.example/telefon-samsung-galaxy-s24/pd/DQ1/"
    SELECTORS = SelectorConfig(
        price=".product-page-pricing .product-new-price",
        original_price=".product-page-pricing .product-old-price",
        availability=".stock-and-genius .label",
        title="h1.page-title",
        mpn=".product-part-number",
        gtin=".product-ean span",
        shipping=".delivery-cost",
    )

    def test_selectors_with_sup_decimals(self) -> None:
        o = extract_offer(fixture("emag_like.html"), self.URL, self.SELECTORS)
        assert (o.method, o.confidence) == ("selectors", 0.95)
        assert (o.price, o.currency, o.original_price) == (D("3899.99"), "RON", D("4699.99"))
        assert (o.availability, o.shipping) == (A.IN_STOCK, D("19.99"))
        assert o.title == "Telefon mobil Samsung Galaxy S24, 8GB RAM, 256GB, 5G, Onyx Black"
        assert (o.mpn, o.gtin) == ("SM-S921BZKGEUE", "8806095300320")

    def test_price_attr_with_default_currency(self) -> None:
        sel = SelectorConfig(price=".product-page-pricing", price_attr="data-product-price")
        o = extract_offer(fixture("emag_like.html"), self.URL, sel, default_currency="ron")
        assert (o.method, o.price, o.currency) == ("selectors", D("3899.99"), "RON")

    def test_heuristics_without_selectors_skip_header_and_carousel(self) -> None:
        o = extract_offer(fixture("emag_like.html"), self.URL)
        assert (o.method, o.confidence) == ("heuristic", 0.4)
        assert (o.price, o.currency, o.original_price) == (D("3899.99"), "RON", D("4699.99"))
        assert o.availability is A.IN_STOCK
        assert o.title == "Telefon mobil Samsung Galaxy S24, 8GB RAM, 256GB, 5G, Onyx Black"

    def test_selector_without_match_falls_back(self) -> None:
        o = extract_offer(fixture("emag_like.html"), self.URL, SelectorConfig(price=".does-not-exist"))
        assert (o.method, o.price) == ("heuristic", D("3899.99"))

    @pytest.mark.parametrize("bad", ["div[[", "p::before", "", "a:nth-child(x)"])
    def test_invalid_selector_falls_back(self, bad: str) -> None:
        o = extract_offer(fixture("emag_like.html"), self.URL, SelectorConfig(price=bad, title=bad))
        assert (o.method, o.price) == ("heuristic", D("3899.99"))


def test_opengraph_only() -> None:
    o = extract_offer(fixture("opengraph_only.html"), "https://nordhaus.example/chairs/nordic-oak")
    assert (o.method, o.confidence) == ("opengraph", 0.75)
    assert (o.price, o.original_price, o.currency) == (D("119.00"), D("149.00"), "GBP")
    assert o.availability is A.IN_STOCK
    assert o.title == "Nordic Oak Dining Chair"  # og:title wins over <h1>
    assert (o.brand, o.sku) == ("Nordhaus", "NH-CH-OAK")
    assert o.image == "https://nordhaus.example/media/chairs/nordic-oak-1200.jpg"


def test_heuristic_only() -> None:
    o = extract_offer(fixture("heuristic_only.html"), "https://werkzeug.example/bs18")
    assert (o.method, o.confidence) == ("heuristic", 0.4)
    # Not the per-unit price, the nav teaser, the sidebar filter or the cross-sell tile.
    assert (o.price, o.currency, o.original_price) == (D("199.00"), "EUR", D("249.00"))
    assert o.availability is A.IN_STOCK  # "Auf Lager", not the cross-sell "Ausverkauft"
    assert o.title == "Akku-Bohrschrauber BS 18 Pro"


def test_out_of_stock_with_price_specification_only() -> None:
    o = extract_offer(
        fixture("out_of_stock.html"), "https://electromax.example/cuptor-microunde-whirlpool-mwp-304"
    )
    assert o.method == "jsonld"
    assert (o.price, o.original_price, o.currency) == (D("549.00"), D("699.00"), "RON")
    assert o.availability is A.OUT_OF_STOCK
    assert (o.gtin, o.mpn, o.sku, o.brand) == ("883049456783", "MWP 304 SB", "EMX-118273", "Whirlpool")
    assert o.image == "https://cdn.electromax.example/p/mwp304.webp"


def test_no_price() -> None:
    o = extract_offer(fixture("no_price.html"), "https://termopro.example/vulcan")
    assert (o.price, o.method, o.confidence) == (None, "none", 0.0)
    assert (o.currency, o.original_price) == (None, None)
    assert o.title == "Centrală termică industrială Vulcan 500 kW"
    assert (o.brand, o.mpn) == ("Vulcan", "VLC-500-IND")
    assert o.availability is A.UNKNOWN


def test_rdfa() -> None:
    o = extract_offer(fixture("rdfa_product.html"), "https://cafe.example/barista-pro")
    assert (o.method, o.confidence) == ("rdfa", 0.85)
    assert (o.price, o.currency, o.availability) == (D("349.00"), "EUR", A.PREORDER)
    assert (o.gtin, o.sku, o.brand) == ("3760123456789", "CM-BP-01", "Café Maison")
    assert o.image == "https://cafe.example/img/barista-pro.jpg"


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.glob("*.html")))
def test_every_fixture_yields_a_valid_offer(name: str) -> None:
    o = extract_offer(fixture(name), URL)
    assert isinstance(o, ExtractedOffer)
    assert o.url == URL
    assert (o.price is None) == (o.method == "none")
    assert o.confidence == CONFIDENCE.get(o.method, 0.0)


# --------------------------------------------------------------------------- JSON-LD details


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://schema.org/InStock", A.IN_STOCK),
        ("https://schema.org/InStock", A.IN_STOCK),
        ("InStock", A.IN_STOCK),
        ("instock", A.IN_STOCK),
        ("schema:InStock", A.IN_STOCK),
        ("In stock", A.IN_STOCK),
        ("https://schema.org/OnlineOnly", A.IN_STOCK),
        ({"@id": "http://schema.org/InStock"}, A.IN_STOCK),
        (["https://schema.org/InStock"], A.IN_STOCK),
        ("https://schema.org/OutOfStock", A.OUT_OF_STOCK),
        ("SoldOut", A.OUT_OF_STOCK),
        ("http://schema.org/SoldOut", A.OUT_OF_STOCK),
        ("Discontinued", A.OUT_OF_STOCK),
        ("PreOrder", A.PREORDER),
        ("https://schema.org/PreSale", A.PREORDER),
        ("BackOrder", A.PREORDER),
        ("http://schema.org/BackOrder", A.PREORDER),
        ("LimitedAvailability", A.LIMITED),
        ("https://schema.org/InStoreOnly", A.LIMITED),
        ("", A.UNKNOWN),
        ("https://schema.org/Whatever", A.UNKNOWN),
    ],
)
def test_jsonld_availability_values(value: Any, expected: Availability) -> None:
    offer = {"@type": "Offer", "price": 10, "priceCurrency": "EUR", "availability": value}
    assert extract_offer(page(product(offers=offer)), URL).availability is expected


def test_jsonld_missing_availability_is_unknown() -> None:
    assert extract_offer(page(product()), URL).availability is A.UNKNOWN


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (19.99, "19.99"),
        (20, "20"),
        ("19.99", "19.99"),
        ("1,299.00", "1299.00"),
        ("1299,00", "1299.00"),
        ("1.299,00", "1299.00"),
        ("€19.99", "19.99"),
        (["24.50"], "24.50"),
    ],
)
def test_jsonld_price_formats(price: Any, expected: str) -> None:
    o = extract_offer(page(product(offers={"@type": "Offer", "price": price, "priceCurrency": "EUR"})), URL)
    assert (o.price, o.currency, o.method) == (D(expected), "EUR", "jsonld")


def test_jsonld_top_level_list_and_webpage_main_entity() -> None:
    listed = [{"@type": "Organization", "name": "Shop"}, product()]
    nested = {"@context": "https://schema.org", "@type": "ItemPage", "mainEntity": product(name="Nested")}
    assert extract_offer(page(listed), URL).price == D("19.99")
    o = extract_offer(page(nested), URL)
    assert (o.price, o.title) == (D("19.99"), "Nested")


def test_jsonld_offer_list_prefers_lowest_in_stock() -> None:
    offers = [
        {"@type": "Offer", "price": "15.00", "priceCurrency": "EUR", "availability": "OutOfStock"},
        {"@type": "Offer", "price": "0", "priceCurrency": "EUR", "availability": "InStock"},
        {"@type": "Offer", "price": "21.00", "priceCurrency": "EUR", "availability": "InStock"},
        {"@type": "Offer", "price": "18.50", "priceCurrency": "EUR", "availability": "InStock"},
    ]
    o = extract_offer(page(product(offers=offers)), URL)
    assert (o.price, o.availability) == (D("18.50"), A.IN_STOCK)


def test_jsonld_offer_list_all_out_of_stock_takes_lowest() -> None:
    offers = [
        {"@type": "Offer", "price": "15.00", "priceCurrency": "EUR", "availability": "OutOfStock"},
        {"@type": "Offer", "price": "12.00", "priceCurrency": "EUR", "availability": "SoldOut"},
    ]
    o = extract_offer(page(product(offers=offers)), URL)
    assert (o.price, o.availability) == (D("12.00"), A.OUT_OF_STOCK)


def test_jsonld_aggregate_offer_with_price_only() -> None:
    o = extract_offer(
        page(product(offers={"@type": "AggregateOffer", "price": 7.5, "priceCurrency": "USD"})), URL
    )
    assert (o.price, o.currency) == (D("7.5"), "USD")


def test_jsonld_price_specification_list_prefers_non_unit_and_non_strikethrough() -> None:
    offer = {
        "@type": "Offer",
        "priceSpecification": [
            {
                "@type": "UnitPriceSpecification",
                "price": 2.5,
                "priceCurrency": "EUR",
                "referenceQuantity": {"value": 1, "unitCode": "KGM"},
            },
            {
                "@type": "UnitPriceSpecification",
                "price": 15,
                "priceCurrency": "EUR",
                "priceType": "StrikethroughPrice",
            },
            {"@type": "UnitPriceSpecification", "price": 12.49, "priceCurrency": "EUR"},
        ],
    }
    o = extract_offer(page(product(offers=offer)), URL)
    assert (o.price, o.original_price, o.currency) == (D("12.49"), D("15"), "EUR")


def test_jsonld_strike_price_not_above_price_is_dropped() -> None:
    offer = {
        "@type": "Offer",
        "price": 20,
        "priceCurrency": "EUR",
        "priceSpecification": {"price": 18, "priceType": "https://schema.org/ListPrice"},
    }
    assert extract_offer(page(product(offers=offer)), URL).original_price is None


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("/img/a.jpg", "https://shop.example/img/a.jpg"),
        (["https://cdn.example/1.jpg", "https://cdn.example/2.jpg"], "https://cdn.example/1.jpg"),
        ({"@type": "ImageObject", "url": "https://cdn.example/obj.jpg"}, "https://cdn.example/obj.jpg"),
        ([{"@type": "ImageObject", "contentUrl": "//cdn.example/c.jpg"}], "https://cdn.example/c.jpg"),
    ],
)
def test_jsonld_image_forms(image: Any, expected: str) -> None:
    assert extract_offer(page(product(image=image)), URL).image == expected


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        ("Acme", "Acme"),
        ({"@type": "Brand", "name": "Acme"}, "Acme"),
        ([{"@type": "Organization", "name": "Acme &amp; Co"}], "Acme & Co"),
    ],
)
def test_jsonld_brand_forms(brand: Any, expected: str) -> None:
    assert extract_offer(page(product(brand=brand)), URL).brand == expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("gtin", "5901234123457", "5901234123457"),
        ("gtin13", "590-1234-12345-7", "5901234123457"),
        ("gtin12", 883049456783, "883049456783"),
        ("gtin14", "00012345600012", "00012345600012"),
        ("gtin8", "96385074", "96385074"),
        ("isbn", "978-3-16-148410-0", "9783161484100"),
        ("gtin13", "12345", None),
    ],
)
def test_jsonld_gtin_keys(key: str, value: Any, expected: str | None) -> None:
    assert extract_offer(page(product(**{key: value})), URL).gtin == expected


def test_jsonld_offer_level_identifiers() -> None:
    offer = {
        "@type": "Offer",
        "price": 5,
        "priceCurrency": "EUR",
        "sku": "OFF-1",
        "gtin13": "4006381333931",
        "mpn": "M-1",
    }
    o = extract_offer(page(product(offers=offer)), URL)
    assert (o.sku, o.gtin, o.mpn) == ("OFF-1", "4006381333931", "M-1")


def test_jsonld_offers_by_id_reference() -> None:
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "Product", "@id": "#p", "name": "Ref", "offers": [{"@id": "#o1"}]},
            {
                "@type": "Offer",
                "@id": "#o1",
                "price": "42.00",
                "priceCurrency": "PLN",
                "availability": "InStock",
            },
        ],
    }
    o = extract_offer(page(graph), URL)
    assert (o.price, o.currency, o.availability, o.title) == (D("42.00"), "PLN", A.IN_STOCK, "Ref")


def test_jsonld_prefers_product_whose_url_matches_page() -> None:
    related = product(
        name="Related", url="https://shop.example/p/other", offers={"price": 5, "priceCurrency": "EUR"}
    )
    main = product(name="Main", url="/p/widget/", offers={"price": 50, "priceCurrency": "EUR"})
    o = extract_offer(page(related, main), URL)
    assert (o.title, o.price) == ("Main", D("50"))


def test_jsonld_shipping_details_list() -> None:
    offer = {
        "@type": "Offer",
        "price": 30,
        "priceCurrency": "EUR",
        "shippingDetails": [
            {
                "@type": "OfferShippingDetails",
                "shippingRate": {"@type": "MonetaryAmount", "value": "0", "currency": "EUR"},
            },
            {
                "@type": "OfferShippingDetails",
                "shippingRate": {"@type": "MonetaryAmount", "value": "9.90", "currency": "EUR"},
            },
        ],
    }
    assert extract_offer(page(product(offers=offer)), URL).shipping == D("0")


def test_jsonld_script_type_variants_and_escaped_json() -> None:
    body = json.dumps(product(name="Esc")).replace('"', "&quot;")
    html = f'<html><head><script type="application/ld+json; charset=utf-8">{body}</script></head></html>'
    o = extract_offer(html, URL)
    assert (o.title, o.price) == ("Esc", D("19.99"))


def test_jsonld_zero_price_falls_back_to_lower_layer() -> None:
    html = page(
        product(offers={"@type": "Offer", "price": 0, "priceCurrency": "EUR"}),
        body='<div class="product"><span class="price">24,90 €</span></div>',
    )
    o = extract_offer(html, URL)
    assert (o.method, o.price, o.currency, o.title) == ("heuristic", D("24.90"), "EUR", "Widget")


def test_jsonld_without_currency_uses_default_currency() -> None:
    html = page(product(offers={"@type": "Offer", "price": "99"}))
    assert extract_offer(html, URL, default_currency="ron").currency == "RON"
    assert extract_offer(html, URL).currency is None


def test_currency_filled_from_lower_layer() -> None:
    html = page(
        product(offers={"@type": "Offer", "price": "99"}),
        head='<meta property="product:price:currency" content="HUF">',
    )
    assert extract_offer(html, URL, default_currency="EUR").currency == "HUF"


def test_selectors_beat_jsonld_and_identity_fields_are_merged() -> None:
    html = page(
        product(gtin13="4006381333931", brand="Acme"),
        body='<div id="buy"><b class="now">17,49 €</b></div>',
    )
    o = extract_offer(html, URL, SelectorConfig(price="#buy .now"))
    assert (o.method, o.price, o.currency) == ("selectors", D("17.49"), "EUR")
    assert (o.title, o.gtin, o.brand) == ("Widget", "4006381333931", "Acme")


def test_selector_currency_and_availability_link() -> None:
    html = (
        '<div class="p"><span class="amt">1.299</span><span class="cur">Lei</span>'
        '<link class="av" href="https://schema.org/OutOfStock"></div>'
    )
    sel = SelectorConfig(price=".amt", currency=".cur", availability=".av")
    o = extract_offer(html, URL, sel)
    assert (o.price, o.currency, o.availability) == (D("1299"), "RON", A.OUT_OF_STOCK)


# --------------------------------------------------------------------------- meta / microdata


def test_itemprop_price_outside_itemscope() -> None:
    html = (
        "<html><body><h1>Lamp</h1>"
        '<span itemprop="price" content="59.90">59,90 €</span>'
        '<meta itemprop="priceCurrency" content="EUR">'
        '<link itemprop="availability" href="http://schema.org/LimitedAvailability">'
        "</body></html>"
    )
    o = extract_offer(html, URL)
    assert (o.method, o.confidence) == ("meta", 0.75)
    assert (o.price, o.currency, o.availability, o.title) == (D("59.90"), "EUR", A.LIMITED, "Lamp")


def test_twitter_label_price() -> None:
    head = (
        '<meta property="product:price:currency" content="EUR">'
        '<meta name="twitter:label1" content="Preț"><meta name="twitter:data1" content="1.299,00 lei">'
        '<meta name="twitter:label2" content="Disponibilitate"><meta name="twitter:data2" content="În stoc">'
    )
    o = extract_offer(page(head=head), URL)
    assert (o.method, o.price, o.currency, o.availability) == ("meta", D("1299.00"), "RON", A.IN_STOCK)


def test_og_price_amount() -> None:
    head = '<meta property="og:price:amount" content="15"><meta property="og:price:currency" content="usd">'
    o = extract_offer(page(head=head), URL)
    assert (o.method, o.price, o.currency) == ("opengraph", D("15"), "USD")


def test_microdata_related_product_does_not_win_over_page_product() -> None:
    html = """
    <div itemscope itemtype="https://schema.org/Product">
      <span itemprop="name">Teaser</span><a itemprop="url" href="/p/other"></a>
      <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
        <meta itemprop="price" content="5"><meta itemprop="priceCurrency" content="EUR"></div>
    </div>
    <div itemscope itemtype="https://schema.org/Product">
      <span itemprop="name">Widget</span><link itemprop="url" href="/p/widget">
      <div itemprop="offers" itemscope itemtype="https://schema.org/Offer">
        <meta itemprop="price" content="25"><meta itemprop="priceCurrency" content="EUR"></div>
    </div>"""
    o = extract_offer(html, URL)
    assert (o.method, o.title, o.price) == ("microdata", "Widget", D("25"))


# --------------------------------------------------------------------------- heuristics


def test_heuristic_woocommerce_del_ins() -> None:
    html = (
        '<div class="summary"><p class="price"><del><span class="amount">30,00&nbsp;€</span></del> '
        '<ins><span class="amount">24,00&nbsp;€</span></ins></p></div>'
    )
    o = extract_offer(html, URL)
    assert (o.price, o.original_price, o.currency) == (D("24.00"), D("30.00"), "EUR")


def test_heuristic_prefers_sale_over_regular() -> None:
    html = (
        '<div class="product"><div class="price__regular">'
        '<span class="price-item price-item--regular">$25.00</span></div>'
        '<div class="price__sale"><s class="price-item price-item--regular">$25.00</s>'
        '<span class="price-item price-item--sale">$19.00</span></div></div>'
    )
    o = extract_offer(html, URL)
    assert (o.price, o.original_price, o.currency) == (D("19.00"), D("25.00"), "USD")


def test_heuristic_camel_case_old_price_class() -> None:
    html = '<div class="pdp"><span class="oldPrice">99 €</span><span class="salePrice">79 €</span></div>'
    o = extract_offer(html, URL)
    assert (o.price, o.original_price) == (D("79"), D("99"))


def test_heuristic_ignores_layout_classes_on_body() -> None:
    html = (
        '<html><body class="has-sidebar menu-open">'
        '<div class="product-info"><span class="price">12,50 €</span></div></body></html>'
    )
    assert extract_offer(html, URL).price == D("12.50")


def test_heuristic_header_inside_article_is_not_page_chrome() -> None:
    html = '<article class="product"><header><h1>Kettle</h1><span class="price">€35</span></header></article>'
    o = extract_offer(html, URL)
    assert (o.price, o.title) == (D("35"), "Kettle")


def test_heuristic_title_falls_back_to_title_tag() -> None:
    html = (
        "<html><head><title> Kettle | Shop </title></head>"
        '<body><h1></h1><p class="price">€35</p></body></html>'
    )
    assert extract_offer(html, URL).title == "Kettle | Shop"


def test_availability_text_only_from_targeted_elements() -> None:
    body = '<p>Usually in stock within 2 days.</p><span class="price">€10</span>'
    assert extract_offer(page(body=body), URL).availability is A.UNKNOWN
    body_with_label = body + '<div class="stock-status">Stoc epuizat</div>'
    assert extract_offer(page(body=body_with_label), URL).availability is A.OUT_OF_STOCK


def test_structured_availability_beats_text() -> None:
    html = page(
        product(offers={"price": 9, "priceCurrency": "EUR", "availability": "InStock"}),
        body='<div class="availability">Out of stock</div>',
    )
    assert extract_offer(html, URL).availability is A.IN_STOCK


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("In stock", A.IN_STOCK),
        ("5 in stock", A.IN_STOCK),
        ("În stoc", A.IN_STOCK),
        ("in stoc furnizor", A.IN_STOCK),
        ("Disponibil", A.IN_STOCK),
        ("Auf Lager", A.IN_STOCK),
        ("Sofort lieferbar", A.IN_STOCK),
        ("En stock", A.IN_STOCK),
        ("Available", A.IN_STOCK),
        ("Dostępny", A.IN_STOCK),
        ("Out of stock", A.OUT_OF_STOCK),
        ("Not in stock", A.OUT_OF_STOCK),
        ("Currently unavailable", A.OUT_OF_STOCK),
        ("Sold out", A.OUT_OF_STOCK),
        ("Stoc epuizat", A.OUT_OF_STOCK),
        ("Indisponibil", A.OUT_OF_STOCK),
        ("Nu este în stoc", A.OUT_OF_STOCK),
        ("Ausverkauft", A.OUT_OF_STOCK),
        ("Nicht verfügbar", A.OUT_OF_STOCK),
        ("Épuisé", A.OUT_OF_STOCK),
        ("Rupture de stock", A.OUT_OF_STOCK),
        ("Agotado", A.OUT_OF_STOCK),
        ("Precomandă", A.PREORDER),
        ("Pre-order now", A.PREORDER),
        ("Preorder", A.PREORDER),
        ("Vorbestellen", A.PREORDER),
        ("On backorder", A.PREORDER),
        ("Stoc limitat", A.LIMITED),
        ("Only 2 left", A.LIMITED),
        ("Ultimele 3 produse", A.LIMITED),
        ("Nur noch 1 Stück", A.LIMITED),
        ("Call us", A.UNKNOWN),
        ("", A.UNKNOWN),
        (None, A.UNKNOWN),
    ],
)
def test_availability_from_text(text: str | None, expected: Availability) -> None:
    assert availability_from_text(text) is expected


# --------------------------------------------------------------------------- JSON-LD loader


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', [{"a": 1}]),
        ('<!-- {"a": 1} -->', [{"a": 1}]),
        ('//<![CDATA[\n{"a": 1}\n//]]>', [{"a": 1}]),
        ('<![CDATA[{"a": 1}]]>', [{"a": 1}]),
        ('{"a": [1, 2,], "b": {"c": 3,},}', [{"a": [1, 2], "b": {"c": 3}}]),
        ('{"a": "line1\nline2\tx"}', [{"a": "line1\nline2\tx"}]),
        ('{"a": 1}\n{"b": 2}', [{"a": 1}, {"b": 2}]),
        ('{"a": 1},{"b": 2};', [{"a": 1}, {"b": 2}]),
        ('[{"a": 1}, {"b": 2}]', [[{"a": 1}, {"b": 2}]]),
        ('{"a": "it\\\'s"}', [{"a": "it's"}]),
        ("{&quot;a&quot;: 1}", [{"a": 1}]),
        ('{"a": 1} {broken', [{"a": 1}]),
        ("", []),
        ("not json at all", []),
    ],
)
def test_load_jsonld(raw: str, expected: list[Any]) -> None:
    assert load_jsonld(raw) == expected


# --------------------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    "html",
    [
        "",
        "   ",
        "<<<>>>",
        "\x00\x01\x02",
        "<html><body><div class='price'>",
        "<script type='application/ld+json'>{{{{</script>",
        '<script type=\'application/ld+json\'>[1, "x", null, {"@type": 5}]</script>',
        '<script type="application/ld+json">{"@type":"Product","offers":{"price":{"x":1}}}</script>',
        '<script type="application/ld+json">'
        '{"@type":"Product","offers":"cheap","image":42,"brand":true}</script>',
        "<div itemscope itemtype='http://schema.org/Product'><span itemprop='price'>abc</span></div>",
        "<div vocab='http://schema.org/' typeof='Product'><span property='price'>-5</span></div>",
        "<div>" * 300 + '<span class="price">€5</span>' + "</div>" * 300,
    ],
)
def test_never_raises_on_garbage(html: str) -> None:
    o = extract_offer(html, URL)
    assert isinstance(o, ExtractedOffer)
    assert o.url == URL
    assert o.price is None or o.price > 0


def test_cyclic_jsonld_references_terminate() -> None:
    graph = {
        "@graph": [
            {
                "@type": "Product",
                "@id": "#p",
                "name": "Loop",
                "offers": {"@id": "#o"},
                "isRelatedTo": {"@id": "#p"},
            },
            {"@type": "Offer", "@id": "#o", "price": 3, "priceCurrency": "EUR", "itemOffered": {"@id": "#p"}},
        ]
    }
    assert extract_offer(page(graph), URL).price == D("3")


def test_failing_layer_does_not_break_others(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(extract_mod, "_jsonld_layer", boom)
    o = extract_offer(fixture("magento_microdata.html"), "https://bergwelt.example/fusion-backpack.html")
    assert (o.method, o.price) == ("microdata", D("49"))


def test_unexpected_failure_returns_empty_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(extract_mod, "BeautifulSoup", boom)
    o = extract_offer("<html></html>", URL)
    assert o == ExtractedOffer(url=URL)
    assert (o.price, o.method, o.confidence) == (None, "none", 0.0)
