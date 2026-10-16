from __future__ import annotations

from decimal import Decimal

import pytest

from pricehunter.analyze import market_snapshot, suggest_price
from pricehunter.models import (
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    Observation,
    Product,
    RepricingRule,
    Settings,
)

D = Decimal
IN, OUT, UNKNOWN = Availability.IN_STOCK, Availability.OUT_OF_STOCK, Availability.UNKNOWN
SETTINGS = Settings(currency_rates={"RON": D("0.2"), "USD": D("0.9")}, base_currency="EUR")


def product(**fields: object) -> Product:
    data: dict[str, object] = {"sku": "SKU1", "name": "Phone X", "currency": "EUR"}
    data.update(fields)
    return Product.model_validate(data)


def obs(
    competitor: str,
    price: str | None,
    *,
    currency: str | None = "EUR",
    availability: Availability = IN,
    shipping: str | None = None,
    score: float = 1.0,
    sku: str = "SKU1",
    url: str | None = None,
) -> Observation:
    return Observation(
        sku=sku,
        competitor=competitor,
        url=url or f"https://{competitor}.example/p",
        price=None if price is None else D(price),
        currency=currency,
        availability=availability,
        shipping=None if shipping is None else D(shipping),
        match_score=score,
    )


def offer(
    competitor: str, price: str, availability: Availability = IN, score: float = 1.0
) -> CompetitorPrice:
    return CompetitorPrice(
        competitor=competitor,
        url=f"https://{competitor}.example/p",
        price=D(price),
        original_currency="EUR",
        availability=availability,
        match_score=score,
    )


def snapshot_of(*offers: CompetitorPrice, our_price: str | None = "1400.00") -> MarketSnapshot:
    return MarketSnapshot(
        sku="SKU1",
        currency="EUR",
        our_price=None if our_price is None else D(our_price),
        offers=sorted(offers, key=lambda o: o.price),
    )


# --------------------------------------------------------------------------- market_snapshot


def test_snapshot_statistics_use_available_offers_only() -> None:
    snap = market_snapshot(
        product(our_price=D("105")),
        [
            obs("c", "125.00"),
            obs("a", "100.00"),
            obs("b", "110.00"),
            obs("d", "90.00", availability=OUT),
        ],
        SETTINGS,
    )
    assert [(o.competitor, o.price) for o in snap.offers] == [
        ("d", D("90.00")),
        ("a", D("100.00")),
        ("b", D("110.00")),
        ("c", D("125.00")),
    ]
    assert snap.sku == "SKU1" and snap.currency == "EUR" and snap.our_price == D("105")
    assert snap.lowest == D("100.00") and snap.lowest_competitor == "a"
    assert snap.highest == D("125.00")
    assert snap.median == D("110.00")
    assert snap.average == D("111.67")
    assert snap.in_stock_offers == 3
    assert snap.price_index == 95.5  # 105 / 110 * 100
    assert snap.competitors_cheaper == 1
    assert snap.our_rank == 2


def test_snapshot_even_median_and_cent_quantization() -> None:
    snap = market_snapshot(product(), [obs("a", "100"), obs("b", "110.01")], SETTINGS)
    assert snap.median == D("105.01")  # 105.005 rounds half up
    assert snap.average == D("105.01")
    assert snap.lowest == D("100.00")
    assert str(snap.lowest) == "100.00"


def test_snapshot_filters_foreign_sku_missing_price_and_low_match() -> None:
    snap = market_snapshot(
        product(),
        [
            obs("a", "100.00"),
            obs("b", "50.00", sku="OTHER"),
            obs("c", None),
            obs("d", "0.00"),
            obs("e", "60.00", score=0.5),
            obs("f", "70.00", score=0.9),
        ],
        SETTINGS,
        min_match_score=0.9,
    )
    assert [o.competitor for o in snap.offers] == ["f", "a"]
    assert snap.offers[0].match_score == 0.9


def test_snapshot_converts_ron_to_eur() -> None:
    settings = Settings(currency_rates={"RON": D("0.201")}, base_currency="EUR")
    snap = market_snapshot(
        product(),
        [obs("emag", "1299.00", currency="RON"), obs("altex", "1300", currency="ron")],
        settings,
    )
    assert [(o.competitor, o.price, o.original_currency) for o in snap.offers] == [
        ("emag", D("261.10"), "RON"),  # 1299 * 0.201 = 261.099
        ("altex", D("261.30"), "RON"),
    ]


def test_snapshot_converts_into_non_base_product_currency() -> None:
    snap = market_snapshot(
        product(currency="RON", our_price=D("480")),
        [
            obs("eu", "100.00", currency="EUR"),  # 100 / 0.2 = 500 RON
            obs("us", "100.00", currency="USD"),  # 100 * 0.9 / 0.2 = 450 RON
            obs("ro", "470.00", currency="RON"),
        ],
        SETTINGS,
    )
    assert snap.currency == "RON"
    assert [(o.competitor, o.price) for o in snap.offers] == [
        ("us", D("450.00")),
        ("ro", D("470.00")),
        ("eu", D("500.00")),
    ]
    assert snap.our_rank == 3 and snap.competitors_cheaper == 2


def test_snapshot_skips_unconvertible_and_assumes_missing_currency() -> None:
    snap = market_snapshot(
        product(),
        [
            obs("x", "10.00", currency="XYZ"),
            obs("gbp", "10.00", currency="GBP"),
            obs("n", "12.00", currency=None),
        ],
        SETTINGS,
    )
    assert [(o.competitor, o.price, o.original_currency) for o in snap.offers] == [
        ("n", D("12.00"), None)
    ]


@pytest.mark.parametrize(
    ("include", "expected"),
    [(True, ["a:105.99", "b:106.00", "ro:205.00"]), (False, ["a:100.00", "b:106.00", "ro:200.00"])],
)
def test_snapshot_shipping(include: bool, expected: list[str]) -> None:
    settings = SETTINGS.model_copy(update={"include_shipping": include})
    snap = market_snapshot(
        product(),
        [
            obs("a", "100.00", shipping="5.99"),
            obs("b", "106.00"),  # unknown shipping counts as zero
            obs("ro", "1000.00", currency="RON", shipping="25.00"),  # (1000 + 25) * 0.2
        ],
        settings,
    )
    assert [f"{o.competitor}:{o.price}" for o in snap.offers] == expected


def test_snapshot_one_offer_per_competitor_prefers_available() -> None:
    snap = market_snapshot(
        product(),
        [
            obs("a", "90.00", availability=OUT, url="https://a.example/1"),
            obs("a", "95.00", url="https://a.example/2"),
            obs("a", "99.00", url="https://a.example/3"),
            obs("b", "85.00", availability=OUT, url="https://b.example/1"),
            obs("b", "80.00", availability=OUT, url="https://b.example/2"),
            obs("c", "97.00", availability=UNKNOWN),
        ],
        SETTINGS,
    )
    by = {o.competitor: o for o in snap.offers}
    assert len(snap.offers) == 3
    assert (by["a"].price, by["a"].url) == (D("95.00"), "https://a.example/2")
    assert (by["b"].price, by["b"].availability) == (D("80.00"), OUT)
    assert by["c"].availability is UNKNOWN
    assert snap.lowest == D("95.00") and snap.lowest_competitor == "a"
    assert snap.highest == D("97.00")
    assert snap.in_stock_offers == 2


def test_snapshot_rank_ties_and_out_of_stock() -> None:
    snap = market_snapshot(
        product(our_price=D("100.00")),
        [obs("a", "100.00"), obs("b", "99.99"), obs("c", "50.00", availability=OUT)],
        SETTINGS,
    )
    assert snap.competitors_cheaper == 1
    assert snap.our_rank == 2

    tie = market_snapshot(product(our_price=D("100")), [obs("a", "100.00")], SETTINGS)
    assert tie.our_rank == 1 and tie.competitors_cheaper == 0
    assert tie.price_index == 100.0


def test_snapshot_without_our_price() -> None:
    snap = market_snapshot(product(), [obs("a", "10.00")], SETTINGS)
    assert snap.our_rank is None
    assert snap.competitors_cheaper == 0
    assert snap.price_index is None
    assert snap.lowest == D("10.00")


def test_snapshot_without_offers() -> None:
    snap = market_snapshot(product(our_price=D("10")), [], SETTINGS)
    assert snap.offers == []
    assert snap.lowest is snap.median is snap.average is snap.highest is None
    assert snap.lowest_competitor is None and snap.price_index is None
    assert snap.our_rank == 1 and snap.competitors_cheaper == 0 and snap.in_stock_offers == 0


def test_snapshot_all_out_of_stock() -> None:
    snap = market_snapshot(
        product(our_price=D("10")), [obs("a", "5.00", availability=OUT)], SETTINGS
    )
    assert len(snap.offers) == 1
    assert snap.lowest is None and snap.median is None
    assert snap.in_stock_offers == 0 and snap.our_rank == 1


# --------------------------------------------------------------------------- suggest_price


def test_beat_lowest_by_amount_with_readable_reason() -> None:
    p = product(our_price=D("1400.00"), cost=D("1000"))
    snap = snapshot_of(offer("Shop X", "1299.00"), offer("Shop Y", "1350.00"))
    s = suggest_price(p, snap, RepricingRule())
    assert s.suggested_price == D("1298.99")
    assert s.reason == "beat lowest (Shop X @ 1299.00) by 0.01; floor 1100.00 from cost+10% margin"
    assert s.floor == D("1100.00") and s.ceiling is None
    assert s.current_price == D("1400.00")
    assert s.strategy == "beat_lowest"
    assert s.change_percent == -7.21  # (1298.99 - 1400) / 1400 = -7.2150 %


def test_beat_lowest_by_percent() -> None:
    s = suggest_price(
        product(our_price=D("1300")),
        snapshot_of(offer("Shop X", "1299.00")),
        RepricingRule(undercut_percent=D("5")),
    )
    assert s.suggested_price == D("1234.05")
    assert s.reason == "beat lowest (Shop X @ 1299.00) by 5% (64.95)"
    assert s.floor is None


def test_zero_undercut_percent_takes_precedence_over_amount() -> None:
    s = suggest_price(
        product(our_price=D("1300")),
        snapshot_of(offer("Shop X", "1299.00")),
        RepricingRule(undercut_percent=D("0")),
    )
    assert s.suggested_price == D("1299.00")


def test_match_lowest_and_median_strategies() -> None:
    snap = snapshot_of(
        offer("a", "100.00"), offer("b", "110.00"), offer("c", "130.00"), our_price="120"
    )
    match = suggest_price(product(our_price=D("120")), snap, RepricingRule(strategy="match_lowest"))
    assert match.suggested_price == D("100.00")
    assert match.reason == "match lowest (a @ 100.00)"

    med = suggest_price(product(our_price=D("120")), snap, RepricingRule(strategy="median"))
    assert med.suggested_price == D("110.00")
    assert med.reason == "median of 3 competitor offers (110.00)"
    assert med.change_percent == -8.33

    even = snapshot_of(offer("a", "100.00"), offer("b", "110.01"))
    assert suggest_price(product(), even, RepricingRule(strategy="median")).suggested_price == D(
        "105.01"
    )


def test_hold_keeps_current_price_but_respects_floor() -> None:
    snap = snapshot_of(offer("a", "50.00"))
    held = suggest_price(product(our_price=D("99.90")), snap, RepricingRule(strategy="hold"))
    assert held.suggested_price == D("99.90")
    assert held.change_percent == 0.0
    assert held.reason == "hold current price (99.90)"

    raised = suggest_price(
        product(our_price=D("99.90"), cost=D("100")), snap, RepricingRule(strategy="hold")
    )
    assert raised.suggested_price == D("110.00")
    assert "raised to floor 110.00 from cost+10% margin" in raised.reason

    unpriced = suggest_price(product(), snap, RepricingRule(strategy="hold"))
    assert unpriced.suggested_price is None and unpriced.change_percent is None


def test_floor_from_cost_margin_clamps_target() -> None:
    s = suggest_price(
        product(our_price=D("1400"), cost=D("1250")),
        snapshot_of(offer("Shop X", "1299.00")),
        RepricingRule(min_margin_percent=D("10")),
    )
    assert s.floor == D("1375.00")
    assert s.suggested_price == D("1375.00")
    assert (
        s.reason
        == "beat lowest (Shop X @ 1299.00) by 0.01; raised to floor 1375.00 from cost+10% margin"
    )


def test_floor_is_highest_of_min_price_and_margin_and_rounds_up() -> None:
    snap = snapshot_of(offer("a", "5.00"))
    by_min_price = suggest_price(
        product(cost=D("1000"), min_price=D("1200")), snap, RepricingRule()
    )
    assert str(by_min_price.floor) == "1200.00" and by_min_price.suggested_price == D("1200.00")
    assert "raised to floor 1200.00 from min_price" in by_min_price.reason

    rounded = suggest_price(
        product(cost=D("9.99")), snap, RepricingRule(min_margin_percent=D("12.5"))
    )
    assert rounded.floor == D("11.24")  # 9.99 * 1.125 = 11.23875, rounded up
    assert "cost+12.5% margin" in rounded.reason


def test_ceiling_caps_target() -> None:
    s = suggest_price(
        product(our_price=D("1400"), max_price=D("1500")),
        snapshot_of(offer("a", "2000.00")),
        RepricingRule(strategy="match_lowest"),
    )
    assert str(s.suggested_price) == "1500.00"
    assert str(s.ceiling) == "1500.00"
    assert s.reason == "match lowest (a @ 2000.00); capped at ceiling 1500.00 (max_price)"


def test_floor_wins_over_ceiling() -> None:
    s = suggest_price(
        product(our_price=D("1050"), cost=D("1000"), max_price=D("1000")),
        snapshot_of(offer("a", "900.00")),
        RepricingRule(),
    )
    assert s.suggested_price == D("1100.00")
    assert "floor 1100.00 (cost+10% margin) exceeds ceiling 1000.00; floor wins" in s.reason


@pytest.mark.parametrize(
    ("lowest", "ending", "expected"),
    [
        ("1299.00", "0.99", "1298.99"),  # target 1298.99 already ends in .99
        ("1299.50", "0.99", "1298.99"),  # target 1299.49 -> largest .99 below it
        ("1299.50", "0.90", "1298.90"),
        ("1300.00", "0.00", "1299.00"),
        ("1300.00", "0.5", "1299.50"),
        ("1300.00", "9.99", "1299.99"),  # only the fractional part matters
    ],
)
def test_price_ending(lowest: str, ending: str, expected: str) -> None:
    s = suggest_price(
        product(our_price=D("1400")),
        snapshot_of(offer("a", lowest)),
        RepricingRule(price_ending=D(ending)),
    )
    assert s.suggested_price == D(expected)
    assert str(s.suggested_price) == expected
    assert "price ending" in s.reason


def test_price_ending_rounds_up_when_below_floor() -> None:
    s = suggest_price(
        product(our_price=D("1400"), min_price=D("1300.00")),
        snapshot_of(offer("a", "1299.00")),
        RepricingRule(price_ending=D("0.99")),
    )
    assert s.suggested_price == D("1300.99")
    assert s.reason.endswith("price ending 0.99")


def test_price_ending_dropped_when_it_cannot_fit() -> None:
    s = suggest_price(
        product(our_price=D("1400"), min_price=D("1300.00"), max_price=D("1300.50")),
        snapshot_of(offer("a", "1299.00")),
        RepricingRule(price_ending=D("0.99")),
    )
    assert s.suggested_price == D("1300.00")
    assert "no price ending in 0.99 fits between floor and ceiling" in s.reason


def test_price_ending_never_goes_negative() -> None:
    s = suggest_price(
        product(our_price=D("1")),
        snapshot_of(offer("a", "0.50")),
        RepricingRule(price_ending=D("0.99"), min_margin_percent=D("0")),
    )
    assert s.suggested_price == D("0.99")


def test_no_considered_offers_keeps_current_price() -> None:
    p = product(our_price=D("100.00"), cost=D("200"))
    oos = snapshot_of(offer("a", "80.00", OUT), offer("b", "70.00", score=0.5))
    s = suggest_price(p, oos, RepricingRule())
    assert s.suggested_price == D("100.00")
    assert s.reason == "no competitor data"
    assert s.change_percent == 0.0
    assert s.floor == D("220.00")

    empty = suggest_price(product(), snapshot_of(our_price=None), RepricingRule())
    assert empty.suggested_price is None and empty.change_percent is None


def test_out_of_stock_considered_when_not_ignored() -> None:
    snap = snapshot_of(offer("a", "80.00", OUT), offer("b", "90.00"))
    ignored = suggest_price(
        product(our_price=D("100")), snap, RepricingRule(strategy="match_lowest")
    )
    assert ignored.suggested_price == D("90.00")
    kept = suggest_price(
        product(our_price=D("100")),
        snap,
        RepricingRule(strategy="match_lowest", ignore_out_of_stock=False),
    )
    assert kept.suggested_price == D("80.00")


def test_match_score_filter_in_suggestion() -> None:
    snap = snapshot_of(offer("wrong-variant", "50.00", score=0.6), offer("b", "90.00", score=0.9))
    s = suggest_price(product(our_price=D("100")), snap, RepricingRule(strategy="match_lowest"))
    assert s.suggested_price == D("90.00")
    loose = suggest_price(
        product(our_price=D("100")),
        snap,
        RepricingRule(strategy="match_lowest", min_match_score=0.5),
    )
    assert loose.suggested_price == D("50.00")


def test_end_to_end_multi_currency() -> None:
    settings = Settings(
        currency_rates={"RON": D("0.2")}, base_currency="EUR", include_shipping=True
    )
    p = product(our_price=D("260.00"), cost=D("200"))
    snap = market_snapshot(
        p,
        [
            obs("emag", "1249.95", currency="RON", shipping="15.00"),  # (1264.95) * 0.2 = 252.99
            obs("altex", "255.00", currency="EUR"),
            obs("flanco", "1000.00", currency="RON", availability=OUT),
        ],
        settings,
    )
    assert snap.lowest == D("252.99") and snap.lowest_competitor == "emag"
    assert snap.competitors_cheaper == 2 and snap.our_rank == 3
    s = suggest_price(p, snap, RepricingRule(price_ending=D("0.99")))
    assert s.suggested_price == D("251.99")
    assert s.reason == (
        "beat lowest (emag @ 252.99) by 0.01; floor 220.00 from cost+10% margin; price ending 0.99"
    )
    assert s.change_percent == -3.08
