"""Tests for product matching: identifiers, text normalisation and wrong-variant guards."""

from __future__ import annotations

import pytest

from pricehunter.match import (
    model_tokens,
    normalize_gtin,
    normalize_text,
    rank_offers,
    score_match,
    spec_tokens,
)
from pricehunter.models import ExtractedOffer, MatchResult, Product

THRESHOLD = 0.80  # Settings.match_threshold default


def offer(
    title: str | None = None,
    *,
    brand: str | None = None,
    gtin: str | None = None,
    mpn: str | None = None,
) -> ExtractedOffer:
    return ExtractedOffer(url="https://shop.example/p/1", title=title, brand=brand, gtin=gtin, mpn=mpn)


IPHONE = Product(sku="IP15-128-BLK", name="iPhone 15 128GB Black", brand="Apple")
SONY = Product(
    sku="WH1000XM5",
    name="Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black",
    brand="Sony",
)
COLA = Product(sku="COLA-ZERO-6", name="Coca-Cola Zero 6 x 0.5L", brand="Coca-Cola")
MILK = Product(sku="ZUZU-35", name="Lapte Zuzu 3.5% grasime 1L", brand="Zuzu")
NIKE = Product(sku="AF1", name="Nike Air Force 1 '07 White", brand="Nike", mpn="CW2288-111")
SSD = Product(sku="990PRO-1", name="Samsung 990 PRO 1TB NVMe M.2 SSD", brand="Samsung", mpn="MZ-V9P1T0BW")


# --------------------------------------------------------------------------- GTIN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4006381333931", "04006381333931"),  # EAN-13
        ("036000291452", "00036000291452"),  # UPC-A
        ("96385074", "00000096385074"),  # GTIN-8
        ("10012345678902", "10012345678902"),  # GTIN-14
        ("04006381333931", "04006381333931"),  # already padded
        (" 4006381333931\n", "04006381333931"),
        ("400-6381-333931", "04006381333931"),
        ("4006 3813 3393 1", "04006381333931"),
        ("EAN: 4006381333931", "04006381333931"),
    ],
)
def test_normalize_gtin_valid(raw: str, expected: str) -> None:
    assert normalize_gtin(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "abc",
        "4006381333932",  # bad check digit
        "036000291453",  # bad check digit (UPC-A)
        "96385075",  # bad check digit (GTIN-8)
        "123",  # too short
        "1234567890",  # 10 digits is not a GTIN length
        "400638133393100",  # 15 digits
        "00000000",  # all zeros passes the checksum but is not an item
    ],
)
def test_normalize_gtin_invalid(raw: str | None) -> None:
    assert normalize_gtin(raw) is None


def test_upc_and_ean_of_same_item_normalize_equal() -> None:
    assert normalize_gtin("036000291452") == normalize_gtin("0036000291452")


# --------------------------------------------------------------------------- text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Apple iPhone 15 128 GB", "apple iphone 15 128gb"),
        ("SSD 1 TB", "ssd 1tb"),
        ('Display 6.1"', "display 6.1in"),
        ("Display 6.1 inch", "display 6.1in"),
        ("Display 6,1 inch", "display 6.1in"),
        ("Apa 500 ml", "apa 500ml"),
        ("Faina 2 kg", "faina 2kg"),
        ("Apa 0,5 L", "apa 0.5l"),
        ("Coca-Cola 6 x 0.5L", "coca cola 6x 0.5l"),
        ("Coca-Cola 0.5L x 6", "coca cola 0.5l 6x"),
        ("Mașină de spălat Bosch Serie|4", "masina de spalat bosch serie 4"),
        ("Lapte 3,5% grăsime", "lapte 3.5pct grasime"),
        ("Baterie 5,000 mAh", "baterie 5000mah"),
        ("Telefon, (Negru)!", "telefon black"),
        ("Sony WH-1000XM5", "sony wh1000xm5"),
        ("Galaxy S24+", "galaxy s24 plus"),
        ("  lots   of\tspace ", "lots of space"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_keeps_hyphenated_words_apart() -> None:
    assert normalize_text("Coca-Cola") == "coca cola"
    assert normalize_text("Galaxy-S24") == "galaxy s24"


def test_model_tokens_strip_separators() -> None:
    assert "wh1000xm5" in model_tokens("Sony WH-1000XM5 Black")
    assert "sms918b" in model_tokens("Samsung Galaxy S23 Ultra SM-S918B")
    assert "i713700k" in model_tokens("Intel Core i7-13700K")
    assert "s2721ds" in model_tokens("Dell S2721DS")


@pytest.mark.parametrize(
    "text",
    [
        "128GB 256 GB 1TB",
        "5G 4K 8K 1080p",
        "Model 2024",
        "WiFi6 USB3 IP68 DDR5 Bluetooth 5.3",
        "Apa Borsec 6x2L",
        "Monitor 1920x1080",
        "Telefon mobil negru",
        "S24",  # too short to be a reliable identifier
    ],
)
def test_model_tokens_exclude_units_numbers_and_standards(text: str) -> None:
    assert model_tokens(text) == set()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("iPhone 15 128 GB", {"gb": {"128"}}),
        ("SSD 1TB", {"tb": {"1"}}),
        ('TV Samsung 55" 4K 120 Hz', {"in": {"55"}, "hz": {"120"}}),
        ("Telefon 8GB RAM 256GB 5000 mAh 25W", {"gb": {"8", "256"}, "mah": {"5000"}, "w": {"25"}}),
        ("Coca-Cola 6 x 0.5L", {"x": {"6"}, "l": {"0.5"}}),
        ("Detergent 2 kg", {"kg": {"2"}}),
        ("Sampon 400 ml", {"ml": {"400"}}),
        ("Telefon mobil", {}),
    ],
)
def test_spec_tokens(text: str, expected: dict[str, set[str]]) -> None:
    assert spec_tokens(text) == expected


# --------------------------------------------------------------------------- identifiers


def test_equal_gtin_in_different_formats_is_a_match() -> None:
    product = IPHONE.model_copy(update={"gtin": "0195949036323"})  # EAN-13
    result = score_match(product, offer("Something with a different title", gtin="00195949036323"))
    assert result == MatchResult(score=1.0, reason="gtin", details={"gtin": 1.0})


def test_different_gtin_is_a_strong_reject_even_with_identical_title() -> None:
    product = IPHONE.model_copy(update={"gtin": "0195949036323"})
    result = score_match(product, offer("Apple iPhone 15 128GB Black", gtin="5901234123457"))
    assert result.reason == "gtin_conflict"
    assert result.score == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [("0195949036324", "0195949036323"), ("0195949036323", "0195949036324")],
)
def test_invalid_gtin_is_ignored_and_title_decides(ours: str, theirs: str) -> None:
    product = IPHONE.model_copy(update={"gtin": ours})
    result = score_match(product, offer("Apple iPhone 15 128GB Black", gtin=theirs))
    assert result.reason == "title"
    assert result.score >= THRESHOLD


def test_equal_mpn_with_different_formatting_is_a_match() -> None:
    result = score_match(SSD, offer("SSD Samsung 990 PRO 1TB", mpn="mz v9p1t0bw"))
    assert (result.score, result.reason) == (0.97, "mpn")


def test_equal_mpn_but_conflicting_specs_is_rejected() -> None:
    result = score_match(SSD, offer("SSD Samsung 990 PRO 2TB", mpn="MZ-V9P1T0BW"))
    assert result.reason == "spec_conflict"
    assert result.score <= 0.4


def test_different_mpn_caps_score() -> None:
    result = score_match(NIKE, offer("Pantofi sport Nike Air Force 1 '07, Negru", mpn="CW2288-001"))
    assert result.reason == "mpn_conflict"
    assert result.score <= 0.3


def test_mpn_with_regional_suffix_is_not_a_conflict() -> None:
    product = IPHONE.model_copy(update={"mpn": "MTP03"})
    result = score_match(product, offer("Apple iPhone 15 128GB Black", mpn="MTP03ZD/A"))
    assert result.reason == "title"
    assert result.score >= THRESHOLD


def test_our_mpn_inside_their_title_counts() -> None:
    product = SONY.model_copy(update={"mpn": "WH1000XM5B"})
    result = score_match(product, offer("Casti wireless Sony WH1000XM5B negru"))
    assert result.details["mpn_in_title"] == 1.0
    assert result.score >= THRESHOLD


# --------------------------------------------------------------------------- electronics


@pytest.mark.parametrize(
    "title",
    [
        "Apple iPhone 15, 128GB, Black",
        "Telefon mobil Apple iPhone 15, 128GB, 5G, Negru",
        "Black 128GB iPhone 15 Apple",  # same words, different order
        "APPLE IPHONE 15 128 GB BLACK",
    ],
)
def test_iphone_same_variant_accepted(title: str) -> None:
    result = score_match(IPHONE, offer(title))
    assert result.reason == "title"
    assert result.score >= THRESHOLD


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("Apple iPhone 15 256GB Black", "spec_conflict"),
        ("Telefon mobil Apple iPhone 15, 256GB, 5G, Negru", "spec_conflict"),
        ("Apple iPhone 14 128GB Black", "model_conflict"),
        ("Apple iPhone 15 Pro 128GB Black", "title"),
        ("Apple iPhone 15 Plus 128GB Black", "title"),
        ("Husa iPhone 15 silicon neagra", "title"),
        ("Husa Apple iPhone 15 128GB silicon", "title"),
        ("Folie sticla securizata pentru iPhone 15", "title"),
        ("Incarcator rapid compatibil iPhone 15", "title"),
        ("Apple iPhone 15 128GB Black - Resigilat", "title"),
        ("Samsung Galaxy S24 128GB Black", "title"),
    ],
)
def test_iphone_wrong_items_rejected(title: str, reason: str) -> None:
    result = score_match(IPHONE, offer(title))
    assert result.score < THRESHOLD, result
    assert result.reason == reason


def test_storage_conflict_is_capped_with_reason() -> None:
    result = score_match(IPHONE, offer("Apple iPhone 15 256GB Black"))
    assert result.score <= 0.4
    assert result.details["spec_conflict"] == 1.0


def test_accessory_penalty_is_reported() -> None:
    result = score_match(IPHONE, offer("Husa Apple iPhone 15 128GB silicon"))
    assert result.details["accessory_penalty"] == 0.6


def test_accessory_words_on_both_sides_are_not_penalised() -> None:
    case = Product(sku="SPG", name="Husa Spigen Ultra Hybrid iPhone 15 Transparent", brand="Spigen")
    same = score_match(case, offer("Husa Spigen Ultra Hybrid pentru iPhone 15, Crystal Clear transparenta"))
    other_phone = score_match(case, offer("Husa Spigen Ultra Hybrid pentru iPhone 15 Pro, transparenta"))
    assert "accessory_penalty" not in same.details
    assert same.score >= THRESHOLD
    assert other_phone.score < THRESHOLD


def test_generic_for_is_not_an_accessory() -> None:
    shampoo = Product(sku="HS", name="Sampon Head & Shoulders Classic Clean 400ml", brand="Head & Shoulders")
    result = score_match(
        shampoo, offer("Sampon antimatreata Head & Shoulders Classic Clean pentru par normal, 400 ml")
    )
    assert "accessory_penalty" not in result.details
    assert result.score >= THRESHOLD


def test_missing_brand_is_penalised() -> None:
    with_brand = score_match(IPHONE, offer("Apple iPhone 15 128GB Black"))
    without = score_match(IPHONE, offer("iPhone 15 128GB Black"))
    assert without.details["brand_penalty"] == 0.75
    assert without.score < with_brand.score
    # Brand given as structured data instead of in the title is fine.
    assert "brand_penalty" not in score_match(IPHONE, offer("iPhone 15 128GB Black", brand="Apple")).details


@pytest.mark.parametrize(
    "title",
    [
        "Casti Sony WH-1000XM5, Noise Cancelling, Bluetooth, Negru",
        "Headphones Black Wireless Noise Cancelling Sony WH1000XM5",
    ],
)
def test_sony_same_model_accepted_with_model_bonus(title: str) -> None:
    result = score_match(SONY, offer(title))
    assert result.details["model_bonus"] == 0.1
    assert result.score >= THRESHOLD


@pytest.mark.parametrize(
    "title",
    [
        "Sony WH-1000XM4 Wireless Noise Cancelling Headphones Black",
        "Casti audio Sony WH-1000XM4, Negru",
    ],
)
def test_sony_previous_generation_rejected(title: str) -> None:
    result = score_match(SONY, offer(title))
    assert result.reason == "model_conflict"
    assert result.score <= 0.45


def test_model_conflict_from_offer_mpn_without_our_mpn() -> None:
    result = score_match(SONY, offer("Sony Headphones Black", mpn="WH1000XM4"))
    assert result.reason == "model_conflict"


COFFEE = Product(
    sku="ECAM22",
    name="De'Longhi Magnifica S ECAM22.110.B Coffee Machine",
    brand="De'Longhi",
    mpn="ECAM22.110.B",
)


@pytest.mark.parametrize(
    "title",
    [
        "delonghi magnifica s ecam22 110 b",  # sitemap slug words
        "DeLonghi Magnifica S ECAM22-110-B espresso",
        "DeLonghi Magnifica S ECAM22.110.B Bean to Cup Coffee Machine",
    ],
)
def test_dotted_model_code_matches_other_spellings(title: str) -> None:
    assert score_match(COFFEE, offer(title)).score >= THRESHOLD


def test_dotted_model_code_still_conflicts_with_a_different_code() -> None:
    result = score_match(COFFEE, offer("DeLonghi Magnifica S ECAM22.120.B Coffee Machine"))
    assert result.reason == "model_conflict"
    assert result.score < THRESHOLD


def test_offer_mpn_does_not_dilute_title_when_we_have_none() -> None:
    result = score_match(SONY, offer("Sony WH-1000XM5 Black", mpn="WH1000XM5B.CE7"))
    assert result.score >= THRESHOLD


def test_sony_accessory_for_the_model_rejected() -> None:
    result = score_match(SONY, offer("Husa pentru casti Sony WH-1000XM5"))
    assert result.score < THRESHOLD
    assert result.details["accessory_penalty"] == 0.6


@pytest.mark.parametrize(
    ("title", "accepted"),
    [
        ("SSD Samsung 990 PRO 1000GB NVMe M.2", True),  # 1 TB == 1000 GB
        ("SSD Samsung 990 PRO, 2TB, PCIe 4.0 NVMe M.2", False),
    ],
)
def test_capacity_units_compare_across_tb_and_gb(title: str, accepted: bool) -> None:
    product = SSD.model_copy(update={"mpn": None})
    assert (score_match(product, offer(title)).score >= THRESHOLD) is accepted


def test_cpu_sibling_rejected() -> None:
    cpu = Product(sku="CPU", name="Intel Core i7-13700K", brand="Intel")
    assert score_match(cpu, offer("Procesor Intel Core i7-13700K, 3.4GHz, LGA1700")).score >= THRESHOLD
    assert score_match(cpu, offer("Procesor Intel Core i9-13900K, 3.0GHz, LGA1700")).score < THRESHOLD


# --------------------------------------------------------------------------- grocery


@pytest.mark.parametrize(
    ("title", "accepted"),
    [
        ("Bax Coca-Cola Zero 0.5L x 6 bucati", True),
        ("Coca Cola Zero Zahar, 6 x 500 ml", True),
        ("Coca-Cola Zero 0.5L", False),  # single bottle, not the 6-pack
        ("Coca-Cola Zero doza 330ml", False),
        ("Coca-Cola Zero 2L", False),
    ],
)
def test_grocery_pack_and_size(title: str, accepted: bool) -> None:
    assert (score_match(COLA, offer(title)).score >= THRESHOLD) is accepted


def test_grocery_romanian_decimal_comma() -> None:
    assert score_match(MILK, offer("Lapte de consum Zuzu 3,5% grăsime, 1 l")).score >= THRESHOLD
    rejected = score_match(MILK, offer("Lapte Zuzu 1.5% grasime 1L"))
    assert rejected.reason == "spec_conflict"


# --------------------------------------------------------------------------- fashion


def test_fashion_matching() -> None:
    assert score_match(NIKE, offer("Pantofi sport Nike Air Force 1 '07, Alb", mpn="cw2288-111")).reason == "mpn"
    assert score_match(NIKE, offer("Sireturi pentru Nike Air Force 1")).score < THRESHOLD

    jeans = Product(sku="501", name="Levi's 501 Original Fit Jeans W32 L32", brand="Levi's")
    assert score_match(jeans, offer("Blugi Levi's 501 Original Fit barbati, W32 L32")).score >= THRESHOLD
    assert score_match(jeans, offer("Blugi Levi's 505 Regular Fit barbati")).score < THRESHOLD


# --------------------------------------------------------------------------- invariants


def test_offer_without_title_or_ids_scores_zero() -> None:
    assert score_match(IPHONE, offer(None)) == MatchResult(score=0.0, reason="no_title", details={})


def test_details_carry_component_scores() -> None:
    details = score_match(IPHONE, offer("Apple iPhone 15 128GB Black")).details
    assert {"title", "token_set", "token_sort", "partial"} <= details.keys()
    assert all(0.0 <= v <= 1.0 for v in details.values())


@pytest.mark.parametrize(
    "o",
    [
        offer("Apple iPhone 15 128GB Black"),
        offer("Headphones Black Wireless Noise Cancelling Sony WH1000XM5", mpn="WH1000XM5"),
        offer("x"),
        offer("!!!"),
        offer("Husa Husa Husa pentru pentru iPhone iPhone 15 15 Pro Max"),
        offer("Apple", brand="Apple", gtin="not-a-gtin", mpn="??"),
    ],
)
@pytest.mark.parametrize("product", [IPHONE, SONY, COLA, MILK, NIKE, SSD])
def test_score_is_always_in_unit_interval(product: Product, o: ExtractedOffer) -> None:
    result = score_match(product, o)
    assert 0.0 <= result.score <= 1.0


def test_rank_offers_sorts_and_filters() -> None:
    offers = [
        offer("Apple iPhone 15 256GB Black"),
        offer("Telefon mobil Apple iPhone 15, 128GB, 5G, Negru"),
        offer("Husa iPhone 15"),
        offer("Apple iPhone 15, 128GB, Black"),
    ]
    ranked = rank_offers(IPHONE, offers, threshold=THRESHOLD)
    assert [o.title for o, _ in ranked] == [
        "Apple iPhone 15, 128GB, Black",
        "Telefon mobil Apple iPhone 15, 128GB, 5G, Negru",
    ]
    everything = [r.score for _, r in rank_offers(IPHONE, offers, threshold=0.0)]
    assert len(everything) == len(offers)
    assert everything == sorted(everything, reverse=True)
    assert rank_offers(IPHONE, [], threshold=0.5) == []


def test_rank_offers_keeps_input_order_on_ties() -> None:
    a = offer("Apple iPhone 15 128GB Black").model_copy(update={"url": "https://a.example/1"})
    b = offer("Apple iPhone 15 128GB Black").model_copy(update={"url": "https://b.example/1"})
    assert [o.url for o, _ in rank_offers(IPHONE, [a, b], threshold=0.5)] == [a.url, b.url]
