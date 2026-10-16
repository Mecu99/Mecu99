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


# --------------------------------------------------------------------------- generations


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Apple AirPods Pro (2nd generation)", "apple airpods pro 2"),
        ("AirPods Pro 2. Generation", "airpods pro 2"),
        ("AirPods Pro 2e génération", "airpods pro 2"),
        ("AirPods Pro generatia a 2-a", "airpods pro 2"),
        ("Canon EOS R6 Mark II", "canon eos r6 mark 2"),
        ("Canon EOS R6 MkII", "canon eos r6 mark 2"),
        ("Sony Alpha A7 IV", "sony alpha a7 4"),
        ("LEGO Technic 42143, 18+", "lego technic 42143 18"),
        ("Joc Monopoly 8+ ani", "joc monopoly 8 ani"),
        ("LEGO Icons, Ages 18+", "lego icons ages 18"),
        ("Samsung Galaxy Note 10+", "samsung galaxy note 10 plus"),
        ("Bosch WAN28263BY 1400 RPM", "bosch wan28263by 1400rpm"),
        ("garantie 24 luni", "garantie 24mo"),
        ("Szkło hartowane", "szklo hartowane"),
    ],
)
def test_normalize_text_generations_ages_and_counts(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_bottles_and_cans_are_pack_counts() -> None:
    assert spec_tokens("Coca-Cola bax 12 sticle 0.5L") == {"pcs": {"12"}, "l": {"0.5"}}
    assert spec_tokens("Bere Ursus 6 doze 0.5L") == {"pcs": {"6"}, "l": {"0.5"}}


# --------------------------------------------------------------------------- used and refurbished


S24 = Product(sku="S24", name="Samsung Galaxy S24 128GB Onyx Black", brand="Samsung")


@pytest.mark.parametrize(
    "condition",
    [
        "- Pre-owned", "- Grade A", "Grad B", "Demo unit", "- Ex-display", "(Reconditioned)",
        "- Like New", "Open Box", "B-Stock", "Second-hand",
        "Gebraucht", "Generalüberholt", "B-Ware", "wie neu",  # DE
        "Reconditionné", "Remis à neuf", "d'occasion",  # FR
        "Reacondicionado", "Seminuevo", "Usado",  # ES
        "Ricondizionato", "Usato", "Rigenerato",  # IT
        "odnowiony", "używany", "powystawowy",  # PL
        "Folosit", "Ca Nou", "(Reînnoit)", "Resigilat", "Recondiționat",  # RO
    ],
)  # fmt: skip
def test_used_refurbished_and_demo_items_rejected(condition: str) -> None:
    result = score_match(S24, offer(f"Samsung Galaxy S24 128GB Onyx Black {condition}"))
    assert result.score < THRESHOLD, result
    assert result.details["condition_penalty"] == 0.6


def test_refurbished_product_of_ours_matches_refurbished_offers() -> None:
    ours = S24.model_copy(update={"name": "Samsung Galaxy S24 128GB Onyx Black Refurbished"})
    result = score_match(ours, offer("Samsung Galaxy S24 128GB Onyx Black - Renewed"))
    assert "condition_penalty" not in result.details
    assert result.score >= THRESHOLD


# --------------------------------------------------------------------------- accessories and spare parts

XM5 = Product(sku="XM5", name="Sony WH-1000XM5", brand="Sony")
V15 = Product(sku="V15", name="Dyson V15 Detect Absolute", brand="Dyson")
AIRPODS = Product(sku="APP2", name="Apple AirPods Pro 2", brand="Apple")


@pytest.mark.parametrize(
    ("product", "title"),
    [
        (XM5, "Etui Sony WH-1000XM5"),
        (XM5, "Hülle für Sony WH-1000XM5"),
        (XM5, "Pernute Sony WH-1000XM5"),
        (XM5, "Ohrpolster Sony WH-1000XM5"),
        (XM5, "Sony WH-1000XM5 Ersatzpolster"),
        (XM5, "Geanta transport Sony WH-1000XM5"),
        (AIRPODS, "Coque Apple AirPods Pro 2"),
        (AIRPODS, "Ear tips Apple AirPods Pro 2"),
        (AIRPODS, "Charging Case for Apple AirPods Pro 2"),
        (V15, "Filtru Dyson V15 Detect Absolute"),
        (V15, "Acumulator Dyson V15 Detect Absolute"),
        (V15, "Akku Dyson V15 Detect Absolute"),
        (V15, "Perie Dyson V15 Detect Absolute"),
        (V15, "Dyson V15 Detect Absolute Replacement Battery"),
        (IPHONE, "Szkło hartowane do Apple iPhone 15"),
        (IPHONE, "Ładowarka do Apple iPhone 15"),
        (IPHONE, "Funda Apple iPhone 15 128GB negro"),
    ],
)
def test_multilingual_accessories_and_spare_parts_rejected(product: Product, title: str) -> None:
    result = score_match(product, offer(title))
    assert result.score < THRESHOLD, result
    assert result.details["accessory_penalty"] == 0.6


@pytest.mark.parametrize(
    ("product", "title"),
    [
        # Part words further into the title describe the product itself.
        (V15, "Aspirator vertical fara fir Dyson V15 Detect Absolute, acumulator, 60 min autonomie"),
        (V15, "Dyson V15 Detect Absolute Cordless Vacuum, HEPA filter, 60 min battery"),
        (XM5, "Sony WH-1000XM5 Wireless Headphones, 30 Hours Battery, Black"),
        (XM5, "Sony WH-1000XM5 Kabellose Bluetooth Noise Cancelling Kopfhörer, schwarz"),
        (AIRPODS, "Apple AirPods Pro (2nd generation) with MagSafe Charging Case (USB-C)"),
        (
            Product(sku="W9", name="Apple Watch Series 9 GPS 41mm", brand="Apple"),
            "Apple Watch Series 9 GPS 41mm Midnight Aluminium Case with Midnight Sport Band",
        ),
        (
            Product(sku="PS5", name="Sony PlayStation 5 Slim", brand="Sony"),
            "Consola de jocuri Sony PlayStation 5 Slim, 1TB SSD",
        ),
    ],
)
def test_part_words_that_do_not_lead_the_title_are_specs(product: Product, title: str) -> None:
    result = score_match(product, offer(title))
    assert "accessory_penalty" not in result.details
    assert result.score >= THRESHOLD, result


def test_part_word_leading_our_own_name_is_not_penalised() -> None:
    brita = Product(sku="BRITA", name="Filtru Brita Maxtra Plus", brand="Brita")
    result = score_match(brita, offer("Filtru de apa Brita Maxtra Plus"))
    assert "accessory_penalty" not in result.details
    assert result.score >= THRESHOLD


# --------------------------------------------------------------------------- sibling models


@pytest.mark.parametrize(
    ("ours", "brand", "theirs"),
    [
        ("Microsoft Xbox Series X", "Microsoft", "Microsoft Xbox Series S"),
        ("Logitech MX Master 3S", "Logitech", "Logitech MX Master 3"),
        ("Logitech MX Master 3", "Logitech", "Logitech MX Master 3S"),
        ("Garmin Fenix 7", "Garmin", "Garmin Fenix 7X Solar"),
        ("Google Pixel 8 128GB", "Google", "Google Pixel 8a 128GB"),
        ("Xiaomi 14 512GB", "Xiaomi", "Xiaomi 14T 512GB"),
        ("Apple AirPods Pro (2nd generation)", "Apple", "Apple AirPods Pro (1st generation)"),
        ("Apple AirPods Pro 2", "Apple", "Apple AirPods Pro (1st generation)"),
        ("Sony Alpha A7 IV Body", "Sony", "Sony Alpha A7 III Body"),
        ("Canon EOS 5D Mark IV", "Canon", "Canon EOS 5D Mark III"),
    ],
)
def test_sibling_model_numbers_and_generations_conflict(ours: str, brand: str, theirs: str) -> None:
    result = score_match(Product(sku="x", name=ours, brand=brand), offer(theirs))
    assert result.reason == "model_conflict", result
    assert result.score <= 0.45


@pytest.mark.parametrize(
    ("ours", "brand", "theirs"),
    [
        ("Nintendo Switch OLED", "Nintendo", "Nintendo Switch"),
        ("Nintendo Switch", "Nintendo", "Nintendo Switch OLED"),
        ("Sony PlayStation 5 Console", "Sony", "Sony PlayStation 5 Digital Edition Console"),
        ("Canon EOS R6 Mark II", "Canon", "Canon EOS R6"),
        ("Apple Watch Series 9 GPS 41mm", "Apple", "Apple Watch Series 9 GPS + Cellular 41mm"),
        ("Apple iPad Air 11 M2 Wi-Fi 128GB", "Apple", "Apple iPad Air 11 M2 Wi-Fi + Cellular 128GB"),
        ("Samsung Galaxy Tab S9 WiFi 128GB", "Samsung", "Samsung Galaxy Tab S9 5G 128GB"),
        ("GoPro HERO12 Black", "GoPro", "GoPro HERO12 Black Creator Edition"),
    ],
)
def test_edition_and_connectivity_siblings_rejected(ours: str, brand: str, theirs: str) -> None:
    result = score_match(Product(sku="x", name=ours, brand=brand), offer(theirs))
    assert result.score < THRESHOLD, result
    assert result.details["variant_penalty"] == 0.7


@pytest.mark.parametrize(
    ("ours", "brand", "theirs"),
    [
        ("Apple AirPods Pro 2", "Apple", "Apple AirPods Pro (2nd generation)"),
        ("Canon EOS R6 Mark II Body", "Canon", "Canon EOS R6 MkII Body"),
        ("Logitech MX Master 3S", "Logitech", "Mouse Logitech MX Master 3S, Wireless, Graphite"),
        ("Garmin Fenix 7X Solar", "Garmin", "Ceas smartwatch Garmin Fenix 7X Solar, 51mm, Slate Gray"),
        ("Microsoft Xbox Series X", "Microsoft", "Consola Microsoft Xbox Series X 1TB, Negru"),
        ("Samsung Galaxy Note 10+ 256GB", "Samsung", "Samsung Galaxy Note10+ 256GB Aura Glow"),
        ("Samsung Galaxy Tab S9 5G 128GB", "Samsung", "Tableta Samsung Galaxy Tab S9, 128GB, 8GB RAM, 5G"),
        ("TP-Link Archer C6 WiFi Router", "TP-Link", "Router TP-Link Archer C6, Dual Band 2.4G/5G WiFi"),
        # The exact model code outweighs how a shop words the variant.
        ("Samsung QE55S95C", "Samsung", "Televizor Samsung OLED QE55S95C, 138 cm, Smart, 4K"),
        ("Samsung Galaxy S24 Ultra SM-S928B 256GB", "Samsung", "Samsung Galaxy S24 SM-S928B 256GB Titanium"),
    ],
)
def test_same_model_in_other_words_accepted(ours: str, brand: str, theirs: str) -> None:
    result = score_match(Product(sku="x", name=ours, brand=brand), offer(theirs))
    assert result.score >= THRESHOLD, result


# --------------------------------------------------------------------------- bundles


PS5 = Product(sku="PS5", name="Sony PlayStation 5 Slim", brand="Sony")
R6_BODY = Product(sku="R6", name="Canon EOS R6 Mark II Body", brand="Canon")


@pytest.mark.parametrize(
    ("product", "title"),
    [
        (PS5, "Consola Sony PlayStation 5 Slim + 2 jocuri"),
        (PS5, "Pachet Consola Sony PlayStation 5 Slim cu controller suplimentar"),
        (PS5, "Sony PlayStation 5 Slim + EA Sports FC 25 Bundle"),
        (R6_BODY, "Canon EOS R6 Mark II Kit RF 24-105mm"),
        (R6_BODY, "Canon EOS R6 Mark II + RF 24-105mm"),
    ],
)
def test_bundles_rejected(product: Product, title: str) -> None:
    result = score_match(product, offer(title))
    assert result.score < THRESHOLD, result
    assert result.details["bundle_penalty"] == 0.6


@pytest.mark.parametrize(
    ("product", "title", "accepted"),
    [
        (COLA.model_copy(update={"name": "Coca-Cola Zero 0.5L"}), "Coca-Cola Zero bax 12 sticle 0.5L", False),
        (COLA, "Pachet 6 sticle Coca-Cola Zero 0.5L", True),  # our own 6-pack
        (Product(sku="KK", name="KitKat Chunky 40g", brand="Nestle"), "Nestle Kit Kat Chunky 40g", True),
        (Product(sku="TVB", name="Xiaomi Mi TV Box S", brand="Xiaomi"), "Set-top box Xiaomi Mi TV Box S", True),
    ],
)
def test_pack_words_against_pack_sizes(product: Product, title: str, accepted: bool) -> None:
    assert (score_match(product, offer(title)).score >= THRESHOLD) is accepted


# --------------------------------------------------------------------------- spaced codes


@pytest.mark.parametrize(
    ("ours", "brand", "theirs"),
    [
        ("Samsung QE55Q80C", "Samsung", "Televizor Samsung QE 55Q80C, 138 cm, QLED"),
        ("LG OLED55C3", "LG", "Televizor LG OLED 55C3, 139 cm, 4K"),
        ("Sony WH-1000XM5", "Sony", "Sony WH 1000 XM5 wireless"),
        ("DeLonghi Magnifica S ECAM22.110.B", "DeLonghi", "Espressor DeLonghi Magnifica S ECAM 22.110.B"),
        ("Bosch Serie 4 WAN28263BY", "Bosch", "Masina de spalat Bosch WAN28263BY, 8 kg, 1400 RPM, Clasa B"),
        ("LEGO Technic 42143 Ferrari Daytona SP3", "LEGO", "LEGO Technic Ferrari Daytona SP3 42143, 18+"),
        ("LEGO Technic 42143", "LEGO", "Set LEGO Technic 42143, 3778 piese, 18+"),
        ("Samsung Galaxy S24 128GB", "Samsung", "Samsung Galaxy S24 128GB, garantie 24 luni"),
    ],
)
def test_spaced_codes_and_spec_numbers_still_match(ours: str, brand: str, theirs: str) -> None:
    result = score_match(Product(sku="x", name=ours, brand=brand), offer(theirs))
    assert result.score >= THRESHOLD, result


@pytest.mark.parametrize(
    ("ours", "brand", "theirs"),
    [
        ("LG OLED55C3", "LG", "Televizor LG OLED 65C3, 164 cm, 4K"),
        ("Samsung QE55Q80C", "Samsung", "Televizor Samsung QE 65Q80C, 163 cm"),
        ("Sony WH-1000XM5", "Sony", "Sony WH 1000 XM4 wireless"),
    ],
)
def test_spaced_codes_of_other_models_still_rejected(ours: str, brand: str, theirs: str) -> None:
    assert score_match(Product(sku="x", name=ours, brand=brand), offer(theirs)).score < THRESHOLD


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
