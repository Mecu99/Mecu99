from __future__ import annotations

from decimal import Decimal

import pytest

from pricehunter.pricing import currency_code, detect_currency, parse_decimal, parse_price

D = Decimal


def _check(text: str | None, amount: str | None, currency: str | None, default: str | None = None) -> None:
    got_amount, got_currency = parse_price(text, default)
    if amount is None:
        assert got_amount is None
    else:
        assert isinstance(got_amount, Decimal)
        assert got_amount == D(amount), text
    assert got_currency == currency, text


@pytest.mark.parametrize(
    ("text", "amount", "currency"),
    [
        # European / Anglo separators with symbols, codes and local names
        ("1.234,56 €", "1234.56", "EUR"),
        ("€1,234.56", "1234.56", "EUR"),
        ("1 234,56 lei", "1234.56", "RON"),
        ("RON 1.299,00", "1299.00", "RON"),
        ("$19.99", "19.99", "USD"),
        ("19,99 zł", "19.99", "PLN"),
        ("Preț: 2.499,99 Lei", "2499.99", "RON"),
        ("1'234.50 CHF", "1234.50", "CHF"),
        ("1\u2019234.50 CHF", "1234.50", "CHF"),
        ("CHF 1'234.\u2013", "1234", "CHF"),
        ("Fr. 19.90", "19.90", "CHF"),
        ("USD 5", "5", "USD"),
        ("US$5", "5", "USD"),
        ("C$ 12.50", "12.50", "CAD"),
        ("A$ 99", "99", "AUD"),
        ("R$ 99,90", "99.90", "BRL"),  # not USD although money.py maps a bare "$" to USD
        ("MX$ 1,299", "1299", "MXN"),
        ("NZ$5", "5", "NZD"),
        ("HK$ 12.50", "12.50", "HKD"),
        ("S$9", "9", "SGD"),
        ("Z$ 5", "5", None),  # unknown dollar prefix: no guess
        ("€ 0,99", "0.99", "EUR"),
        ("£1,000", "1000", "GBP"),
        ("¥1,000", "1000", "JPY"),
        ("₹ 2,499", "2499", "INR"),
        ("12.990 Kč", "12990", "CZK"),
        ("1 990 Ft", "1990", "HUF"),
        ("29,90 лв.", "29.90", "BGN"),
        ("EUR1.234,56", "1234.56", "EUR"),
        ("1299lei", "1299", "RON"),
        ("49,90 EUR", "49.90", "EUR"),
        ("BRL 79,90", "79.90", "BRL"),
        ("99,00 €\n  inkl. MwSt.", "99.00", "EUR"),
        # thousands vs decimal ambiguity
        ("1,299", "1299", None),
        ("1.299", "1299", None),
        ("12,999", "12999", None),
        ("123.456", "123456", None),
        ("12,99", "12.99", None),
        ("12.99", "12.99", None),
        ("12.9", "12.9", None),
        ("0.299", "0.299", None),
        ("0,299", "0.299", None),
        ("1234.567", "1234.567", None),
        ("1299", "1299", None),
        ("1.299.000", "1299000", None),
        ("1,234,567.89", "1234567.89", None),
        ("1.234.567,89", "1234567.89", None),
        (".99", "0.99", None),
        (",5", "0.5", None),
        ("1.5", "1.5", None),
        ("19,-", "19", None),
        ("1.299,-", "1299", None),
        ("999999999999999.99", "999999999999999.99", None),
        ("123 456 789,10 lei", "123456789.10", "RON"),
        # ranges, "from" prices and several numbers
        ("From $10", "10", "USD"),
        ("10 - 20 EUR", "10", "EUR"),
        ("10\u201320 €", "10", "EUR"),
        ("$10 - $20", "10", "USD"),
        ("de la 99,99 lei", "99.99", "RON"),
        ("ab 49,90 €", "49.90", "EUR"),
        ("Was €30 now €20", "30", "EUR"),
        # a number next to a currency beats an earlier bare number
        ("iPhone 15 Pro 999 €", "999", "EUR"),
        ("2 x 19,99 lei", "19.99", "RON"),
        ("Model 2024 - €19.99", "19.99", "EUR"),
        ("Model X - 5 €", "5", "EUR"),
        ("19,99 lei (4 €)", "19.99", "RON"),
        # ambiguous local names ("ft" = feet, "tl", "Fr." = from) lose to any other marker
        ("Cable 5 ft $9.99", "9.99", "USD"),
        ("Hose 10 ft - 24,99 lei", "24.99", "RON"),
        ("99 TL", "99", "TRY"),
        ("1 990 Ft", "1990", "HUF"),
        # percentages and dates are never prices
        ("-20% 79,99 lei", "79.99", "RON"),
        ("Save 15%: 85,00 €", "85.00", "EUR"),
        ("Valabil până la 15.10.2026: 19,99 lei", "19.99", "RON"),
        # zero / free
        ("0", "0", None),
        ("0,00 lei", "0.00", "RON"),
        ("Free", "0", None),
        ("FREE", "0", None),
        ("Gratuit", "0", None),
        ("Livrare gratuită", "0", None),
        ("Kostenlos", "0", None),
        ("gratis", "0", None),
        ("Free shipping: $4.99", "4.99", "USD"),
        # word-boundary aware currency detection
        ("electronics 1299", "1299", None),
        ("Ronaldo jersey 99", "99", None),
        ("Leipzig edition 25", "25", None),
        ("Promotion 50", "50", None),
    ],
)
def test_parse_price(text: str, amount: str, currency: str | None) -> None:
    _check(text, amount, currency)


@pytest.mark.parametrize(
    "space",
    [" ", "\u00a0", "\u202f", "\u2009", "\u2007", "\u200a"],
    ids=["space", "nbsp", "narrow-nbsp", "thin", "figure", "hair"],
)
def test_parse_price_unicode_group_separators(space: str) -> None:
    _check(f"1{space}234,56{space}lei", "1234.56", "RON")
    _check(f"€{space}1{space}234{space}567.89", "1234567.89", "EUR")


def test_parse_price_ignores_zero_width_and_soft_hyphen() -> None:
    _check("1\u200b.299,\u00ad99 lei", "1299.99", "RON")
    _check("\ufeff$19.99", "19.99", "USD")


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "abc",
        "Lei",
        "€",
        "Preț la cerere",
        "Call for price",
        "-5",
        "-€5",
        "€-5",
        "\u221219,99 lei",
        "-1.299,00 RON",
        "20%",
        "15.10.2026",
        "....",
        ",,,",
        "١٢٣",
    ],
)
def test_parse_price_rejects(text: str | None) -> None:
    assert parse_price(text) == (None, None)
    assert parse_price(text, "EUR") == (None, None)


@pytest.mark.parametrize(
    ("text", "default", "amount", "currency"),
    [
        ("1299", "EUR", "1299", "EUR"),
        ("1299", "ron", "1299", "RON"),
        ("1299", "Lei", "1299", "RON"),
        ("1299", "€", "1299", "EUR"),
        ("19,99 lei", "EUR", "19.99", "RON"),
        ("$5", "EUR", "5", "USD"),
        ("Free", "eur", "0", "EUR"),
        ("1299", None, "1299", None),
    ],
)
def test_parse_price_default_currency(
    text: str, default: str | None, amount: str, currency: str | None
) -> None:
    _check(text, amount, currency, default)


def test_parse_price_currency_far_from_number() -> None:
    # Capitalised ISO codes and symbols count anywhere; local words only next to a number.
    _check("Price in RON: 1.299,00", "1299.00", "RON")
    _check("Total (€): 5", "5", "EUR")
    _check("lei \u2014 see 1.299", "1299", None)


@pytest.mark.parametrize(
    "text",
    ["1" * 400, "1." * 200, "€" * 50 + "5", "a1b2c3", "1,2,3,4", "12 34 56", "$$$1", "1e10", "0x1F"],
)
def test_parse_price_never_raises(text: str) -> None:
    amount, _ = parse_price(text)
    assert amount is None or (isinstance(amount, Decimal) and amount >= 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("€", "EUR"),
        ("EUR", "EUR"),
        ("Lei", "RON"),
        ("lei", "RON"),
        ("RON", "RON"),
        ("zł", "PLN"),
        ("Kč", "CZK"),
        ("Ft", "HUF"),
        ("лв.", "BGN"),
        ("US$", "USD"),
        ("$", "USD"),
        ("Euro (EUR)", "EUR"),
        ("Preț în lei", "RON"),
        ("NZD", "NZD"),
        ("R$", "BRL"),
        ("5 ft, $", "USD"),
        ("Ft", "HUF"),
        ("electronics", None),
        ("SKU", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_currency(text: str | None, expected: str | None) -> None:
    assert detect_currency(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RON", "RON"),
        ("ron", "RON"),
        ("Lei", "RON"),
        ("€", "EUR"),
        ("eur", "EUR"),
        ("XAF", "XAF"),  # unknown but well-formed ISO code passes through
        (" usd ", "USD"),
        ("", None),
        (None, None),
    ],
)
def test_currency_code(raw: str | None, expected: str | None) -> None:
    assert currency_code(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1299, "1299"),
        (0, "0"),
        (19.99, "19.99"),
        (24.9, "24.90"),
        (D("49.50"), "49.50"),
        ("1299.00", "1299.00"),
        ("1299", "1299"),
        ("19.99", "19.99"),
        ("1.299", "1.299"),  # '.' is the decimal separator in structured data
        ("0.5", "0.5"),
        (".5", "0.5"),
        ("1,299.00", "1299.00"),
        ("1.299,00", "1299.00"),
        ("1299,00", "1299.00"),
        ("1,299", "1299"),
        (" 1 299.50 ", "1299.50"),
        ("1\u00a0299,50", "1299.50"),
        ("€19.99", "19.99"),
        ("19.99 EUR", "19.99"),
        ("1.299,99 lei", "1299.99"),
        ("1e3", "1000"),
        ("2.5E1", "25"),
        ("1E-2", "0.01"),
    ],
)
def test_parse_decimal(value: object, expected: str) -> None:
    result = parse_decimal(value)
    assert isinstance(result, Decimal)
    assert result == D(expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "",
        "  ",
        "abc",
        "-5",
        -5,
        -0.01,
        D("-1"),
        float("nan"),
        float("inf"),
        "NaN",
        [],
        {},
        "1.2.3",
    ],
)
def test_parse_decimal_rejects(value: object) -> None:
    assert parse_decimal(value) is None


def test_parse_decimal_float_has_no_binary_noise() -> None:
    assert str(parse_decimal(0.1)) == "0.1"
    assert str(parse_decimal(1299.99)) == "1299.99"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1e20, "100000000000000000000"), ("1e30", "1" + "0" * 30), (10**30, "1" + "0" * 30)],
)
def test_parse_decimal_huge_values_are_plain_digits(value: object, expected: str) -> None:
    assert str(parse_decimal(value)) == expected


@pytest.mark.parametrize(
    ("text", "preferred", "amount", "currency"),
    [
        ("$449.99", "CAD", "449.99", "CAD"),
        ("$449.99", "cad", "449.99", "CAD"),
        ("$449.99", "EUR", "449.99", "USD"),  # EUR is no "$" currency
        ("US$449.99", "CAD", "449.99", "USD"),  # explicit markers stay what they say
        ("C$449.99", "AUD", "449.99", "CAD"),
        ("17 499 lei", "MDL", "17499", "MDL"),
        ("17 499 LEI", "MDL", "17499", "MDL"),
        ("17 499 lei", "RON", "17499", "RON"),
        ("¥1,200", "CNY", "1200", "CNY"),
        ("1.299 kr", "NOK", "1299", "NOK"),
        ("1.299 kr", None, "1299", None),  # "kr" alone names no currency
        ("449.99 USD", "CAD", "449.99", "USD"),
    ],
)
def test_parse_price_preferred_currency_decides_shared_markers(
    text: str, preferred: str | None, amount: str, currency: str | None
) -> None:
    assert parse_price(text, preferred=preferred) == (D(amount), currency)


def test_parse_price_default_currency_also_decides_shared_markers() -> None:
    assert parse_price("$5", "CAD") == (D("5"), "CAD")
    assert parse_price("5", preferred="CAD") == (D("5"), None)  # preferred is no fallback


@pytest.mark.parametrize(
    ("raw", "preferred", "expected"),
    [
        ("$", "CAD", "CAD"),
        ("lei", "MDL", "MDL"),
        ("USD", "CAD", "USD"),
        ("€", "CAD", "EUR"),
        ("$", None, "USD"),
    ],
)
def test_currency_code_with_preferred(raw: str, preferred: str | None, expected: str) -> None:
    assert currency_code(raw, preferred) == expected
    assert detect_currency(raw, preferred) == expected


@pytest.mark.parametrize(
    ("value", "currency", "expected"),
    [
        ("12.990", "HUF", "12990"),
        ("129.990", "HUF", "129990"),
        ("1.299", "RON", "1299"),
        ("1.299", "lei", "1299"),
        ("19.990", "EUR", "19990"),
        ("1.299", "KWD", "1.299"),  # three minor units: a real third decimal
        ("1.299", "bhd", "1.299"),
        ("1.299", None, "1.299"),  # unknown currency: '.' stays the decimal separator
        ("0.990", "EUR", "0.990"),
        ("1299.990", "EUR", "1299.990"),
        ("12.99", "HUF", "12.99"),
        ("1.299.000", "HUF", "1299000"),
        (12.99, "HUF", "12.99"),
    ],
)
def test_parse_decimal_dot_thousands_for_known_currency(
    value: object, currency: str | None, expected: str
) -> None:
    assert parse_decimal(value, currency) == D(expected)
