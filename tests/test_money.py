from __future__ import annotations

from decimal import Decimal

import pytest

from pricehunter.money import convert, normalize_currency, quantize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("lei", "RON"),
        ("Leu", "RON"),
        ("eur", "EUR"),
        ("€", "EUR"),
        ("zł", "PLN"),
        ("xaf", "XAF"),  # unknown but well-formed ISO code passes through
        ("dollars", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_currency_prefers_aliases_over_the_three_letter_rule(
    raw: str | None, expected: str | None
) -> None:
    assert normalize_currency(raw) == expected


def test_convert_through_the_base_currency() -> None:
    rates = {"RON": Decimal("0.201"), "USD": Decimal("0.92")}
    assert convert(Decimal("1000"), "RON", "EUR", rates, "EUR") == Decimal("201.000")
    via_base = convert(Decimal("100"), "RON", "USD", rates, "EUR")
    assert via_base is not None and quantize(via_base) == Decimal("21.85")
    assert convert(Decimal("5"), None, "EUR", rates, "EUR") == Decimal("5")
    assert convert(Decimal("5"), "GBP", "EUR", rates, "EUR") is None
