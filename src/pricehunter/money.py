"""Currency normalisation and conversion helpers."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# Symbol / local-name -> ISO-4217. Ambiguous "$" defaults to USD; "kr" is left out on purpose.
CURRENCY_ALIASES: dict[str, str] = {
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "$": "USD", "us$": "USD", "usd": "USD",
    "£": "GBP", "gbp": "GBP",
    "lei": "RON", "ron": "RON", "leu": "RON",
    "zł": "PLN", "zl": "PLN", "pln": "PLN",
    "kč": "CZK", "czk": "CZK",
    "ft": "HUF", "huf": "HUF",
    "лв": "BGN", "лв.": "BGN", "bgn": "BGN",
    "chf": "CHF", "fr.": "CHF",
    "¥": "JPY", "jpy": "JPY",
    "₹": "INR", "inr": "INR",
    "c$": "CAD", "ca$": "CAD", "cad": "CAD",
    "a$": "AUD", "au$": "AUD", "aud": "AUD",
    "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "₺": "TRY", "try": "TRY", "tl": "TRY",
    "₴": "UAH", "uah": "UAH",
    "mdl": "MDL",
}

CENT = Decimal("0.01")


def normalize_currency(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if len(s) == 3 and s.isalpha():
        return s.upper()
    return CURRENCY_ALIASES.get(s.lower())


def convert(
    amount: Decimal,
    from_ccy: str | None,
    to_ccy: str,
    rates: dict[str, Decimal],
    base: str,
) -> Decimal | None:
    """Convert ``amount`` into ``to_ccy``.

    ``rates`` maps a currency to how many units of ``base`` one unit is worth;
    ``base`` itself is implicitly 1. Returns ``None`` when a rate is unknown.
    A missing ``from_ccy`` is assumed to already be ``to_ccy``.
    """
    src = (from_ccy or to_ccy).upper()
    dst = to_ccy.upper()
    if src == dst:
        return amount
    table = {k.upper(): Decimal(str(v)) for k, v in rates.items()}
    table[base.upper()] = Decimal(1)
    if src not in table or dst not in table or table[dst] == 0:
        return None
    return amount * table[src] / table[dst]


def quantize(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)
