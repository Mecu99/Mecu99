"""Parse human-written and structured-data prices into ``Decimal`` + ISO currency.

Handles the separators real shops use (``1.234,56``, ``1,234.56``, ``1 234,56``
with any Unicode space, ``1'234.50``), currency symbols / codes / local names
anywhere in the text, ranges, "from" prices, percentages and "free" labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .money import CURRENCY_ALIASES, normalize_currency

# ISO codes accepted when written in capitals even if they have no alias in money.py.
_ISO_CODES = frozenset(CURRENCY_ALIASES.values()) | frozenset(
    "AED ARS BAM BRL BYN CLP CNY COP EGP GEL HKD HRK IDR ILS ISK"
    " KRW KZT MKD MXN MYR NZD PEN RSD RUB SAR SGD THB VND ZAR".split()
)

# Markers money.CURRENCY_ALIASES lacks; there a bare "$" means USD, so "R$" must not.
_EXTRA_ALIASES = {"r$": "BRL", "mx$": "MXN", "nz$": "NZD", "hk$": "HKD", "s$": "SGD"}
# Markers several currencies share. A preferred currency (the competitor's configured one)
# among a marker's candidates beats the alias default ("$" -> USD, "lei" -> RON); "kr" has none.
_SHARED_MARKERS: dict[str, frozenset[str]] = {
    "$": frozenset("USD CAD AUD NZD MXN SGD HKD ARS CLP COP TWD".split()),
    "lei": frozenset({"RON", "MDL"}),
    "leu": frozenset({"RON", "MDL"}),
    "¥": frozenset({"JPY", "CNY"}),
    "kr": frozenset({"SEK", "NOK", "DKK", "ISK"}),
}
# ISO 4217 currencies with three minor units. For any other currency a machine-readable
# "12.990" cannot be 12.99 written with three decimals: it is 12990 (Shopify's money formats).
_THREE_DECIMALS = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})
_DOT_THOUSANDS_RE = re.compile(r"[1-9][0-9]{0,2}\.[0-9]{3}")
# Local names that double as units or abbreviations ("5 ft", "Fr." = from): used only as a last resort.
_AMBIGUOUS_WORDS = frozenset({"ft", "tl", "fr."})
# Strength of a currency marker. Symbols and capitalised ISO codes count anywhere in
# the text; local names only next to a number, and ambiguous ones only if nothing else is.
_SYMBOL, _WORD, _AMBIGUOUS = 2, 1, 0

# Zero-width / soft-hyphen characters vanish; typographic apostrophes become "'".
_TRANSLATE = {
    **dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None),
    **dict.fromkeys(map(ord, "\u2019\u2018\u02bc\u2032`"), "'"),
}

# A number token. Alternatives, in order: space/apostrophe-grouped thousands
# ("1 234,56", "1'234.50"), dot/comma groups ("1.234,56", "1299"), bare ".99".
_NUM_RE = re.compile(
    r"(?<![0-9.,])(?:"
    r"[0-9]{1,3}(?:[ '][0-9]{3}(?![0-9]))+(?:[.,][0-9]+)?"
    r"|[0-9]+(?:[.,][0-9]+)*"
    r"|[.,][0-9]+"
    r")"
)
# Currency markers: letter-prefixed "$" ("US$", "R$"), symbols, or any word (looked up in the alias tables).
_CCY_TOKEN_RE = re.compile(r"(?<![^\W\d_])[a-z]{1,2}\$|\$|[€£¥₹₺₴]|[^\W\d_]+\.?", re.IGNORECASE)
# A number Python's Decimal reads as-is: "1299", "19.99", ".5", "1e3".
_PLAIN_NUMBER_RE = re.compile(r"[0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?", re.IGNORECASE)
_RANGE_GAP_RE = re.compile(r"\s*(?:-|\u2013|\u2014|~|to|bis|la|à)\s*", re.IGNORECASE)
# A minus sign attached to the number, optionally with a symbol in between ("-5", "€-5", "-€ 5").
_NEGATIVE_RE = re.compile(r"(?<!\w)[-\u2212](?:[^\w\s]{1,3} ?)?$")
_FREE_RE = re.compile(r"\b(?:free|gratuit[aăe]?|gratis|kostenlos|gratuito|offert)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _CcyToken:
    start: int
    end: int
    code: str
    rank: int  # _SYMBOL, _WORD or _AMBIGUOUS


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TRANSLATE)).strip()


def _marker_code(tok: str, preferred: str | None) -> str | None:
    low = tok.lower()
    shared = _SHARED_MARKERS.get(low) or _SHARED_MARKERS.get(low.rstrip("."), frozenset())
    if preferred in shared:
        return preferred
    code = _EXTRA_ALIASES.get(low) or CURRENCY_ALIASES.get(low) or CURRENCY_ALIASES.get(low.rstrip("."))
    if code is None and tok.isupper() and tok in _ISO_CODES:
        code = tok
    return code


def _currency_tokens(s: str, preferred: str | None = None) -> list[_CcyToken]:
    tokens: list[_CcyToken] = []
    for m in _CCY_TOKEN_RE.finditer(s):
        tok = m.group()
        low = tok.lower()
        code = _marker_code(tok, preferred)
        if not code:
            continue
        if not tok[0].isalpha() or tok.endswith("$") or (len(tok) == 3 and tok.isupper()):
            rank = _SYMBOL
        else:
            rank = _AMBIGUOUS if low in _AMBIGUOUS_WORDS else _WORD
        tokens.append(_CcyToken(m.start(), m.end(), code, rank))
    return tokens


def _adjacent(s: str, num: re.Match[str], tok: _CcyToken) -> bool:
    lo, hi = (tok.end, num.start()) if tok.end <= num.start() else (num.end(), tok.start)
    return lo <= hi and not s[lo:hi].strip()


def _valid_groups(groups: list[str]) -> bool:
    head, rest = groups[0], groups[1:]
    if not rest:
        return head == "" or head.isdigit()
    return head.isdigit() and len(head) <= 3 and all(len(g) == 3 and g.isdigit() for g in rest)


def _number(token: str, *, dot_decimal: bool = False) -> Decimal | None:
    """Turn one number token into a Decimal, deciding which separator is the decimal one.

    A lone separator followed by exactly three digits is a thousands separator
    ("1.299" -> 1299) unless the integer part is 0/empty, longer than 3 digits,
    the number is space-grouped, or ``dot_decimal`` says '.' is always decimal.
    """
    grouped = " " in token or "'" in token
    t = token.replace(" ", "").replace("'", "")
    frac = ""
    if "." in t and "," in t:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        whole, frac = t.rsplit(dec, 1)
        groups = whole.split("," if dec == "." else ".")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        groups = t.split(sep)
        if len(groups) == 2:
            head, tail = groups
            thousands = (
                not grouped
                and len(tail) == 3
                and head not in ("", "0")
                and len(head) <= 3
                and not (dot_decimal and sep == ".")
            )
            if not thousands:
                groups, frac = [head], tail
    else:
        groups = [t]
    if not _valid_groups(groups) or (frac and not frac.isdigit()):
        return None
    whole = "".join(groups) or "0"
    return Decimal(f"{whole}.{frac}" if frac else whole)


def detect_currency(text: str | None, preferred: str | None = None) -> str | None:
    """ISO code of the first, strongest currency marker in ``text`` (symbols and ISO codes first).

    A marker several currencies share ("$", "lei", "kr") means ``preferred`` when it is one of them.
    """
    if not text:
        return None
    best = max(_currency_tokens(_clean(text), _iso(preferred)), key=lambda t: t.rank, default=None)
    return best.code if best else None


def currency_code(raw: str | None, preferred: str | None = None) -> str | None:
    """ISO code for a currency *field* such as ``"RON"``, ``"Lei"``, ``"€"`` or ``"us$"``.

    Markers recognised by :func:`detect_currency` (including letter-prefixed dollars such
    as ``"R$"``) come first; unknown but well-formed ISO codes still pass through.
    """
    return detect_currency(raw, preferred) or normalize_currency(raw)


def _iso(preferred: str | None) -> str | None:
    return currency_code(preferred) if preferred else None


def parse_price(
    text: str | None, default_currency: str | None = None, preferred: str | None = None
) -> tuple[Decimal | None, str | None]:
    """Parse a displayed price such as ``"1.234,56 €"`` or ``"From $10"``.

    Returns ``(amount, currency)``. The amount is the first price in the text,
    preferring one written next to a currency marker (so "iPhone 15 - 999 €"
    yields 999, while the range "10 - 20 EUR" yields 10). Percentages are
    ignored; negative amounts and text without a number give ``(None, None)``;
    "Free"-style labels give ``Decimal(0)``. The currency is the marker next to
    the chosen number, else any unambiguous marker, else ``default_currency``.
    ``preferred`` (else ``default_currency``) decides what a marker several
    currencies share means: "$" with ``"CAD"`` is CAD, "lei" with ``"MDL"`` MDL.
    Local names that double as units ("5 ft") only count when no other marker
    sits next to a number.
    """
    if not text:
        return None, None
    s = _clean(text)
    tokens = _currency_tokens(s, _iso(preferred or default_currency))
    fallback = next((t.code for t in tokens if t.rank == _SYMBOL), None) or currency_code(default_currency)
    nums = [
        (m, value)
        for m in _NUM_RE.finditer(s)
        if not s[m.end() :].lstrip().startswith("%") and (value := _number(m.group())) is not None
    ]
    if not nums:
        return (Decimal(0), fallback) if _FREE_RE.search(s) else (None, None)

    def currency_next_to(m: re.Match[str], min_rank: int) -> str | None:
        return next((t.code for t in tokens if t.rank >= min_rank and _adjacent(s, m, t)), None)

    def pick(min_rank: int) -> tuple[re.Match[str], Decimal, str] | None:
        for i, (m, value) in enumerate(nums):
            currency = currency_next_to(m, min_rank)
            if currency is None and i + 1 < len(nums):
                nxt = nums[i + 1][0]
                if _RANGE_GAP_RE.fullmatch(s[m.end() : nxt.start()]):
                    currency = currency_next_to(nxt, min_rank)
            if currency:
                return m, value, currency
        return None

    chosen, amount, currency = pick(_WORD) or pick(_AMBIGUOUS) or (*nums[0], fallback)
    if _NEGATIVE_RE.search(s[: chosen.start()]):
        return None, None
    return amount, currency


def parse_decimal(value: object, currency: str | None = None) -> Decimal | None:
    """Parse a machine-readable price (JSON-LD ``price``, ``content`` attributes).

    Accepts int/float/Decimal/str. In strings '.' is the decimal separator, but
    "1,299.00" and "1299,00" are tolerated; strings with extra text fall back to
    :func:`parse_price`. When ``currency`` is known and has fewer than three minor
    units, "12.990" is 12990 (no such currency has a third decimal to write).
    Returns ``None`` for negative, non-finite or garbage values.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        number: Decimal | None = value
    elif isinstance(value, (int, float)):
        number = Decimal(str(value))
    elif isinstance(value, str):
        s = _clean(value)
        code = _iso(currency)
        if code and code not in _THREE_DECIMALS and _DOT_THOUSANDS_RE.fullmatch(s):
            number = Decimal(s.replace(".", ""))
        elif _PLAIN_NUMBER_RE.fullmatch(s):
            number = Decimal(s)
        elif _NUM_RE.fullmatch(s):
            number = _number(s, dot_decimal=True)
        else:
            number = parse_price(s)[0]
    else:
        return None
    if number is None or not number.is_finite() or number < 0:
        return None
    exponent = number.as_tuple().exponent
    if isinstance(exponent, int) and exponent > 0:
        number = Decimal(f"{number:f}")  # "1E+3" -> "1000"
    return number
