"""Decide whether a competitor offer is the same item as a product in our catalogue.

Identifiers win when both sides carry them (GTIN, then MPN). Otherwise a fuzzy title score is
adjusted by guards against the classic wrong-variant matches: different storage or pack sizes,
different model numbers, "Pro"/"Max" siblings, and accessories sold *for* the product.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal
from functools import lru_cache

from rapidfuzz import fuzz

from .models import ExtractedOffer, MatchResult, Product

__all__ = [
    "model_tokens",
    "normalize_gtin",
    "normalize_text",
    "rank_offers",
    "score_match",
    "spec_tokens",
    "title_similarity",
]

# --------------------------------------------------------------------------- GTIN

_GTIN_RUN = re.compile(r"\d(?:[\d \t.\-]*\d)?", re.ASCII)
_GTIN_LENGTHS = frozenset({8, 12, 13, 14})


def _gs1_check_digit(body: str) -> int:
    """GS1 mod-10 check digit: weights 3,1,3,... starting from the rightmost body digit."""
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10


def normalize_gtin(value: str | None) -> str | None:
    """Return ``value`` as a check-digit-valid, zero-padded GTIN-14, or ``None``.

    Accepts GTIN-8, UPC-A, EAN-13 and GTIN-14, with spaces, dots or hyphens inside the number and
    labels around it (``"EAN: 590 1234 123457"``).
    """
    if not value:
        return None
    for run in _GTIN_RUN.findall(str(value)):
        digits = re.sub(r"\D", "", run, flags=re.ASCII)
        if (
            len(digits) in _GTIN_LENGTHS
            and digits.strip("0")
            and _gs1_check_digit(digits[:-1]) == int(digits[-1])
        ):
            return digits.zfill(14)
    return None


# --------------------------------------------------------------------------- text

# Spelling -> canonical unit. Only applied right after a number ("128 GB", "6,1 inch").
_UNIT_ALIASES: dict[str, str] = {
    "gb": "gb", "gib": "gb", "tb": "tb", "tib": "tb", "mb": "mb",
    "inch": "in", "inches": "in", "in": "in", "inci": "in", "toli": "in", "zoll": "in",
    "ml": "ml", "cl": "cl", "l": "l", "ltr": "l", "litre": "l", "litres": "l",
    "liter": "l", "liters": "l", "litri": "l", "litru": "l",
    "kg": "kg", "kgs": "kg", "g": "g", "gr": "g", "gram": "g", "grams": "g", "grame": "g",
    "mg": "mg",
    "w": "w", "watt": "w", "watts": "w", "kw": "kw", "wh": "wh",
    "mah": "mah", "v": "v", "volt": "v", "volts": "v",
    "hz": "hz", "khz": "khz", "mhz": "mhz", "ghz": "ghz",
    "mm": "mm", "cm": "cm", "m": "m",
    "mp": "mp", "mpx": "mp",
    "pcs": "pcs", "pc": "pcs", "buc": "pcs", "bucati": "pcs", "piece": "pcs", "pieces": "pcs",
    "pack": "pcs", "pk": "pcs", "%": "pct",
}  # fmt: skip
_CANONICAL_UNITS = frozenset(_UNIT_ALIASES.values()) | {"x"}


def _alternation(words: Iterable[str]) -> str:
    return "|".join(sorted(map(re.escape, words), key=len, reverse=True))


_NUM = r"(?<![\w.,])(\d+(?:[.,]\d+)?)"
_END = r"(?![a-z0-9])"
_COMMA_NUM = re.compile(r"(?<![\w.,])\d+,\d+(?![\w,])")
_PLUS = re.compile(r"(?<=[a-z0-9])\+(?![a-z0-9])")
_N_IN_1 = re.compile(r"(?<!\w)(\d)\s*-?\s*in\s*-?\s*1(?!\d)")
_MULT_PREFIX = re.compile(r"(?<![\w.,])(\d+)\s*[x×*]\s*(?=\d)")
_INCH_MARK = re.compile(_NUM + r"\s*(?:\"|''|′′|″|”|“)")
_UNIT = re.compile(_NUM + r"\s*(" + _alternation(_UNIT_ALIASES) + r")" + _END)
_MULT_SUFFIX = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:" + _alternation(_CANONICAL_UNITS) + r"))\s*[x×*]\s*(\d+)" + _END
)
_CODE = re.compile(r"(?<![\w-])[a-z0-9]+(?:[-/][a-z0-9]+)+(?![\w-])")
_PUNCT = re.compile(r"[^\w.]+|_")
_LOOSE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SPEC_TOKEN = re.compile(r"(\d+(?:\.\d+)?)(" + _alternation(_CANONICAL_UNITS) + r")")


def _fold(text: str) -> str:
    """Casefold and strip diacritics (NFKD), so "Mașină" and "masina" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _canon_number(raw: str) -> str:
    """``"6,1"`` -> ``"6.1"``, ``"5,000"`` -> ``"5000"``, ``"0,50"`` -> ``"0.5"``.

    A separator followed by exactly three digits after a non-zero 1-3 digit head is read as a
    thousands separator; anything else is a decimal point.
    """
    head, sep, tail = raw.replace(",", ".").partition(".")
    if sep and len(tail) == 3 and head != "0" and len(head) <= 3:
        head, tail = head + tail, ""
    value = Decimal(f"{head}.{tail}" if tail else head)
    return format(value.normalize(), "f")


def _unit_repl(m: re.Match[str]) -> str:
    return _canon_number(m[1]) + _UNIT_ALIASES[m[2]]


def _code_repl(m: re.Match[str]) -> str:
    """Glue hyphenated model codes ("WH-1000XM5" -> "wh1000xm5", "SM-S918B" -> "sms918b").

    Only when some part mixes letters and digits and the others are short, so "Galaxy-S24" and
    "Coca-Cola" stay separate words.
    """
    parts = re.split(r"[-/]", m[0])
    if any(map(_mixed, parts)) and all(_mixed(p) or len(p) <= 3 for p in parts):
        return "".join(parts)
    return m[0]


# Romanian colour names -> English, so "negru" and "black" compare equal.
_COLOURS = {
    "negru": "black", "neagra": "black", "alb": "white", "alba": "white", "rosu": "red",
    "rosie": "red", "albastru": "blue", "albastra": "blue", "verde": "green", "gri": "grey",
    "gray": "grey", "argintiu": "silver", "auriu": "gold", "roz": "pink", "mov": "purple",
    "galben": "yellow", "galbena": "yellow", "portocaliu": "orange", "maro": "brown",
}  # fmt: skip


@lru_cache(maxsize=8192)
def normalize_text(text: str) -> str:
    """Canonical form for fuzzy comparison.

    Lowercases, strips diacritics, glues numbers to units (``"128 GB"`` -> ``"128gb"``,
    ``'6.1"'`` -> ``"6.1in"``, ``"0,5 L"`` -> ``"0.5l"``), writes pack multipliers as ``"6x"``
    (``"6 x 0.5 l"`` and ``"0.5 l x 6"`` both give ``"6x 0.5l"``-style tokens), glues hyphenated
    model codes, turns a trailing ``+`` into ``plus``, maps Romanian colour names to English and
    collapses all other punctuation to single spaces.
    """
    s = _fold(text)
    s = _PLUS.sub(" plus", s)
    s = _N_IN_1.sub(r"\1in1", s)
    s = _MULT_PREFIX.sub(r"\1x ", s)
    s = _INCH_MARK.sub(lambda m: _canon_number(m[1]) + "in ", s)
    s = _UNIT.sub(_unit_repl, s)
    s = _MULT_SUFFIX.sub(r"\1 \2x", s)
    s = _COMMA_NUM.sub(lambda m: _canon_number(m[0]), s)
    s = _CODE.sub(_code_repl, s)
    s = _LOOSE_DOT.sub(" ", _PUNCT.sub(" ", s))
    return " ".join(_COLOURS.get(w, w) for w in s.split())


@lru_cache(maxsize=8192)
def _spec_tokens(text: str) -> tuple[tuple[str, frozenset[str]], ...]:
    specs: dict[str, set[str]] = {}
    for token in normalize_text(text).split():
        if m := _SPEC_TOKEN.fullmatch(token):
            specs.setdefault(m[2], set()).add(m[1])
    return tuple((unit, frozenset(values)) for unit, values in specs.items())


def spec_tokens(text: str) -> dict[str, set[str]]:
    """Numeric specs keyed by canonical unit: ``"iPhone 15 128 GB"`` -> ``{"gb": {"128"}}``.

    Pack multipliers are reported under ``"x"`` (``"6 x 0.5L"`` -> ``{"x": {"6"}, "l": {"0.5"}}``).
    """
    return {unit: set(values) for unit, values in _spec_tokens(text)}


# Units that measure the same quantity, scaled to a common base for conflict checks.
_UNIT_FAMILY: dict[str, tuple[str, Decimal]] = {
    "tb": ("gb", Decimal(1000)), "gb": ("gb", Decimal(1)),
    "l": ("ml", Decimal(1000)), "cl": ("ml", Decimal(10)), "ml": ("ml", Decimal(1)),
    "kg": ("g", Decimal(1000)), "g": ("g", Decimal(1)), "mg": ("g", Decimal("0.001")),
    "kw": ("w", Decimal(1000)), "w": ("w", Decimal(1)),
    "ghz": ("hz", Decimal(10**9)), "mhz": ("hz", Decimal(10**6)),
    "khz": ("hz", Decimal(1000)), "hz": ("hz", Decimal(1)),
    "m": ("mm", Decimal(1000)), "cm": ("mm", Decimal(10)), "mm": ("mm", Decimal(1)),
    "x": ("pcs", Decimal(1)), "pcs": ("pcs", Decimal(1)),
}  # fmt: skip


def _canonical_specs(text: str) -> dict[str, set[Decimal]]:
    out: dict[str, set[Decimal]] = {}
    for unit, values in _spec_tokens(text):
        family, factor = _UNIT_FAMILY.get(unit, (unit, Decimal(1)))
        out.setdefault(family, set()).update(Decimal(v) * factor for v in values)
    return out


def _spec_conflicts(a: str, b: str) -> list[str]:
    """Unit families where neither side's values contain the other's (128gb vs 256gb).

    A subset is fine: titles often add RAM ("8GB RAM, 256GB") that the other side leaves out.
    A pack count on one side only conflicts when both sides state the same unit size
    ("6 x 0.5 L" vs "0.5 L" is a multipack versus a single bottle).
    """
    sa, sb = _canonical_specs(a), _canonical_specs(b)
    conflicts = {u for u in sa.keys() & sb.keys() if not (sa[u] <= sb[u] or sb[u] <= sa[u])}
    pa, pb = sa.get("pcs", set()), sb.get("pcs", set())
    one_sided_pack = set() if pa and pb else pa | pb
    same_size = any(sa.get(f, set()) & sb.get(f, set()) for f in ("ml", "g"))
    if same_size and any(n > 1 for n in one_sided_pack):
        conflicts.add("pcs")
    return sorted(conflicts)


# --------------------------------------------------------------------------- model numbers

_CHUNK = re.compile(r"[a-z0-9]+(?:[-/_][a-z0-9]+)*")
_UNITS_RE = _alternation(set(_UNIT_ALIASES) | {"x"})
# Quantities, not model numbers: "128gb", "6x2l", "1920x1080", "4k", "1080p".
_UNIT_LIKE = re.compile(
    rf"(?:\d+(?:\.\d+)?x)?\d+(?:\.\d+)?(?:{_UNITS_RE})?|[1-9]\d?k|\d{{3,4}}[pi]"
)
# Standards and ratings that look like model numbers but say nothing about which model it is.
_GENERIC_MODEL = re.compile(
    r"(?:wi-?fi|wlan|usb|ddr|lpddr|gddr|pcie|hdmi|bt|ip|ipx|hdr|dvb[tsc]|h|gen|cat|rev|ver"
    r"|android|ios|win|windows)\d+[a-z]?|\d+in\d+|\d+(?:st|nd|rd|th)"
)


def _mixed(token: str) -> bool:
    return (
        any(c.isdigit() for c in token)
        and any(c.isalpha() for c in token)
        and not _UNIT_LIKE.fullmatch(token)
        and not _GENERIC_MODEL.fullmatch(token)
    )


def _is_model(token: str) -> bool:
    return len(token) >= 4 and _mixed(token)


@lru_cache(maxsize=8192)
def _model_tokens(text: str) -> frozenset[str]:
    tokens: set[str] = set()
    for chunk in _CHUNK.findall(_fold(text)):
        parts = re.split(r"[-/_]", chunk)
        tokens.update(p for p in parts if _is_model(p))
        if len(parts) > 1 and any(_mixed(p) for p in parts):
            # "WH-1000XM5" -> "wh1000xm5"; "SM-S918B/DS" -> "sms918bds" and "sms918b".
            groups = [chunk, *chunk.split("/")]
            tokens.update(j for g in groups if _is_model(j := re.sub(r"[-/_]", "", g)))
    return frozenset(tokens)


def model_tokens(text: str) -> set[str]:
    """Identifier-like tokens (letters and digits, length >= 4) with ``-``/``/`` removed.

    ``"Sony WH-1000XM5"`` -> ``{"wh1000xm5", "1000xm5"}``. Units (``128gb``, ``4k``), pure numbers
    (years) and generic standards (``wifi6``, ``ip68``, ``ddr5``) are excluded.
    """
    return set(_model_tokens(text))


def _related(x: str, y: str) -> bool:
    """Equal, or one long-enough code contains the other ("mtp03" ~ "mtp03zda")."""
    return x == y or (min(len(x), len(y)) >= 5 and (x in y or y in x))


def _tokens_overlap(a: Iterable[str], b: Iterable[str]) -> bool:
    return any(_related(x, y) for x in a for y in b)


def _shape(token: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", token)


def _respelled(token: str, other_compact: str) -> bool:
    """A dotted code the other side writes without dots ("ecam22.110" vs a slug's "ecam22 110")."""
    plain = token.replace(".", "")
    return "." in token and len(plain) >= 5 and plain in other_compact


def _number_conflict(a: str, b: str) -> bool:
    """Same-shaped numeric tokens that differ: "iphone 15" vs "iphone 14", "s24" vs "s23"."""
    na = {t for t in a.split() if any(c.isdigit() for c in t)}
    nb = {t for t in b.split() if any(c.isdigit() for c in t)}
    ca, cb = re.sub(r"[\s.]", "", a), re.sub(r"[\s.]", "", b)
    only_b = {_shape(t) for t in nb - na if not _respelled(t, ca)}
    return any(_shape(t) in only_b for t in na - nb if not _respelled(t, cb))


# --------------------------------------------------------------------------- guards

_VARIANT_WORDS = frozenset({"pro", "max", "plus", "ultra", "mini", "lite", "light", "se", "fe", "zero"})
# Phrases where a variant word describes a feature, not a sibling model.
_NOT_A_VARIANT = re.compile(
    r"\b(?:ultra hd|ultra high definition|mini (?:usb|hdmi|jack|displayport|dp)|plus size)\b"
)
_ACCESSORY = re.compile(
    r"\b(?:husa|huse|case|cases|cover|covers|carcasa|bumper|charger|chargers|incarcator\w*"
    r"|cable|cables|cablu\w*|adapter|adaptor|screen protector|tempered glass|sticla securizata"
    r"|folie|folii|strap|straps|curea|curele|replacement|spare|compatib\w*|refurbished|refurb"
    r"|renewed|resigilat\w*|reconditionat\w*|used|second hand|open box|bundle)\b"
)
# "for"/"pentru" are common words; they only signal an accessory when followed by the product.
_FOR = re.compile(r"\b(?:for|pentru)\s+(\w+)")


def _accessory_terms(text: str, anchors: frozenset[str]) -> set[str]:
    """Accessory/condition keywords, plus ``"for"`` when the text is *for* something: it says
    "for <one of our product's words>" or names an accessory (a case is always for a phone)."""
    terms = set(_ACCESSORY.findall(text))
    if terms or any(w in anchors for w in _FOR.findall(text)):
        terms.add("for")
    return terms


def _anchors(norm: str, raw: str, brand: str | None) -> frozenset[str]:
    words = {w for w in norm.split() if len(w) >= 4 and w.isalpha()}
    return frozenset(words | set(normalize_text(brand or "").split()) | _model_tokens(raw))


def _variant_mismatch(a: str, b: str) -> set[str]:
    """Variant words ("pro", "max", ...) present on one side only, ignoring compounds
    like "airfryer" vs "air fryer" by checking the other side with spaces removed."""
    a, b = _NOT_A_VARIANT.sub(" ", a), _NOT_A_VARIANT.sub(" ", b)
    ta, tb = set(a.split()), set(b.split())
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    return {w for w in (ta ^ tb) & _VARIANT_WORDS if w not in (cb if w in ta else ca)}


def _contains(haystack: str, needle: str) -> bool:
    """Whole-phrase match on normalized text, or compact match for longer needles."""
    if not needle:
        return True
    if f" {needle} " in f" {haystack} ":
        return True
    compact = needle.replace(" ", "")
    return len(compact) >= 4 and compact in haystack.replace(" ", "")


def _compose(brand: str | None, title: str | None, mpn: str | None) -> tuple[str, str]:
    """``brand + title + mpn`` as (raw, normalized), skipping parts the title already contains.

    The raw form keeps hyphenated codes ("WH-1000XM5") intact for :func:`model_tokens`.
    """
    raw, norm = title or "", normalize_text(title or "")
    nb, nm = normalize_text(brand or ""), normalize_text(mpn or "")
    if brand and nb and not _contains(norm, nb):
        raw, norm = f"{brand} {raw}", f"{nb} {norm}"
    if mpn and nm and not _contains(norm, nm):
        raw, norm = f"{raw} {mpn}", f"{norm} {nm}"
    return raw.strip(), norm.strip()


def _norm_mpn(mpn: str | None) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", _fold(mpn or ""))
    return key or None


def _brands_compatible(a: str | None, b: str | None) -> bool:
    na, nb = normalize_text(a or "").replace(" ", ""), normalize_text(b or "").replace(" ", "")
    return not na or not nb or na in nb or nb in na


# --------------------------------------------------------------------------- scoring


def title_similarity(a: str, b: str) -> dict[str, float]:
    """Blend of rapidfuzz ratios (0..1) on already-normalized strings.

    ``token_set`` forgives extra words in long shop titles, ``token_sort`` penalises them a little
    and ``partial`` rewards a contiguous common run.
    """
    token_set = fuzz.token_set_ratio(a, b) / 100
    token_sort = fuzz.token_sort_ratio(a, b) / 100
    partial = fuzz.partial_ratio(a, b) / 100
    return {
        "title": 0.5 * token_set + 0.3 * token_sort + 0.2 * partial,
        "token_set": token_set,
        "token_sort": token_sort,
        "partial": partial,
    }


def _result(score: float, reason: str, details: dict[str, float]) -> MatchResult:
    return MatchResult(
        score=round(min(1.0, max(0.0, score)), 4),
        reason=reason,
        details={k: round(v, 4) for k, v in details.items()},
    )


def score_match(product: Product, offer: ExtractedOffer) -> MatchResult:
    """How likely ``offer`` is the very same item as ``product`` (0..1, with the deciding reason).

    1. GTINs on both sides decide alone: equal -> 1.0 ``gtin``; different -> 0.05 ``gtin_conflict``.
    2. Equal MPNs (same or unknown brand) -> 0.97 ``mpn`` unless the specs disagree.
    3. Otherwise a fuzzy title score with penalties for a missing brand, accessories and variant
       words, a bonus for a shared model number, and caps for conflicting model numbers
       (``model_conflict``), specs (``spec_conflict``) or MPNs (``mpn_conflict``).
    """
    pg, og = normalize_gtin(product.gtin), normalize_gtin(offer.gtin)
    if pg and og:
        if pg == og:
            return _result(1.0, "gtin", {"gtin": 1.0})
        return _result(0.05, "gtin_conflict", {"gtin": 0.0})

    p_raw, p_text = _compose(product.brand, product.name, product.mpn)
    # Their MPN only adds noise to the title comparison when we have none to compare it with.
    o_raw, o_text = _compose(offer.brand, offer.title, offer.mpn if product.mpn else None)
    spec_conflicts = _spec_conflicts(p_text, o_text)

    pm, om = _norm_mpn(product.mpn), _norm_mpn(offer.mpn)
    if pm and om and pm == om and len(pm) >= 4 and _brands_compatible(product.brand, offer.brand):
        if spec_conflicts:
            return _result(0.4, "spec_conflict", {"mpn": 1.0, "spec_conflict": 1.0})
        return _result(0.97, "mpn", {"mpn": 1.0})
    if not o_text:
        return _result(0.0, "no_title", {})

    details = title_similarity(p_text, o_text)
    score, reason = details["title"], "title"
    pt, ot = _model_tokens(p_raw), _model_tokens(f"{o_raw} {offer.mpn or ''}")
    model_clash = bool(pt and ot and not _tokens_overlap(pt, ot))
    if pt and ot and not model_clash:
        score = min(1.0, score + 0.1)
        details["model_bonus"] = 0.1
    # Our MPN spelled out in their title is nearly as good as a structured MPN.
    if pm and not om and len(pm) >= 5 and pm in o_text.replace(" ", ""):
        score = max(score, 0.9)
        details["mpn_in_title"] = 1.0

    # Penalties multiply after the boosts so an accessory "for WH-1000XM5" cannot climb back.
    if product.brand and not _contains(o_text, normalize_text(product.brand)):
        score *= 0.75
        details["brand_penalty"] = 0.75
    anchors = _anchors(p_text, p_raw, product.brand)
    if _accessory_terms(o_text, anchors) - _accessory_terms(p_text, anchors):
        score *= 0.6
        details["accessory_penalty"] = 0.6
    if _variant_mismatch(p_text, o_text):
        score *= 0.7
        details["variant_penalty"] = 0.7

    if model_clash or _number_conflict(p_text, o_text):
        score = min(score, 0.45)
        reason = "model_conflict"
        details["model_conflict"] = 1.0
    if spec_conflicts:
        score = min(score, 0.4)
        reason = "spec_conflict"
        details["spec_conflict"] = 1.0
    if pm and om and not _related(pm, om):
        score = min(score, 0.3)
        reason = "mpn_conflict"
        details["mpn_conflict"] = 1.0
    return _result(score, reason, details)


def rank_offers(
    product: Product, offers: list[ExtractedOffer], threshold: float
) -> list[tuple[ExtractedOffer, MatchResult]]:
    """Offers scoring at least ``threshold``, best first (ties keep input order)."""
    scored = [(offer, score_match(product, offer)) for offer in offers]
    accepted = [pair for pair in scored if pair[1].score >= threshold]
    return sorted(accepted, key=lambda pair: pair[1].score, reverse=True)
