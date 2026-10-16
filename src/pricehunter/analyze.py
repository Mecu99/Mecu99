"""Market position from competitor observations, and repricing suggestions."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from statistics import median

from .models import (
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    Observation,
    PriceSuggestion,
    Product,
    RepricingRule,
    Settings,
)
from .money import CENT, convert, normalize_currency, quantize

_ZERO = Decimal(0)


def product_currency(product: Product) -> str:
    """ISO code of ``product.currency``, which the config may spell ``eur``, ``€`` or ``lei``."""
    return normalize_currency(product.currency) or product.currency.upper()


def _available(offer: CompetitorPrice) -> bool:
    """Anything not explicitly out of stock can be bought (UNKNOWN counts as available)."""
    return offer.availability is not Availability.OUT_OF_STOCK


def _num(value: Decimal) -> str:
    """Render a Decimal without exponent or trailing zeros: ``Decimal("10.0")`` -> ``"10"``."""
    return format(value.normalize(), "f")


def _offer(obs: Observation, product: Product, settings: Settings) -> CompetitorPrice | None:
    """Convert one observation into the product's currency; ``None`` if it cannot be priced."""
    if obs.price is None or obs.price <= 0:
        return None
    amount = obs.price
    if settings.include_shipping and obs.shipping is not None:
        amount += obs.shipping
    currency = obs.currency.upper() if obs.currency else None
    rates, base = settings.currency_rates, settings.base_currency
    converted = convert(amount, currency, product_currency(product), rates, base)
    if converted is None:
        return None
    return CompetitorPrice(
        competitor=obs.competitor,
        url=obs.url,
        price=quantize(converted),
        original_currency=currency,
        availability=obs.availability,
        match_score=obs.match_score,
    )


def _best_offer(offers: list[CompetitorPrice]) -> CompetitorPrice:
    """Cheapest available offer, or the cheapest overall if none is available."""
    pool = [o for o in offers if _available(o)] or offers
    return min(pool, key=lambda o: (o.price, -o.match_score, o.url))


def market_snapshot(
    product: Product,
    observations: list[Observation],
    settings: Settings,
    min_match_score: float = 0.0,
) -> MarketSnapshot:
    """Summarise the market for ``product`` from its competitors' latest observations.

    Prices are converted into the product's currency (with shipping when
    ``settings.include_shipping``); observations without a usable price, below
    ``min_match_score`` or in a currency without a known rate (see :func:`missing_rates`)
    are ignored. Each competitor contributes one offer. The summary statistics, rank and
    ``in_stock_offers`` count only offers that are not out of stock.
    """
    by_competitor: dict[str, list[CompetitorPrice]] = {}
    for obs in observations:
        if obs.sku != product.sku or obs.match_score < min_match_score:
            continue
        offer = _offer(obs, product, settings)
        if offer is not None:
            by_competitor.setdefault(obs.competitor, []).append(offer)

    offers = sorted(
        (_best_offer(group) for group in by_competitor.values()),
        key=lambda o: (o.price, o.competitor),
    )
    available = [o for o in offers if _available(o)]
    prices = [o.price for o in available]
    our_price = product.our_price
    snapshot = MarketSnapshot(
        sku=product.sku,
        currency=product_currency(product),
        our_price=our_price,
        offers=offers,
        in_stock_offers=len(available),
    )
    if prices:
        snapshot.lowest = prices[0]
        snapshot.lowest_competitor = available[0].competitor
        snapshot.highest = prices[-1]
        snapshot.median = quantize(median(prices))
        snapshot.average = quantize(sum(prices, _ZERO) / len(prices))
    if our_price is not None:
        snapshot.competitors_cheaper = sum(1 for p in prices if p < our_price)
        snapshot.our_rank = snapshot.competitors_cheaper + 1
        if snapshot.median:
            snapshot.price_index = round(float(our_price / snapshot.median * 100), 1)
    return snapshot


def missing_rates(
    product: Product, observations: list[Observation], settings: Settings
) -> dict[str, set[str]]:
    """Currencies lacking a rate that :func:`market_snapshot` needs for ``product``, each with
    the competitors whose prices are left out for it (a wrong or missing rate is otherwise silent).
    """
    target = product_currency(product)
    known = {code.upper() for code in settings.currency_rates} | {settings.base_currency.upper()}
    missing: dict[str, set[str]] = {}
    for obs in observations:
        if obs.sku != product.sku or not obs.price:
            continue
        source = (obs.currency or target).upper()
        rates, base = settings.currency_rates, settings.base_currency
        if convert(obs.price, source, target, rates, base) is not None:
            continue
        for code in {source, target} - known or {source}:
            missing.setdefault(code, set()).add(obs.competitor)
    return missing


# --------------------------------------------------------------------------- repricing


def _floor(product: Product, rule: RepricingRule) -> tuple[Decimal | None, str]:
    """The highest applicable lower bound (rounded up to cents) and where it comes from."""
    bounds: list[tuple[Decimal, str]] = []
    if product.min_price is not None:
        bounds.append((product.min_price, "min_price"))
    if product.cost is not None:
        margin = product.cost * (1 + rule.min_margin_percent / 100)
        bounds.append((margin, f"cost+{_num(rule.min_margin_percent)}% margin"))
    if not bounds:
        return None, ""
    value, source = max(bounds, key=lambda b: b[0])
    return value.quantize(CENT, rounding=ROUND_CEILING), source


def _with_ending(
    price: Decimal, ending: Decimal, floor: Decimal | None, ceiling: Decimal | None
) -> Decimal | None:
    """Largest price with fractional part ``ending`` that is <= ``price``; if that drops
    below the floor (or zero), the smallest such price >= floor. ``None`` if neither
    fits under the ceiling."""
    candidate = price.to_integral_value(rounding=ROUND_FLOOR) + ending
    if candidate > price:
        candidate -= 1
    lower = floor if floor is not None else _ZERO
    if candidate < lower:
        candidate = lower.to_integral_value(rounding=ROUND_FLOOR) + ending
        if candidate < lower:
            candidate += 1
    if ceiling is not None and candidate > ceiling:
        return None
    return quantize(candidate)


def _target(
    product: Product, rule: RepricingRule, considered: list[CompetitorPrice]
) -> tuple[Decimal | None, str]:
    """Unclamped price for the rule's strategy, plus a description of how it was chosen."""
    cheapest = min(considered, key=lambda o: (o.price, o.competitor))
    lowest = f"lowest ({cheapest.competitor} @ {cheapest.price})"
    if rule.strategy == "beat_lowest":
        if rule.undercut_percent is not None:
            cut = cheapest.price * rule.undercut_percent / 100
            how = f"{_num(rule.undercut_percent)}% ({quantize(cut)})"
        else:
            cut, how = rule.undercut_amount, str(rule.undercut_amount)
        return cheapest.price - cut, f"beat {lowest} by {how}"
    if rule.strategy == "match_lowest":
        return cheapest.price, f"match {lowest}"
    if rule.strategy == "median":
        mid = quantize(median(o.price for o in considered))
        return mid, f"median of {len(considered)} competitor offers ({mid})"
    if product.our_price is None:
        return None, "hold: no current price"
    return product.our_price, f"hold current price ({product.our_price})"


def suggest_price(
    product: Product, snapshot: MarketSnapshot, rule: RepricingRule
) -> PriceSuggestion:
    """Suggest a price for ``product`` following ``rule``, within its floor and ceiling.

    The floor is the higher of ``product.min_price`` and cost plus the minimum margin
    (rounded up to cents); the ceiling is ``product.max_price`` (rounded down). When the
    floor exceeds the ceiling the floor wins. Without any qualifying competitor offer
    the current price is kept.
    """
    current = product.our_price
    floor, floor_source = _floor(product, rule)
    ceiling = None if product.max_price is None else product.max_price.quantize(CENT, ROUND_FLOOR)
    considered = [
        o
        for o in snapshot.offers
        if o.match_score >= rule.min_match_score
        and not (rule.ignore_out_of_stock and o.availability is Availability.OUT_OF_STOCK)
    ]

    def result(suggested: Decimal | None, reason: str) -> PriceSuggestion:
        change = None
        if suggested is not None and current:
            change = round(float((suggested - current) / current * 100), 2)
        return PriceSuggestion(
            sku=product.sku,
            current_price=current,
            suggested_price=suggested,
            floor=floor,
            ceiling=ceiling,
            strategy=rule.strategy,
            reason=reason,
            change_percent=change,
        )

    if not considered:
        return result(current, "no competitor data")

    target, explanation = _target(product, rule, considered)
    if target is None:
        return result(None, explanation)
    reasons = [explanation]
    price = max(quantize(target), _ZERO)

    if floor is not None and ceiling is not None and floor > ceiling:
        price = floor
        reasons.append(f"floor {floor} ({floor_source}) exceeds ceiling {ceiling}; floor wins")
        ceiling_bound = None
    else:
        ceiling_bound = ceiling
        if floor is not None:
            if price < floor:
                price = floor
                reasons.append(f"raised to floor {floor} from {floor_source}")
            else:
                reasons.append(f"floor {floor} from {floor_source}")
        if ceiling is not None and price > ceiling:
            price = ceiling
            reasons.append(f"capped at ceiling {ceiling} (max_price)")

    if rule.price_ending is not None:
        ending = quantize(rule.price_ending % 1)
        ended = _with_ending(price, ending, floor, ceiling_bound)
        if ended is None:
            reasons.append(f"no price ending in {ending} fits between floor and ceiling")
        else:
            price = ended
            reasons.append(f"price ending {ending}")

    return result(price, "; ".join(reasons))


def reprice(
    product: Product,
    observations: list[Observation],
    settings: Settings,
    rule: RepricingRule,
    min_match_score: float = 0.0,
) -> PriceSuggestion:
    """:func:`suggest_price` over a market built only from the listings ``rule`` trusts.

    Each competitor's offer is chosen among its listings scoring at least
    ``rule.min_match_score``, so a cheaper but doubtful listing cannot hide the same
    competitor's confident one from the repricing.
    """
    trusted = market_snapshot(
        product, observations, settings, max(min_match_score, rule.min_match_score)
    )
    return suggest_price(product, trusted, rule)
