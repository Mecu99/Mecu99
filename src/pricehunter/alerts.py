"""Turn price movements into alerts and deliver them to notification channels."""

from __future__ import annotations

import asyncio
import os
import re
import smtplib
import ssl
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import (
    Alert,
    AlertRule,
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    NotifierConfig,
    Observation,
    Product,
)
from .money import normalize_currency, quantize

SLACK_CHUNK_CHARS = 3500
DISCORD_CHUNK_CHARS = 2000
HTTP_TIMEOUT = 10.0
SMTP_TIMEOUT = 30.0

_BUYABLE = frozenset({Availability.IN_STOCK, Availability.LIMITED, Availability.PREORDER})

# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True, slots=True)
class _Market:
    product: Product
    pairs: Sequence[tuple[Observation, Observation | None]]
    snapshot: MarketSnapshot
    prev_snapshot: MarketSnapshot | None

    @property
    def label(self) -> str:
        return f"{self.product.name} ({self.product.sku})"


def _money(amount: Decimal | None, currency: str | None) -> str:
    if amount is None:
        return "n/a"
    return f"{quantize(amount)} {currency or ''}".rstrip()


def _pct(new: Decimal, old: Decimal) -> float:
    """Signed change from ``old`` to ``new`` in percent (``old`` must be non-zero)."""
    return float((new - old) / old * 100)


def _wants(rule: AlertRule, competitor: str | None) -> bool:
    return rule.competitors is None or competitor in rule.competitors


def _ccy(obs: Observation) -> str | None:
    return normalize_currency(obs.currency) or obs.currency


def _in_stock(obs: Observation) -> bool | None:
    """Whether the listing can be bought: ``None`` when the reading says nothing.

    A priced listing without an availability signal counts as buyable, as in
    :mod:`pricehunter.analyze`; a failed or unpriced UNKNOWN reading does not.
    """
    if obs.error is not None:
        return None
    if obs.availability is Availability.OUT_OF_STOCK:
        return False
    if obs.availability in _BUYABLE or obs.price is not None:
        return True
    return None


def _undercut_by(offer: CompetitorPrice, ours: Decimal | None, threshold: float) -> float | None:
    """How far (in %) a buyable ``offer`` is below ``ours``, if at least ``threshold``."""
    if ours is None or ours <= 0 or offer.availability is Availability.OUT_OF_STOCK:
        return None
    if offer.price >= ours:
        return None
    below = -_pct(offer.price, ours)
    return below if below >= threshold else None


def _undercut(m: _Market, rule: AlertRule) -> Iterator[Alert]:
    ours, currency = m.product.our_price, m.snapshot.currency
    prev = m.prev_snapshot
    prev_offers = {o.competitor: o for o in prev.offers} if prev else {}
    prev_ours = prev.our_price if prev and prev.our_price is not None else ours
    for offer in m.snapshot.offers:
        if not _wants(rule, offer.competitor):
            continue
        below = _undercut_by(offer, ours, rule.threshold_percent)
        if below is None:
            continue
        before = prev_offers.get(offer.competitor)
        if (
            prev is not None
            and before is not None
            and prev.currency == currency
            and _undercut_by(before, prev_ours, rule.threshold_percent) is not None
        ):
            continue  # already undercutting us at the previous scan
        yield Alert(
            type="undercut",
            sku=m.product.sku,
            competitor=offer.competitor,
            url=offer.url,
            message=(
                f"{offer.competitor} undercuts {m.label}: {_money(offer.price, currency)}"
                f" vs our {_money(ours, currency)} ({-below:+.1f}%)"
            ),
            old_price=ours,
            new_price=offer.price,
            change_percent=round(-below, 2),
        )


def _price_change(m: _Market, rule: AlertRule) -> Iterator[Alert]:
    falling = rule.type == "price_drop"
    for cur, prev in m.pairs:
        if prev is None or cur.error is not None or not _wants(rule, cur.competitor):
            continue
        if cur.price is None or prev.price is None or prev.price <= 0 or _ccy(cur) != _ccy(prev):
            continue
        change = _pct(cur.price, prev.price)
        if change == 0 or (change < 0) != falling or abs(change) < rule.threshold_percent:
            continue
        verb = "dropped" if falling else "raised"
        currency = _ccy(cur)
        yield Alert(
            type=rule.type,
            sku=m.product.sku,
            competitor=cur.competitor,
            url=cur.url,
            message=(
                f"{cur.competitor} {verb} the price of {m.label} from"
                f" {_money(prev.price, currency)} to {_money(cur.price, currency)} ({change:+.1f}%)"
            ),
            old_price=prev.price,
            new_price=cur.price,
            change_percent=round(change, 2),
        )


def _stock_change(m: _Market, rule: AlertRule) -> Iterator[Alert]:
    restocked = rule.type == "back_in_stock"
    wanted = (False, True) if restocked else (True, False)
    for cur, prev in m.pairs:
        if prev is None or not _wants(rule, cur.competitor):
            continue
        if (_in_stock(prev), _in_stock(cur)) != wanted:
            continue
        if restocked:
            text = f"{cur.competitor} has {m.label} back in stock at {_money(cur.price, _ccy(cur))}"
        else:
            text = (
                f"{cur.competitor} is out of stock of {m.label}"
                f" (last price {_money(prev.price, _ccy(prev))})"
            )
        yield Alert(
            type=rule.type,
            sku=m.product.sku,
            competitor=cur.competitor,
            url=cur.url,
            message=text,
            old_price=prev.price,
            new_price=cur.price,
        )


def _new_lowest(m: _Market, rule: AlertRule) -> Iterator[Alert]:
    snap, prev = m.snapshot, m.prev_snapshot
    if prev is None or prev.lowest is None or snap.lowest is None or prev.currency != snap.currency:
        return
    if snap.lowest >= prev.lowest or not _wants(rule, snap.lowest_competitor):
        return
    change = _pct(snap.lowest, prev.lowest)
    if -change < rule.threshold_percent:
        return
    url = next(
        (
            o.url
            for o in snap.offers
            if o.competitor == snap.lowest_competitor and o.price == snap.lowest
        ),
        None,
    )
    yield Alert(
        type="new_lowest",
        sku=m.product.sku,
        competitor=snap.lowest_competitor,
        url=url,
        message=(
            f"New market low for {m.label}: {_money(snap.lowest, snap.currency)}"
            f" at {snap.lowest_competitor} (was {_money(prev.lowest, prev.currency)},"
            f" {change:+.1f}%)"
        ),
        old_price=prev.lowest,
        new_price=snap.lowest,
        change_percent=round(change, 2),
    )


_EVALUATORS: dict[str, Callable[[_Market, AlertRule], Iterator[Alert]]] = {
    "undercut": _undercut,
    "price_drop": _price_change,
    "price_rise": _price_change,
    "out_of_stock": _stock_change,
    "back_in_stock": _stock_change,
    "new_lowest": _new_lowest,
}


def evaluate_alerts(
    product: Product,
    pairs: Sequence[tuple[Observation, Observation | None]],
    snapshot: MarketSnapshot,
    prev_snapshot: MarketSnapshot | None,
    rules: list[AlertRule],
) -> list[Alert]:
    """Alerts for one product, in rule order, without duplicates.

    ``pairs`` holds (current observation, previous successful observation of the same
    listing or ``None``); ``snapshot``/``prev_snapshot`` are the market now and at the
    previous scan, in the product's currency.

    * ``undercut``: a buyable competitor offer is at least ``threshold_percent`` below
      ``our_price`` and was not already that far below at the previous scan.
    * ``price_drop`` / ``price_rise``: a listing's price moved by at least
      ``threshold_percent`` (0 = any change) in the same currency.
    * ``out_of_stock`` / ``back_in_stock``: a listing's availability flipped.
    * ``new_lowest``: the market low fell (by at least ``threshold_percent``).
    """
    market = _Market(product, pairs, snapshot, prev_snapshot)
    alerts: list[Alert] = []
    seen: set[tuple[Any, ...]] = set()
    for rule in rules:
        if rule.skus is not None and product.sku not in rule.skus:
            continue
        for alert in _EVALUATORS[rule.type](market, rule):
            key = (
                alert.type,
                alert.sku,
                alert.competitor,
                alert.url,
                alert.old_price,
                alert.new_price,
            )
            if key not in seen:
                seen.add(key)
                alerts.append(alert)
    return alerts


# --------------------------------------------------------------------------- notifiers


class Notifier(Protocol):
    async def send(self, alerts: list[Alert]) -> None:
        """Deliver ``alerts``; raises on delivery failure (``dispatch`` collects errors)."""
        ...


def _is_web_url(url: str | None) -> bool:
    if not url:
        return False
    parts = urlsplit(url.strip())
    return parts.scheme.lower() in {"http", "https"} and bool(parts.netloc)


def _summary(alerts: list[Alert]) -> str:
    n = len(alerts)
    return f"{n} price alert{'' if n == 1 else 's'}"


def _chunks(lines: list[str], limit: int) -> list[str]:
    """Join ``lines`` with newlines into messages of at most ``limit`` characters."""
    chunks: list[str] = []
    current = ""
    for line in lines:
        pieces = [line[i : i + limit] for i in range(0, len(line), limit)] or [""]
        for piece in pieces:
            candidate = f"{current}\n{piece}" if current else piece
            if len(candidate) > limit:
                chunks.append(current)
                candidate = piece
            current = candidate
    if current:
        chunks.append(current)
    return chunks


@asynccontextmanager
async def _client_scope(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """Use ``client`` as is, or a short-lived one that is closed afterwards."""
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as own:
        yield own


class ConsoleNotifier:
    """Print alerts as a table on the terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    async def send(self, alerts: list[Alert]) -> None:
        table = Table(title=f"PriceHunter: {_summary(alerts)}", show_lines=False)
        for column in ("Type", "SKU", "Competitor", "Change", "Message"):
            table.add_column(column, overflow="fold")
        for a in alerts:
            # change_percent is already rounded to 2 places; rounding again could contradict the message.
            change = "" if a.change_percent is None else f"{a.change_percent:+.2f}%"
            # Text() keeps product/competitor names from being parsed as rich markup.
            table.add_row(
                Text(a.type), Text(a.sku), Text(a.competitor or ""), Text(change), Text(a.message)
            )
        self.console.print(table)


class FileNotifier:
    """Append alerts to a JSON-lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def send(self, alerts: list[Alert]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.writelines(a.model_dump_json() + "\n" for a in alerts)


class _HttpNotifier:
    def __init__(self, url: str, client: httpx.AsyncClient | None = None) -> None:
        self.url = url
        self.client = client

    def payloads(self, alerts: list[Alert]) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def send(self, alerts: list[Alert]) -> None:
        async with _client_scope(self.client) as client:
            for payload in self.payloads(alerts):  # sequential, so chunks arrive in order
                response = await client.post(self.url, json=payload)
                response.raise_for_status()


class WebhookNotifier(_HttpNotifier):
    """POST ``{"alerts": [...]}`` with every alert as JSON."""

    def payloads(self, alerts: list[Alert]) -> list[dict[str, Any]]:
        return [{"alerts": [a.model_dump(mode="json") for a in alerts]}]


def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackNotifier(_HttpNotifier):
    """Slack incoming webhook: a bullet list, split into messages of <= 3500 chars."""

    def payloads(self, alerts: list[Alert]) -> list[dict[str, Any]]:
        lines = [f"*PriceHunter*: {_summary(alerts)}"]
        for a in alerts:
            link = f" <{_slack_escape(a.url.strip())}|view>" if a.url and _is_web_url(a.url) else ""
            lines.append(f"• {_slack_escape(a.message)}{link}")
        return [{"text": chunk} for chunk in _chunks(lines, SLACK_CHUNK_CHARS)]


_DISCORD_MARKDOWN = re.compile(r"([\\*_~`|])")


class DiscordNotifier(_HttpNotifier):
    """Discord webhook: a bullet list, split into messages of <= 2000 chars, no pings."""

    def payloads(self, alerts: list[Alert]) -> list[dict[str, Any]]:
        lines = [f"**PriceHunter**: {_summary(alerts)}"]
        for a in alerts:
            # <url> links without generating an embed preview per alert.
            text = _DISCORD_MARKDOWN.sub(r"\\\1", a.message)
            link = f" <{a.url.strip()}>" if a.url and _is_web_url(a.url) else ""
            lines.append(f"- {text}{link}")
        return [
            {"content": chunk, "allowed_mentions": {"parse": []}}
            for chunk in _chunks(lines, DISCORD_CHUNK_CHARS)
        ]


class EmailNotifier:
    """Plain-text e-mail over SMTP (STARTTLS when configured, implicit TLS on port 465)."""

    def __init__(self, cfg: NotifierConfig) -> None:
        self.cfg = cfg

    def message(self, alerts: list[Alert]) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = f"PriceHunter: {_summary(alerts)}"
        msg["From"] = self.cfg.email_from or ""
        msg["To"] = ", ".join(self.cfg.email_to)
        body = [f"PriceHunter found {_summary(alerts)}:", ""]
        for a in alerts:
            body.append(f"- {a.message}")
            if a.url:
                body.append(f"  {a.url}")
        msg.set_content("\n".join(body) + "\n")
        return msg

    async def send(self, alerts: list[Alert]) -> None:
        password: str | None = None
        if self.cfg.smtp_password_env:
            password = os.environ.get(self.cfg.smtp_password_env)
            if password is None:
                raise RuntimeError(f"environment variable {self.cfg.smtp_password_env} is not set")
        await asyncio.to_thread(self._deliver, self.message(alerts), password)

    def _deliver(self, msg: EmailMessage, password: str | None) -> None:
        cfg = self.cfg
        host = cfg.smtp_host or ""
        smtp: smtplib.SMTP
        if cfg.smtp_port == 465:
            smtp = smtplib.SMTP_SSL(
                host, cfg.smtp_port, timeout=SMTP_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            smtp = smtplib.SMTP(host, cfg.smtp_port, timeout=SMTP_TIMEOUT)
        with smtp:
            if cfg.smtp_starttls and cfg.smtp_port != 465:
                smtp.starttls(context=ssl.create_default_context())
            if cfg.smtp_user and password is not None:
                smtp.login(cfg.smtp_user, password)
            smtp.send_message(msg)


_HTTP_NOTIFIERS: dict[str, type[_HttpNotifier]] = {
    "webhook": WebhookNotifier,
    "slack": SlackNotifier,
    "discord": DiscordNotifier,
}


def build_notifier(cfg: NotifierConfig, client: httpx.AsyncClient | None = None) -> Notifier:
    """Create the notifier ``cfg`` describes; ``ValueError`` if it is misconfigured."""
    if cfg.type == "console":
        return ConsoleNotifier()
    if cfg.type == "file":
        if not cfg.path:
            raise ValueError("file notifier needs 'path'")
        return FileNotifier(cfg.path)
    if cfg.type in _HTTP_NOTIFIERS:
        if not _is_web_url(cfg.url):
            raise ValueError(f"{cfg.type} notifier needs an http(s) 'url'")
        return _HTTP_NOTIFIERS[cfg.type](cfg.url or "", client)
    if cfg.type == "email":
        missing = [f for f in ("smtp_host", "email_from", "email_to") if not getattr(cfg, f)]
        if missing:
            raise ValueError(f"email notifier needs {', '.join(repr(f) for f in missing)}")
        if bool(cfg.smtp_user) != bool(cfg.smtp_password_env):
            raise ValueError(
                "email notifier needs both 'smtp_user' and 'smtp_password_env', or neither"
            )
        return EmailNotifier(cfg)
    raise ValueError(f"unknown notifier type {cfg.type!r}")


def _describe(exc: BaseException) -> str:
    """Error text without full webhook URLs, which embed secrets."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {exc.request.url.host}"
    if isinstance(exc, httpx.RequestError):
        return f"{type(exc).__name__} contacting {exc.request.url.host}: {exc}"
    return f"{type(exc).__name__}: {exc}"


async def dispatch(
    alerts: list[Alert],
    configs: list[NotifierConfig],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Send ``alerts`` through every configured notifier concurrently.

    Does nothing without alerts. Never raises: configuration and delivery problems
    come back as one error string per failing notifier.
    """
    if not alerts or not configs:
        return []
    errors: list[str] = []
    async with _client_scope(client) as http:
        notifiers: list[tuple[str, Notifier]] = []
        for cfg in configs:
            try:
                notifiers.append((cfg.type, build_notifier(cfg, http)))
            except ValueError as exc:
                errors.append(str(exc))
        results = await asyncio.gather(
            *(notifier.send(alerts) for _, notifier in notifiers), return_exceptions=True
        )
    for (name, _), result in zip(notifiers, results):
        if isinstance(result, BaseException):
            errors.append(f"{name} notifier failed: {_describe(result)}")
    return errors
