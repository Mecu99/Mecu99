from __future__ import annotations

import io
import json
import re
import smtplib
from collections.abc import Sequence
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import respx
from rich.console import Console
from typing_extensions import Self

from pricehunter.alerts import (
    DISCORD_CHUNK_CHARS,
    SLACK_CHUNK_CHARS,
    ConsoleNotifier,
    DiscordNotifier,
    EmailNotifier,
    FileNotifier,
    SlackNotifier,
    WebhookNotifier,
    build_notifier,
    dispatch,
    evaluate_alerts,
)
from pricehunter.models import (
    Alert,
    AlertRule,
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    NotifierConfig,
    Observation,
    Product,
)

D = Decimal
A = Availability
PRODUCT = Product(sku="SKU-1", name="Widget Pro", our_price=D("100.00"), currency="EUR")


def obs(
    competitor: str,
    price: str | None,
    availability: Availability = A.IN_STOCK,
    *,
    currency: str | None = "EUR",
    error: str | None = None,
) -> Observation:
    return Observation(
        sku="SKU-1",
        competitor=competitor,
        url=f"https://{competitor.lower()}.example/widget",
        price=D(price) if price is not None else None,
        currency=currency,
        availability=availability,
        error=error,
    )


def offer(competitor: str, price: str, availability: Availability = A.IN_STOCK) -> CompetitorPrice:
    return CompetitorPrice(
        competitor=competitor,
        url=f"https://{competitor.lower()}.example/widget",
        price=D(price),
        original_currency="EUR",
        availability=availability,
        match_score=1.0,
    )


def snap(*offers: CompetitorPrice, our_price: str | None = "100.00") -> MarketSnapshot:
    ordered = sorted(offers, key=lambda o: o.price)
    available = [o for o in ordered if o.availability is not A.OUT_OF_STOCK]
    return MarketSnapshot(
        sku="SKU-1",
        currency="EUR",
        our_price=D(our_price) if our_price is not None else None,
        offers=ordered,
        lowest=available[0].price if available else None,
        lowest_competitor=available[0].competitor if available else None,
        in_stock_offers=len(available),
    )


def run(
    rules: list[AlertRule],
    *,
    pairs: Sequence[tuple[Observation, Observation | None]] = (),
    now: MarketSnapshot | None = None,
    before: MarketSnapshot | None = None,
    product: Product = PRODUCT,
) -> list[Alert]:
    return evaluate_alerts(product, pairs, now or snap(), before, rules)


def alert(competitor: str = "Acme", message: str = "Acme dropped Widget", **kw: Any) -> Alert:
    return Alert(
        type=kw.pop("type", "price_drop"),
        sku="SKU-1",
        competitor=competitor,
        url=kw.pop("url", "https://acme.example/widget"),
        message=message,
        old_price=D("100.00"),
        new_price=D("89.99"),
        change_percent=-10.01,
        **kw,
    )


# --------------------------------------------------------------------------- undercut


def test_undercut_fires_on_first_scan_with_readable_message() -> None:
    now = snap(offer("Acme", "89.99"), offer("Bolt", "105.00"))
    [a] = run([AlertRule(type="undercut")], now=now)
    assert (a.type, a.sku, a.competitor, a.url) == (
        "undercut",
        "SKU-1",
        "Acme",
        "https://acme.example/widget",
    )
    assert (a.old_price, a.new_price, a.change_percent) == (D("100.00"), D("89.99"), -10.01)
    assert a.message == "Acme undercuts Widget Pro (SKU-1): 89.99 EUR vs our 100.00 EUR (-10.0%)"


def test_undercut_threshold_and_stock() -> None:
    now = snap(offer("Acme", "97.00"), offer("Bolt", "80.00", A.OUT_OF_STOCK), offer("Cart", "95"))
    alerts = run([AlertRule(type="undercut", threshold_percent=5)], now=now)
    assert [a.competitor for a in alerts] == ["Cart"]  # 3% too small, Bolt not buyable
    equal = snap(offer("Acme", "100.00"))
    assert run([AlertRule(type="undercut")], now=equal) == []


def test_undercut_counts_unknown_availability_as_buyable() -> None:
    now = snap(offer("Acme", "90", A.UNKNOWN))
    assert [a.competitor for a in run([AlertRule(type="undercut")], now=now)] == ["Acme"]


def test_undercut_not_repeated_while_competitor_stays_cheaper() -> None:
    rule = AlertRule(type="undercut")
    before = snap(
        offer("Acme", "95.00"), offer("Bolt", "110.00"), offer("Cart", "90", A.OUT_OF_STOCK)
    )
    now = snap(
        offer("Acme", "93.00"), offer("Bolt", "99.00"), offer("Cart", "90"), offer("Dune", "98")
    )
    # Acme was already cheaper; Bolt was dearer; Cart was not buyable; Dune is new.
    assert sorted(str(a.competitor) for a in run([rule], now=now, before=before)) == [
        "Bolt",
        "Cart",
        "Dune",
    ]


def test_undercut_fires_when_crossing_threshold() -> None:
    rule = AlertRule(type="undercut", threshold_percent=5)
    before = snap(offer("Acme", "98.00"))  # cheaper, but only by 2%
    now = snap(offer("Acme", "90.00"))
    assert [a.competitor for a in run([rule], now=now, before=before)] == ["Acme"]


def test_undercut_compares_previous_offer_with_previous_own_price() -> None:
    # We were at 80 then (Acme 85 was dearer); we raised to 100, so Acme now undercuts us.
    before = snap(offer("Acme", "85.00"), our_price="80.00")
    now = snap(offer("Acme", "85.00"))
    assert len(run([AlertRule(type="undercut")], now=now, before=before)) == 1


def test_undercut_needs_our_price() -> None:
    unpriced = PRODUCT.model_copy(update={"our_price": None})
    now = snap(offer("Acme", "1.00"), our_price=None)
    assert run([AlertRule(type="undercut")], now=now, product=unpriced) == []


def test_undercut_uses_the_products_own_price() -> None:
    cheaper = PRODUCT.model_copy(update={"our_price": D("80.00")})
    now = snap(offer("Acme", "90.00"))  # snapshot still says 100.00
    assert run([AlertRule(type="undercut")], now=now, product=cheaper) == []


# --------------------------------------------------------------------------- price moves


def test_price_drop_and_rise() -> None:
    pairs = [
        (obs("Acme", "90.00"), obs("Acme", "100.00")),
        (obs("Bolt", "110.00"), obs("Bolt", "100.00")),
        (obs("Cart", "50.00"), obs("Cart", "50.00")),
    ]
    rules = [AlertRule(type="price_drop"), AlertRule(type="price_rise")]
    drop, rise = run(rules, pairs=pairs)
    assert (drop.type, drop.competitor, drop.change_percent) == ("price_drop", "Acme", -10.0)
    assert (drop.old_price, drop.new_price) == (D("100.00"), D("90.00"))
    assert drop.message == (
        "Acme dropped the price of Widget Pro (SKU-1) from 100.00 EUR to 90.00 EUR (-10.0%)"
    )
    assert (rise.type, rise.competitor, rise.change_percent) == ("price_rise", "Bolt", 10.0)
    assert "raised" in rise.message and "+10.0%" in rise.message


@pytest.mark.parametrize(
    ("threshold", "fires"), [(0, True), (4.99, True), (5, True), (5.01, False), (20, False)]
)
def test_price_drop_threshold(threshold: float, fires: bool) -> None:
    pairs = [(obs("Acme", "95.00"), obs("Acme", "100.00"))]
    alerts = run([AlertRule(type="price_drop", threshold_percent=threshold)], pairs=pairs)
    assert bool(alerts) is fires


def test_price_change_needs_comparable_readings() -> None:
    rules = [AlertRule(type="price_drop"), AlertRule(type="price_rise")]
    pairs: list[tuple[Observation, Observation | None]] = [
        (obs("Acme", "90"), None),  # nothing to compare with
        (obs("Bolt", "90", currency="USD"), obs("Bolt", "100")),  # currency changed
        (obs("Cart", None), obs("Cart", "100")),  # no current price
        (obs("Dune", "90", error="HTTP 500"), obs("Dune", "100")),  # failed reading
        (obs("Echo", "90", currency="€"), obs("Echo", "100", currency="eur")),  # same currency
    ]
    assert [a.competitor for a in run(rules, pairs=pairs)] == ["Echo"]


# --------------------------------------------------------------------------- stock


def test_out_of_stock_and_back_in_stock() -> None:
    pairs = [
        (obs("Acme", "90", A.OUT_OF_STOCK), obs("Acme", "90")),
        (obs("Bolt", "95", A.IN_STOCK), obs("Bolt", "99", A.OUT_OF_STOCK)),
        (obs("Cart", "95", A.LIMITED), obs("Cart", "95", A.PREORDER)),  # still buyable
        (obs("Dune", "95", A.OUT_OF_STOCK), obs("Dune", "95", A.OUT_OF_STOCK)),
        (obs("Echo", "95", A.OUT_OF_STOCK), None),
    ]
    rules = [AlertRule(type="out_of_stock"), AlertRule(type="back_in_stock")]
    out, back = run(rules, pairs=pairs)
    assert (out.type, out.competitor) == ("out_of_stock", "Acme")
    assert out.message == "Acme is out of stock of Widget Pro (SKU-1) (last price 90.00 EUR)"
    assert (back.type, back.competitor, back.old_price, back.new_price) == (
        "back_in_stock",
        "Bolt",
        D("99"),
        D("95"),
    )
    assert back.message == "Bolt has Widget Pro (SKU-1) back in stock at 95.00 EUR"


def test_stock_transitions_with_unknown_availability() -> None:
    rules = [AlertRule(type="out_of_stock"), AlertRule(type="back_in_stock")]
    pairs = [
        # Priced with no stock signal = buyable, so these are real transitions.
        (obs("Acme", None, A.OUT_OF_STOCK), obs("Acme", "90", A.UNKNOWN)),
        (obs("Bolt", "90", A.UNKNOWN), obs("Bolt", "90", A.OUT_OF_STOCK)),
        # No price and no stock signal, or a failed fetch, tells us nothing.
        (obs("Cart", None, A.UNKNOWN), obs("Cart", "90", A.OUT_OF_STOCK)),
        (obs("Dune", None, A.UNKNOWN, error="timeout"), obs("Dune", "90", A.OUT_OF_STOCK)),
    ]
    assert [(a.type, a.competitor) for a in run(rules, pairs=pairs)] == [
        ("out_of_stock", "Acme"),
        ("back_in_stock", "Bolt"),
    ]


# --------------------------------------------------------------------------- new lowest


def test_new_lowest() -> None:
    before = snap(offer("Acme", "90.00"), offer("Bolt", "95.00"))
    now = snap(offer("Acme", "90.00"), offer("Bolt", "85.00"))
    [a] = run([AlertRule(type="new_lowest")], now=now, before=before)
    assert (a.type, a.competitor, a.url) == ("new_lowest", "Bolt", "https://bolt.example/widget")
    assert (a.old_price, a.new_price, a.change_percent) == (D("90.00"), D("85.00"), -5.56)
    assert (
        a.message
        == "New market low for Widget Pro (SKU-1): 85.00 EUR at Bolt (was 90.00 EUR, -5.6%)"
    )


@pytest.mark.parametrize(
    ("before", "now", "rule"),
    [
        (None, snap(offer("Acme", "80")), AlertRule(type="new_lowest")),
        (snap(), snap(offer("Acme", "80")), AlertRule(type="new_lowest")),  # no previous low
        (snap(offer("Acme", "80")), snap(offer("Acme", "80")), AlertRule(type="new_lowest")),
        (snap(offer("Acme", "80")), snap(offer("Acme", "85")), AlertRule(type="new_lowest")),
        (
            snap(offer("Acme", "80")),
            snap(offer("Acme", "79")),
            AlertRule(type="new_lowest", threshold_percent=5),
        ),
        (
            snap(offer("Acme", "80")),
            snap(offer("Bolt", "70")),
            AlertRule(type="new_lowest", competitors=["Acme"]),
        ),
    ],
)
def test_new_lowest_not_fired(
    before: MarketSnapshot | None, now: MarketSnapshot, rule: AlertRule
) -> None:
    assert run([rule], now=now, before=before) == []


# --------------------------------------------------------------------------- filters, dedupe


def test_competitor_and_sku_filters() -> None:
    pairs = [(obs("Acme", "90"), obs("Acme", "100")), (obs("Bolt", "80"), obs("Bolt", "100"))]
    only_bolt = AlertRule(type="price_drop", competitors=["Bolt"])
    assert [a.competitor for a in run([only_bolt], pairs=pairs)] == ["Bolt"]
    other_sku = AlertRule(type="price_drop", skus=["SKU-2"])
    assert run([other_sku], pairs=pairs) == []
    this_sku = AlertRule(type="price_drop", skus=["SKU-1"], competitors=["Acme", "Bolt"])
    assert len(run([this_sku], pairs=pairs)) == 2
    now = snap(offer("Acme", "80"), offer("Bolt", "80"))
    undercut = AlertRule(type="undercut", competitors=["Acme"])
    assert [a.competitor for a in run([undercut], now=now)] == ["Acme"]


def test_overlapping_rules_are_deduplicated() -> None:
    pairs = [(obs("Acme", "80"), obs("Acme", "100"))]
    rules = [
        AlertRule(type="price_drop"),
        AlertRule(type="price_drop", threshold_percent=10, competitors=["Acme"]),
        AlertRule(type="price_drop", skus=["SKU-1"]),
    ]
    assert len(run(rules, pairs=pairs)) == 1


def test_all_types_together_keep_rule_order() -> None:
    before = snap(offer("Acme", "100"), offer("Bolt", "105"))
    now = snap(offer("Acme", "90"), offer("Bolt", "105", A.OUT_OF_STOCK))
    pairs = [
        (obs("Acme", "90"), obs("Acme", "100")),
        (obs("Bolt", "105", A.OUT_OF_STOCK), obs("Bolt", "105")),
    ]
    rules = [
        AlertRule(type="new_lowest"),
        AlertRule(type="out_of_stock"),
        AlertRule(type="undercut"),
        AlertRule(type="price_drop"),
    ]
    assert [(a.type, a.competitor) for a in run(rules, pairs=pairs, now=now, before=before)] == [
        ("new_lowest", "Acme"),
        ("out_of_stock", "Bolt"),
        ("undercut", "Acme"),
        ("price_drop", "Acme"),
    ]


# --------------------------------------------------------------------------- notifiers


def recorder(status: int = 200) -> tuple[list[httpx.Request], httpx.AsyncClient]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status)

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def test_webhook_posts_all_alerts_as_json() -> None:
    seen, client = recorder()
    alerts = [alert(), alert("Bolt", "Bolt rose", type="price_rise", url=None)]
    async with client:
        await WebhookNotifier("https://hooks.example/x", client).send(alerts)
    [request] = seen
    assert (request.method, str(request.url)) == ("POST", "https://hooks.example/x")
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content)
    assert [a["competitor"] for a in body["alerts"]] == ["Acme", "Bolt"]
    first = body["alerts"][0]
    assert (first["old_price"], first["new_price"], first["change_percent"]) == (
        "100.00",
        "89.99",
        -10.01,
    )
    assert [Alert.model_validate(a) for a in body["alerts"]] == alerts


async def test_slack_bullets_escaping_and_links() -> None:
    seen, client = recorder()
    alerts = [
        alert(message="Acme <b>&</b> deal"),
        alert("Bolt", "Bolt moved", url="javascript:alert(1)"),
    ]
    async with client:
        await SlackNotifier("https://hooks.slack.com/services/T/B/C", client).send(alerts)
    [request] = seen
    text = json.loads(request.content)["text"]
    assert text.splitlines() == [
        "*PriceHunter*: 2 price alerts",
        "• Acme &lt;b&gt;&amp;&lt;/b&gt; deal <https://acme.example/widget|view>",
        "• Bolt moved",
    ]


async def test_slack_chunks_long_messages_in_order() -> None:
    seen, client = recorder()
    alerts = [alert(message=f"alert {i:03d} " + "x" * 180) for i in range(60)]
    async with client:
        await SlackNotifier("https://hooks.slack.com/services/T/B/C", client).send(alerts)
    texts = [json.loads(r.content)["text"] for r in seen]
    assert len(texts) > 1
    assert all(len(t) <= SLACK_CHUNK_CHARS for t in texts)
    joined = "\n".join(texts)
    assert [line[2:11] for line in joined.splitlines()[1:]] == [f"alert {i:03d}" for i in range(60)]


async def test_discord_chunks_and_disables_mentions() -> None:
    seen, client = recorder()
    alerts = [alert(message=f"@everyone Widget_{i} " + "y" * 150) for i in range(40)]
    alerts.append(alert(message="z" * 2500))  # longer than one Discord message on its own
    async with client:
        await DiscordNotifier("https://discord.com/api/webhooks/1/abc", client).send(alerts)
    payloads = [json.loads(r.content) for r in seen]
    assert len(payloads) > 2
    assert all(len(p["content"]) <= DISCORD_CHUNK_CHARS for p in payloads)
    assert all(p["allowed_mentions"] == {"parse": []} for p in payloads)
    content = "".join(p["content"] for p in payloads)
    assert payloads[0]["content"].startswith(
        "**PriceHunter**: 41 price alerts\n- @everyone Widget\\_0 "
    )
    assert "<https://acme.example/widget>" in content
    assert content.count("z") == 2500


async def test_http_notifier_raises_on_error_status() -> None:
    _, client = recorder(status=500)
    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await WebhookNotifier("https://hooks.example/x", client).send([alert()])


async def test_file_notifier_appends_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "out" / "alerts.jsonl"
    notifier = FileNotifier(path)
    await notifier.send([alert()])
    await notifier.send([alert("Bolt"), alert("Cart")])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [Alert.model_validate_json(line).competitor for line in lines] == [
        "Acme",
        "Bolt",
        "Cart",
    ]
    assert json.loads(lines[0])["new_price"] == "89.99"


async def test_console_notifier_prints_table_without_markup() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None)
    await ConsoleNotifier(console).send([alert(message="Acme [bold]cheap[/] deal")])
    out = buffer.getvalue()
    assert "PriceHunter: 1 price alert" in out
    assert "Acme [bold]cheap[/] deal" in out
    assert "-10.01%" in out


class FakeSMTP:
    instances: ClassVar[list[FakeSMTP]] = []

    def __init__(self, host: str, port: int, **kwargs: Any) -> None:
        self.host, self.port, self.kwargs = host, port, kwargs
        self.calls: list[str] = []
        self.login_args: tuple[str, str] | None = None
        self.sent: list[EmailMessage] = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append("quit")

    def starttls(self, **kwargs: Any) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append("login")
        self.login_args = (user, password)

    def send_message(self, msg: EmailMessage) -> None:
        self.calls.append("send")
        self.sent.append(msg)


@pytest.fixture
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def email_cfg(**kw: Any) -> NotifierConfig:
    base: dict[str, Any] = {
        "type": "email",
        "smtp_host": "smtp.example.com",
        "smtp_user": "bot",
        "smtp_password_env": "PH_SMTP_PASSWORD",
        "email_from": "bot@example.com",
        "email_to": ["a@example.com", "b@example.com"],
    }
    return NotifierConfig(**(base | kw))


async def test_email_notifier_starttls_login_and_body(
    fake_smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PH_SMTP_PASSWORD", "s3cret")
    await EmailNotifier(email_cfg()).send([alert(), alert("Bolt", "Bolt rose", url=None)])
    [smtp] = fake_smtp.instances
    assert (smtp.host, smtp.port) == ("smtp.example.com", 587)
    assert smtp.calls == ["starttls", "login", "send", "quit"]
    assert smtp.login_args == ("bot", "s3cret")
    [msg] = smtp.sent
    assert msg["Subject"] == "PriceHunter: 2 price alerts"
    assert (msg["From"], msg["To"]) == ("bot@example.com", "a@example.com, b@example.com")
    assert msg.get_content_type() == "text/plain"
    body = msg.get_content()
    assert "- Acme dropped Widget\n  https://acme.example/widget\n- Bolt rose\n" in body


async def test_email_notifier_without_auth_or_tls(fake_smtp: type[FakeSMTP]) -> None:
    cfg = email_cfg(smtp_user=None, smtp_password_env=None, smtp_starttls=False, smtp_port=25)
    await EmailNotifier(cfg).send([alert()])
    assert fake_smtp.instances[0].calls == ["send", "quit"]


async def test_email_notifier_port_465_uses_implicit_tls(
    fake_smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PH_SMTP_PASSWORD", "pw")
    await EmailNotifier(email_cfg(smtp_port=465)).send([alert()])
    [smtp] = fake_smtp.instances
    assert "context" in smtp.kwargs
    assert smtp.calls == ["login", "send", "quit"]


async def test_email_notifier_missing_password_env(
    fake_smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PH_SMTP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="PH_SMTP_PASSWORD"):
        await EmailNotifier(email_cfg()).send([alert()])
    assert fake_smtp.instances == []


# --------------------------------------------------------------------------- build / dispatch


@pytest.mark.parametrize(
    ("cfg", "kind"),
    [
        (NotifierConfig(type="console"), ConsoleNotifier),
        (NotifierConfig(type="file", path="alerts.jsonl"), FileNotifier),
        (NotifierConfig(type="webhook", url="https://hooks.example/x"), WebhookNotifier),
        (NotifierConfig(type="slack", url="https://hooks.slack.com/services/x"), SlackNotifier),
        (NotifierConfig(type="discord", url="https://discord.com/api/webhooks/x"), DiscordNotifier),
        (email_cfg(), EmailNotifier),
    ],
)
def test_build_notifier(cfg: NotifierConfig, kind: type) -> None:
    assert isinstance(build_notifier(cfg), kind)


def test_build_notifier_passes_client() -> None:
    client = httpx.AsyncClient()
    notifier = build_notifier(NotifierConfig(type="webhook", url="https://h.example/x"), client)
    assert isinstance(notifier, WebhookNotifier) and notifier.client is client


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (NotifierConfig(type="file"), "'path'"),
        (NotifierConfig(type="webhook"), "http(s) 'url'"),
        (NotifierConfig(type="slack", url="hooks.slack.com/x"), "http(s) 'url'"),
        (NotifierConfig(type="discord", url="javascript:alert(1)"), "http(s) 'url'"),
        (email_cfg(smtp_host=None), "'smtp_host'"),
        (email_cfg(email_from=None, email_to=[]), "'email_from', 'email_to'"),
        (email_cfg(smtp_password_env=None), "smtp_password_env"),
    ],
)
def test_build_notifier_rejects_misconfiguration(cfg: NotifierConfig, message: str) -> None:
    with pytest.raises(ValueError, match=f"{cfg.type} notifier.*{re.escape(message)}"):
        build_notifier(cfg)


async def test_dispatch_noop_without_alerts() -> None:
    seen, client = recorder()
    async with client:
        cfgs = [
            NotifierConfig(type="webhook", url="https://hooks.example/x"),
            NotifierConfig(type="file"),
        ]
        assert await dispatch([], cfgs, client=client) == []
    assert seen == []


async def test_dispatch_collects_errors_without_leaking_webhook_secrets(tmp_path: Path) -> None:
    received: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "down.example":
            raise httpx.ConnectError("connection refused")
        received.append(str(request.url))
        return httpx.Response(403 if request.url.host == "hooks.slack.com" else 204)

    cfgs = [
        NotifierConfig(type="slack", url="https://hooks.slack.com/services/T0/B0/TOPSECRET"),
        NotifierConfig(type="webhook", url="https://ok.example/hook"),
        NotifierConfig(type="webhook", url="https://down.example/hook"),
        NotifierConfig(type="email"),  # misconfigured
        NotifierConfig(type="file", path=str(tmp_path / "alerts.jsonl")),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        errors = await dispatch([alert()], cfgs, client=client)
    assert sorted(received) == [
        "https://hooks.slack.com/services/T0/B0/TOPSECRET",
        "https://ok.example/hook",
    ]
    assert len(errors) == 3
    assert errors[0] == "email notifier needs 'smtp_host', 'email_from', 'email_to'"
    assert "slack notifier failed: HTTP 403 from hooks.slack.com" in errors
    assert (
        "webhook notifier failed: ConnectError contacting down.example: connection refused"
        in errors
    )
    assert not any("TOPSECRET" in e for e in errors)
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 1


async def test_dispatch_reports_email_failure(
    fake_smtp: type[FakeSMTP], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PH_SMTP_PASSWORD", raising=False)
    errors = await dispatch([alert()], [email_cfg()])
    assert errors == [
        "email notifier failed: RuntimeError: environment variable PH_SMTP_PASSWORD is not set"
    ]


@respx.mock
async def test_dispatch_creates_its_own_client() -> None:
    route = respx.post("https://hooks.example/x").mock(return_value=httpx.Response(200))
    errors = await dispatch(
        [alert()], [NotifierConfig(type="webhook", url="https://hooks.example/x")]
    )
    assert errors == []
    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content)["alerts"][0]["sku"] == "SKU-1"
