"""Command-line interface: ``pricehunter <command>`` (run ``pricehunter --help``)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .analyze import product_currency
from .config import load_config
from .demo import DemoMarket
from .engine import (
    build_snapshots,
    load_history,
    lookback_start,
    market_warnings,
    run_scan,
)
from .engine import probe as probe_page
from .example_config import EXAMPLE_CONFIG
from .models import (
    AppConfig,
    Availability,
    Competitor,
    MarketSnapshot,
    PriceSuggestion,
    ScanResult,
    ScanStats,
)
from .report import MAX_ALERTS_SHOWN, build_rows, render_html, write_csv, write_json
from .storage import Storage

app = typer.Typer(
    name="pricehunter",
    help="Competitor price intelligence: discover, match, track and reprice.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

DEFAULT_CONFIG = Path("pricehunter.yaml")
ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        envvar="PRICEHUNTER_CONFIG",
        help="Configuration file (YAML or JSON). Defaults to ./pricehunter.yaml.",
    ),
]
SkuOption = Annotated[list[str] | None, typer.Option("--sku", help="Only this SKU (repeatable).")]
REPORT_FORMATS = (".html", ".htm", ".csv", ".json")
EXIT_SCAN_ERRORS = 3  # `scan --strict`: the run finished but reported errors
_SPARK_BARS = "▁▂▃▄▅▆▇█"
_INTERVAL = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])", re.IGNORECASE)
_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# --------------------------------------------------------------------------- helpers


def _fail(message: str) -> typer.Exit:
    err_console.print(Text.assemble(("Error: ", "bold red"), message))
    return typer.Exit(1)


def _load(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except FileNotFoundError:
        raise _fail(
            f"config file not found: {path} (create one with: pricehunter init {path})"
        ) from None
    except (ValidationError, yaml.YAMLError, ValueError) as exc:
        raise _fail(f"invalid config {path}:\n{exc}") from None


def _check_skus(config: AppConfig, skus: list[str] | None) -> None:
    unknown = [s for s in skus or [] if config.product(s) is None]
    if unknown:
        raise _fail(f"unknown SKU(s): {', '.join(unknown)}")


def parse_interval(text: str) -> float:
    """Seconds in an interval such as ``30s``, ``15m``, ``2h`` or ``1d``."""
    match = _INTERVAL.fullmatch(text.strip())
    seconds = float(match[1]) * _SECONDS[match[2].lower()] if match else 0.0
    if seconds <= 0:
        raise typer.BadParameter("use a positive number with a unit, e.g. 30s, 15m, 2h or 1d")
    return seconds


def sparkline(values: list[Decimal]) -> str:
    """A text sparkline (``▁▃█``) scaled between the lowest and highest value."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_BARS[3] * len(values)
    top = len(_SPARK_BARS) - 1
    return "".join(_SPARK_BARS[int((v - lo) / (hi - lo) * top)] for v in values)


def _money(amount: Decimal | None, currency: str | None = None) -> str:
    if amount is None:
        return "—"
    return f"{amount:,.2f}" + (f" {currency}" if currency else "")


def _percent(change: float | None, places: int = 1) -> Text:
    if change is None:
        return Text("")
    return Text(
        f"{change:+.{places}f}%", style="green" if change > 0 else "red" if change < 0 else ""
    )


def _index(value: float | None) -> Text:
    if value is None:
        return Text("—", style="dim")
    return Text(
        f"{value:.1f}", style="green" if value <= 100 else "yellow" if value <= 105 else "red"
    )


def _two_lines(first: str, second: str | None, style: str = "") -> Text:
    text = Text(first, style=style)
    if second:
        text.append(f"\n{second}", style="dim")
    return text


def _market_table(
    config: AppConfig, snapshots: list[MarketSnapshot], suggestions: list[PriceSuggestion]
) -> Table:
    competitors = sorted({o.competitor for s in snapshots for o in s.offers}, key=str.casefold)
    table = Table(
        title="Market position",
        caption="Competitor prices in each product's currency. "
        "Struck = out of stock; bold = cheapest.",
        show_lines=True,
    )
    table.add_column("Product")
    for name in ("Ours", "Lowest", "Index", "Rank", "Suggested", *competitors):
        table.add_column(name, justify="right")
    suggestion_by_sku = {s.sku: s for s in suggestions}
    for snap in snapshots:
        product = config.product(snap.sku)
        suggestion = suggestion_by_sku.get(snap.sku)
        suggested = Text(_money(suggestion.suggested_price) if suggestion else "—")
        if suggestion and suggestion.change_percent is not None:
            suggested.append("\n").append_text(_percent(suggestion.change_percent))
        offers = {o.competitor: o for o in snap.offers}
        cells: list[Text] = []
        for name in competitors:
            offer = offers.get(name)
            if offer is None:
                cells.append(Text("—", style="dim"))
            elif offer.availability is Availability.OUT_OF_STOCK:
                cells.append(Text(_money(offer.price), style="dim strike"))
            else:
                cheapest = name == snap.lowest_competitor and offer.price == snap.lowest
                cells.append(Text(_money(offer.price), style="bold green" if cheapest else ""))
        rank = f"{snap.our_rank}/{snap.in_stock_offers + 1}" if snap.our_rank else "—"
        table.add_row(
            _two_lines(product.name if product else snap.sku, snap.sku, "bold"),
            Text(_money(snap.our_price, snap.currency)),
            _two_lines(_money(snap.lowest), snap.lowest_competitor),
            _index(snap.price_index),
            Text(rank),
            suggested,
            *cells,
        )
    return table


def _stats_line(stats: ScanStats) -> Text:
    return Text.assemble(
        (f"Run #{stats.run_id}: ", "bold"),
        f"{stats.products} products · {stats.pages_fetched} pages fetched · "
        f"{stats.candidates_checked} candidates checked · {stats.listings_matched} new listings · "
        f"{stats.observations} prices read · ",
        (f"{stats.alerts} alerts", "bold yellow" if stats.alerts else ""),
        " · ",
        (f"{stats.errors} errors", "bold red" if stats.errors else ""),
    )


def _print_problems(problems: list[str]) -> None:
    for problem in problems:
        console.print(Text(f"  ! {problem}", style="yellow"))


def _print_result(config: AppConfig, result: ScanResult, *, table: bool = True) -> None:
    if table:
        console.print(_market_table(config, result.snapshots, result.suggestions))
    console.print(_stats_line(result.stats))
    _print_problems(result.errors[:10])
    if len(result.errors) > 10:
        console.print(Text(f"  ... and {len(result.errors) - 10} more", style="yellow"))


def _scan(
    config: AppConfig, storage: Storage, *, verbose: bool = False, **options: Any
) -> ScanResult:
    """Run one scan with a spinner (and every progress line when ``verbose``)."""
    with console.status("Scanning competitors…") as status:

        def progress(message: str) -> None:
            status.update(Text(f"Scanning: {message}"))
            if verbose:
                console.print(Text(message, style="dim"))

        return asyncio.run(run_scan(config, storage, progress=progress, **options))


def _check_report_path(path: Path) -> None:
    if path.suffix.lower() not in REPORT_FORMATS:
        raise typer.BadParameter(
            f"unsupported format {path.suffix or '(none)'!r}; use .html, .csv or .json"
        )


def _write_report(
    path: Path,
    config: AppConfig,
    storage: Storage,
    result: ScanResult | None = None,
    *,
    title: str = "PriceHunter report",
) -> None:
    """Write an HTML (with history), CSV or JSON report, chosen by the file extension."""
    if result is not None:
        snapshots, suggestions = result.snapshots, result.suggestions
    else:
        snapshots, suggestions = build_snapshots(config, storage)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        write_csv(path, build_rows(snapshots, suggestions, config.products))
    elif suffix == ".json":
        write_json(
            path,
            result
            or ScanResult(
                stats=next(iter(storage.runs(1)), ScanStats()),
                observations=storage.latest_observations(),
                snapshots=snapshots,
                suggestions=suggestions,
                alerts=storage.recent_alerts(),
            ),
        )
    else:
        html = render_html(
            snapshots,
            suggestions,
            storage.recent_alerts(MAX_ALERTS_SHOWN),
            config.products,
            history=load_history(config, storage),
            title=title,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    console.print(f"Report written to {path}")


def _print_listings(storage: Storage) -> None:
    table = Table(title="Matched competitor listings (decoys and wrong variants were rejected)")
    for name in ("SKU", "Competitor", "Score", "Why", "URL"):
        table.add_column(name, justify="right" if name == "Score" else "left")
    for listing in storage.listings():
        table.add_row(
            Text(listing.sku),
            Text(listing.competitor),
            f"{listing.match_score:.2f}",
            Text(listing.match_reason),
            Text(urlsplit(listing.url).path, style="dim"),
        )
    console.print(table)


def _host_key(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    return f"{host}:{parts.port}" if parts.port else host


def _competitor_for(config: AppConfig, url: str) -> Competitor | None:
    """The configured competitor whose site serves ``url`` (``www.`` ignored)."""
    key = _host_key(url)
    return next((c for c in config.competitors if _host_key(c.base_url) == key), None)


def _version(value: bool) -> None:
    if value:
        console.print(f"pricehunter {__version__}")
        raise typer.Exit()


# --------------------------------------------------------------------------- commands


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    """Competitor price intelligence: discover, match, track and reprice."""


@app.command()
def init(
    path: Annotated[
        Path, typer.Argument(help="Where to write the example config.")
    ] = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a fully commented example configuration."""
    if path.exists() and not force:
        raise _fail(f"{path} already exists; use --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    console.print(
        f"Wrote {path}. Edit the competitors and products, then run: pricehunter scan -c {path}"
    )


@app.command()
def scan(
    config: ConfigOption = DEFAULT_CONFIG,
    sku: SkuOption = None,
    competitor: Annotated[
        list[str] | None, typer.Option("--competitor", help="Only this competitor (repeatable).")
    ] = None,
    no_discover: Annotated[
        bool, typer.Option("--no-discover", help="Only re-read known listings; skip discovery.")
    ] = False,
    no_notify: Annotated[
        bool, typer.Option("--no-notify", help="Do not send alerts to notifiers.")
    ] = False,
    report: Annotated[
        Path | None, typer.Option("--report", help="Also write a report: .html, .csv or .json.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print every step.")] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help=f"Exit with code {EXIT_SCAN_ERRORS} when the run reports errors (cron, systemd).",
        ),
    ] = False,
) -> None:
    """Discover competitor listings, read their prices, analyse and alert."""
    cfg = _load(config)
    if report is not None:
        _check_report_path(report)
    with Storage(cfg.settings.database) as storage:
        result = _scan(
            cfg,
            storage,
            verbose=verbose,
            skus=sku,
            competitors=competitor,
            discover=not no_discover,
            notify=not no_notify,
        )
        _print_result(cfg, result)
        if report is not None:
            _write_report(report, cfg, storage, result)
    if strict and result.errors:
        raise typer.Exit(EXIT_SCAN_ERRORS)


@app.command()
def watch(
    config: ConfigOption = DEFAULT_CONFIG,
    every: Annotated[
        str, typer.Option("--every", help="Interval between scans: 30s, 15m, 2h, 1d.")
    ] = "60m",
    max_runs: Annotated[
        int | None, typer.Option("--max-runs", min=1, help="Stop after this many scans.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print every step.")] = False,
) -> None:
    """Scan repeatedly on an interval (Ctrl-C to stop). Config edits apply from the next scan."""
    seconds = parse_interval(every)
    cfg = _load(config)
    runs = 0
    try:
        with Storage(cfg.settings.database) as storage:
            while True:
                result = _scan(cfg, storage, verbose=verbose)
                _print_result(cfg, result)
                runs += 1
                if max_runs is not None and runs >= max_runs:
                    break
                console.print(Text(f"Next scan in {every} (Ctrl-C to stop)", style="dim"))
                asyncio.run(asyncio.sleep(seconds))
                try:
                    cfg = load_config(config)
                except (OSError, ValidationError, yaml.YAMLError, ValueError) as exc:
                    console.print(
                        Text(f"Keeping the previous config; reload failed: {exc}", style="yellow")
                    )
    except KeyboardInterrupt:
        console.print("Stopped.")


@app.command()
def prices(config: ConfigOption = DEFAULT_CONFIG, sku: SkuOption = None) -> None:
    """Market position from stored prices (no network)."""
    cfg = _load(config)
    _check_skus(cfg, sku)
    with Storage(cfg.settings.database) as storage:
        snapshots, suggestions = build_snapshots(cfg, storage, skus=sku)
        warnings = market_warnings(cfg, storage, skus=sku)
    if not any(s.offers for s in snapshots) and not warnings:
        console.print("No competitor prices stored yet. Run: pricehunter scan")
        return
    console.print(_market_table(cfg, snapshots, suggestions))
    _print_problems(warnings)


@app.command()
def suggest(config: ConfigOption = DEFAULT_CONFIG, sku: SkuOption = None) -> None:
    """Repricing suggestions from stored prices."""
    cfg = _load(config)
    _check_skus(cfg, sku)
    with Storage(cfg.settings.database) as storage:
        _, suggestions = build_snapshots(cfg, storage, skus=sku)
        warnings = market_warnings(cfg, storage, skus=sku)
    table = Table(title=f"Repricing ({cfg.repricing.strategy})", show_lines=True)
    for name in ("Product", "Current", "Suggested", "Change", "Floor", "Ceiling", "Reason"):
        table.add_column(name, justify="left" if name in ("Product", "Reason") else "right")
    for s in suggestions:
        product = cfg.product(s.sku)
        table.add_row(
            _two_lines(product.name if product else s.sku, s.sku, "bold"),
            _money(s.current_price, product_currency(product) if product else None),
            Text(_money(s.suggested_price), style="bold"),
            _percent(s.change_percent),
            _money(s.floor),
            _money(s.ceiling),
            Text(s.reason),
        )
    console.print(table)
    _print_problems(warnings)


@app.command()
def history(
    sku: Annotated[str, typer.Argument(help="SKU of one of your products.")],
    config: ConfigOption = DEFAULT_CONFIG,
    competitor: Annotated[
        str | None, typer.Option("--competitor", help="Only this competitor.")
    ] = None,
    days: Annotated[float, typer.Option("--days", min=0, help="How far back to look.")] = 30,
) -> None:
    """Price history of one product, with a sparkline per competitor listing."""
    cfg = _load(config)
    _check_skus(cfg, [sku])
    with Storage(cfg.settings.database) as storage:
        rows = storage.history(sku, competitor, since=lookback_start(days))
    if not rows:
        console.print(f"No price history for {sku} in the last {days:g} days.")
        return
    table = Table(title=f"Price history: {sku}")
    for name in ("When (UTC)", "Competitor", "Price", "Stock", "Match"):
        table.add_column(name, justify="right" if name in ("Price", "Match") else "left")
    for obs in rows:
        table.add_row(
            obs.observed_at.strftime("%Y-%m-%d %H:%M"),
            Text(obs.competitor),
            _money(obs.price, obs.currency),
            Text(
                obs.availability.value.replace("_", " "),
                style="red" if obs.availability is Availability.OUT_OF_STOCK else "",
            ),
            f"{obs.match_score:.2f}",
        )
    console.print(table)
    series: dict[tuple[str, str], list[Decimal]] = {}
    currency: dict[tuple[str, str], str | None] = {}
    for obs in rows:
        if obs.price is not None:
            series.setdefault((obs.competitor, obs.url), []).append(obs.price)
            currency[(obs.competitor, obs.url)] = obs.currency
    for (name, url), values in series.items():
        ccy = currency[(name, url)]
        console.print(
            Text.assemble(
                (f"{name:<14}", "bold"),
                (sparkline(values), "cyan"),
                f"  {_money(values[0])} → {_money(values[-1], ccy)}",
                (f"  (min {_money(min(values))}, max {_money(max(values))})  {url}", "dim"),
            )
        )


@app.command()
def alerts(
    config: ConfigOption = DEFAULT_CONFIG,
    limit: Annotated[int, typer.Option("--limit", min=1, help="How many alerts to show.")] = 20,
    sku: Annotated[str | None, typer.Option("--sku", help="Only alerts for this SKU.")] = None,
) -> None:
    """Recent alerts, newest first."""
    cfg = _load(config)
    with Storage(cfg.settings.database) as storage:
        recent = storage.recent_alerts(limit=limit, sku=sku)
    if not recent:
        console.print("No alerts yet.")
        return
    table = Table(title="Recent alerts")
    for name in ("When (UTC)", "Type", "SKU", "Competitor", "Change", "Message"):
        table.add_column(name, justify="right" if name == "Change" else "left")
    for a in recent:
        table.add_row(
            a.created_at.strftime("%Y-%m-%d %H:%M"),
            Text(a.type, style="bold"),
            Text(a.sku),
            Text(a.competitor or ""),
            _percent(a.change_percent, places=2),
            Text(a.message),
        )
    console.print(table)


@app.command()
def export(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output file: .html, .csv or .json.")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Export stored data as an HTML report (with history), CSV table or JSON."""
    _check_report_path(out)
    cfg = _load(config)
    with Storage(cfg.settings.database) as storage:
        _write_report(out, cfg, storage)


@app.command()
def probe(
    url: Annotated[str, typer.Argument(help="Product page to fetch.")],
    config: ConfigOption = DEFAULT_CONFIG,
    competitor: Annotated[
        str | None,
        typer.Option("--competitor", help="Use this competitor's selectors and headers."),
    ] = None,
    sku: Annotated[
        str | None, typer.Option("--sku", help="Also score the page against this product.")
    ] = None,
    render: Annotated[
        bool, typer.Option("--render", help="Render with headless Chromium (pricehunter[js]).")
    ] = False,
) -> None:
    """Fetch one page and show everything PriceHunter extracts (great for writing selectors)."""
    if config.exists():
        cfg = _load(config)
    elif competitor or sku:
        raise _fail(f"config file not found: {config} (needed for --competitor/--sku)")
    else:
        cfg = AppConfig()
    target = cfg.competitor(competitor) if competitor else _competitor_for(cfg, url)
    if competitor and target is None:
        raise _fail(f"unknown competitor {competitor!r}")
    product = cfg.product(sku) if sku else None
    if sku and product is None:
        raise _fail(f"unknown SKU {sku!r}")
    result, offer, match = asyncio.run(
        probe_page(url, target, cfg.settings, product, render=True if render else None)
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(overflow="fold")
    table.add_row("URL", Text(url))
    if result.final_url != url:
        table.add_row("Final URL", Text(result.final_url))
    status = f"{result.status} ({'rendered' if result.rendered else 'plain HTTP'})"
    table.add_row("Status", Text(status, style="green" if result.ok else "red"))
    table.add_row("Competitor", Text(target.name if target else "(none: generic extraction)"))
    if not result.ok:
        console.print(table)
        raise _fail(f"fetch failed: {result.error}")
    for field in ("title", "price", "currency", "availability", "original_price", "shipping",
                  "gtin", "mpn", "brand", "sku", "image"):  # fmt: skip
        value = getattr(offer, field)
        shown = value.value if isinstance(value, Availability) else value
        table.add_row(
            field, Text("—" if shown is None else str(shown), style="dim" if shown is None else "")
        )
    table.add_row(
        "method", Text(f"{offer.method} (confidence {offer.confidence:.2f})", style="cyan")
    )
    if match is not None:
        threshold = cfg.settings.match_threshold
        accepted = match.score >= threshold
        verdict = f"{'accepted' if accepted else 'rejected'} at threshold {threshold:.2f}"
        table.add_row(
            f"match vs {sku}",
            Text(
                f"{match.score:.2f} ({match.reason}): {verdict}",
                style="green" if accepted else "red",
            ),
        )
        table.add_row("match details", Text(json.dumps(match.details)))
    console.print(table)
    if offer.price is None:
        console.print(
            Text(
                "No price found. Add selectors.price for this competitor"
                + (
                    " or try --render if the page builds its price with JavaScript."
                    if not render
                    else "."
                ),
                style="yellow",
            )
        )


@app.command()
def serve(
    config: ConfigOption = DEFAULT_CONFIG,
    host: Annotated[str, typer.Option("--host", help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
) -> None:
    """Run the web dashboard and JSON API."""
    import uvicorn

    from .web import create_app

    _load(config)  # fail early on a missing or invalid config
    console.print(f"PriceHunter dashboard on http://{host}:{port}/ (Ctrl-C to stop)")
    uvicorn.run(create_app(str(config)), host=host, port=port)


@app.command()
def demo(
    days: Annotated[
        int, typer.Option("--days", min=1, help="Simulated days to scan (prices change daily).")
    ] = 3,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the HTML report.")
    ] = Path("demo-report.html"),
    keep_db: Annotated[
        bool,
        typer.Option(
            "--keep-db", help="Keep the demo database and config instead of deleting them."
        ),
    ] = False,
) -> None:
    """Scan three fake local shops over a few days: discovery, matching, alerts and a report."""
    workdir = Path(tempfile.mkdtemp(prefix="pricehunter-demo-"))
    try:
        with DemoMarket() as market:
            cfg = market.config(workdir / "demo.db")
            console.print(Text("PriceHunter demo: three fake shops on this machine", style="bold"))
            for c in cfg.competitors:
                console.print(Text(f"  {c.name:<10} {c.base_url}", style="dim"))
            with Storage(cfg.settings.database) as storage:
                for day in range(1, days + 1):
                    market.set_day(day)
                    console.rule(f"Day {day}")
                    result = _scan(cfg, storage)
                    if day == 1:
                        _print_listings(storage)
                    _print_result(cfg, result)
                _write_report(out, cfg, storage, title="PriceHunter demo report")
            if keep_db:
                config_path = workdir / "demo.yaml"
                config_path.write_text(
                    yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
                )
                console.print(
                    f"Demo data kept in {workdir}. "
                    f"Explore it with: pricehunter prices -c {config_path}"
                )
    finally:
        if not keep_db:
            shutil.rmtree(workdir, ignore_errors=True)
