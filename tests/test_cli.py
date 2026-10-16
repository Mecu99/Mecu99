"""CLI tests (typer's CliRunner) against the local demo marketplace."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from pricehunter.cli import app, parse_interval, sparkline
from pricehunter.config import load_config
from pricehunter.demo import DemoMarket
from pricehunter.example_config import EXAMPLE_CONFIG
from pricehunter.models import AppConfig

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "examples" / "pricehunter.example.yaml"


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("COLUMNS", "220")  # wide tables, so assertions see whole cells
    monkeypatch.delenv("PRICEHUNTER_CONFIG", raising=False)
    return CliRunner()


def write_config(config: AppConfig, path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path


@dataclass
class DemoEnv:
    market: DemoMarket
    config: Path
    scans: list[str]


@pytest.fixture(scope="module")
def demo_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoEnv]:
    """Two days of scans through the CLI; later tests only read the stored data."""
    runner = CliRunner()
    workdir = tmp_path_factory.mktemp("cli")
    with DemoMarket() as market:
        cfg = market.config(workdir / "demo.db", requests_per_minute=0)
        cfg.notifiers = []
        path = write_config(cfg, workdir / "pricehunter.yaml")
        scans = []
        for day in (1, 2):
            market.set_day(day)
            result = runner.invoke(app, ["scan", "-c", str(path)], env={"COLUMNS": "220"})
            assert result.exit_code == 0, result.output
            scans.append(result.output)
        yield DemoEnv(market, path, scans)


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400), ("1.5h", 5400), (" 10M ", 600)],
)
def test_parse_interval(text: str, seconds: float) -> None:
    assert parse_interval(text) == seconds


@pytest.mark.parametrize("text", ["60", "0m", "-5m", "soon", "5w", ""])
def test_parse_interval_rejects_bad_values(text: str) -> None:
    with pytest.raises(typer.BadParameter):
        parse_interval(text)


def test_sparkline_scales_between_min_and_max() -> None:
    assert sparkline([Decimal(1), Decimal(5), Decimal(9)]) == "▁▄█"
    assert sparkline([Decimal("3.5")] * 3) == "▄▄▄"
    assert sparkline([]) == ""


# --------------------------------------------------------------------------- init


def test_init_writes_the_example_config_and_refuses_to_overwrite(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "conf" / "pricehunter.yaml"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8") == EXAMPLE_CONFIG

    config = load_config(target)
    assert [c.name for c in config.competitors] == ["ExampleShop", "MagazinExemplu"]
    assert len(config.products) == 2 and len(config.alerts) == 6
    assert config.settings.database == str(target.parent / "pricehunter.db")

    target.write_text("# mine\n", encoding="utf-8")
    refused = runner.invoke(app, ["init", str(target)])
    assert refused.exit_code == 1 and "already exists" in refused.output
    assert target.read_text(encoding="utf-8") == "# mine\n"
    forced = runner.invoke(app, ["init", str(target), "--force"])
    assert forced.exit_code == 0 and target.read_text(encoding="utf-8") == EXAMPLE_CONFIG


def test_example_file_is_the_init_template() -> None:
    assert EXAMPLE_FILE.read_text(encoding="utf-8") == EXAMPLE_CONFIG


def test_missing_or_invalid_config_is_a_clear_error(runner: CliRunner, tmp_path: Path) -> None:
    missing = runner.invoke(app, ["prices", "-c", str(tmp_path / "nope.yaml")])
    assert missing.exit_code == 1 and "config file not found" in missing.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("settings:\n  concurrency: lots\n", encoding="utf-8")
    invalid = runner.invoke(app, ["prices", "-c", str(bad)])
    assert (
        invalid.exit_code == 1
        and "invalid config" in invalid.output
        and "concurrency" in invalid.output
    )


def test_config_path_comes_from_the_environment(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "env.yaml"
    config.write_text("products: [{sku: A1, name: Widget, our_price: '10.00'}]\n", encoding="utf-8")
    result = runner.invoke(app, ["prices"], env={"PRICEHUNTER_CONFIG": str(config)})
    assert result.exit_code == 0, result.output
    assert "No competitor prices stored yet" in result.output


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and re.fullmatch(r"pricehunter \d+\.\d+\.\d+\n", result.output)


# --------------------------------------------------------------------------- scan & stored data


def test_scan_prints_the_market_and_the_run(demo_env: DemoEnv) -> None:
    day1, day2 = demo_env.scans
    for out in (day1, day2):
        assert "Market position" in out and "Sony WH-1000XM5" in out and "PretMic" in out
    assert "Run #1:" in day1 and "12 new listings" in day1 and "0 alerts" in day1
    assert "Run #2:" in day2 and "12 alerts" in day2 and "0 errors" in day2


def test_scan_rejects_an_unknown_report_format(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["scan", "-c", str(demo_env.config), "--report", str(tmp_path / "r.pdf")]
    )
    assert result.exit_code == 2 and "unsupported format" in result.output


def test_scan_writes_a_report_for_the_selected_products(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path
) -> None:
    report = tmp_path / "scan.json"
    args = [
        "scan",
        "-c",
        str(demo_env.config),
        "--sku",
        "DYSON-V15",
        "--no-discover",
        "--no-notify",
    ]
    result = runner.invoke(app, [*args, "--report", str(report)])
    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert [s["sku"] for s in data["snapshots"]] == ["DYSON-V15"]
    assert {o["competitor"] for o in data["observations"]} == {"PretMic", "BudgetBay"}


def test_prices_shows_stored_competitor_prices(runner: CliRunner, demo_env: DemoEnv) -> None:
    result = runner.invoke(app, ["prices", "-c", str(demo_env.config)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "299.99" in out  # MegaShop's day-2 headphones price
    assert "351.75" in out  # PretMic's 1749.99 RON in EUR
    for sku in ("SONY-XM5", "IPHONE15-128", "SAMSUNG-Q80C-55", "DELONGHI-ECAM22", "DYSON-V15"):
        assert sku in out
    only = runner.invoke(app, ["prices", "-c", str(demo_env.config), "--sku", "DYSON-V15"])
    assert "DYSON-V15" in only.output and "SONY-XM5" not in only.output
    unknown = runner.invoke(app, ["prices", "-c", str(demo_env.config), "--sku", "NOPE"])
    assert unknown.exit_code == 1 and "unknown SKU" in unknown.output


def test_suggest_explains_each_price(runner: CliRunner, demo_env: DemoEnv) -> None:
    result = runner.invoke(app, ["suggest", "-c", str(demo_env.config)])
    assert result.exit_code == 0, result.output
    assert "Repricing (beat_lowest)" in result.output
    assert "298.99" in result.output and "beat lowest (MegaShop @ 299.99)" in result.output


def test_prices_and_suggest_name_currencies_without_a_rate(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path
) -> None:
    cfg = load_config(demo_env.config)
    cfg.settings.currency_rates = {}  # PretMic quotes RON
    path = write_config(cfg, tmp_path / "no-rates.yaml")
    for command in ("prices", "suggest"):
        result = runner.invoke(app, [command, "-c", str(path)])
        assert result.exit_code == 0, result.output
        assert "no currency rate for RON" in result.output
        assert "prices from PretMic are left out" in result.output
    clean = runner.invoke(app, ["prices", "-c", str(demo_env.config)])
    assert "no currency rate" not in clean.output


def test_scan_strict_exits_non_zero_when_the_run_reports_errors(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path
) -> None:
    cfg = load_config(demo_env.config)
    cfg.settings.database = str(tmp_path / "strict.db")
    path = write_config(cfg, tmp_path / "strict.yaml")
    args = ["scan", "-c", str(path), "--sku", "NOPE", "--no-discover", "--no-notify"]
    lenient = runner.invoke(app, args)
    assert lenient.exit_code == 0, lenient.output
    assert "1 errors" in lenient.output and "unknown SKU 'NOPE'" in lenient.output
    strict = runner.invoke(app, [*args, "--strict"])
    assert strict.exit_code == 3 and "unknown SKU 'NOPE'" in strict.output


def test_history_lists_readings_with_sparklines(runner: CliRunner, demo_env: DemoEnv) -> None:
    result = runner.invoke(app, ["history", "SONY-XM5", "-c", str(demo_env.config)])
    assert result.exit_code == 0, result.output
    assert "Price history: SONY-XM5" in result.output
    assert re.search(r"MegaShop\s+█▁\s+345\.00 → 299\.99 EUR", result.output)
    one = runner.invoke(
        app, ["history", "SONY-XM5", "-c", str(demo_env.config), "--competitor", "PretMic"]
    )
    assert "1,749.99 RON" in one.output and "MegaShop" not in one.output
    none = runner.invoke(app, ["history", "SONY-XM5", "-c", str(demo_env.config), "--days", "0"])
    assert "No price history" in none.output
    forever = ["history", "SONY-XM5", "-c", str(demo_env.config), "--days", "1000000"]
    everything = runner.invoke(app, forever)
    assert everything.exit_code == 0, everything.output
    assert "Price history: SONY-XM5" in everything.output


def test_alerts_lists_recent_alerts(runner: CliRunner, demo_env: DemoEnv) -> None:
    result = runner.invoke(app, ["alerts", "-c", str(demo_env.config), "--limit", "50"])
    assert result.exit_code == 0, result.output
    for kind in ("undercut", "price_drop", "out_of_stock", "price_rise", "new_lowest"):
        assert kind in result.output
    one = runner.invoke(app, ["alerts", "-c", str(demo_env.config), "--sku", "DELONGHI-ECAM22"])
    assert "price_rise" in one.output and "SONY-XM5" not in one.output


@pytest.mark.parametrize("suffix", [".html", ".csv", ".json"])
def test_export_by_extension(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path, suffix: str
) -> None:
    out = tmp_path / f"export{suffix}"
    result = runner.invoke(app, ["export", "-c", str(demo_env.config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8-sig")
    if suffix == ".html":
        assert (
            '<svg class="spark"' in text and "Sony WH-1000XM5" in text and "Recent alerts" in text
        )
    elif suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        assert len(rows) == 5 and {"MegaShop", "PretMic", "BudgetBay"} <= set(rows[0])
    else:
        data = json.loads(text)
        assert len(data["snapshots"]) == 5 and data["alerts"] and data["stats"]["run_id"] >= 2
    bad = runner.invoke(
        app, ["export", "-c", str(demo_env.config), "--out", str(tmp_path / "x.txt")]
    )
    assert bad.exit_code == 2


# --------------------------------------------------------------------------- probe


def test_probe_shows_extracted_fields_and_match(runner: CliRunner, demo_env: DemoEnv) -> None:
    url = demo_env.market.product_url("PretMic", "2001")
    result = runner.invoke(app, ["probe", url, "-c", str(demo_env.config), "--sku", "SONY-XM5"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Competitor" in out and "PretMic" in out  # recognised from the URL's host
    assert "1749.99" in out and "RON" in out and "selectors (confidence 0.95)" in out
    assert "in_stock" in out and "accepted at threshold 0.80" in out

    decoy = demo_env.market.product_url("PretMic", "2004")
    rejected = runner.invoke(
        app, ["probe", decoy, "-c", str(demo_env.config), "--sku", "IPHONE15-128"]
    )
    assert rejected.exit_code == 0 and "rejected" in rejected.output


def test_probe_without_a_config_uses_generic_extraction(
    runner: CliRunner, demo_env: DemoEnv, tmp_path: Path
) -> None:
    missing = tmp_path / "none.yaml"
    url = demo_env.market.product_url("MegaShop", "1001")
    result = runner.invoke(app, ["probe", url, "-c", str(missing)])
    assert result.exit_code == 0, result.output
    out = result.output  # the shared demo market is on day 2
    assert "299.99" in out and "jsonld" in out and "generic extraction" in out

    pretmic = demo_env.market.product_url("PretMic", "2001")
    no_price = runner.invoke(app, ["probe", pretmic, "-c", str(missing)])
    assert "No price found" in no_price.output  # this shop needs selectors
    needs_config = runner.invoke(app, ["probe", url, "-c", str(missing), "--sku", "X"])
    assert needs_config.exit_code == 1


def test_probe_reports_a_failed_fetch(runner: CliRunner, demo_env: DemoEnv) -> None:
    url = demo_env.market.base_url("MegaShop") + "/p/nothing-here-1"
    result = runner.invoke(app, ["probe", url, "-c", str(demo_env.config)])
    assert result.exit_code == 1 and "fetch failed: HTTP 404" in result.output


# --------------------------------------------------------------------------- watch & demo


def test_watch_repeats_scans(runner: CliRunner, demo_env: DemoEnv, tmp_path: Path) -> None:
    config = load_config(demo_env.config)
    config.settings.database = str(tmp_path / "watch.db")
    path = write_config(config, tmp_path / "watch.yaml")
    result = runner.invoke(app, ["watch", "-c", str(path), "--every", "1s", "--max-runs", "2"])
    assert result.exit_code == 0, result.output
    assert (
        "Run #1:" in result.output and "Run #2:" in result.output and "Run #3" not in result.output
    )
    assert "Next scan in 1s" in result.output
    bad = runner.invoke(app, ["watch", "-c", str(path), "--every", "often"])
    assert bad.exit_code == 2


def test_demo_runs_days_and_writes_a_report(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["demo", "--days", "2", "--out", str(out)])
    assert result.exit_code == 0, result.output
    text = result.output
    assert "Day 1" in text and "Day 2" in text and "Day 3" not in text
    assert "Matched competitor listings" in text and "pinned" in text
    assert "PriceHunter: 12 price alerts" in text  # console notifier on day 2
    assert "Run #2:" in text and "0 errors" in text
    html = out.read_text(encoding="utf-8")
    assert "PriceHunter demo report" in html and '<svg class="spark"' in html
    assert "Husa" not in html and "256GB" not in html  # decoys never become offers
    assert "Demo data kept" not in text


def test_demo_can_keep_its_data(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["demo", "--days", "1", "--out", str(tmp_path / "r.html"), "--keep-db"]
    )
    assert result.exit_code == 0, result.output
    match = re.search(r"pricehunter prices -c (\S+demo\.yaml)", result.output.replace("\n", ""))
    assert match, result.output
    kept = Path(match[1])
    try:
        prices = runner.invoke(app, ["prices", "-c", str(kept)])
        assert prices.exit_code == 0 and "SONY-XM5" in prices.output and "345.00" in prices.output
    finally:
        for f in sorted(kept.parent.iterdir()):
            f.unlink()
        kept.parent.rmdir()
