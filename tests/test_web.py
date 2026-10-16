"""Web dashboard and JSON API tests (FastAPI TestClient, plus one real-browser check)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
import yaml
from bs4 import BeautifulSoup

from pricehunter import web
from pricehunter.demo import DemoMarket
from pricehunter.models import AppConfig, ScanResult, ScanStats
from pricehunter.web import create_app

with warnings.catch_warnings():  # starlette suggests httpx2, which is not a dependency here
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient


@pytest.fixture
def market() -> Iterator[DemoMarket]:
    with DemoMarket() as m:
        yield m


def write_config(config: AppConfig, path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path


@pytest.fixture
def config_path(market: DemoMarket, tmp_path: Path) -> Path:
    config = market.config(tmp_path / "demo.db", requests_per_minute=0)
    config.notifiers = []
    return write_config(config, tmp_path / "pricehunter.yaml")


@pytest.fixture
def client(config_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(str(config_path))) as c:  # the context keeps the event loop alive
        yield c


def wait_for_scan(client: TestClient, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status: dict[str, Any] = client.get("/api/scan/status").json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def test_scan_in_the_background_then_read_the_api(client: TestClient, market: DemoMarket) -> None:
    assert client.get("/api/scan/status").json()["running"] is False
    started = client.post("/api/scan")
    assert started.status_code == 202 and started.json()["running"] is True

    status = wait_for_scan(client)
    assert status["error"] is None and status["errors"] == [] and status["finished_at"]
    assert status["stats"]["observations"] == 13 and status["stats"]["listings_matched"] == 12

    snapshots = client.get("/api/snapshots").json()
    assert [s["sku"] for s in snapshots] == [
        "SONY-XM5", "IPHONE15-128", "SAMSUNG-Q80C-55", "DELONGHI-ECAM22", "DYSON-V15",
    ]  # fmt: skip
    sony = snapshots[0]
    assert (sony["lowest"], sony["lowest_competitor"], sony["our_price"]) == (
        "345.00",
        "MegaShop",
        "349.00",
    )
    assert {o["competitor"]: o["original_currency"] for o in sony["offers"]}["PretMic"] == "RON"
    suggestions = client.get("/api/suggestions").json()
    assert suggestions[0]["sku"] == "SONY-XM5" and suggestions[0]["suggested_price"] == "343.99"
    history = client.get("/api/history/SONY-XM5").json()
    assert len(history) == 3 and {h["competitor"] for h in history} == {
        "MegaShop",
        "PretMic",
        "BudgetBay",
    }
    assert len(client.get("/api/history/SONY-XM5", params={"competitor": "PretMic"}).json()) == 1
    runs = client.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["finished_at"] is not None
    assert client.get("/api/alerts").json() == []

    market.set_day(2)
    assert client.post("/api/scan", params={"discover": "false"}).status_code == 202
    assert wait_for_scan(client)["stats"]["alerts"] == 12
    alerts = client.get("/api/alerts", params={"limit": 5}).json()
    assert len(alerts) == 5 and {"type", "sku", "message", "change_percent"} <= set(alerts[0])
    assert all(
        a["sku"] == "DYSON-V15"
        for a in client.get("/api/alerts", params={"sku": "DYSON-V15"}).json()
    )
    assert [r["run_id"] for r in client.get("/api/runs").json()] == [2, 1]


def test_only_one_scan_at_a_time(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    async def slow_scan(config: AppConfig, storage: object, **options: Any) -> ScanResult:
        options["progress"]("halfway")
        while not release.is_set():
            await asyncio.sleep(0.01)
        return ScanResult(
            stats=ScanStats(run_id=42, products=len(config.products)), errors=["note"]
        )

    monkeypatch.setattr(web, "run_scan", slow_scan)
    assert client.post("/api/scan").status_code == 202
    conflict = client.post("/api/scan")
    assert conflict.status_code == 409 and conflict.json()["detail"] == "a scan is already running"
    status = client.get("/api/scan/status").json()
    assert status["running"] is True and status["progress"] == "halfway"

    release.set()
    done = wait_for_scan(client)
    assert done["stats"]["run_id"] == 42 and done["errors"] == ["note"] and done["progress"] is None
    assert client.post("/api/scan").status_code == 202  # free again
    wait_for_scan(client)


def test_a_crashing_scan_is_reported_in_the_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*args: object, **kwargs: object) -> ScanResult:
        raise RuntimeError("disk full")

    monkeypatch.setattr(web, "run_scan", broken)
    assert client.post("/api/scan").status_code == 202
    status = wait_for_scan(client)
    assert status["error"] == "RuntimeError: disk full" and status["running"] is False


def test_bad_requests_are_rejected(client: TestClient) -> None:
    assert client.get("/api/history/NOPE").status_code == 404
    unknown = client.post("/api/scan", params={"sku": ["SONY-XM5", "NOPE"], "competitor": "Ghost"})
    assert (
        unknown.status_code == 400
        and unknown.json()["detail"] == "unknown SKU or competitor: NOPE, Ghost"
    )
    assert client.get("/api/alerts", params={"limit": 0}).status_code == 422
    assert client.get("/api/history/SONY-XM5", params={"days": -1}).status_code == 422
    assert client.get("/api/scan/status").json()["running"] is False


def test_dashboard_page_has_controls_and_a_strict_csp(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    soup = BeautifulSoup(response.text, "html.parser")
    assert soup.title is not None and soup.title.string == "PriceHunter dashboard"
    assert soup.find(id="run-scan") is not None and soup.find(id="refresh") is not None
    meta = soup.find("meta", attrs={"http-equiv": "Content-Security-Policy"})
    assert meta is not None
    csp = str(meta["content"])
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp
    for script in soup.find_all("script"):
        digest = base64.b64encode(hashlib.sha256(str(script.string).encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    assert "Dyson V15 Detect Absolute Cordless Vacuum" in response.text


def test_config_edits_are_picked_up(client: TestClient, config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["products"][0]["our_price"] = "333.00"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    stat = config_path.stat()
    os.utime(config_path, (stat.st_atime, stat.st_mtime + 5))  # make the change visible to mtime
    assert client.get("/api/snapshots").json()[0]["our_price"] == "333.00"

    config_path.write_text("products: [oops", encoding="utf-8")
    os.utime(config_path, (stat.st_atime, stat.st_mtime + 10))
    assert client.get("/api/snapshots").json()[0]["our_price"] == "333.00"  # last good config


async def test_dashboard_run_scan_button_works_in_a_browser(config_path: Path) -> None:
    async_api = pytest.importorskip("playwright.async_api")
    server = uvicorn.Server(
        uvicorn.Config(create_app(str(config_path)), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with async_api.async_playwright() as pw:
            try:
                browser = await pw.chromium.launch()
            except async_api.Error as exc:  # browser not installed in this environment
                pytest.skip(f"chromium unavailable: {exc}")
            page = await browser.new_page()
            csp_errors: list[str] = []
            page.on(
                "console",
                lambda m: (
                    csp_errors.append(m.text) if "Content Security Policy" in m.text else None
                ),
            )
            await page.goto(f"http://127.0.0.1:{port}/")
            assert await page.locator("td.offer").count() == 0  # nothing scanned yet
            async with page.expect_navigation(timeout=30_000):  # the page reloads when done
                await page.click("#run-scan")
            await page.wait_for_selector("td.offer")
            assert await page.locator("td.offer").count() == 13
            assert await page.locator("#products tbody tr").count() == 5
            assert csp_errors == []
            await browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
