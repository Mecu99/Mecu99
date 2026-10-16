"""Web dashboard and JSON API (``pricehunter serve``).

``GET /`` renders the HTML report with live data, price history and Refresh / Run scan
buttons. The JSON API under ``/api`` exposes the same data and starts background scans.
Decimal amounts are serialised as strings so no precision is lost.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import load_config
from .engine import build_snapshots, load_history, lookback_start, run_scan
from .models import (
    Alert,
    AppConfig,
    MarketSnapshot,
    Observation,
    PriceSuggestion,
    ScanStats,
    utcnow,
)
from .report import MAX_ALERTS_SHOWN, render_html
from .storage import Storage

log = logging.getLogger(__name__)


class ScanStatus(BaseModel):
    """State of the most recent background scan."""

    running: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: str | None = None
    stats: ScanStats | None = None
    errors: list[str] = Field(default_factory=list)
    error: str | None = None


class _State:
    """One app's Storage, its config (re-read when the file changes) and the scan task."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._mtime = config_path.stat().st_mtime
        self._config = load_config(config_path)
        # The database stays the one configured at start-up, even if the file changes later.
        self.storage = Storage(self._config.settings.database)
        self.status = ScanStatus()
        self.task: asyncio.Task[None] | None = None

    @property
    def config(self) -> AppConfig:
        """The current config; an invalid edit keeps the last good one (and is logged once)."""
        try:
            mtime = self.config_path.stat().st_mtime
            if mtime != self._mtime:
                self._mtime = mtime
                self._config = load_config(self.config_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            log.warning("keeping the previous config: %s", exc)
        return self._config

    @property
    def scanning(self) -> bool:
        return self.task is not None and not self.task.done()

    async def scan(self, config: AppConfig, **options: Any) -> None:
        def progress(message: str) -> None:
            self.status.progress = message

        try:
            result = await run_scan(config, self.storage, progress=progress, **options)
        except Exception as exc:  # run_scan reports problems as data; this is the last resort
            log.exception("background scan failed")
            update: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            update = {"stats": result.stats, "errors": result.errors, "progress": None}
        self.status = self.status.model_copy(
            update={"running": False, "finished_at": utcnow(), **update}
        )

    async def close(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        self.storage.close()


def create_app(config_path: str) -> FastAPI:
    """The dashboard + API app for the config at ``config_path`` (one Storage per app)."""
    state = _State(Path(config_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await state.close()

    app = FastAPI(title="PriceHunter", version=__version__, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        config = state.config
        snapshots, suggestions = build_snapshots(config, state.storage)
        return render_html(
            snapshots,
            suggestions,
            state.storage.recent_alerts(MAX_ALERTS_SHOWN),
            config.products,
            history=load_history(config, state.storage),
            title="PriceHunter dashboard",
            dashboard=True,
        )

    @app.get("/api/snapshots")
    def snapshots() -> list[MarketSnapshot]:
        return build_snapshots(state.config, state.storage)[0]

    @app.get("/api/suggestions")
    def suggestions() -> list[PriceSuggestion]:
        return build_snapshots(state.config, state.storage)[1]

    # ``path``: SKUs may contain "/" (sent as such or as %2F).
    @app.get("/api/history/{sku:path}")
    def history(
        sku: str,
        competitor: str | None = None,
        days: Annotated[float, Query(gt=0, description="How far back to look.")] = 30,
    ) -> list[Observation]:
        if state.config.product(sku) is None:
            raise HTTPException(404, f"unknown SKU {sku!r}")
        return state.storage.history(sku, competitor, since=lookback_start(days))

    @app.get("/api/alerts")
    def alerts(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100, sku: str | None = None
    ) -> list[Alert]:
        return state.storage.recent_alerts(limit=limit, sku=sku)

    @app.get("/api/runs")
    def runs(limit: Annotated[int, Query(ge=1, le=1000)] = 20) -> list[ScanStats]:
        return state.storage.runs(limit)

    @app.post("/api/scan", status_code=202)
    async def start_scan(
        sku: Annotated[list[str] | None, Query(description="Only these SKUs.")] = None,
        competitor: Annotated[
            list[str] | None, Query(description="Only these competitors.")
        ] = None,
        discover: bool = True,
        notify: bool = True,
    ) -> ScanStatus:
        """Start a scan in the background; 409 while one is already running."""
        if state.scanning:
            raise HTTPException(409, "a scan is already running")
        config = state.config
        unknown = [s for s in sku or [] if config.product(s) is None]
        unknown += [c for c in competitor or [] if config.competitor(c) is None]
        if unknown:
            raise HTTPException(400, f"unknown SKU or competitor: {', '.join(unknown)}")
        state.status = ScanStatus(running=True, started_at=utcnow(), progress="starting")
        state.task = asyncio.create_task(
            state.scan(config, skus=sku, competitors=competitor, discover=discover, notify=notify)
        )
        return state.status

    @app.get("/api/scan/status")
    def scan_status() -> ScanStatus:
        return state.status

    return app
