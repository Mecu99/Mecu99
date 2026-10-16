"""SQLite persistence for runs, listings, discovery bookkeeping, observations and alerts.

Storage conventions:

* ``Decimal`` values are stored as TEXT (``str(value)``) so they round-trip exactly,
  including their exponent (``Decimal("1299.00")`` comes back as ``1299.00``).
* Datetimes are stored as fixed-width ISO-8601 UTC text (microsecond precision), so
  lexical order equals chronological order. Naive datetimes are taken to be UTC;
  every datetime read back is timezone-aware UTC.
* ``Availability`` is stored as its enum value.

A single connection is shared by all callers (``check_same_thread=False``). Every
statement runs under one lock: writes stay atomic, and readers on other
threads (e.g. the web dashboard's threadpool) never observe a half-applied batch.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from .models import Alert, Availability, Listing, Observation, ScanStats, utcnow

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    stats       TEXT
);
CREATE TABLE IF NOT EXISTS listings (
    sku           TEXT NOT NULL,
    competitor    TEXT NOT NULL,
    url           TEXT NOT NULL,
    match_score   REAL NOT NULL,
    match_reason  TEXT NOT NULL,
    pinned        INTEGER NOT NULL DEFAULT 0,
    active        INTEGER NOT NULL DEFAULT 1,
    discovered_at TEXT NOT NULL,
    last_seen_at  TEXT,
    PRIMARY KEY (sku, competitor, url)
);
CREATE TABLE IF NOT EXISTS discovery (
    sku           TEXT NOT NULL,
    competitor    TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY (sku, competitor)
);
CREATE TABLE IF NOT EXISTS observations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    sku            TEXT NOT NULL,
    competitor     TEXT NOT NULL,
    url            TEXT NOT NULL,
    title          TEXT,
    price          TEXT,
    currency       TEXT,
    availability   TEXT NOT NULL,
    original_price TEXT,
    shipping       TEXT,
    match_score    REAL NOT NULL,
    match_reason   TEXT NOT NULL,
    method         TEXT NOT NULL,
    observed_at    TEXT NOT NULL,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS ix_observations_listing
    ON observations (sku, competitor, url, observed_at);
CREATE INDEX IF NOT EXISTS ix_observations_observed_at ON observations (observed_at);
CREATE INDEX IF NOT EXISTS ix_observations_run ON observations (run_id);
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    type           TEXT NOT NULL,
    sku            TEXT NOT NULL,
    competitor     TEXT,
    url            TEXT,
    message        TEXT NOT NULL,
    old_price      TEXT,
    new_price      TEXT,
    change_percent REAL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alerts_created_at ON alerts (created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_sku ON alerts (sku, created_at);
"""

# An error-free reading is the listing's current state, even without a price (an out-of-stock
# page that hides it); price history only needs the readings that carry a price.
_SUCCESSFUL = "o.error IS NULL"
_PRICED = "o.error IS NULL AND o.price IS NOT NULL"

_OBS_COLUMNS = (
    "run_id, sku, competitor, url, title, price, currency, availability, original_price,"
    " shipping, match_score, match_reason, method, observed_at, error"
)


# --------------------------------------------------------------------------- codecs


def _ts(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _ts_or_none(value: datetime | None) -> str | None:
    return None if value is None else _ts(value)


def _parse_ts(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _parse_ts_or_none(text: str | None) -> datetime | None:
    return None if text is None else _parse_ts(text)


def _dec(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _parse_dec(text: str | None) -> Decimal | None:
    return None if text is None else Decimal(text)


def _obs_params(obs: Observation) -> tuple[Any, ...]:
    return (
        obs.run_id,
        obs.sku,
        obs.competitor,
        obs.url,
        obs.title,
        _dec(obs.price),
        obs.currency,
        obs.availability.value,
        _dec(obs.original_price),
        _dec(obs.shipping),
        obs.match_score,
        obs.match_reason,
        obs.method,
        _ts(obs.observed_at),
        obs.error,
    )


def _obs_from_row(row: sqlite3.Row) -> Observation:
    return Observation(
        id=row["id"],
        run_id=row["run_id"],
        sku=row["sku"],
        competitor=row["competitor"],
        url=row["url"],
        title=row["title"],
        price=_parse_dec(row["price"]),
        currency=row["currency"],
        availability=Availability(row["availability"]),
        original_price=_parse_dec(row["original_price"]),
        shipping=_parse_dec(row["shipping"]),
        match_score=row["match_score"],
        match_reason=row["match_reason"],
        method=row["method"],
        observed_at=_parse_ts(row["observed_at"]),
        error=row["error"],
    )


def _listing_from_row(row: sqlite3.Row) -> Listing:
    return Listing(
        sku=row["sku"],
        competitor=row["competitor"],
        url=row["url"],
        match_score=row["match_score"],
        match_reason=row["match_reason"],
        pinned=bool(row["pinned"]),
        active=bool(row["active"]),
        discovered_at=_parse_ts(row["discovered_at"]),
        last_seen_at=_parse_ts_or_none(row["last_seen_at"]),
    )


def _alert_from_row(row: sqlite3.Row) -> Alert:
    return Alert(
        type=row["type"],
        sku=row["sku"],
        competitor=row["competitor"],
        url=row["url"],
        message=row["message"],
        old_price=_parse_dec(row["old_price"]),
        new_price=_parse_dec(row["new_price"]),
        change_percent=row["change_percent"],
        created_at=_parse_ts(row["created_at"]),
    )


def _run_from_row(row: sqlite3.Row) -> ScanStats:
    base = ScanStats.model_validate_json(row["stats"]) if row["stats"] else ScanStats()
    return base.model_copy(
        update={
            "run_id": row["id"],
            "started_at": _parse_ts(row["started_at"]),
            "finished_at": _parse_ts_or_none(row["finished_at"]),
        }
    )


def _where(clauses: Iterable[str]) -> str:
    parts = list(clauses)
    return f"WHERE {' AND '.join(parts)}" if parts else ""


# --------------------------------------------------------------------------- storage


class Storage:
    """Thread-safe SQLite store. Use ``":memory:"`` for an ephemeral database."""

    def __init__(self, path: str) -> None:
        self.path = path
        in_memory = path == ":memory:"
        target = path if in_memory else str(Path(path).expanduser())
        if not in_memory:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(target, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not in_memory:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        with self._write() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO schema_version (version)"
                " SELECT ? WHERE NOT EXISTS (SELECT 1 FROM schema_version)",
                (SCHEMA_VERSION,),
            )

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Serialise a write transaction: commit on success, roll back on error."""
        with self._lock, self._conn:
            yield self._conn

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    @property
    def schema_version(self) -> int:
        return int(self._query("SELECT version FROM schema_version")[0]["version"])

    # -- runs -----------------------------------------------------------------------

    def start_run(self) -> int:
        with self._write() as conn:
            cur = conn.execute("INSERT INTO runs (started_at) VALUES (?)", (_ts(utcnow()),))
        return cast(int, cur.lastrowid)

    def finish_run(self, run_id: int, stats: ScanStats) -> None:
        """Record final stats; ``finished_at`` defaults to now when the stats lack it."""
        finished = stats.finished_at or utcnow()
        payload = stats.model_copy(update={"run_id": run_id, "finished_at": finished})
        with self._write() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, stats = ? WHERE id = ?",
                (_ts(finished), payload.model_dump_json(), run_id),
            )

    def runs(self, limit: int = 20) -> list[ScanStats]:
        """Newest runs first. Unfinished runs come back with zeroed counters."""
        rows = self._query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return [_run_from_row(r) for r in rows]

    # -- listings -------------------------------------------------------------------

    def upsert_listing(self, listing: Listing) -> None:
        """Insert, or update an existing listing while keeping its original ``discovered_at``.

        A ``last_seen_at`` of ``None`` does not erase a previously recorded value.
        """
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO listings (sku, competitor, url, match_score, match_reason,
                                      pinned, active, discovered_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (sku, competitor, url) DO UPDATE SET
                    match_score  = excluded.match_score,
                    match_reason = excluded.match_reason,
                    pinned       = excluded.pinned,
                    active       = excluded.active,
                    last_seen_at = COALESCE(excluded.last_seen_at, listings.last_seen_at)
                """,
                (
                    listing.sku,
                    listing.competitor,
                    listing.url,
                    listing.match_score,
                    listing.match_reason,
                    int(listing.pinned),
                    int(listing.active),
                    _ts(listing.discovered_at),
                    _ts_or_none(listing.last_seen_at),
                ),
            )

    def listings(
        self, sku: str | None = None, competitor: str | None = None, active_only: bool = True
    ) -> list[Listing]:
        """Listings ordered by sku, competitor, then best match first."""
        clauses: list[str] = []
        params: list[Any] = []
        if sku is not None:
            clauses.append("sku = ?")
            params.append(sku)
        if competitor is not None:
            clauses.append("competitor = ?")
            params.append(competitor)
        if active_only:
            clauses.append("active = 1")
        rows = self._query(
            f"SELECT * FROM listings {_where(clauses)}"
            " ORDER BY sku, competitor, match_score DESC, url",
            params,
        )
        return [_listing_from_row(r) for r in rows]

    def deactivate_listing(self, sku: str, competitor: str, url: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE listings SET active = 0 WHERE sku = ? AND competitor = ? AND url = ?",
                (sku, competitor, url),
            )

    # -- discovery bookkeeping ------------------------------------------------------

    def mark_discovered(self, sku: str, competitor: str, when: datetime | None = None) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO discovery (sku, competitor, discovered_at) VALUES (?, ?, ?)
                ON CONFLICT (sku, competitor) DO UPDATE SET discovered_at = excluded.discovered_at
                """,
                (sku, competitor, _ts(when or utcnow())),
            )

    def last_discovered(self, sku: str, competitor: str) -> datetime | None:
        rows = self._query(
            "SELECT discovered_at FROM discovery WHERE sku = ? AND competitor = ?",
            (sku, competitor),
        )
        return _parse_ts(rows[0]["discovered_at"]) if rows else None

    # -- observations ---------------------------------------------------------------

    def add_observation(self, obs: Observation) -> int:
        """Store one observation and return its id (``obs.id`` is ignored)."""
        return self.add_observations([obs])[0]

    def add_observations(self, obs: list[Observation]) -> list[int]:
        """Store observations in one transaction; ids are returned in input order."""
        ids: list[int] = []
        with self._write() as conn:
            for o in obs:
                cur = conn.execute(
                    f"INSERT INTO observations ({_OBS_COLUMNS})"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    _obs_params(o),
                )
                ids.append(cast(int, cur.lastrowid))
        return ids

    def latest_observations(
        self,
        sku: str | None = None,
        *,
        before_run: int | None = None,
        successful_only: bool = True,
        max_fallback_age: timedelta | None = None,
    ) -> list[Observation]:
        """The most recent observation per (sku, competitor, url), ordered by those keys.

        "Most recent" means greatest ``observed_at``, ties broken by insertion order.
        Filters apply *before* picking the latest row, so with ``successful_only`` (no
        error; an out-of-stock reading without a price counts) a failed fetch falls back
        to the last good reading of that URL, and ``before_run`` (``run_id < before_run``)
        rebuilds the market as a previous run saw it. ``max_fallback_age`` drops such a
        fallback once it is that much older than the URL's latest attempt, so a listing
        that keeps failing stops counting.

        URLs whose listing has been deactivated (a rejected match or a dead page) are
        excluded. Observations with no listing row at all are kept, so ad-hoc checks
        of a URL that was never registered as a listing are not silently dropped.
        """
        clauses = ["(l.active IS NULL OR l.active = 1)"]
        params: list[Any] = []
        if sku is not None:
            clauses.append("o.sku = ?")
            params.append(sku)
        if before_run is not None:
            clauses.append("o.run_id < ?")
            params.append(before_run)
        if successful_only:
            clauses.append(_SUCCESSFUL)
        rows = self._query(
            f"""
            SELECT latest.*, (
                SELECT MAX(a.observed_at) FROM observations AS a
                WHERE a.sku = latest.sku AND a.competitor = latest.competitor
                    AND a.url = latest.url {"AND a.run_id < ?" if before_run is not None else ""}
            ) AS last_attempt
            FROM (
                SELECT o.*, ROW_NUMBER() OVER (
                    PARTITION BY o.sku, o.competitor, o.url
                    ORDER BY o.observed_at DESC, o.id DESC
                ) AS rn
                FROM observations AS o
                LEFT JOIN listings AS l
                    ON l.sku = o.sku AND l.competitor = o.competitor AND l.url = o.url
                {_where(clauses)}
            ) AS latest
            WHERE rn = 1
            ORDER BY sku, competitor, url
            """,
            # The attempts subquery comes first in the statement, so its parameter does too.
            [*([before_run] if before_run is not None else []), *params],
        )
        found = [(_obs_from_row(r), _parse_ts(r["last_attempt"])) for r in rows]
        return [
            obs
            for obs, last_attempt in found
            if max_fallback_age is None or last_attempt - obs.observed_at <= max_fallback_age
        ]

    def previous_observation(
        self, sku: str, competitor: str, url: str, before_id: int
    ) -> Observation | None:
        """The latest error-free observation of this URL with ``id < before_id``."""
        rows = self._query(
            f"""
            SELECT o.* FROM observations AS o
            WHERE o.sku = ? AND o.competitor = ? AND o.url = ? AND o.id < ? AND {_SUCCESSFUL}
            ORDER BY o.observed_at DESC, o.id DESC
            LIMIT 1
            """,
            (sku, competitor, url, before_id),
        )
        return _obs_from_row(rows[0]) if rows else None

    def history(
        self, sku: str, competitor: str | None = None, since: datetime | None = None
    ) -> list[Observation]:
        """Priced, error-free observations of ``sku`` (any listing state), oldest first."""
        clauses = ["o.sku = ?", _PRICED]
        params: list[Any] = [sku]
        if competitor is not None:
            clauses.append("o.competitor = ?")
            params.append(competitor)
        if since is not None:
            clauses.append("o.observed_at >= ?")
            params.append(_ts(since))
        rows = self._query(
            f"SELECT o.* FROM observations AS o {_where(clauses)} ORDER BY o.observed_at, o.id",
            params,
        )
        return [_obs_from_row(r) for r in rows]

    def prune(self, older_than: datetime) -> int:
        """Delete observations observed before ``older_than``; return how many."""
        with self._write() as conn:
            cur = conn.execute("DELETE FROM observations WHERE observed_at < ?", (_ts(older_than),))
        return cur.rowcount

    # -- alerts ---------------------------------------------------------------------

    def add_alerts(self, run_id: int | None, alerts: list[Alert]) -> None:
        with self._write() as conn:
            conn.executemany(
                """
                INSERT INTO alerts (run_id, type, sku, competitor, url, message,
                                    old_price, new_price, change_percent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        a.type,
                        a.sku,
                        a.competitor,
                        a.url,
                        a.message,
                        _dec(a.old_price),
                        _dec(a.new_price),
                        a.change_percent,
                        _ts(a.created_at),
                    )
                    for a in alerts
                ],
            )

    def recent_alerts(self, limit: int = 100, sku: str | None = None) -> list[Alert]:
        """Newest alerts first, optionally for one sku."""
        clauses: list[str] = []
        params: list[Any] = []
        if sku is not None:
            clauses.append("sku = ?")
            params.append(sku)
        rows = self._query(
            f"SELECT * FROM alerts {_where(clauses)} ORDER BY created_at DESC, id DESC LIMIT ?",
            [*params, limit],
        )
        return [_alert_from_row(r) for r in rows]
