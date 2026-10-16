from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from pricehunter.models import Alert, Availability, Listing, Observation, ScanStats
from pricehunter.storage import SCHEMA_VERSION, Storage

D = Decimal
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db() -> Iterator[Storage]:
    with Storage(":memory:") as storage:
        yield storage


def obs(
    *,
    run_id: int | None = None,
    sku: str = "SKU1",
    competitor: str = "shopA",
    url: str = "https://a.example/p1",
    price: str | None = "100.00",
    minutes: int = 0,
    error: str | None = None,
) -> Observation:
    return Observation(
        run_id=run_id,
        sku=sku,
        competitor=competitor,
        url=url,
        price=None if price is None else D(price),
        currency="EUR",
        observed_at=T0 + timedelta(minutes=minutes),
        error=error,
    )


def listing(url: str = "https://a.example/p1", **extra: object) -> Listing:
    fields: dict[str, object] = {
        "sku": "SKU1",
        "competitor": "shopA",
        "url": url,
        "match_score": 0.9,
        "match_reason": "gtin",
        "discovered_at": T0,
    }
    fields.update(extra)
    return Listing.model_validate(fields)


# --------------------------------------------------------------------------- schema


def test_file_database_uses_wal_and_foreign_keys(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ph.db"
    with Storage(str(path)) as storage:
        conn = storage._conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert path.exists()


def test_memory_database_has_foreign_keys(db: Storage) -> None:
    assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        db.add_observation(obs(run_id=999))


def test_schema_is_idempotent_and_data_survives_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "ph.db")
    with Storage(path) as storage:
        run_id = storage.start_run()
        storage.add_observation(obs(run_id=run_id))
    with Storage(path) as storage:
        assert storage.schema_version == SCHEMA_VERSION
        rows = storage._conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        assert rows[0] == 1
        assert len(storage.history("SKU1")) == 1
        assert storage.runs()[0].run_id == run_id


# --------------------------------------------------------------------------- runs


def test_runs_round_trip_newest_first(db: Storage) -> None:
    first = db.start_run()
    second = db.start_run()
    assert second > first
    finished = T0 + timedelta(hours=1)
    db.finish_run(
        first,
        ScanStats(products=3, pages_fetched=12, observations=7, errors=1, finished_at=finished),
    )

    runs = db.runs()
    assert [r.run_id for r in runs] == [second, first]
    unfinished, done = runs
    assert unfinished.finished_at is None
    assert unfinished.observations == 0
    assert done.products == 3 and done.pages_fetched == 12
    assert done.observations == 7 and done.errors == 1
    assert done.finished_at == finished
    assert done.started_at.tzinfo is not None
    assert db.runs(limit=1)[0].run_id == second


def test_finish_run_defaults_finished_at_to_now(db: Storage) -> None:
    run_id = db.start_run()
    before = datetime.now(timezone.utc)
    db.finish_run(run_id, ScanStats())
    finished_at = db.runs()[0].finished_at
    assert finished_at is not None and finished_at >= before - timedelta(seconds=1)


# --------------------------------------------------------------------------- listings


def test_upsert_listing_updates_fields_but_keeps_discovered_at(db: Storage) -> None:
    seen = T0 + timedelta(days=1)
    db.upsert_listing(listing(last_seen_at=seen))
    db.upsert_listing(
        listing(
            match_score=0.5,
            match_reason="title",
            pinned=True,
            discovered_at=T0 + timedelta(days=5),
            last_seen_at=None,
        )
    )
    (stored,) = db.listings()
    assert stored.match_score == 0.5
    assert stored.match_reason == "title"
    assert stored.pinned is True
    assert stored.discovered_at == T0
    assert stored.last_seen_at == seen  # None does not erase it

    later = T0 + timedelta(days=9)
    db.upsert_listing(listing(last_seen_at=later, active=False))
    (stored,) = db.listings(active_only=False)
    assert stored.last_seen_at == later
    assert stored.active is False
    assert db.listings() == []


def test_listings_filters_and_deactivation(db: Storage) -> None:
    db.upsert_listing(listing("https://a.example/1", match_score=0.8))
    db.upsert_listing(listing("https://a.example/2", match_score=0.95))
    db.upsert_listing(listing("https://b.example/1", competitor="shopB"))
    db.upsert_listing(listing("https://a.example/9", sku="SKU2"))

    assert [x.url for x in db.listings(sku="SKU1", competitor="shopA")] == [
        "https://a.example/2",
        "https://a.example/1",
    ]
    assert {x.competitor for x in db.listings(sku="SKU1")} == {"shopA", "shopB"}
    assert [x.sku for x in db.listings(competitor="shopA")] == ["SKU1", "SKU1", "SKU2"]

    db.deactivate_listing("SKU1", "shopA", "https://a.example/2")
    assert [x.url for x in db.listings(sku="SKU1", competitor="shopA")] == ["https://a.example/1"]
    inactive = [x for x in db.listings(active_only=False) if not x.active]
    assert [x.url for x in inactive] == ["https://a.example/2"]


# --------------------------------------------------------------------------- discovery


def test_mark_and_last_discovered(db: Storage) -> None:
    assert db.last_discovered("SKU1", "shopA") is None
    db.mark_discovered("SKU1", "shopA", T0)
    assert db.last_discovered("SKU1", "shopA") == T0
    db.mark_discovered("SKU1", "shopA", T0 + timedelta(days=2))
    assert db.last_discovered("SKU1", "shopA") == T0 + timedelta(days=2)
    assert db.last_discovered("SKU1", "shopB") is None

    db.mark_discovered("SKU2", "shopA")
    when = db.last_discovered("SKU2", "shopA")
    assert when is not None and when.tzinfo is not None
    assert abs(datetime.now(timezone.utc) - when) < timedelta(minutes=1)


def test_datetimes_are_stored_as_utc(db: Storage) -> None:
    bucharest = timezone(timedelta(hours=3))
    db.mark_discovered("SKU1", "shopA", datetime(2026, 5, 1, 15, 0, tzinfo=bucharest))
    db.mark_discovered("SKU1", "shopB", datetime(2026, 5, 1, 15, 0))  # noqa: DTZ001 - naive means UTC
    raw = dict(db._conn.execute("SELECT competitor, discovered_at FROM discovery").fetchall())
    assert raw["shopA"] == "2026-05-01T12:00:00.000000+00:00"
    assert raw["shopB"] == "2026-05-01T15:00:00.000000+00:00"
    shop_a = db.last_discovered("SKU1", "shopA")
    assert shop_a == datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    assert shop_a is not None and shop_a.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- observations


def test_observation_round_trip_is_exact(db: Storage) -> None:
    run_id = db.start_run()
    original = Observation(
        run_id=run_id,
        sku="SKU1",
        competitor="shopA",
        url="https://a.example/p1",
        title="Phone X 128GB",
        price=D("1299.00"),
        currency="RON",
        availability=Availability.LIMITED,
        original_price=D("12345678901234567890.123456789"),
        shipping=D("0.10"),
        match_score=0.875,
        match_reason="gtin",
        method="jsonld",
        observed_at=datetime(2026, 3, 4, 5, 6, 7, 891011, tzinfo=timezone.utc),
    )
    obs_id = db.add_observation(original)
    (stored,) = db.history("SKU1")
    assert stored == original.model_copy(update={"id": obs_id})
    assert str(stored.price) == "1299.00"
    assert str(stored.shipping) == "0.10"
    assert stored.original_price == D("12345678901234567890.123456789")
    assert stored.availability is Availability.LIMITED
    raw = db._conn.execute("SELECT price, availability FROM observations").fetchone()
    assert tuple(raw) == ("1299.00", "limited")


def test_add_observations_returns_ids_in_order(db: Storage) -> None:
    ids = db.add_observations([obs(minutes=i, price=f"{i}.00") for i in range(1, 4)])
    assert len(ids) == 3 and ids == sorted(ids)
    assert db.add_observations([]) == []
    assert db.add_observation(obs(minutes=10)) > ids[-1]
    assert [o.id for o in db.history("SKU1")][:3] == ids


def test_failed_observation_round_trip(db: Storage) -> None:
    db.add_observation(obs(price=None, error="HTTP 503"))
    (latest,) = db.latest_observations(successful_only=False)
    assert latest.price is None and latest.error == "HTTP 503"
    assert latest.availability is Availability.UNKNOWN
    assert db.latest_observations() == []


def test_latest_observations_across_runs(db: Storage) -> None:
    url_a1, url_a2, url_b = "https://a.example/1", "https://a.example/2", "https://b.example/1"
    r1, r2, r3 = db.start_run(), db.start_run(), db.start_run()
    db.add_observations(
        [
            obs(run_id=r1, url=url_a1, price="100.00", minutes=0),
            obs(run_id=r1, url=url_a2, price="110.00", minutes=0),
            obs(run_id=r1, url=url_b, competitor="shopB", price="120.00", minutes=0),
            obs(run_id=r1, sku="SKU2", url=url_a1, price="5.00", minutes=0),
            obs(run_id=r2, url=url_a1, price="95.00", minutes=60),
            obs(run_id=r2, url=url_b, competitor="shopB", price=None, error="timeout", minutes=60),
            obs(run_id=r3, url=url_a1, price="90.00", minutes=120),
        ]
    )

    latest = {(o.competitor, o.url): o.price for o in db.latest_observations("SKU1")}
    # The failed shopB fetch falls back to its last successful price.
    assert latest == {
        ("shopA", url_a1): D("90.00"),
        ("shopA", url_a2): D("110.00"),
        ("shopB", url_b): D("120.00"),
    }

    raw = {
        (o.competitor, o.url): o.error
        for o in db.latest_observations("SKU1", successful_only=False)
    }
    assert raw[("shopB", url_b)] == "timeout"

    previous = {
        (o.competitor, o.url): o.price for o in db.latest_observations("SKU1", before_run=r3)
    }
    assert previous[("shopA", url_a1)] == D("95.00")
    first = {(o.competitor, o.url): o.price for o in db.latest_observations("SKU1", before_run=r2)}
    assert first[("shopA", url_a1)] == D("100.00")
    assert db.latest_observations("SKU1", before_run=r1) == []

    everything = db.latest_observations()
    assert [(o.sku, o.competitor, o.url) for o in everything] == [
        ("SKU1", "shopA", url_a1),
        ("SKU1", "shopA", url_a2),
        ("SKU1", "shopB", url_b),
        ("SKU2", "shopA", url_a1),
    ]


def test_latest_observations_tie_on_time_uses_insertion_order(db: Storage) -> None:
    db.add_observation(obs(price="10.00"))
    second = db.add_observation(obs(price="11.00"))
    (latest,) = db.latest_observations()
    assert latest.id == second and latest.price == D("11.00")


def test_latest_observations_orders_by_observed_at_not_id(db: Storage) -> None:
    db.add_observation(obs(price="20.00", minutes=30))
    db.add_observation(obs(price="10.00", minutes=0))  # back-filled older reading
    (latest,) = db.latest_observations()
    assert latest.price == D("20.00")


def test_latest_observations_excludes_deactivated_listings(db: Storage) -> None:
    good, bad, orphan = "https://a.example/good", "https://a.example/bad", "https://a.example/adhoc"
    db.upsert_listing(listing(good))
    db.upsert_listing(listing(bad))
    db.add_observations([obs(url=good), obs(url=bad), obs(url=orphan)])
    assert {o.url for o in db.latest_observations()} == {good, bad, orphan}

    db.deactivate_listing("SKU1", "shopA", bad)
    assert {o.url for o in db.latest_observations()} == {good, orphan}
    assert {o.url for o in db.latest_observations(successful_only=False)} == {good, orphan}
    # History is kept for deactivated listings.
    assert {o.url for o in db.history("SKU1")} == {good, bad, orphan}


def test_previous_observation(db: Storage) -> None:
    url = "https://a.example/p1"
    first = db.add_observation(obs(price="100.00", minutes=0))
    db.add_observation(obs(url="https://a.example/other", price="1.00", minutes=1))
    db.add_observation(obs(price=None, error="503", minutes=2))
    second = db.add_observation(obs(price="90.00", minutes=3))
    current = db.add_observation(obs(price="80.00", minutes=4))

    prev = db.previous_observation("SKU1", "shopA", url, current)
    assert prev is not None and prev.id == second and prev.price == D("90.00")
    prev = db.previous_observation("SKU1", "shopA", url, second)  # skips the failed one
    assert prev is not None and prev.id == first
    assert db.previous_observation("SKU1", "shopA", url, first) is None
    assert db.previous_observation("SKU1", "shopB", url, current) is None


def test_out_of_stock_reading_without_a_price_is_the_current_state(db: Storage) -> None:
    """Shops often hide the price once an item sells out: that reading must not fall back."""
    url = "https://a.example/p1"
    db.add_observation(obs(price="299.00", minutes=0))
    sold_out = obs(price=None, minutes=60).model_copy(
        update={"availability": Availability.OUT_OF_STOCK}
    )
    sold_out_id = db.add_observation(sold_out)

    (latest,) = db.latest_observations()
    assert latest.id == sold_out_id and latest.price is None  # not the old 299.00

    restocked = db.add_observation(obs(price="299.00", minutes=120))
    prev = db.previous_observation("SKU1", "shopA", url, restocked)
    assert prev is not None and prev.id == sold_out_id
    assert [o.price for o in db.history("SKU1")] == [D("299.00"), D("299.00")]  # priced only


def test_fallback_to_the_last_good_reading_is_bounded(db: Storage) -> None:
    r1, r2, r3 = db.start_run(), db.start_run(), db.start_run()
    day = 24 * 60
    db.add_observations(
        [
            obs(run_id=r1, price="249.00", minutes=0),
            obs(run_id=r2, price=None, error="HTTP 403", minutes=3 * day),
            obs(run_id=r3, price=None, error="HTTP 403", minutes=10 * day),
        ]
    )
    week = timedelta(days=7)
    assert db.latest_observations(max_fallback_age=week) == []  # failing for 10 days
    assert [o.price for o in db.latest_observations(max_fallback_age=timedelta(days=11))] == [
        D("249.00")
    ]
    # The market as run 3 saw it: failing for 3 days only.
    previous = db.latest_observations(before_run=r3, max_fallback_age=week)
    assert [o.price for o in previous] == [D("249.00")]
    # A good reading, however old, is never dropped when it is the latest attempt.
    db.add_observation(obs(price="239.00", minutes=11 * day))
    assert [o.price for o in db.latest_observations(max_fallback_age=week)] == [D("239.00")]


def test_database_path_expands_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with Storage("~/data/ph.db") as storage:
        storage.start_run()
    assert (tmp_path / "data" / "ph.db").is_file()


def test_history_filters_and_order(db: Storage) -> None:
    db.add_observations(
        [
            obs(price="3.00", minutes=30),
            obs(price="1.00", minutes=10),
            obs(price=None, error="boom", minutes=15),
            obs(competitor="shopB", url="https://b.example/1", price="2.00", minutes=20),
            obs(sku="SKU2", price="9.00", minutes=5),
        ]
    )
    assert [o.price for o in db.history("SKU1")] == [D("1.00"), D("2.00"), D("3.00")]
    assert [o.price for o in db.history("SKU1", "shopA")] == [D("1.00"), D("3.00")]
    since = T0 + timedelta(minutes=20)
    assert [o.price for o in db.history("SKU1", since=since)] == [D("2.00"), D("3.00")]
    assert db.history("NOPE") == []


def test_prune_deletes_only_old_observations(db: Storage) -> None:
    db.add_observations([obs(minutes=m) for m in (0, 10, 20, 30)])
    assert db.prune(T0 + timedelta(minutes=20)) == 2
    assert [o.observed_at for o in db.history("SKU1")] == [
        T0 + timedelta(minutes=20),
        T0 + timedelta(minutes=30),
    ]
    assert db.prune(T0) == 0


# --------------------------------------------------------------------------- alerts


def test_alerts_round_trip_newest_first(db: Storage) -> None:
    run_id = db.start_run()
    alerts = [
        Alert(
            type="price_drop",
            sku="SKU1",
            competitor="shopA",
            url="https://a.example/p1",
            message="shopA dropped",
            old_price=D("100.00"),
            new_price=D("89.90"),
            change_percent=-10.1,
            created_at=T0 + timedelta(minutes=i),
        )
        for i in range(3)
    ]
    db.add_alerts(run_id, alerts)
    db.add_alerts(None, [Alert(type="undercut", sku="SKU2", message="x", created_at=T0)])
    db.add_alerts(None, [])

    recent = db.recent_alerts()
    assert len(recent) == 4
    assert recent[0] == alerts[2]
    assert [a.created_at for a in recent] == sorted((a.created_at for a in recent), reverse=True)
    assert str(recent[0].new_price) == "89.90"
    assert [a.sku for a in db.recent_alerts(sku="SKU2")] == ["SKU2"]
    assert db.recent_alerts(limit=2) == [alerts[2], alerts[1]]
    linked = db._conn.execute("SELECT COUNT(*) FROM alerts WHERE run_id = ?", (run_id,))
    assert linked.fetchone()[0] == 3


# --------------------------------------------------------------------------- concurrency


def test_concurrent_writes_and_reads(tmp_path: Path) -> None:
    with Storage(str(tmp_path / "ph.db")) as storage:
        run_id = storage.start_run()
        errors: list[BaseException] = []

        def writer(n: int) -> None:
            try:
                for i in range(25):
                    storage.add_observation(
                        obs(run_id=run_id, url=f"https://a.example/{n}", minutes=i)
                    )
            except BaseException as exc:  # noqa: BLE001 - surfaced via the assertion below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(25):
                    storage.latest_observations()
                    storage.runs()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(storage.history("SKU1")) == 6 * 25
        assert len(storage.latest_observations()) == 6
