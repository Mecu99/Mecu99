"""Export scan results: CSV/JSON tables and a self-contained HTML report."""

from __future__ import annotations

import base64
import csv
import hashlib
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from html import escape
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import urlsplit

from .models import (
    Alert,
    Availability,
    CompetitorPrice,
    MarketSnapshot,
    Observation,
    PriceSuggestion,
    Product,
    ScanResult,
    utcnow,
)
from .money import normalize_currency, quantize

BASE_COLUMNS = (
    "sku",
    "name",
    "our_price",
    "currency",
    "lowest",
    "lowest_competitor",
    "median",
    "highest",
    "price_index",
    "our_rank",
    "competitors_cheaper",
    "in_stock_offers",
    "suggested_price",
    "change_percent",
    "reason",
)
MAX_ALERTS_SHOWN = 50

# --------------------------------------------------------------------------- tabular


def _competitors(snapshots: Iterable[MarketSnapshot]) -> list[str]:
    return sorted({o.competitor for s in snapshots for o in s.offers}, key=str.casefold)


def _column(competitor: str) -> str:
    """Competitor column name; prefixed only if it would clobber a base column."""
    return f"competitor:{competitor}" if competitor in BASE_COLUMNS else competitor


def build_rows(
    snapshots: list[MarketSnapshot],
    suggestions: list[PriceSuggestion],
    products: list[Product] | None = None,
) -> list[dict[str, Any]]:
    """One row per snapshot: market summary, suggestion, then each competitor's price.

    Missing summary values are ``None``; a competitor without an offer for the
    product gets ``""``. Every row has the same competitor columns.
    """
    names = {p.sku: p.name for p in products or []}
    by_sku = {s.sku: s for s in suggestions}
    competitors = _competitors(snapshots)
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        suggestion = by_sku.get(snap.sku)
        row: dict[str, Any] = {
            "sku": snap.sku,
            "name": names.get(snap.sku),
            "our_price": snap.our_price,
            "currency": snap.currency,
            "lowest": snap.lowest,
            "lowest_competitor": snap.lowest_competitor,
            "median": snap.median,
            "highest": snap.highest,
            "price_index": snap.price_index,
            "our_rank": snap.our_rank,
            "competitors_cheaper": snap.competitors_cheaper,
            "in_stock_offers": snap.in_stock_offers,
            "suggested_price": suggestion.suggested_price if suggestion else None,
            "change_percent": suggestion.change_percent if suggestion else None,
            "reason": suggestion.reason if suggestion else None,
        }
        prices: dict[str, Decimal] = {o.competitor: o.price for o in snap.offers}
        for name in competitors:
            row[_column(name)] = prices.get(name, "")
        rows.append(row)
    return rows


def _csv_cell(value: Any) -> Any:
    """Neutralise spreadsheet formulas (``=``, ``+``, ``-``, ``@``) in text cells."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with a BOM (so Excel detects UTF-8); header = union of keys in first-seen order."""
    header = list(dict.fromkeys(key for row in rows for key in row))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow([_csv_cell(h) for h in header])
        for row in rows:
            writer.writerow([_csv_cell(row.get(key)) for key in header])


def write_json(path: str | Path, result: ScanResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- html helpers


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _safe_href(url: str | None) -> str | None:
    """``url`` if it is an absolute http(s) URL, else ``None`` (blocks ``javascript:`` etc.)."""
    if not url:
        return None
    url = url.strip()
    parts = urlsplit(url)
    return url if parts.scheme.lower() in {"http", "https"} and parts.netloc else None


def _link(url: str | None, label: str) -> str:
    """``label`` (already escaped) wrapped in a link when ``url`` is safe."""
    href = _safe_href(url)
    if href is None:
        return label
    return f'<a href="{_e(href)}" rel="noopener noreferrer nofollow" target="_blank">{label}</a>'


def _price(value: Decimal | None) -> str:
    return "—" if value is None else format(quantize(value), ",f")


def _sort_key(value: object) -> str:
    return "" if value is None else _e(value)


def _index_class(index: float) -> str:
    if index <= 100:
        return "good"
    return "warn" if index <= 105 else "bad"


def _lowest_series(observations: list[Observation], currency: str) -> list[Decimal]:
    """Lowest buyable price per scan run (per day for observations without a run id)."""
    groups: dict[tuple[str, int | date], tuple[datetime, Decimal]] = {}
    for obs in observations:
        if obs.price is None or obs.error or obs.availability is Availability.OUT_OF_STOCK:
            continue
        if (normalize_currency(obs.currency) or currency) != currency:
            continue
        key = ("run", obs.run_id) if obs.run_id is not None else ("day", obs.observed_at.date())
        seen = groups.get(key)
        groups[key] = (
            (min(seen[0], obs.observed_at), min(seen[1], obs.price))
            if seen
            else (obs.observed_at, obs.price)
        )
    return [price for _, price in sorted(groups.values(), key=lambda g: g[0])]


def _sparkline(points: list[Decimal], currency: str) -> str:
    width, height, pad = 120, 32, 3
    values = [float(p) for p in points]
    lo, hi = min(values), max(values)
    step = (width - 2 * pad) / (len(values) - 1) if len(values) > 1 else 0.0

    def xy(i: int, v: float) -> str:
        x = pad + i * step if len(values) > 1 else width / 2
        y = height / 2 if hi == lo else pad + (hi - v) / (hi - lo) * (height - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    coords = [xy(i, v) for i, v in enumerate(values)]
    last_x, last_y = coords[-1].split(",")
    label = (
        f"Lowest competitor price over {len(points)} scan{'s' if len(points) != 1 else ''}:"
        f" {_price(points[0])} to {_price(points[-1])} {currency}"
    )
    line = f'<polyline points="{" ".join(coords)}"/>' if len(coords) > 1 else ""
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}"'
        f' role="img" aria-label="{_e(label)}"><title>{_e(label)}</title>{line}'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.5"/></svg>'
    )


# --------------------------------------------------------------------------- html sections


def _kpis(snapshots: list[MarketSnapshot]) -> str:
    priced = [s for s in snapshots if s.our_price is not None and s.in_stock_offers]
    cheapest = sum(1 for s in priced if s.competitors_cheaper == 0)
    undercut = sum(1 for s in snapshots if s.competitors_cheaper > 0)
    indices = [s.price_index for s in snapshots if s.price_index is not None]
    avg = fmean(indices) if indices else None
    avg_html = (
        f'<span class="idx idx-{_index_class(avg)}">{avg:.1f}</span>' if avg is not None else "—"
    )
    tiles = [
        ("Products tracked", str(len(snapshots)), "with a market snapshot"),
        ("We're cheapest", str(cheapest), f"of {len(priced)} with in-stock competition"),
        ("Undercut", str(undercut), "a competitor sells for less"),
        ("Avg price index", avg_html, "100 = market median"),
    ]
    return "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div><div class="tile-note">{note}</div></div>'
        for label, value, note in tiles
    )


def _offer_cell(offer: CompetitorPrice | None, cheapest: bool) -> str:
    if offer is None:
        return '<td class="num muted" data-sort="">—</td>'
    out = offer.availability is Availability.OUT_OF_STOCK
    label = _price(offer.price)
    if out:
        label = f'<s title="Out of stock">{label}</s><span class="sr">, out of stock</span>'
    classes = ["num", "offer"] + (["cheapest"] if cheapest else []) + (["oos"] if out else [])
    title = ' title="Cheapest in-stock offer"' if cheapest else ""
    return (
        f'<td class="{" ".join(classes)}" data-sort="{_sort_key(offer.price)}"{title}>'
        f"{_link(offer.url, label)}</td>"
    )


def _product_row(
    snap: MarketSnapshot,
    product: Product | None,
    suggestion: PriceSuggestion | None,
    competitors: list[str],
    history: dict[str, list[Observation]] | None,
) -> str:
    name = product.name if product else snap.sku
    cells = [
        (
            f'<td class="product" data-sort="{_e(name.casefold())}">'
            f'<div class="name">{_e(name)}</div><div class="sub">{_e(snap.sku)}</div></td>'
        ),
        (
            f'<td class="num" data-sort="{_sort_key(snap.our_price)}">{_price(snap.our_price)}'
            f' <span class="ccy">{_e(snap.currency)}</span></td>'
        ),
        (
            f'<td class="num" data-sort="{_sort_key(snap.lowest)}">{_price(snap.lowest)}'
            f'<div class="sub">{_e(snap.lowest_competitor or "")}</div></td>'
        ),
        f'<td class="num" data-sort="{_sort_key(snap.median)}">{_price(snap.median)}</td>',
    ]
    if snap.price_index is None:
        cells.append('<td class="num muted" data-sort="">—</td>')
    else:
        cells.append(
            f'<td class="num" data-sort="{snap.price_index}"><span class="idx'
            f' idx-{_index_class(snap.price_index)}">{snap.price_index:.1f}</span></td>'
        )
    rank = "—" if snap.our_rank is None else f"{snap.our_rank} / {snap.in_stock_offers + 1}"
    cells.append(f'<td class="num" data-sort="{_sort_key(snap.our_rank)}">{rank}</td>')
    if suggestion is None or suggestion.suggested_price is None:
        cells.append('<td class="num muted" data-sort="">—</td>')
    else:
        change = suggestion.change_percent
        direction = "flat" if not change else ("up" if change > 0 else "down")
        change_html = "" if change is None else f'<div class="sub {direction}">{change:+.1f}%</div>'
        cells.append(
            f'<td class="num" data-sort="{_sort_key(change)}" title="{_e(suggestion.reason)}">'
            f"{_price(suggestion.suggested_price)}{change_html}</td>"
        )
    offers = {o.competitor: o for o in snap.offers}
    for competitor in competitors:
        offer = offers.get(competitor)
        cheapest = (
            offer is not None
            and offer.competitor == snap.lowest_competitor
            and offer.price == snap.lowest
            and offer.availability is not Availability.OUT_OF_STOCK
        )
        cells.append(_offer_cell(offer, cheapest))
    if history is not None:
        series = _lowest_series(history.get(snap.sku, []), snap.currency)
        spark = _sparkline(series, snap.currency) if series else "—"
        cells.append(f'<td class="trend" data-sort="">{spark}</td>')
    return "<tr>" + "".join(cells) + "</tr>"


_NUM_CLASS = ' class="num"'


def _table(
    snapshots: list[MarketSnapshot],
    suggestions: list[PriceSuggestion],
    products: list[Product],
    history: dict[str, list[Observation]] | None,
) -> str:
    competitors = _competitors(snapshots)
    headers: list[tuple[str, str]] = [
        ("Product", "text"),
        ("Our price", "num"),
        ("Lowest", "num"),
        ("Median", "num"),
        ("Index", "num"),
        ("Rank", "num"),
        ("Suggested", "num"),
    ] + [(c, "num") for c in competitors]
    head = "".join(
        f'<th scope="col"{_NUM_CLASS if kind == "num" else ""} data-type="{kind}">'
        f'<button type="button">{_e(label)}</button></th>'
        for label, kind in headers
    )
    if history is not None:
        head += '<th scope="col">Trend</th>'
    by_sku = {p.sku: p for p in products}
    suggestion_by_sku = {s.sku: s for s in suggestions}
    body = "".join(
        _product_row(s, by_sku.get(s.sku), suggestion_by_sku.get(s.sku), competitors, history)
        for s in snapshots
    )
    if not body:
        span = len(headers) + (history is not None)
        body = f'<tr><td colspan="{span}" class="muted empty">No products scanned yet.</td></tr>'
    return (
        f'<div class="table-wrap"><table id="products"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


_ALERT_TYPES = {
    "undercut": ("Undercut", "bad"),
    "price_drop": ("Price drop", "warn"),
    "price_rise": ("Price rise", "good"),
    "out_of_stock": ("Out of stock", "good"),
    "back_in_stock": ("Back in stock", "warn"),
    "new_lowest": ("New lowest", "bad"),
}


def _alerts(alerts: list[Alert]) -> str:
    if not alerts:
        return '<p class="muted">No alerts.</p>'
    recent = sorted(alerts, key=lambda a: a.created_at, reverse=True)[:MAX_ALERTS_SHOWN]
    items = []
    for a in recent:
        label, tone = _ALERT_TYPES.get(a.type, (a.type, "neutral"))
        link = f" {_link(a.url, 'view listing')}" if _safe_href(a.url) else ""
        items.append(
            f'<li><span class="badge badge-{tone}">{_e(label)}</span>'
            f'<span class="msg">{_e(a.message)}{link}</span>'
            f'<time datetime="{_e(a.created_at.isoformat())}">'
            f"{_e(a.created_at.strftime('%Y-%m-%d %H:%M'))}</time></li>"
        )
    more = (
        f'<p class="muted">Showing the {MAX_ALERTS_SHOWN} most recent of {len(alerts)} alerts.</p>'
        if len(alerts) > MAX_ALERTS_SHOWN
        else ""
    )
    return f'<ul class="alerts">{"".join(items)}</ul>{more}'


_CSS = """
:root{color-scheme:light dark;--bg:#f5f6f8;--surface:#fff;--text:#1b1f24;--muted:#5d6672;
--border:#dfe3e8;--accent:#2563c9;--good:#17663a;--good-bg:#dcf3e4;--warn:#7a5200;
--warn-bg:#fdefc3;--bad:#b42318;--bad-bg:#fde4e1;--hl:#e3edff;--neutral-bg:#eceff3}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--surface:#171b21;--text:#e6e9ee;
--muted:#9aa4b1;--border:#2a313a;--accent:#7aaeff;--good:#5fd38a;--good-bg:#11301d;
--warn:#f0c25a;--warn-bg:#3a2c08;--bad:#ff8a80;--bad-bg:#3f1714;--hl:#1c2a44;--neutral-bg:#252b33}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
a{color:var(--accent)}.muted{color:var(--muted)}
.meta{margin:0;color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile-value{font-size:26px;font-weight:600;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.tile-note{color:var(--muted);font-size:12px}
.table-wrap{position:relative;overflow-x:auto;-webkit-overflow-scrolling:touch;background:var(--surface);
border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:720px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
tbody tr:last-child td{border-bottom:0}
th{font-size:12px;color:var(--muted);white-space:nowrap;background:var(--surface)}
th button{all:unset;cursor:pointer;font-weight:600}
th button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
th[aria-sort=ascending] button::after{content:" \\25B2"}
th[aria-sort=descending] button::after{content:" \\25BC"}
.num{text-align:right;white-space:nowrap}
th:first-child,td:first-child{position:sticky;left:0;background:var(--surface);z-index:1}
td.product{min-width:160px;max-width:280px}.name{font-weight:600}
.sub{font-size:12px;color:var(--muted)}.ccy{font-size:11px;color:var(--muted)}
.sub.up{color:var(--good)}.sub.down{color:var(--bad)}
.idx{display:inline-block;padding:1px 8px;border-radius:999px;font-weight:600}
.idx-good{color:var(--good);background:var(--good-bg)}
.idx-warn{color:var(--warn);background:var(--warn-bg)}
.idx-bad{color:var(--bad);background:var(--bad-bg)}
td.cheapest{background:var(--hl);font-weight:600}
td.oos,td.oos a{color:var(--muted)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.spark{display:block}.spark polyline{fill:none;stroke:var(--accent);stroke-width:1.5;
stroke-linejoin:round;stroke-linecap:round}.spark circle{fill:var(--accent)}
.empty{text-align:center;padding:24px}
.alerts{list-style:none;margin:0;padding:0;background:var(--surface);border:1px solid var(--border);
border-radius:10px}
.alerts li{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline;padding:10px 14px;
border-bottom:1px solid var(--border)}
.alerts li:last-child{border-bottom:0}
.alerts .msg{flex:1 1 240px;overflow-wrap:anywhere}
.alerts time{color:var(--muted);font-size:12px;white-space:nowrap}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;white-space:nowrap}
.badge-good{color:var(--good);background:var(--good-bg)}
.badge-warn{color:var(--warn);background:var(--warn-bg)}
.badge-bad{color:var(--bad);background:var(--bad-bg)}
.badge-neutral{color:var(--text);background:var(--neutral-bg)}
footer{margin-top:32px;color:var(--muted);font-size:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin-top:14px}
.toolbar button{font:inherit;font-weight:600;padding:6px 14px;border-radius:8px;cursor:pointer;
border:1px solid var(--border);background:var(--surface);color:var(--text)}
.toolbar button.primary{background:var(--accent);border-color:var(--accent);color:var(--surface)}
.toolbar button:disabled{opacity:.6;cursor:progress}
.toolbar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

_JS = """
(function () {
  var table = document.getElementById("products");
  if (!table || !table.tBodies.length) return;
  var body = table.tBodies[0];
  Array.prototype.forEach.call(table.tHead.rows[0].cells, function (th) {
    var button = th.querySelector("button");
    if (!button) return;
    button.addEventListener("click", function () {
      var asc = th.getAttribute("aria-sort") !== "ascending";
      var numeric = th.getAttribute("data-type") === "num";
      var col = th.cellIndex;
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (h) {
        h.removeAttribute("aria-sort");
      });
      th.setAttribute("aria-sort", asc ? "ascending" : "descending");
      var key = function (row) {
        var cell = row.cells[col];
        return cell ? cell.getAttribute("data-sort") || "" : "";
      };
      var rows = Array.prototype.slice.call(body.rows).filter(function (r) {
        return r.cells.length > 1;
      });
      rows.sort(function (a, b) {
        var x = key(a), y = key(b);
        if (x === "" || y === "") return (x === "") - (y === "");
        var d = numeric ? parseFloat(x) - parseFloat(y)
                        : x.localeCompare(y, undefined, {numeric: true, sensitivity: "base"});
        return asc ? d : -d;
      });
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
})();
"""


# Dashboard controls (``pricehunter serve``): start a scan through the web API, show its
# progress and reload when it is done. URLs are relative so the app can live under a prefix.
_DASHBOARD_JS = """
(function () {
  var run = document.getElementById("run-scan");
  var refresh = document.getElementById("refresh");
  var status = document.getElementById("scan-status");
  if (!run || !refresh || !status) return;
  function show(text) { status.textContent = text; }
  function poll(reloadWhenDone) {
    fetch("api/scan/status", {cache: "no-store"})
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (s.running) {
          run.disabled = true;
          show(s.progress || "Scanning\u2026");
          setTimeout(function () { poll(true); }, 1500);
        } else if (reloadWhenDone) {
          run.disabled = false;
          if (s.error) { show("Scan failed: " + s.error); return; }
          show("Scan finished, reloading\u2026");
          location.reload();
        }
      })
      .catch(function () { run.disabled = false; show("Lost contact with the server."); });
  }
  refresh.addEventListener("click", function () { location.reload(); });
  run.addEventListener("click", function () {
    run.disabled = true;
    show("Starting scan\u2026");
    fetch("api/scan", {method: "POST"})
      .then(function (r) {
        if (r.status === 202 || r.status === 409) { poll(true); return; }
        run.disabled = false;
        show("Could not start a scan (HTTP " + r.status + ").");
      })
      .catch(function () { run.disabled = false; show("Lost contact with the server."); });
  });
  poll(false);
})();
"""

_TOOLBAR = (
    '<div class="toolbar"><button type="button" id="refresh">Refresh</button>'
    '<button type="button" id="run-scan" class="primary">Run scan</button>'
    '<span id="scan-status" class="muted" role="status" aria-live="polite"></span></div>'
)


def _csp_hash(source: str) -> str:
    digest = base64.b64encode(hashlib.sha256(source.encode("utf-8")).digest()).decode("ascii")
    return f"'sha256-{digest}'"


def _csp(dashboard: bool) -> str:
    """Only the report's own style and scripts may run, even if escaping were ever bypassed.

    The dashboard additionally needs its script and same-origin ``fetch`` calls to the API.
    """
    scripts = [_csp_hash(_JS)] + ([_csp_hash(_DASHBOARD_JS)] if dashboard else [])
    connect = "; connect-src 'self'" if dashboard else ""
    return (
        f"default-src 'none'; style-src {_csp_hash(_CSS)}; script-src {' '.join(scripts)}"
        f"{connect}; base-uri 'none'; form-action 'none'"
    )


def render_html(
    snapshots: list[MarketSnapshot],
    suggestions: list[PriceSuggestion],
    alerts: list[Alert],
    products: list[Product],
    history: dict[str, list[Observation]] | None = None,
    title: str = "PriceHunter report",
    generated_at: datetime | None = None,
    *,
    dashboard: bool = False,
) -> str:
    """A self-contained HTML report (no external assets) of the market position.

    All scraped or configured text is escaped and only http(s) URLs become links.
    ``history`` maps a sku to its observations; when given, each row gets a
    sparkline of the lowest competitor price per scan (in the product currency).
    ``dashboard`` adds Refresh / Run scan controls that drive the web API of
    ``pricehunter serve`` (``POST api/scan``, ``GET api/scan/status``).
    """
    when = generated_at or utcnow()
    currencies = sorted({s.currency for s in snapshots})
    meta = (
        f"Generated {_e(when.strftime('%Y-%m-%d %H:%M %Z').strip())} · "
        f"{len(_competitors(snapshots))} competitors · prices in "
        f"{_e(', '.join(currencies) or 'product currency')}"
    )
    toolbar = _TOOLBAR if dashboard else ""
    dashboard_script = f"<script>{_DASHBOARD_JS}</script>" if dashboard else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{_csp(dashboard)}">
<meta name="color-scheme" content="light dark">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header><h1>{_e(title)}</h1><p class="meta">{meta}</p></header>{toolbar}
<section class="kpis" aria-label="Summary">{_kpis(snapshots)}</section>
<section><h2>Market position</h2>
<p class="muted">Index = our price ÷ market median × 100. Highlighted cells are the cheapest
in-stock offer; struck-through prices are out of stock. Click a column header to sort.</p>
{_table(snapshots, suggestions, products, history)}</section>
<section><h2>Recent alerts</h2>{_alerts(alerts)}</section>
<footer>PriceHunter</footer>
</main>
<script>{_JS}</script>{dashboard_script}
</body>
</html>
"""
