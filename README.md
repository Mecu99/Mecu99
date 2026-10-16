# PriceHunter

PriceHunter is a competitor price-intelligence tool for online shops. You give it your own
catalogue and a list of competitor sites. It finds each product's page on those sites, reads the
price reliably, checks that the page sells the *same* item (not the 256GB version, not a case
"for" it), keeps a price history, shows where you stand in the market, suggests new prices and
alerts you when something changes.

It is a Python package with a command-line tool, a web dashboard with a JSON API, and a
self-contained demo you can run offline in a few seconds.

## Features

- **Discovery.** PriceHunter finds competitor product pages by itself, through the site's own
  search (GTIN, then brand + MPN, then name, then your keywords) or through its sitemaps,
  including sitemap index files and `.gz` sitemaps. Known URLs can be pinned instead.
- **Robust extraction.** It reads prices from JSON-LD (including `@graph`, product groups,
  aggregate offers and broken JSON), microdata, RDFa, OpenGraph and meta tags, your CSS
  selectors, and finally from heuristics. It understands `1.299,99 €`, `$1,299.99`,
  `1 299,99 zł`, `1.299<sup>99</sup> Lei` and many more formats. Stock status, strike-through
  prices, shipping, GTIN, MPN and brand are read too.
- **Careful matching.** Matching by GTIN or MPN is exact. Otherwise a fuzzy title score is
  combined with guards against wrong variants: storage, size and pack-count conflicts,
  different model numbers and generations (XM4 vs XM5, MX Master 3 vs 3S, Series X vs S,
  AirPods 1st vs 2nd generation, Mark II), Pro/Max/Plus/OLED and Wi-Fi/cellular siblings,
  bundles, accessories and spare parts, and refurbished, used, open-box or demo items, with
  keywords for English, Romanian, German, French, Spanish, Italian and Polish shop titles.
- **History.** Every reading is stored in SQLite, so you can see trends, sparklines and which
  listing was read when.
- **Market position.** For each product you get the lowest, median, average and highest
  competitor price, a price index, your rank and the number of cheaper competitors. Prices are
  converted between currencies, with optional shipping.
- **Repricing suggestions.** Strategies are `beat_lowest`, `match_lowest`, `median` and `hold`,
  always within a cost-plus-margin floor and your min/max prices, with optional `.99` endings.
- **Alerts.** You can be alerted when you are undercut, on price drops and rises, when a
  competitor goes out of stock or comes back, and on a new market low. Alerts go to the
  console, a JSON-lines file, a webhook, Slack, Discord or e-mail.
- **Reports.** A self-contained HTML report (sortable, mobile friendly, light/dark, sparklines),
  plus CSV and JSON exports.
- **Polite by design.** PriceHunter honours robots.txt (including Crawl-delay), rate-limits per
  host, retries with backoff and honours `Retry-After`. Optional headless-browser rendering is
  available for JavaScript-only shops.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[js]"            # drop [js] if no shop needs JavaScript rendering
playwright install chromium       # only for [js]

pricehunter demo                  # 3 fake local shops, 3 days of price changes, demo-report.html
```

The demo starts three small shops on `127.0.0.1`, scans them on three simulated days and writes
`demo-report.html`:

- **MegaShop** has JSON-LD pages and a site search.
- **PretMic** is a Romanian shop with `1.749<sup>99</sup> Lei` prices and no structured data,
  so it is read through CSS selectors.
- **BudgetBay** uses microdata and OpenGraph and is found only through its sitemap.

The shops also sell decoys (WH-1000XM4 instead of XM5, iPhone 15 *256GB*, silicone cases, a
Dyson V12), which the matcher rejects. On day 2 and day 3 you will see undercut, price drop and
rise, out-of-stock, back-in-stock and new-lowest alerts.

Then set it up for real:

```bash
pricehunter init                  # writes a fully commented pricehunter.yaml
$EDITOR pricehunter.yaml          # your competitors and products
pricehunter probe https://competitor.example/some-product --sku YOUR-SKU   # check extraction
pricehunter scan --report report.html
pricehunter watch --every 2h      # or schedule `scan` with cron / systemd (see below)
pricehunter serve                 # dashboard on http://127.0.0.1:8000/
```

Every command reads `./pricehunter.yaml` by default. Use `-c/--config PATH` or set
`PRICEHUNTER_CONFIG` to point at another file.

## Commands

| Command | What it does |
|---|---|
| `init [PATH] [--force]` | Write the commented example config (refuses to overwrite without `--force`). |
| `scan [--sku S]... [--competitor C]... [--no-discover] [--no-notify] [--report out.html\|.csv\|.json] [--strict] [-v]` | Discover, read prices, analyse, alert. Prints the market table and run statistics. With `--strict` it exits with code 3 when the run reported errors. |
| `watch [--every 60m] [--max-runs N]` | Scan on an interval (`30s`, `15m`, `2h`, `1d`). Ctrl-C stops it; config edits apply from the next scan. |
| `prices [--sku S]...` | Market position from stored data (no network). Warns about currencies without a rate. |
| `suggest [--sku S]...` | Repricing suggestions with the reasoning behind each one. |
| `history SKU [--competitor C] [--days 30]` | Every reading, plus a text sparkline per competitor listing. |
| `alerts [--limit 20] [--sku S]` | Recent alerts, newest first. |
| `export --out PATH` | HTML report (with history), CSV table or JSON, chosen by the file extension. |
| `probe URL [--competitor NAME] [--sku SKU] [--render]` | Fetch one page and show every extracted field, the method used and the match score. Works without a config. |
| `serve [--host 127.0.0.1] [--port 8000]` | Web dashboard and JSON API. |
| `demo [--days 3] [--out demo-report.html] [--keep-db]` | The offline demo. With `--keep-db`, the database and a config are kept so you can explore them with the other commands. |

## Configuration reference

`pricehunter init` writes [`examples/pricehunter.example.yaml`](examples/pricehunter.example.yaml),
which shows every option with comments. YAML and JSON are both accepted. `${VAR}` and
`${VAR:-default}` are replaced from the environment, so keep webhook URLs and other secrets out
of the file. Relative `database` and `file` notifier paths are resolved next to the config file.

### `settings`

| Key | Default | Meaning |
|---|---|---|
| `database` | `pricehunter.db` | SQLite file. |
| `concurrency` | `8` | Maximum requests in flight overall. |
| `user_agent` | `PriceHunter/0.1 (+…)` | Identify yourself honestly and include a contact URL. |
| `respect_robots` | `true` | Obey robots.txt Disallow rules and Crawl-delay. |
| `timeout` | `20` | Seconds per request. |
| `max_retries` | `3` | Retries on network errors, 429 and 5xx (exponential backoff, `Retry-After` honoured, capped at 60 s). |
| `proxies` | `[]` | HTTP(S) proxies used in rotation, also for rendered pages. |
| `match_threshold` | `0.80` | Minimum match score to accept a discovered page, and to keep it. |
| `max_candidates_per_competitor` | `8` | Candidate pages checked per product and competitor during discovery. |
| `base_currency` | `EUR` | Currency the rates are expressed in. |
| `currency_rates` | `{}` | Units of `base_currency` per unit of another currency, e.g. `RON: "0.201"`. Prices in a currency without a rate are left out of the market, and every scan reports it. |
| `rediscover_after_hours` | `168` | Look for new or better pages after this long. |
| `include_shipping` | `false` | Add competitor shipping to their price before comparing. |

### `competitors[]`

| Key | Meaning |
|---|---|
| `name` | Unique name, used in tables, alerts and `competitor_urls`. |
| `base_url` | The shop's home, e.g. `https://www.shop.example`. |
| `search_url` | Search URL template, absolute or relative to `base_url`, with `{query}`, e.g. `/search?q={query}`. |
| `sitemap_url` | Sitemap (or sitemap index) used when there is no search. Without it, robots.txt `Sitemap:` lines or `/sitemap.xml` are used. |
| `product_url_pattern` | Regex a URL must match to count as a product page during discovery. |
| `selectors` | CSS selectors for sites without (or with wrong) structured data. See the next section. |
| `render_js` | Render pages with headless Chromium (needs the `[js]` extra). |
| `currency` | Currency to assume when the page shows none, and what a symbol several currencies share means on this site (`$` in a CAD shop, `lei` in a Moldovan one). ISO codes on the page always win. |
| `requests_per_minute` | Per-host pace (default 30). A robots.txt Crawl-delay can only lower it. |
| `headers` | Extra request headers, e.g. `Accept-Language`. |
| `enabled` | `false` skips the competitor without deleting it. Its stored prices leave the market until it is enabled again; so do those of a competitor removed from the config. |

**`selectors`** hold CSS selectors for `price`, `original_price`, `title`, `currency`,
`availability`, `gtin`, `mpn`, `brand` and `shipping`. Set `price_attr` to read the price from
an attribute (e.g. `content` or `data-price`) instead of the text. `search_result_link`
selects the product links on a search results page. Selectors always win over structured data.
Any field you leave out is still taken from the other extraction layers.

### `products[]`

| Key | Meaning |
|---|---|
| `sku` | Your unique SKU. |
| `name` | Product name, used for search queries and title matching. |
| `brand`, `gtin`, `mpn` | Identifiers. A GTIN (EAN/UPC) makes matching exact, and an MPN nearly so. |
| `our_price`, `cost`, `currency` | Your current price, your cost (used for the margin floor) and their currency (default `EUR`; `eur`, `€` or `lei` are read as ISO codes). |
| `keywords` | Extra search queries for discovery. |
| `min_price`, `max_price` | Repricing bounds for this product. |
| `competitor_urls` | Known pages, keyed by competitor name. Pinned pages always count as a match (score 1.0, reason `pinned`), and they replace discovery and any listings it found for that competitor. A pinned URL that starts redirecting to another page (a delisted product sent to its category) must pass the matcher; otherwise the scan reports an error and reads no price. |

Quote prices as strings (`"349.00"`) to keep them exact.

### `repricing`

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `beat_lowest` | `beat_lowest`, `match_lowest`, `median` or `hold`. |
| `undercut_amount` | `0.01` | How much `beat_lowest` undercuts by... |
| `undercut_percent` | `null` | ...or by this percentage, which takes precedence when set (0 means match). |
| `min_margin_percent` | `10` | Floor = cost × (1 + margin/100). The floor is also at least `min_price`. |
| `ignore_out_of_stock` | `true` | Out-of-stock competitors do not count. |
| `min_match_score` | `0.85` | Only offers matched at least this well are used for repricing. Each competitor's best offer is chosen among those. |
| `price_ending` | `null` | Make prices end in e.g. `0.99`, within the floor and ceiling. |

### `alerts[]`

Each rule has a `type`, an optional `threshold_percent` and optional `competitors` and `skus`
filters (`null` means all).

| Type | Fires when |
|---|---|
| `undercut` | A buyable competitor offer is at least `threshold_percent` below `our_price`, and was not already that far below at the previous scan. |
| `price_drop` / `price_rise` | A listing's price changed by at least `threshold_percent` (0 means any change). |
| `out_of_stock` / `back_in_stock` | A listing's availability flipped. Failed readings never trigger this. |
| `new_lowest` | The market's lowest price fell (by at least `threshold_percent`). |

### `notifiers[]`

| Type | Keys |
|---|---|
| `console` | none |
| `file` | `path`: alerts are appended as JSON lines. |
| `webhook` | `url`: receives a POST of `{"alerts": [...]}`. |
| `slack`, `discord` | `url`: an incoming webhook. Messages are split to fit, and Discord mentions are disabled. |
| `email` | `smtp_host`, `smtp_port` (587, or 465 for implicit TLS), `smtp_user`, `smtp_password_env` (the *name* of the environment variable holding the password), `smtp_starttls`, `email_from`, `email_to`. |

A failing notifier is reported in the scan errors and never stops the scan. Webhook URLs are
never echoed back.

## How it works

**1. Discovery.** Discovery runs for each product and enabled competitor. It runs when the pair
has no discovered listing yet, or when the last discovery is older than `rediscover_after_hours`.
Pinned `competitor_urls` replace discovery for that pair.

- With a `search_url`, PriceHunter runs the queries GTIN, "brand MPN", name and keywords, in
  that order. A search that redirects straight to a product page is the strongest signal.
  Otherwise links are taken from the results page: `search_result_link` when configured, or
  heuristics that drop navigation, cart, account, category, pagination and asset links.
- Without a search, the sitemap is crawled once per run. Index files, `.gz` files and
  robots.txt `Sitemap:` lines are followed. Only URLs that mention one of the scanned products
  are kept, so shops with millions of pages are covered too, and they are ranked by how well
  their slug matches the product. If a safety limit stops the crawl early (200,000 matching
  URLs or 500 sitemap files), the scan reports it as an error.
- Tracking and session parameters (`utm_*`, `gclid`, Shopify `_pos`/`_sid`, eMAG `X-Search-*`,
  `?ref=search`, ...) are dropped from candidate URLs.
- Up to `max_candidates_per_competitor` candidates are fetched and scored. The best two at or
  above `match_threshold` become listings. Two are kept because shops sometimes list the same
  item twice; any more would be noise. One page reached under several addresses counts once,
  judged by its `<link rel="canonical">`. On rediscovery the current listings compete with the
  new matches, so a better page replaces the weakest listing.

**2. Extraction.** Each page is read through these layers, most trusted first:

| Layer | Confidence |
|---|---|
| Your selectors | 0.95 |
| JSON-LD | 0.90 |
| Microdata / RDFa | 0.85 |
| OpenGraph / meta | 0.75 |
| Heuristics | 0.40 |

The first layer that has a price decides the price, currency and method. Identity fields come
from whichever layer has them. Each page is fetched at most once per run, even when search
results, candidates and listings point to it.

On a page offering several variants, the price is the one of the variant that is your product
(by GTIN or MPN, then by name, e.g. "256GB Black"), and new offers win over used or refurbished
ones. Two kinds of page give no price instead of a guess, and discovery skips them: category and
search pages (method `listing`), and pages whose variants differ in price while none of them is
recognisably yours (method `ambiguous`; pin that variant's own URL instead). Instalment, shipping,
savings, member-only and ex-VAT amounts shown around the price are ignored.

**3. Matching.**

| Situation | Score |
|---|---|
| Equal GTINs | 1.0 |
| Different GTINs | 0.05 |
| Equal MPNs | 0.97, unless the specs disagree |

Otherwise a blend of fuzzy title ratios is adjusted:

| Adjustment | Effect |
|---|---|
| Missing brand | × 0.75 |
| Accessory or spare-part words that only the competitor's title has (or a part noun such as "Akku" or "Filtru" leading it) | × 0.6 |
| Bundle signs that only the competitor's title has ("+ 2 games", "Kit", "Pachet") | × 0.6 |
| Used, refurbished, open-box or demo condition that only the competitor's title has | × 0.6 |
| Pro/Max/Plus/Mini/OLED/Edition/Mark or Wi-Fi vs cellular mismatch, unless both titles carry the same full model code | × 0.7 |
| The same full model code on both sides (e.g. `WAN28263BY`) and nothing else against it | at least 0.90 |
| Model number or generation conflict (3 vs 3S, Series X vs S, 1st vs 2nd generation) | capped at 0.45 |
| Spec conflict (128GB vs 256GB, 6-pack vs single) | capped at 0.40 |
| MPN conflict | capped at 0.30 |

Model codes a shop writes with spaces (`QE 55Q80C`, `WH 1000 XM5`) are compared as one code.
Discovery keeps at most two listings per product and competitor. Once a GTIN or MPN match is
found there, title-only look-alikes are not kept next to it. A new title-only match priced
under 40% of your price is skipped as a likely accessory or spare part.

Every scan re-scores the listings. A page that stops matching is deactivated and recorded with
the error `match lost`. A page answering 404/410 is deactivated, so the next scan rediscovers.

**4. Storage.** Data lives in SQLite (WAL mode) in five tables: `runs`, `listings`, `discovery`,
`observations` and `alerts`. Decimals are stored exactly. A failed fetch is stored as an
observation with an `error`, so the market falls back to that listing's last good reading, for
up to 7 days of failures. An out-of-stock page that hides its price is a good reading: that
listing leaves the market until it is back in stock.

**5. Market and repricing.** PriceHunter takes the latest good reading of every active listing.
It converts each price to the product's currency, adding shipping if configured, and keeps the
best offer per competitor. Lowest, median, average, highest, price index (our price ÷ median ×
100) and rank are computed over in-stock offers. The suggestion follows the strategy, then is
clamped between the floor and ceiling, then gets the price ending. Its reason spells out each
step, for example
`beat lowest (MegaShop @ 299.99) by 1.00; floor 285.60 from cost+12% margin; price ending 0.99`.

**6. Alerts.** Alerts compare each new reading with the previous one of the same listing, and
the new market with the market before this run. They are stored and then sent to the notifiers.

## Adding a competitor

1. Add it under `competitors` with `base_url` and a `search_url` (search for something on the
   site and copy the URL, replacing your query with `{query}`). If the site has no usable
   search, set `sitemap_url` instead.
2. Run `pricehunter probe <a product page URL> --competitor <Name> --sku <one of your SKUs>`.
   It shows what each field resolved to, which layer found the price, and the match score.
3. If the price or stock is missing or wrong, open the page in your browser's inspector and add
   `selectors`, then probe again. For prices split across elements (`1.299<sup>99</sup> Lei`),
   select the element containing both parts.
4. If the price only appears after JavaScript runs (single-page apps), try `probe --render`.
   If that works, set `render_js: true` for the competitor. Rendering is slower, so use it only
   where it is needed.
5. If discovery picks up junk links, set `product_url_pattern` (e.g. `"/p/"`) or
   `selectors.search_result_link`. If a product is hard to match automatically, pin its URL in
   `competitor_urls`.

## Scheduling

- **Built in:** `pricehunter watch --every 2h` (run it under `tmux`, `nohup` or a service manager).
- **cron**, every 2 hours at minute 7:
  ```cron
  7 */2 * * *  cd /srv/pricehunter && .venv/bin/pricehunter scan -c pricehunter.yaml --report reports/latest.html >> scan.log 2>&1
  ```
- **systemd timer:**
  ```ini
  # /etc/systemd/system/pricehunter.service
  [Service]
  Type=oneshot
  WorkingDirectory=/srv/pricehunter
  ExecStart=/srv/pricehunter/.venv/bin/pricehunter scan -c pricehunter.yaml --report reports/latest.html

  # /etc/systemd/system/pricehunter.timer
  [Timer]
  OnCalendar=*-*-* 00/2:07:00
  Persistent=true
  [Install]
  WantedBy=timers.target
  ```
  Enable it with `systemctl enable --now pricehunter.timer`.

Add `--strict` to have `scan` exit with code 3 when the run reported errors (pages that failed,
notifiers that could not deliver, unknown SKUs), so systemd marks the run as failed.

Daily or a few times a day is plenty for most markets.

## Web dashboard and API

`pricehunter serve` serves the HTML report with live data and history. It has **Refresh** and
**Run scan** buttons, and the page runs under a strict, hash-based Content-Security-Policy. The
JSON API (Decimal amounts are serialised as strings):

| Endpoint | |
|---|---|
| `GET /api/snapshots` | Market snapshot per product. |
| `GET /api/suggestions` | Repricing suggestions. |
| `GET /api/history/{sku}?competitor=&days=30` | Readings, oldest first. |
| `GET /api/alerts?limit=100&sku=` | Recent alerts. |
| `GET /api/runs?limit=20` | Scan runs and their statistics. |
| `POST /api/scan?sku=&competitor=&discover=true&notify=true` | Start a background scan (`202`, or `409` while one is running). |
| `GET /api/scan/status` | Progress and result of the latest scan. |

The server binds to `127.0.0.1` by default and has no authentication. Put it behind a reverse
proxy with authentication before exposing it.

## Responsible use

PriceHunter is meant for monitoring publicly displayed prices, the way a shopper would see them.

- Keep `respect_robots: true`. Pages disallowed by robots.txt are never fetched, and a
  Crawl-delay slows PriceHunter down further. While a site's robots.txt cannot be read (a
  server error, 429 or network failure), nothing on that site is fetched, as RFC 9309 asks,
  unless a copy read earlier is still at hand. It is retried after five minutes.
- Keep `requests_per_minute` low (10–30 per site), scan only what you need, and scan no more
  often than prices actually change.
- Use a truthful `user_agent` with contact details, so site owners can reach you.
- Read and respect each site's terms of service. Some forbid automated access entirely.
- PriceHunter deliberately has no CAPTCHA solving, fingerprint spoofing or other ways around
  bot protection. If a site blocks you, respect that.

## Project layout

```
src/pricehunter/
  models.py        shared data contracts (pydantic)
  config.py        YAML/JSON loading, ${ENV} expansion, path resolution
  money.py         currency normalisation and conversion
  fetch.py         polite fetcher: robots.txt, rate limits, retries, proxies, Playwright rendering
  pricing.py       price-string parsing ("1.299,99 €", "$1,299.99", ...)
  extract.py       layered offer extraction (selectors, JSON-LD, microdata/RDFa, OpenGraph, heuristics)
  discover.py      candidate discovery via site search and sitemaps
  match.py         product matching and wrong-variant guards
  storage.py       SQLite persistence
  analyze.py       market snapshots and repricing
  alerts.py        alert rules and notifiers
  report.py        HTML / CSV / JSON reports
  engine.py        scan orchestration (run_scan, build_snapshots, probe)
  cli.py           the `pricehunter` command
  web.py           FastAPI dashboard and API
  demo.py          offline demo marketplace (three fake shops)
  example_config.py  the template written by `pricehunter init`
examples/pricehunter.example.yaml
tests/             pytest suite (fully offline: mock transports, local servers, HTML fixtures)
```

## Development

```bash
pip install -e ".[js,dev]"
pytest -q
```

The tests never touch the internet. They use `httpx.MockTransport`, local `http.server`
instances (including the demo marketplace), HTML fixtures and, where Chromium is installed, a
real headless browser.
