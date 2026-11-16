# Public Data Intelligence

Research pages built only from free, public data for the **United States** and six
**Southeast Asian** economies (Indonesia, Vietnam, Thailand, Malaysia, the Philippines,
Singapore). A static site on GitHub Pages, built with Quarto and refreshed by scheduled
GitHub Actions. There is no server, no database and no Streamlit.

## Current state

| Order | Module | Status |
|---|---|---|
| 0 | Skeleton: Quarto site, Pages deploy, scheduled Actions, source registry, failure Issues, heartbeat | Built; pipeline and failure Issue proven on GitHub, Pages deploy not yet run |
| 1 | **Tech Value**: which processors beat their specifications, and CPU and GPU performance per dollar | Benchmarks, model and price crawler run on live data; prices come from one small retailer (see below) |
| 2 | **Macro Monitor**: which series moved unusually against their own history and against neighbouring economies | Built and run once on live data; exit criterion needs one full monthly cycle |
| Harden | Heartbeat with a data-freshness check, parser tests on real response slices, monthly "what changed" brief, IMF series for Southeast Asia; energy demand chosen as the one public sector to cover | Built; exit criterion needs 30 days on schedule (see below) |
| Sector | Energy Demand: monthly electricity demand for the US and five Southeast Asian economies | Built 15 November 2026; one source, observations only |
| 3 | Trade and Supply Chain | Built 15 November 2026: share shift and lead-time test live; mirror gaps, Comtrade and commodity prices not built (see below) |
| 4 | Federal Procurement | First slice built (computer hardware); exit criterion not met, see below |
| 5 | What's unusual this week: one list of every module's flags | Built 16 November 2026; exit criterion needs four scheduled Mondays |

Modules are built one at a time, in this order. The next one does not start until the current
one meets its exit criterion. New ideas go in [`BACKLOG.md`](BACKLOG.md), not into the build.

### What has and has not been checked

- **SPEC and Blender collectors run against the live sources** (first on 14 November 2026:
  13,388 SPEC results, 348,813 usable Blender launches). About 14% of Blender launches could
  not be read by the parser. A slice of the live snapshot (15 November 2026) showed why:
  launches from Blender 2.x in the newer file layout report render time instead of samples
  per minute. They are now filed as `old_schema` (a different measurement), not `unreadable`.
  The weekly run's skip counts will say whether that explains all of the 14%.
- **The model test passed on live data.** On the newest generation, mean absolute log error
  was 0.488 for the cores-times-clock baseline, 0.222 for the model validation picked
  (an AutoGluon ensemble) and 0.102 for the plain regression, against a fold spread of 0.143.
  The plain regression was not the pick and the pick was not changed afterwards; see the
  failures log on the site.
- **Prices come from a crawler, and coverage is thin.** Of 24 US retailers tested from
  GitHub's servers on 14 November 2026, 17 answered with a bot wall or not at all (Newegg,
  Amazon, Best Buy, Walmart, B&H, Micro Center among them), 4 disallow search pages in
  robots.txt, and 3 could be used. The crawler covers one of those, Central Computers, and
  found prices for 34 of the 80 most-benchmarked models on its first run. It reads robots.txt
  on every run, says what it is and does nothing to get past a bot wall. Zones and PCNation
  turned out to be unusable, and a probe of 30 more stores found none that lists parts, so
  the crawler covers one store. Amazon US and Best Buy listings are added once a week through
  Bright Data's Web Scraper API (secret `BRIGHTDATA_API_KEY`, free allowance of 5,000 records
  a month, about 3,500 used by both). Raw Amazon and Best Buy rows are never committed:
  Amazon's terms exclude collecting and reusing its prices, and Best Buy's rows are treated
  the same way, so they live in a private Actions cache and only the derived tables
  (lowest price per model, history, fair price) are published. The ranking shows the lowest listing, the median listing
  and the number of stores.
- The collector for Best Buy's own API is built but inactive: Best Buy issues keys to company email
  addresses only. eBay's API licence does not allow stored price history or price statistics.
- **Parsers are tested on slices of real responses** in
  [`tests/fixtures/live/`](tests/fixtures/README.md), captured from every public source on
  15 November 2026. Amazon and Best Buy (Bright Data) keep hand-written samples, because their
  rows are never committed here.
- Nobody has read the reuse terms of SPEC or Blender Open Data yet. Each is marked
  `terms_checked: unconfirmed` in [`sources/registry.yaml`](sources/registry.yaml).

## Tech Value exit criterion

Written before any result was seen. The module is done when all of these hold on live data:

1. A model trained on older processor generations predicts the **newest generation** better
   than the **cores-times-clock baseline**, and the error of both is published.
   - Target: log of the SPECrate 2017 Integer base score of one published result.
   - Baseline it must beat: `log(score) = a + log(cores x MHz)`, one fitted constant.
   - Split, by the first hardware-availability date a processor appears with: train before
     2023-01-01, validation 2023-01-01 to 2024-05-31, holdout from 2024-06-01.
   - Models are fitted on train, picked on validation, and the pick is scored on the holdout
     once.
   - "Better" means the holdout improvement in mean absolute log error is larger than the
     fold-to-fold spread (standard deviation of the improvement across four expanding
     time-ordered folds before the holdout). If not, the baseline is published and the page
     says "no signal found". Two honest iterations without improvement and the descriptive
     pages ship as they are, with the failure recorded.
2. The ranking page shows the price date beside every price.
3. The daily price collector has run on schedule with nobody touching it, and a deliberately
   broken collector has opened an Issue and left the site intact.

The split dates and thresholds live in [`pdi/tech_value/config.py`](pdi/tech_value/config.py).
Changing them after looking at holdout results breaks the test; record any change in the
failures log.

## Macro Monitor exit criterion

Written on 15 November 2026, before any flag or forecast was computed. The module is done when
all of these hold on live data:

1. **Every series updates without manual work for one full monthly cycle**: each series in
   [`pdi/macro/config.py`](pdi/macro/config.py) gains its next month from a scheduled run
   that nobody started by hand.
2. **Every flag the rule raised in the last 24 months is listed with whether it held.**
   - Move: 100 x change in log level, or change in value for rates.
   - Usual move: median over the previous ten years (same calendar month for seasonal series).
   - Flag: surprise of 2.5 spreads or more, where the spread is 1.4826 x the median absolute
     deviation of past surprises. Against neighbours: the same test on the gap between a
     country's surprise and the median of at least three other Southeast Asian economies.
   - Held or reversed: a flag reversed if the next three months gave back more than half of
     the flagged surprise.
3. **The forecast test has run and its result is published either way.** AutoGluon TimeSeries
   is fitted before 2023-11, picked on 2023-11 to 2024-10 and scored once on the months from
   2024-11. It must beat the best of four rules of thumb (last move, move twelve months ago,
   average move, average move for that calendar month) by more than the spread of its
   improvement across the four quarters of the validation year. Forecasts are published,
   labelled as predictions, only on runs where it does. Otherwise the page says "no signal
   found".

Flags are observations: arithmetic on published values, no model. Changing any threshold
after looking at the backtest breaks the test; record any change in the failures log.

### First live run, 15 November 2026

- 51 series collected from GitHub's runners with no key: 26 US, 20 from the BIS, 5 from
  OpenDOSM. Four BIS series do not exist (Vietnam's policy rate, consumer prices and real
  effective exchange rate, and Singapore's policy rate).
- Backtest, 24 months to October 2026: 61 own-history flags in 1,177 readings (5.2%; a bell
  curve would give 1.2%), 42 held, 10 reversed, 9 too early. 18 neighbour flags, 14 held.
- Forecast test: **no signal found**. The picked model scored 0.638 on the holdout against
  0.658 for the best rule of thumb, an improvement of 0.020 against a fold spread of 0.052.
  No forecast is published.
- Still open: criterion 1, one full monthly cycle on schedule.

### What the Macro Monitor does not cover yet

- Southeast Asia comes from the BIS (policy rate, consumer prices, exchange rate, real
  effective exchange rate) plus five Malaysian series from OpenDOSM. Exports, imports,
  industrial output, reserves and GDP for Indonesia, Singapore, Thailand, Vietnam and the
  Philippines are not wired in. Vietnam has only its exchange rate; Singapore has no policy
  rate.
- The backtest uses today's revised figures. First-seen values are stored from the first run
  (`data/store/macro/vintages.parquet`), so an honest real-time backtest is possible later.
- "Release delayed" is not shown yet when a series misses its usual release date.

## Harden exit criterion

Written on 15 November 2026. The phase is done when both hold:

1. **30 days with no silent failure.** For 30 consecutive days, every day either has a
   successful daily run on main whose published tables pass `pdi/health.py`, or the heartbeat
   Issue was open for it. A day with no run and no Issue is a silent failure and restarts the
   count. The monthly brief lists the days with no successful run.
2. **A stranger can reproduce one headline number from the methodology page.** Someone who
   has not seen this repository follows the steps under "Reproduce one number by hand" on the
   methodology page, with only Python and pandas, and gets the same spreads for US consumer
   prices as `site/data/v1/macro/series.json` for that day. CI already checks that the script
   agrees with the pipeline (`tests/test_reproduce.py`); the criterion needs a person.

### What Harden added, 15 November 2026

- **Heartbeat.** Runs twice a day, filters to runs on main, also checks the weekly run, and
  reads every published table: a source older than its allowance (48 hours for daily sources,
  9 days for weekly ones), a file the daily run stopped rewriting, or an empty key table
  counts as a failure even when every workflow is green. GitHub starts scheduled runs late on
  free accounts (the run due 09:17 UTC on 14 November started at about 15:52 UTC), so the
  48-hour window leaves room for delays of several hours without false alarms.
- **Second daily slot.** On 15 November the 09:17 start had still not appeared by 16:20 UTC
  and the day was started by hand (the late scheduled run then arrived at 16:26). The daily
  workflow now has a second schedule at 19:47 UTC that runs the pipeline only if no daily run
  has succeeded on main that day, so one dropped start does not cost a day.
- **IMF series.** Exports and imports of goods and official reserves excluding gold for all six
  Southeast Asian economies, and Vietnam's consumer prices (19 series, from 2011, because Singapore's trade figures change basis in 2004 and again in 2010). Trade and
  reserves join the neighbour comparison, since the IMF compiles them on one definition.
  Industrial production was checked and left out: the IMF's table ends in 2025 or earlier for
  all six. The four BIS series that do not exist were taken out of the catalogue. The catalogue
  is now 70 series; adding series changes the backtest counts and the forecast test
  population, and the next weekly run re-scores both with the same rules.
- **Monthly brief** under `site/briefs/`, written by code from the stored tables.
- **Parser tests on real slices**, and a workflow to refresh them.

## Trade and Supply Chain

### Exit criterion

Written on 15 November 2026, before any trade figure was collected. The module is done when
both hold:

1. **The share-shift figures match Census published totals within 1%.** For every partner and
   month, the product chapters the pages are built from add up to the all-commodity total
   Census publishes for that partner, within 1%. The collector checks this on every run and
   stores nothing when it fails; the result is on the trade page.
2. **The lead-time claim is shown with a measured lead, or dropped.** The claim that
   US-reported imports say something about an economy's own exports before its own figure is
   out is tested once, out of sample, against a persistence baseline, with the rules in
   [`pdi/trade/config.py`](pdi/trade/config.py). An economy that passes is shown as a
   prediction; one that does not is listed as "no signal found" and nothing on the site uses
   US imports as an early reading for it.

### What is built

- **Collector.** US general imports at customs value by two-digit HS chapter, monthly from
  2013, for China, the six Southeast Asian economies and all countries together (Census
  international trade API, sixteen calls at most once every six days).
- **Share shift.** Each partner's share of US imports over twelve months, in total and by
  chapter, against calendar 2017 and against a year earlier; and for each chapter of at least
  $5 billion a year, China's change beside each Southeast Asian economy's. Observations only.
- **Lead-time test.** Per economy, with the IMF export series the Macro Monitor already
  collects.

### What is not built, and why

- **Mirror gaps** (US-reported imports against each economy's reported exports to the US). It
  needs exports by destination; the IMF series in hand are totals to all countries. Backlog.
- **UN Comtrade intra-regional flows.** Needs a Comtrade key and a reading of its terms. Backlog.
- **Commodity price context.** Not started. Backlog.

## Federal Procurement

First slice, chosen on 15 November 2026: **computer hardware bought by every federal agency**
(computers and servers, storage, data centre equipment, desktops, laptops and printers), from
October 2016, from USAspending.gov.

### Exit criterion

Written before the module's figures were computed: **monthly totals match USAspending's own
figure within 1%.** For every month, the transactions downloaded for the pages must add up to
within 1% of the monthly figure USAspending's own search gives for the same filters. Every
month has to pass.

**Result on 15 November 2026: not met.** 95 of 121 months are within 1%; 26 are not, and the
worst (January 2026) is 14.2% apart. Over the whole period the search figure is 0.8% higher
than the download. The limit has not been moved.

**Second check, added after the first failed: 25 of 26 missed months reconcile row by row.**
The transactions behind the search figure were matched to the download one by one on award,
modification, date and amount. In every month checked, every transaction the search lists is
in the download; the search lists 288 transactions a second time, worth $442.6 million, and
with those removed the two agree within 0.0004%. October 2024 could not be listed (dropped
connection) and is retried weekly. So the gap is double counting in USAspending's search, but
the original criterion stays recorded as not met. The pages show the download
figures with this verdict beside them, and the source is marked amber.

### What is built

- **Collector** (weekly run). One transaction file per fiscal year; sixteen columns, none
  about a person or an address. Only sums are stored; the transactions are not committed.
- **Three kinds of money kept apart**: promised (obligation), ceiling and paid (outlay).
- **Supplier name cleaning**: parent company, then identical cleaned names. Names are kept
  only for the 25 largest suppliers of a fiscal year.
- **Metrics**: value by month, kind of hardware and agency; concentration; new entrants;
  single-bidder share; promised against paid by the year an award began. All observations.
- **Pages**: computer hardware by month, suppliers and competition, promised against paid.

Rules are in [`pdi/procurement/config.py`](pdi/procurement/config.py). What is left is in
[BACKLOG.md](BACKLOG.md).

## Energy Demand

Chosen on 15 November 2026 as the one public sector to cover. Monthly electricity demand
from Ember's monthly electricity data (CC BY 4.0) for the United States, Malaysia, the
Philippines, Singapore, Thailand and Vietnam. Indonesia has no monthly demand at the source.

- The page shows demand per day, its change on a year earlier, an index with 2019 = 100, and
  whether the latest monthly move was unusual by the Macro Monitor's rule (2.5 spreads, same
  calendar month). Every figure is an observation; nothing is fitted or forecast.
- The collector stops the run if a required country disappears, if a value is outside 0.5 to
  1,000 TWh, or if demand stops matching generation plus net imports.
- The 30 MB file is fetched at most once every six days; the publisher updates twice a month.
- Not built: oil demand from JODI. Its terms of use reserve all rights and grant no reuse of
  the data, so nothing is built on it until JODI says otherwise. A cross-check of the US
  figures against the EIA is in the backlog (it needs an EIA key).

## What's unusual this week

One page that lists, across every module, what moved unusually, each item with its finding
label, the date it refers to, and a link to the page and data file behind it. Written on 16
November 2026, before the page was first built.

### Exit criterion

Written before the page was first built: **the page builds on schedule, and no untested finding
is labelled a prediction.** Both parts must hold:

1. The page is rebuilt by the scheduled runs on main, with no run started by hand, on four
   Mondays in a row (the first possible four are 23 November to 14 December 2026).
2. At every build, an item labelled prediction must carry a passed out-of-sample test from its
   module; the build stops otherwise. Checked in code at every build, and by a test.

### Fixed rules, one per module

Set before any item was computed. A rule changed later is a new rule with its own failures-log
row.

| Module | An item is listed when |
|---|---|
| Macro Monitor | a series' latest month carries a flag under the module's own rule: 2.5 spreads against its own ten-year history, or against its neighbours |
| Energy Demand | an economy's latest month is flagged under the same rule (the move against the same calendar month in up to ten earlier years, 2.5 spreads) |
| Trade | the month-on-month change in US imports from a partner, in its latest month, is 2.5 spreads from what that calendar month usually brings, by the same rule |
| Federal Procurement | the month-on-month change in obligations, in the latest month that is not provisional, is 2.5 spreads from what that calendar month usually brings, by the same rule |
| Tech Value | a model's lowest listed price on the latest collection day is 5% or more above or below its price seven days earlier (or its earliest price within those seven days) |

"New this week" means the item was first seen by a build in the seven days before this one.
Items present at the very first build are shown as standing, not new, since nothing can be
called new without an earlier build to compare with.

### What is built

- **Shared format.** `pdi.common.findings.item` and `write_items`: each module's export writes
  `site/data/v1/<module>/findings.json` with the same columns. `findings.item` refuses a
  prediction without a passed test.
- **Builder** (`python -m pdi.unusual`, every daily and weekly run). Reads the five files,
  stops the run on an untested prediction or an unknown label, records first-seen times in
  `data/store/unusual/seen.json`, and writes `site/data/v1/unusual/week.json`.
- **Page** `site/unusual.qmd`, first in the navigation: new this week, still standing, and the
  rules.
- **Trial build on the stored data of 16 November 2026** (an observation, not the first
  scheduled build): 8 items. Six Macro Monitor flags, US imports from Vietnam in September
  (2.5 spreads), and federal hardware obligations in July (down 47.9%, 2.9 spreads). No energy
  or price item. No item is a prediction. The first scheduled build on main sets the baseline.

## First-time setup

These are the steps only the repository owner can do.

1. **Enable Pages.** Settings, Pages, set *Source* to **GitHub Actions**.
2. **Nothing to set up for prices.** The crawler needs no key. If you later get a Best Buy API key (it needs a company email
   address), add it as the `BESTBUY_API_KEY` Actions secret and read Best Buy's API terms on
   caching and display first. Without it the daily run skips the price collector with a
   warning.
   For the Macro Monitor, add a free **`FRED_API_KEY`** secret (register at
   https://fred.stlouisfed.org/docs/api/api_key.html). Without it the collector still works,
   through FRED's public CSV download, but cannot check each series' notes for a copyright
   mark. For Trade and Supply Chain, add a free **`CENSUS_API_KEY`** secret (request one at
   https://api.census.gov/data/key_signup.html). Without it the collector is skipped and the
   trade pages keep the last stored data.
3. **Run the weekly workflow once by hand.** Actions, *Weekly benchmarks and model selection*,
   Run workflow. It collects SPEC and Blender data, trains AutoGluon, commits the derived
   tables and deploys the site. If it fails, it opens an Issue that says which step broke.
   A manual run on any branch other than `main` is a dry run: it tests the pipeline and
   publishes nothing.
4. If `main` is protected, allow `github-actions[bot]` to push to it. The scheduled jobs commit
   derived tables to `main`.

## How it works

Every scheduled run goes through the same six steps: collect, validate, transform, model,
export, build and deploy.

| Workflow | Schedule (UTC) | What it does |
|---|---|---|
| `daily.yml` | 09:17 every day, and 19:47 if no run has succeeded that day | Price snapshot, macro series and flags, rebuild exported tables, write last month's brief on the first run of a month, deploy |
| `weekly.yml` | Monday 05:43 | Refresh SPEC and Blender tables, federal hardware contracts (USAspending), retrain AutoGluon (Tech Value and the macro forecast test), deploy |
| `heartbeat.yml` | 03:05 and 21:05 every day | Opens one Issue if the daily run has not succeeded on main in 48 hours, the weekly run in 9 days, or any published table is stale or empty (`pdi/health.py`); closes it when all checks pass again |
| `samples.yml` | by hand, on a branch | Refreshes the real response slices the parser tests use and commits them to that branch |
| `site.yml` | called by the two above | Renders Quarto and publishes to Pages |
| `ci.yml` | pull requests | Lint, tests, site render |

A failed pipeline job opens an Issue and stops before the site job, so the site keeps showing
the last good data.

```
sources/registry.yaml     every source: URL, owner, licence, frequency, lag, status colour
pdi/common/               fetch (raw snapshot + checksum), registry, export envelope, paths
pdi/tech_value/           one file per source (spec, blender, bestbuy), names, value, model,
                          automl, export
pdi/macro/                config (the series catalogue and every threshold), collect, store,
                          flags, automl, export
pdi/site.py               the only way pages read data
data/raw/                 untouched responses; git-ignored, kept as workflow artifacts
data/store/               tidy derived tables, committed by the workflows
site/                     Quarto project
site/data/v1/<module>/    the contract: what the pages read, one JSON file per table
tests/                    parser, model and export tests
```

### Design rules

- **`site/data/v1/` is a contract.** Pages read only from it, by fixed paths and column names,
  through `pdi.site.load`. Every file is `{"meta": ..., "data": [...]}` and the metadata carries
  source, retrieval date, frequency, typical lag and limitations. A later API can serve the
  same shapes and the pages will not change.
- **Raw data is never edited and never published.** Fixes happen in transform code.
- **Secrets live in Actions secrets only.** URLs are redacted before they are written anywhere.
- **The naive baseline is always a row on the leaderboard.**

### One repository instead of two

The plan document describes a private pipeline repository and a public site repository. This
build keeps everything in this one public repository for now. The consequences:

- Raw responses cannot be committed, because everything here is public. They are kept as
  workflow artifacts for 90 days instead. SPEC and Blender data can be downloaded again in
  full at any time. Prices cannot, so the tidy daily price table in
  `data/store/tech_value/prices/` is the permanent record.
- That price table is public. Check Best Buy's terms before relying on this, and move
  `data/store/` to a private repository if they do not allow it.

## Adding the next module

1. Add its sources to `sources/registry.yaml`.
2. Create `pdi/<module>/` with collectors that use `pdi.common.fetch.fetch`, a transform that
   validates with a Pandera schema and writes to `data/store/<module>/`, and an `export.py`
   that calls `pdi.common.export.write_table`.
3. Add the module's steps to `daily.yml` or `weekly.yml`.
4. Add pages under `site/<module>/` that read with `pdi.site.load` and print
   `pdi.site.source_note` under every table and chart, then add them to the navbar in
   `site/_quarto.yml`.
5. Write the module's exit criterion in this README before building it.

## Running it locally

```bash
uv sync --extra site --extra dev          # add --extra automl for the training job
uv run pytest -q

uv run python -m pdi.tech_value.spec      # collect + tidy SPEC CPU 2017
uv run python -m pdi.tech_value.blender   # collect + tidy Blender Open Data
BESTBUY_API_KEY=... uv run python -m pdi.tech_value.bestbuy
uv run python -m pdi.tech_value.automl    # needs --extra automl
uv run python -m pdi.tech_value.export    # writes site/data/v1/tech-value/

uv run python -m pdi.macro.collect        # FRED, BIS, OpenDOSM, IMF, Indeed; no key needed
uv run python -m pdi.macro.automl         # needs --extra automl
uv run python -m pdi.macro.export         # flags and writes site/data/v1/macro/

CENSUS_API_KEY=... uv run python -m pdi.trade.collect   # US imports by chapter
uv run python -m pdi.trade.export        # shares, totals check, lead-time test
uv run python -m pdi.energy.collect       # Ember electricity demand; no key needed
uv run python -m pdi.energy.export
uv run python -m pdi.health               # is every published table fresh?
uv run python -m pdi.brief --month 2026-11 --force   # write one monthly brief
uv run python -m pdi.samples              # refresh the live test slices (needs network)
python scripts/reproduce_headline.py      # one headline number, without this repo's code

QUARTO_PYTHON=.venv/bin/python quarto preview site
```
