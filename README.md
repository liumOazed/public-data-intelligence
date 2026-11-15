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
| 3 | Trade and Supply Chain | Not started |
| 4 | Federal Procurement | Not started |

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
- **IMF series.** Exports and imports of goods and official reserves excluding gold for all six
  Southeast Asian economies, and Vietnam's consumer prices (19 series, from 2011, because Singapore's trade figures change basis in 2004 and again in 2010). Trade and
  reserves join the neighbour comparison, since the IMF compiles them on one definition.
  Industrial production was checked and left out: the IMF's table ends in 2025 or earlier for
  all six. The four BIS series that do not exist were taken out of the catalogue. The catalogue
  is now 70 series; adding series changes the backtest counts and the forecast test
  population, and the next weekly run re-scores both with the same rules.
- **Monthly brief** under `site/briefs/`, written by code from the stored tables.
- **Parser tests on real slices**, and a workflow to refresh them.

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
   mark. No Census key is needed until the Trade module.
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
| `daily.yml` | 09:17 every day | Price snapshot, macro series and flags, rebuild exported tables, write last month's brief on the first run of a month, deploy |
| `weekly.yml` | Monday 05:43 | Refresh SPEC and Blender tables, retrain AutoGluon (Tech Value and the macro forecast test), deploy |
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

uv run python -m pdi.health               # is every published table fresh?
uv run python -m pdi.brief --month 2026-11 --force   # write one monthly brief
uv run python -m pdi.samples              # refresh the live test slices (needs network)
python scripts/reproduce_headline.py      # one headline number, without this repo's code

QUARTO_PYTHON=.venv/bin/python quarto preview site
```
