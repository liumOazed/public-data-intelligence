# Public Data Intelligence

Research pages built only from free, public data for the **United States** and six
**Southeast Asian** economies (Indonesia, Vietnam, Thailand, Malaysia, the Philippines,
Singapore). A static site on GitHub Pages, built with Quarto and refreshed by scheduled
GitHub Actions. There is no server, no database and no Streamlit.

## Current state

| Order | Module | Status |
|---|---|---|
| 0 | Skeleton: Quarto site, Pages deploy, scheduled Actions, source registry, failure Issues, heartbeat | Built; pipeline and failure Issue proven on GitHub, Pages deploy not yet run |
| 1 | **Tech Value**: which processors beat their specifications, and CPU and GPU performance per dollar | Benchmarks and model run on live data; **no price source yet** (see below) |
| 2 | Macro Monitor | Not started |
| 3 | Trade and Supply Chain | Not started |
| 4 | Federal Procurement | Not started |

Modules are built one at a time, in this order. The next one does not start until the current
one meets its exit criterion. New ideas go in [`BACKLOG.md`](BACKLOG.md), not into the build.

### What has and has not been checked

- **SPEC and Blender collectors run against the live sources** (first on 14 November 2026:
  13,388 SPEC results, 348,813 usable Blender launches). About 14% of Blender launches could
  not be read by the parser; the reason has not been looked into yet.
- **The model test passed on live data.** On the newest generation, mean absolute log error
  was 0.488 for the cores-times-clock baseline, 0.222 for the model validation picked
  (an AutoGluon ensemble) and 0.102 for the plain regression, against a fold spread of 0.143.
  The plain regression was not the pick and the pick was not changed afterwards; see the
  failures log on the site.
- **There is no price source.** Best Buy only issues API keys to company email addresses, and
  eBay's API licence does not allow stored price history, price statistics or price modelling.
  The Best Buy collector is built and tested on a hand-written sample only; it has never run
  live and its two category ids are from memory. The value page shows benchmark scores and
  says no prices have been collected. Exit criteria 2 and 3 below stay open until a permitted
  price source exists.
- The test samples in [`tests/fixtures/`](tests/fixtures/README.md) are still hand-written.
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

## First-time setup

These are the steps only the repository owner can do.

1. **Enable Pages.** Settings, Pages, set *Source* to **GitHub Actions**.
2. **Prices are optional for now.** If you get a Best Buy API key (it needs a company email
   address), add it as the `BESTBUY_API_KEY` Actions secret and read Best Buy's API terms on
   caching and display first. Without it the daily run skips the price collector with a
   warning.
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
| `daily.yml` | 09:17 every day | Price snapshot, rebuild exported tables, deploy |
| `weekly.yml` | Monday 05:43 | Refresh SPEC and Blender tables, retrain AutoGluon, deploy |
| `heartbeat.yml` | 21:05 every day | Opens an Issue if the daily run has not succeeded in 48 hours |
| `site.yml` | called by the two above | Renders Quarto and publishes to Pages |
| `ci.yml` | pull requests | Lint, tests, site render |

A failed pipeline job opens an Issue and stops before the site job, so the site keeps showing
the last good data.

```
sources/registry.yaml     every source: URL, owner, licence, frequency, lag, status colour
pdi/common/               fetch (raw snapshot + checksum), registry, export envelope, paths
pdi/tech_value/           one file per source (spec, blender, bestbuy), names, value, model,
                          automl, export
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

QUARTO_PYTHON=.venv/bin/python quarto preview site
```
