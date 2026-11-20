# Backlog

Ideas that came up while a module was being built. They are written here instead of being
built, and are looked at again when the current module meets its exit criterion.

## Tech Value

- A second price source. 54 US stores were tested and only Central Computers allows a polite crawler and lists parts. The Best Buy API (needs a non-free email address) is the best remaining option.
- Read Central Computers' terms of use and record the date in the registry.
- Confirm on the weekly run's skip counts that Blender 2.x launches (render time, not samples
  per minute) explain the whole 14% once filed as `old_schema`.

- Check the hand-entered CPU table (pdi/tech_value/cpu_specs.py) against a second source;
  Core Ultra 5 250K Plus and 7 270K Plus were entered from launch coverage.
- Ask the Blender Foundation whether the weekly snapshot download is fine, now that
  opendata.blender.org's robots.txt disallows /snapshots/.
- Ask Primate Labs (Geekbench) and PassMark whether derived, attributed figures may be published.
- Add power (TDP) to the specification model. The SPEC results dump does not carry it.
- Look at OpenBenchmarking.org as a second consumer benchmark source, after reading its terms.
- Rerun AutoGluon's holdout comparison with a linear model in its model list.

## Macro Monitor

- National sources for Indonesia (BPS WebAPI, needs a free key) and Singapore (SingStat Table
  Builder, no key): industrial output and GDP, which the IMF tables do not carry fresh.
  Trade and reserves now come from the IMF for all six.
- Ask the IMF (copyright@imf.org) to confirm that a daily call for 19 series through the public
  API is not the "bulk download" its general terms forbid.
- Show "release delayed" when a series misses its usual release date.
- Rerun the backtest on first-seen values once `vintages.parquet` holds a year of history.
- Read the OpenDOSM licence on open.dosm.gov.my itself (it was read on an archived copy).
- Install CPU-only PyTorch in the training job; the default wheel carries GPU libraries.

## Harden

- Ask JODI whether derived figures from JODI-Oil may be published with credit. Its terms of
  use (read 15 November 2026) reserve all rights and say nothing about the data. Oil demand
  would add Indonesia to Energy Demand.
- Cross-check Ember's US electricity demand against the EIA's own monthly figure (needs a free
  EIA API key as a repo secret).
- Add an Energy Demand section to the monthly brief.
- Add a second public sector only after energy demand has run for a month.

## Trade and Supply Chain

- Mirror gaps: find each economy's exports to the US by month (IMF trade by partner, or
  national sources), then compare with US-reported imports from it.
- UN Comtrade for flows between the six economies: get a key, read the terms first.
- Commodity price context (World Bank monthly prices; EIA needs a key).
- Four-digit HS headings for the chapters where the shift is largest (85, 84, 94, 61, 62, 64):
  a shift inside a chapter is invisible at two digits.
- Rerun the lead-time test on first-published values once a year of vintages exists for both
  series.
- Add a trade section to the monthly brief.

## Federal Procurement

- Report to USAspending that its search figure counts some transactions twice (288 second
  copies worth $442.6 million in 25 months; the procurement page lists them by month).
- A second slice: add networks and mobile devices as their own series from October 2020, or
  software.
- Maker against reseller: the contract holder is usually a reseller. The download has a
  free-text description that sometimes names the maker; it needs a tested extractor.
- Fair-opportunity field for orders under umbrella agreements, to replace the number of
  offers where that describes the agreement and not the order.
- Supplier cleaning by hand for the largest 50: companies filed under two parents.
- Add a procurement section to the monthly brief.

## Later modules, in order
