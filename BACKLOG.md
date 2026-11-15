# Backlog

Ideas that came up while a module was being built. They are written here instead of being
built, and are looked at again when the current module meets its exit criterion.

## Tech Value

- A second price source. 54 US stores were tested and only Central Computers allows a polite crawler and lists parts. The Best Buy API (needs a non-free email address) is the best remaining option.
- Read Central Computers' terms of use and record the date in the registry.
- Confirm on the weekly run's skip counts that Blender 2.x launches (render time, not samples
  per minute) explain the whole 14% once filed as `old_schema`.

- Add a consumer-CPU specification table (cores, clocks, cache, power) so the residual model
  can be fitted on desktop parts as well as SPEC's server-heavy results. The retailer API
  returns specifications and is the likely source.
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

- **Sector chosen (15 November 2026): energy demand only.** Monthly electricity demand from
  Ember (CC BY 4.0; US, Malaysia, the Philippines, Singapore, Thailand and Vietnam from 2017-19;
  no monthly figure for Indonesia), oil product demand from JODI-Oil for all seven, and the EIA
  as a check on the US figures. Build it as its own module after the Harden PR merges.
- Read JODI's terms of use before building on JODI-Oil (the download page says the data are
  free but states no reuse terms).
- Add a second public sector only after energy demand has run for a month.

## Later modules, in order

1. Trade and Supply Chain
2. Federal Procurement
