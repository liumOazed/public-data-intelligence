# Backlog

Ideas that came up while a module was being built. They are written here instead of being
built, and are looked at again when the current module meets its exit criterion.

## Tech Value

- A second price source. 54 US stores were tested and only Central Computers allows a polite crawler and lists parts. The Best Buy API (needs a non-free email address) is the best remaining option.
- Read Central Computers' terms of use and record the date in the registry.
- Find out why about 14% of Blender Open Data launches are unreadable by the parser.

- Replace the hand-written test samples in `tests/fixtures/` with small slices of real
  responses once each collector has run.
- Add a consumer-CPU specification table (cores, clocks, cache, power) so the residual model
  can be fitted on desktop parts as well as SPEC's server-heavy results. The retailer API
  returns specifications and is the likely source.
- Add power (TDP) to the specification model. The SPEC results dump does not carry it.
- Look at OpenBenchmarking.org as a second consumer benchmark source, after reading its terms.
- Rerun AutoGluon's holdout comparison with a linear model in its model list.

## Macro Monitor

- National sources for Indonesia (BPS WebAPI, needs a free key) and Singapore (SingStat Table
  Builder, no key): exports, imports, industrial output, reserves, GDP. IMF series for
  Thailand, Vietnam and the Philippines belong to the Harden phase.
- Show "release delayed" when a series misses its usual release date.
- Rerun the backtest on first-seen values once `vintages.parquet` holds a year of history.
- Read the OpenDOSM licence on open.dosm.gov.my itself (it was read on an archived copy).
- Install CPU-only PyTorch in the training job; the default wheel carries GPU libraries.

## Later modules, in order

1. Trade and Supply Chain
2. Federal Procurement
