# Backlog

Ideas that came up while a module was being built. They are written here instead of being
built, and are looked at again when the current module meets its exit criterion.

## Tech Value

- Find a US price source whose terms allow stored daily prices. Best Buy needs a company
  email for a key. eBay's API licence rules out price history and price statistics.
- Find out why about 14% of Blender Open Data launches are unreadable by the parser.

- Replace the hand-written test samples in `tests/fixtures/` with small slices of real
  responses once each collector has run.
- Add a consumer-CPU specification table (cores, clocks, cache, power) so the residual model
  can be fitted on desktop parts as well as SPEC's server-heavy results. The retailer API
  returns specifications and is the likely source.
- Add power (TDP) to the specification model. The SPEC results dump does not carry it.
- Look at OpenBenchmarking.org as a second consumer benchmark source, after reading its terms.
- Rerun AutoGluon's holdout comparison with a linear model in its model list.

## Later modules, in order

1. Macro Monitor
2. Trade and Supply Chain
3. Federal Procurement
