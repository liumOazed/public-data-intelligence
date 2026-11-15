# Test fixtures

## `live/`: slices of real responses

Captured by `python -m pdi.samples` (the "Refresh test samples" workflow, run by hand on a
branch). Each file keeps the source's own bytes, header and quoting; only rows are dropped.
The `.meta.json` beside each file records the URL (keys redacted), the retrieval time, the size
and checksum of the full response, and how it was sliced. The tests in
`tests/test_live_samples.py` check format and plausibility, not exact numbers, so a refresh
breaks a test only when a source's format or units changed. A changed slice arrives as a pull
request, so the change is read before it reaches main.

| File | Source |
|---|---|
| `spec_rint2017.csv` | SPEC CPU 2017 results dump, up to 3 rows per vendor group and one without a score |
| `blender_launches.jsonl` | Blender Open Data snapshot, up to 4 launches per outcome |
| `centralcomputer_cpu.html`, `centralcomputer_gpu.html` | First result tiles of two Central Computers searches |
| `fred_*.csv`, `fred_*.json` | FRED public CSV, series notes and observations from the API |
| `bis_*.csv` | BIS exchange rates and policy rates for the six economies |
| `opendosm_ipi.json` | OpenDOSM industrial production, last three months |
| `indeed_US.csv` | Indeed Job Postings Index, last 80 days of each variable |
| `imf_*.csv` | IMF trade, reserves and consumer prices |

Nothing here is published on the site. Raw responses beyond these slices are kept only as
workflow artifacts.

## `synthetic/`: hand-written samples

Kept for two reasons only:

- **Amazon and Best Buy through Bright Data** (`brightdata_*.json`) and the **Best Buy Products
  API** (`bestbuy_cpu_page.json`). Their rows are never committed to this public repository, and
  there is no Best Buy API key, so these stay in the shape of the API's answer.
- **Edge cases a live slice may not contain**: a withdrawn SPEC result, a whole PC listed as a
  part, an out-of-stock tile, a refurbished listing. None of the numbers in them is real.
