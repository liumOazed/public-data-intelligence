# Test fixtures

These files are **hand-written samples in the shape of each source's response**, not saved
downloads. They exist so the parsers can be tested without network access. None of the numbers
in them is a real benchmark result or a real price, and nothing here is ever published to the
site.

When a collector has run against the live source, replace the matching sample with a small
slice of a real response (and re-run the tests) so the parser is pinned to the true format.

| File | Shape of |
|---|---|
| `spec_rint2017_sample.csv` | SPEC CPU 2017 results dump (`op=dump&format=csvdump`) |
| `blender_sample.jsonl` | Blender Open Data snapshot, one launch per line |
| `bestbuy_cpu_page.json` | Best Buy Products API page |
