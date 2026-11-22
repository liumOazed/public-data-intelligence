"""Capture small slices of real source responses as parser test fixtures.

Each collector's parser is tested against a slice of a response the live source actually
sent, so a test fails when the parser and the real format disagree, not only when the parser
disagrees with somebody's idea of the format. A slice keeps the response's own bytes, header
and quoting; only rows are dropped. Each slice has a `.meta.json` beside it with where and when
it was fetched and the size and checksum of the full response.

Not captured, on purpose:

* Amazon and Best Buy (Bright Data). Their rows are never committed to this public repository
  (see the README), so their fixtures stay hand-written in the shape of the API's answer.
* The Best Buy Products API. There is no key to call it with.

Run as `python -m pdi.samples` from a machine that can reach the sources (the "Refresh test
samples" workflow does it on GitHub's runners). Slices go to tests/fixtures/live/.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests

from pdi.common.fetch import USER_AGENT, redact, utc_now
from pdi.macro import config as macro
from pdi.tech_value import blender, retail
from pdi.tech_value import config as tv

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "live"


def _get(url: str, *, params=None, headers=None, timeout=300, stream=False) -> requests.Response:
    last = None
    for attempt in range(3):
        if attempt:
            time.sleep(2**attempt)
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
                timeout=timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            last = type(exc).__name__
            continue
        if resp.status_code == 200:
            return resp
        last = f"HTTP {resp.status_code}"
    raise RuntimeError(f"{redact(url)}: {last}")


def _save(name: str, body: bytes, *, url: str, full: bytes | None = None, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes(body)
    whole = full if full is not None else body
    meta = {
        "url": redact(url),
        "retrieved_at": utc_now().isoformat(),
        "full_response_bytes": len(whole),
        "full_response_sha256": hashlib.sha256(whole).hexdigest(),
        "slice": note,
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {len(body):,} bytes ({note})")


def _csv_slice(text: str, keep) -> str:
    """Header plus the rows `keep(row_dict)` selects, written back with the source's quoting."""
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    quoting = csv.QUOTE_ALL if text.lstrip().startswith('"') else csv.QUOTE_MINIMAL
    out = io.StringIO()
    w = csv.writer(out, quoting=quoting, lineterminator="\n")
    w.writerow(header)
    for r in body:
        if keep(dict(zip(header, r, strict=False))):
            w.writerow(r)
    return out.getvalue()


# --- Tech Value ---------------------------------------------------------------------------------


def spec() -> None:
    resp = _get(tv.SPEC_ENDPOINT, params={"conf": tv.SPEC_CONF, "op": "dump", "format": "csvdump"})
    text = resp.content.decode("utf-8", errors="replace")
    picked: dict[str, int] = {}

    def keep(r: dict) -> bool:
        proc = r.get("Processor", "")
        score = r.get("Baseline", "").strip()
        if score in ("", "0", "--"):
            group = "no score"
        elif "Intel" in proc:
            group = "intel"
        elif "AMD" in proc:
            group = "amd"
        else:
            group = "other"
        if "  " in proc:
            group += " double space"
        if picked.get(group, 0) >= 3:
            return False
        picked[group] = picked.get(group, 0) + 1
        return True

    _save(
        "spec_rint2017.csv",
        _csv_slice(text, keep).encode(),
        url=resp.url,
        full=resp.content,
        note=f"header and up to 3 rows per group {sorted(picked)}",
    )


def spec_power() -> None:
    """SPECpower_ssj2008 results dump: header and up to 4 rows per processor maker."""
    resp = _get(
        tv.SPEC_ENDPOINT, params={"conf": "power_ssj2008", "op": "dump", "format": "csvdump"}
    )
    text = resp.content.decode("utf-8", errors="replace")
    picked: dict[str, int] = {}

    def keep(r: dict) -> bool:
        blob = " ".join(r.values())
        group = "intel" if "Intel" in blob else "amd" if "AMD" in blob else "other"
        if picked.get(group, 0) >= 4:
            return False
        picked[group] = picked.get(group, 0) + 1
        return True

    _save(
        "spec_power_ssj2008.csv",
        _csv_slice(text, keep).encode(),
        url=resp.url,
        full=resp.content,
        note=f"header and up to 4 rows per group {sorted(picked)}",
    )


def mlperf_inference() -> None:
    """MLPerf Inference summaries: closed rows with power, two rows without, one open power row."""
    from pdi.tech_value import mlperf

    for round_ in ("5.0", "5.1"):
        resp = _get(mlperf.URL.format(round=round_))
        rows = resp.json()
        keep = [r for r in rows if r.get("has_power") and r.get("Category") == "closed"]
        keep += [r for r in rows if not r.get("has_power")][:2]
        keep += [r for r in rows if r.get("has_power") and r.get("Category") == "open"][:1]
        for r in keep:
            for k in ("Notes", "Details", "Code"):
                r.pop(k, None)
        _save(
            f"mlperf_inference_v{round_}.json",
            json.dumps(keep, indent=1).encode(),
            url=resp.url,
            full=resp.content,
            note="closed-division rows with a power measurement, two rows without, one "
            "open-division power row; Notes, Details and Code fields dropped",
        )


def blender_launches() -> None:
    resp = _get(tv.BLENDER_SNAPSHOT_URL, timeout=900, stream=True)
    tmp = OUT.parent / "_blender.zip"
    OUT.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(1 << 20):
            fh.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    picked: dict[str, int] = {}
    lines = []
    archive = zipfile.ZipFile(tmp)
    members = [n for n in archive.namelist() if n.lower().endswith((".jsonl", ".json"))]
    biggest = max(members, key=lambda n: archive.getinfo(n).file_size)
    with archive.open(biggest) as handle:
        for raw in handle:
            try:
                got = blender.parse_launch(json.loads(raw))
            except (ValueError, KeyError, TypeError):
                got = "unreadable"
            group = got if isinstance(got, str) else f"ok {got[0][1]}"
            if picked.get(group, 0) < 4:
                picked[group] = picked.get(group, 0) + 1
                lines.append(raw.rstrip(b"\n"))
            if len(picked) >= 8 and all(v >= 4 for v in picked.values()):
                break
            if sum(picked.values()) >= 60:
                break
    tmp.unlink()
    body = b"\n".join(lines) + b"\n"
    OUT.joinpath("blender_launches.jsonl.meta.json")  # written by _save below
    _save(
        "blender_launches.jsonl",
        body,
        url=resp.url,
        note=f"up to 4 launches per outcome {picked} from {biggest}",
    )
    # _save hashed the slice; record the zip's own size and checksum instead.
    meta_path = OUT / "blender_launches.jsonl.meta.json"
    meta = json.loads(meta_path.read_text())
    meta |= {"full_response_bytes": size, "full_response_sha256": digest.hexdigest()}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def central_computers() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": tv.CRAWLER_USER_AGENT, "Accept": "text/html"})
    robots = retail.Robots(session)
    from bs4 import BeautifulSoup

    for key, name in (("rtx 5070", "gpu"), ("ryzen 7 9700x", "cpu")):
        url = retail.RETAILERS[0].search_url.format(q=quote_plus(key))
        if not robots.allowed(url):
            print(f"central computers: robots.txt does not allow {url}; not captured")
            return
        time.sleep(tv.CRAWLER_PAUSE_S)
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("li.product-item, div.product-item")[:6]
        # The result tiles exactly as served, inside the list element that held them.
        parent = items[0].parent if items else None
        wrapper = f'<{parent.name} class="{" ".join(parent.get("class", []))}">' if parent else ""
        closing = f"</{parent.name}>" if parent else ""
        html = (
            "<!-- Slice of a Central Computers search page: first result tiles, as served. -->\n"
            f"<html><body>{wrapper}\n"
            + "\n".join(str(i) for i in items)
            + f"\n{closing}</body></html>\n"
        )
        _save(
            f"centralcomputer_{name}.html",
            html.encode(),
            url=url,
            full=resp.content,
            note=f"first {len(items)} result tiles for '{key}'",
        )


# --- Macro Monitor --------------------------------------------------------------------------


def fred() -> None:
    code = "CPIAUCSL"
    resp = _get(macro.FRED_CSV, params={"id": code, "cosd": "2024-01-01"})
    _save("fred_CPIAUCSL.csv", resp.content, url=resp.url, note="whole answer from 2024-01")
    key = os.environ.get(macro.FRED_KEY_ENV, "").strip()
    if not key:
        print("FRED API: no key; API samples not refreshed")
        return
    base = {"api_key": key, "file_type": "json"}
    info = _get(f"{macro.FRED_API}/series", params={"series_id": code, **base})
    _save("fred_CPIAUCSL_series.json", info.content, url=info.url, note="whole answer")
    for sid, extra in (
        ("UNRATE", {}),
        ("DTWEXBGS", {"frequency": "m", "aggregation_method": "avg"}),
    ):
        obs = _get(
            f"{macro.FRED_API}/series/observations",
            params={"series_id": sid, "observation_start": "2025-01-01", **base, **extra},
        )
        _save(f"fred_{sid}_observations.json", obs.content, url=obs.url, note="from 2025-01")


def bis() -> None:
    for ind in ("fx_usd", "policy_rate"):
        flow, template, *_ = macro.BIS_INDICATORS[ind]
        key = template.replace("{c}", "+".join(macro.SEA))
        resp = _get(
            f"{macro.BIS_API}/{flow}/{key}/all",
            params={"format": "csv", "detail": "dataonly", "startPeriod": "2026-01"},
        )
        _save(f"bis_{flow}.csv", resp.content, url=resp.url, note="whole answer from 2026-01")


def opendosm() -> None:
    resp = _get(macro.DOSM_API, params={"id": "ipi", "limit": 100000})
    rows = resp.json()
    last = sorted({r["date"] for r in rows})[-3:]
    body = json.dumps([r for r in rows if r["date"] in last], indent=1).encode()
    _save("opendosm_ipi.json", body, url=resp.url, full=resp.content, note=f"rows for {last}")


def indeed() -> None:
    resp = _get(macro.INDEED_URL)
    lines = resp.text.splitlines()
    header, rows = lines[0], lines[1:]
    col = header.split(",").index("variable")
    # The file interleaves several variables; keep the last 80 days of each.
    by_var: dict[str, list[str]] = {}
    for line in rows:
        by_var.setdefault(line.split(",")[col].strip(), []).append(line)
    kept = [line for v in by_var.values() for line in v[-80:]]
    body = ("\n".join([header, *kept]) + "\n").encode()
    _save(
        "indeed_US.csv", body, url=resp.url, full=resp.content,
        note=f"header and the last 80 rows of each variable {sorted(by_var)}",
    )  # fmt: skip


def imf() -> None:
    for name, flow, key in (
        ("imf_ITG_exports.csv", "ITG", "THA+VNM+PHL.XG.FOB_USD.M"),
        ("imf_IL_reserves.csv", "IL", "THA+VNM+PHL.RXF11_REVS.USD.M"),
        ("imf_CPI.csv", "CPI", "VNM+THA.CPI._T.IX.M"),
    ):
        resp = _get(
            f"{macro.IMF_API}/IMF.STA,{flow}/{key}",
            params={"startPeriod": "2025-01"},
            headers={"Accept": macro.IMF_CSV_ACCEPT},
        )
        _save(name, resp.content, url=resp.url, note="whole answer from 2025-01")


def ember() -> None:
    from pdi.energy import config as energy

    resp = _get(energy.EMBER_URL, timeout=600)
    text = resp.content.decode("utf-8-sig")
    rows = (energy.DEMAND_ROW, energy.GENERATION_ROW, energy.NET_IMPORTS_ROW, "Coal")
    codes = set(energy.ISO3.values())

    def keep(r: dict) -> bool:
        return (
            r.get("ISO 3 code") in codes
            and r.get("Electricity source") in rows
            and r.get("Date", "") >= "2024-01"
        )

    _save(
        "ember_monthly.csv",
        _csv_slice(text, keep).encode(),
        url=resp.url,
        full=resp.content,
        note=f"header and rows from 2024-01 for {sorted(codes)}, sources {list(rows)}",
    )


def census() -> None:
    from pdi.trade import config as trade

    key = os.environ.get(trade.KEY_ENV, "").strip()
    if not key:
        print("Census: no key; trade samples not refreshed")
        return
    code = trade.PARTNERS["VN"][0]
    common = {"time": "from 2026-01", "CTY_CODE": code, "key": key}
    hs2 = _get(
        trade.API,
        params={"get": f"I_COMMODITY,I_COMMODITY_SDESC,{trade.VALUE}", "COMM_LVL": "HS2", **common},
    )
    _save("census_VN_hs2.json", hs2.content, url=hs2.url, note="Vietnam, chapters, from 2026-01")
    total = _get(trade.API, params={"get": trade.VALUE, **common})
    _save("census_VN_total.json", total.content, url=total.url, note="Vietnam, total, from 2026-01")


def usaspending() -> None:
    """One month of one product code: the transaction file, and the published monthly figure."""
    from pdi.procurement import collect as proc
    from pdi.procurement import config as pc

    first = (pd.Timestamp(date.today()) - pd.DateOffset(months=5)).replace(day=1)
    last = first + pd.offsets.MonthEnd(0)
    filters = {**proc.filters(first.date(), last.date()), "psc_codes": [USASPENDING_SAMPLE_PSC]}
    job = proc.call("POST", pc.DOWNLOAD, {"filters": filters, "columns": pc.COLUMNS,
                                          "file_format": "csv"})  # fmt: skip
    for _ in range(80):
        status = proc.call("GET", f"{pc.DOWNLOAD_STATUS}?file_name={job['file_name']}")
        if status.get("status") in ("finished", "failed"):
            break
        time.sleep(15)
    if status.get("status") != "finished":
        raise RuntimeError(f"USAspending download {status.get('status')}")
    resp = _get(status["file_url"], timeout=600)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        member = next(n for n in z.namelist() if "PrimeTransactions" in n)
        text = z.read(member).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=pc.COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # A sole trader can be registered under a person's name. Names without a company
        # ending are blanked, so no individual's name is committed.
        for col in ("recipient_name", "recipient_parent_name"):
            if not _COMPANY.search(row.get(col, "")):
                row[col] = ""
        writer.writerow(row)
    note = f"{USASPENDING_SAMPLE_PSC}, {first:%Y-%m}"
    _save("usaspending_transactions.csv", out.getvalue().encode(), url=status["file_url"],
          full=resp.content, note=f"{note}: configured columns only; names without a company "
          "ending blanked")  # fmt: skip
    answer = proc.call("POST", pc.OVER_TIME, {"group": "month", "filters": filters})
    _save("usaspending_over_time.json", json.dumps(answer).encode(), url=pc.API + pc.OVER_TIME,
          note=f"{note}: whole answer")  # fmt: skip


USASPENDING_SAMPLE_PSC = "7K20"
_COMPANY = re.compile(
    r"(?i)\b(inc|llc|l\.l\.c|corp|corporation|company|co|ltd|limited|lp|l\.p|llp|plc|group|"
    r"technolog\w*|systems|solutions|services|federal|government|international|enterprises)\b"
)

CAPTURES = {
    "spec": spec,
    "spec_power": spec_power,
    "mlperf": mlperf_inference,
    "blender": blender_launches,
    "centralcomputer": central_computers,
    "fred": fred,
    "bis": bis,
    "opendosm": opendosm,
    "indeed": indeed,
    "imf": imf,
    "ember": ember,
    "census": census,
    "usaspending": usaspending,
}


def main() -> int:
    failed = []
    for name, fn in CAPTURES.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every source, then fail
            failed.append(f"{name}: {exc}")
            print(f"::warning title=Sample not refreshed::{name}: {exc}")
    print(f"captured on {date.today().isoformat()}; {len(failed)} sources failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
