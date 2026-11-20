"""SPECworkstation CPU subsystem scores, read from the SPEC GWPG pages the GPU collector saves.

SPECworkstation 3.0, 3.1 and 4.0 publish one score per hardware subsystem (CPU, graphics,
storage) next to the industry-vertical scores. The CPU subsystem score is a SPEC-published CPU
measurement for desktop and workstation processors that SPEC CPU 2017 rarely covers (Core i7 and
i9, Core Ultra, Threadripper PRO), so it fills part of the CPU audit queue.

It is a different benchmark from SPECrate 2017 Integer: the scores are never mixed with SPEC CPU
2017, never fed to the CPU over- and under-performer models, and compared only within one
SPECworkstation version. Every row records the suite and version, the submitter, the system, the
CPU and GPU as SPEC wrote them, the disclosure link and the retrieval time.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from pdi.common.fetch import latest
from pdi.tech_value import config

_LOOKS_LIKE_CPU = re.compile(r"\b(intel|amd|xeon|core|ryzen|epyc|threadripper|cortex)\b", re.I)


def metric_name(suite: str) -> str:
    return f"{suite} CPU subsystem score"


def clean_cpu(raw: str) -> str:
    """SPEC's CPU text without trademarks or a trailing nominal clock."""
    s = re.sub(r"\(R\)|\(TM\)|®|™", "", str(raw))
    s = re.sub(r"\s+CPU\s*@.*$|\s+@\s*[\d.]+\s*GHz.*$", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def parse_page(suite: str, base_url: str, html: str, retrieved_at: str | None = None) -> list[dict]:
    """Rows of one SPECworkstation summary page that carry a readable CPU subsystem score."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        header = None
        start = 0
        for i, tr in enumerate(trs[:3]):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if "CPU Model" in cells and "CPU" in cells:
                header, start = cells, i + 1
                break
        if header is None:
            continue
        col = {name: header.index(name) for name in ("CPU Model", "CPU")}
        gpu_col = header.index("Graphics Device") if "Graphics Device" in header else None
        for tr in trs[start:]:
            vals = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(vals) != len(header) or vals == header:
                continue
            cpu_raw = vals[col["CPU Model"]]
            try:
                score = float(vals[col["CPU"]])
            except ValueError:
                continue
            if not _LOOKS_LIKE_CPU.search(cpu_raw) or not 0 < score < 1000:
                continue
            link = tr.find("a", href=True)
            out.append(
                {
                    "suite": suite,
                    "metric": metric_name(suite),
                    "system": vals[0],
                    "submitter": vals[1],
                    "cpu_raw": cpu_raw,
                    "processor": clean_cpu(cpu_raw),
                    "multi_socket": bool(re.match(r"^\s*\d+\s*x\s", cpu_raw, re.I)),
                    "gpu_raw": vals[gpu_col] if gpu_col is not None else "",
                    "score": score,
                    "disclosure_url": urljoin(base_url, link["href"]) if link else base_url,
                    "retrieved_at": retrieved_at,
                }
            )
    return out


def parse(bundle: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict] = []
    for suite, item in bundle.items():
        if not suite.startswith("SPECworkstation"):
            continue
        rows.extend(parse_page(suite, item["url"], item["html"], item.get("retrieved_at")))
    return pd.DataFrame(rows)


def transform() -> pd.DataFrame:
    """Read the latest SPEC GWPG bundle and write the SPECworkstation CPU table."""
    manifest = latest(config.RAW_DIR / "gwpg", "*_gwpg_bundle.json")
    df = parse(json.loads(manifest.read_text(encoding="utf-8")))
    if df.empty:
        raise ValueError("No SPECworkstation CPU subsystem scores were parsed.")
    config.WS_CPU_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.WS_CPU_TABLE, index=False)
    return df


if __name__ == "__main__":
    print(f"SPECworkstation CPU rows: {len(transform())}")
