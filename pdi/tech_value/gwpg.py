"""SPEC Graphics & Workstation Performance Group (GWPG) GPU collector, parser, and residual model.

Collects the 15 official SPEC GWPG result summary pages (SPECviewperf 15/2020/13/12.1,
SPECworkstation 4.0/3.1/3.0, and SPECapc Maya 2024 / Solidworks 2022 / Creo 9 / 3ds Max 2020)
from spec.org/gwpg/, enriches each canonical GPU with hardware specifications (shader count,
boost clock, VRAM, memory bus width, memory bandwidth, architecture, release year), and
computes the 3-view out-of-fold residual consensus ranking for GPUs.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import statsmodels.api as sm
from bs4 import BeautifulSoup
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from pdi.common.fetch import fetch, latest, read_meta
from pdi.tech_value import config

# Hardware specification lookup for canonical GPUs tested across SPEC GWPG suites:
# (shaders, boost_mhz, vram_gb, bus_bits, bandwidth_gbps, architecture, release_year, is_mobile)
GPU_SPECS: dict[str, tuple[int, int, float, int, float, str, int, int]] = {
    # NVIDIA Blackwell (2025)
    "NVIDIA RTX PRO 6000 Blackwell": (24064, 2617, 96.0, 512, 1792.0, "Blackwell", 2025, 0),
    "NVIDIA RTX PRO 5000 Blackwell": (14080, 2580, 48.0, 384, 1344.0, "Blackwell", 2025, 0),
    "NVIDIA RTX PRO 5000 Blackwell Laptop": (10496, 2175, 24.0, 256, 896.0, "Blackwell", 2025, 1),
    "NVIDIA RTX PRO 3000 Blackwell Laptop": (7680, 2100, 24.0, 192, 672.0, "Blackwell", 2025, 1),
    "NVIDIA RTX PRO 2000 Blackwell Laptop": (4608, 2055, 16.0, 128, 448.0, "Blackwell", 2025, 1),
    # NVIDIA Ada Lovelace (2022-2024)
    "NVIDIA RTX 6000 Ada": (18176, 2505, 48.0, 384, 960.0, "Ada Lovelace", 2022, 0),
    "NVIDIA RTX 5000 Ada": (12800, 2550, 32.0, 256, 576.0, "Ada Lovelace", 2023, 0),
    "NVIDIA RTX 5000 Ada Laptop": (9728, 2115, 16.0, 256, 512.0, "Ada Lovelace", 2023, 1),
    "NVIDIA RTX 4500 Ada": (7680, 2580, 24.0, 192, 432.0, "Ada Lovelace", 2023, 0),
    "NVIDIA RTX 4000 Ada": (6144, 2175, 20.0, 160, 360.0, "Ada Lovelace", 2023, 0),
    "NVIDIA RTX 4000 Ada Laptop": (7424, 2175, 12.0, 192, 432.0, "Ada Lovelace", 2023, 1),
    "NVIDIA RTX 3500 Ada": (5120, 2145, 12.0, 192, 432.0, "Ada Lovelace", 2023, 1),
    "NVIDIA RTX 3000 Ada": (4608, 2115, 8.0, 128, 256.0, "Ada Lovelace", 2023, 1),
    "NVIDIA RTX 2000 Ada": (2816, 2130, 16.0, 128, 224.0, "Ada Lovelace", 2024, 0),
    "NVIDIA RTX 2000 Ada Laptop": (3072, 2115, 8.0, 128, 256.0, "Ada Lovelace", 2023, 1),
    "NVIDIA RTX 1000 Ada": (2560, 2040, 6.0, 96, 192.0, "Ada Lovelace", 2023, 1),
    "NVIDIA GeForce RTX 4080": (9728, 2505, 16.0, 256, 716.8, "Ada Lovelace", 2022, 0),
    "NVIDIA GeForce RTX 4070 Ti": (7680, 2610, 12.0, 192, 504.2, "Ada Lovelace", 2023, 0),
    # NVIDIA Ampere (2020-2022)
    "NVIDIA RTX A6000": (10752, 1800, 48.0, 384, 768.0, "Ampere", 2020, 0),
    "NVIDIA RTX A5500": (10240, 1665, 24.0, 384, 768.0, "Ampere", 2022, 0),
    "NVIDIA RTX A5000": (8192, 1695, 24.0, 384, 768.0, "Ampere", 2021, 0),
    "NVIDIA RTX A4500": (7168, 1650, 20.0, 320, 640.0, "Ampere", 2021, 0),
    "NVIDIA RTX A4000": (6144, 1560, 16.0, 256, 448.0, "Ampere", 2021, 0),
    "NVIDIA RTX A3000": (4096, 1560, 6.0, 192, 264.0, "Ampere", 2021, 1),
    "NVIDIA RTX A2000": (3328, 1200, 12.0, 192, 288.0, "Ampere", 2021, 0),
    "NVIDIA RTX A1000": (2048, 1455, 4.0, 128, 192.0, "Ampere", 2022, 1),
    "NVIDIA GeForce RTX 3080": (8704, 1710, 10.0, 320, 760.3, "Ampere", 2020, 0),
    # NVIDIA Turing (2018-2021)
    "NVIDIA Quadro RTX 8000": (4608, 1770, 48.0, 384, 672.0, "Turing", 2018, 0),
    "NVIDIA Quadro RTX 6000": (4608, 1770, 24.0, 384, 672.0, "Turing", 2018, 0),
    "NVIDIA Quadro RTX 5000": (3072, 1815, 16.0, 256, 448.0, "Turing", 2018, 0),
    "NVIDIA Quadro RTX 4000": (2304, 1545, 8.0, 256, 416.0, "Turing", 2018, 0),
    "NVIDIA Quadro RTX 3000": (1920, 1380, 6.0, 192, 336.0, "Turing", 2019, 1),
    "NVIDIA T1000": (896, 1395, 4.0, 128, 160.0, "Turing", 2021, 0),
    "NVIDIA T600": (640, 1335, 4.0, 128, 160.0, "Turing", 2021, 1),
    "NVIDIA T500": (896, 1695, 4.0, 64, 80.0, "Turing", 2020, 1),
    "NVIDIA Quadro T2000": (1024, 1575, 4.0, 128, 128.0, "Turing", 2019, 1),
    "NVIDIA Quadro T1000": (768, 1455, 4.0, 128, 128.0, "Turing", 2019, 1),
    "NVIDIA GeForce RTX 2080": (2944, 1800, 8.0, 256, 448.0, "Turing", 2018, 0),
    # NVIDIA Pascal (2016-2017)
    "NVIDIA Quadro GP100": (3584, 1442, 16.0, 4096, 717.0, "Pascal", 2017, 0),
    "NVIDIA Quadro P6000": (3840, 1530, 24.0, 384, 432.0, "Pascal", 2016, 0),
    "NVIDIA Quadro P5200": (2560, 1569, 16.0, 256, 230.0, "Pascal", 2018, 1),
    "NVIDIA Quadro P5000": (2560, 1733, 16.0, 256, 288.0, "Pascal", 2016, 0),
    "NVIDIA Quadro P4200": (2304, 1645, 8.0, 256, 224.0, "Pascal", 2018, 1),
    "NVIDIA Quadro P4000": (1792, 1480, 8.0, 256, 243.0, "Pascal", 2017, 0),
    "NVIDIA Quadro P3200": (1792, 1544, 6.0, 192, 168.0, "Pascal", 2018, 1),
    "NVIDIA Quadro P3000": (1280, 1215, 6.0, 192, 168.0, "Pascal", 2017, 1),
    "NVIDIA Quadro P2200": (1280, 1493, 5.0, 160, 200.0, "Pascal", 2019, 0),
    "NVIDIA Quadro P2000": (1024, 1470, 5.0, 160, 140.0, "Pascal", 2017, 0),
    "NVIDIA Quadro P1000": (640, 1480, 4.0, 128, 82.0, "Pascal", 2017, 0),
    "NVIDIA Quadro P620": (512, 1354, 2.0, 128, 80.0, "Pascal", 2018, 0),
    "NVIDIA Quadro P600": (384, 1557, 2.0, 128, 64.0, "Pascal", 2017, 0),
    # NVIDIA Maxwell (2015)
    "NVIDIA Quadro M6000": (3072, 1140, 24.0, 384, 317.0, "Maxwell", 2015, 0),
    "NVIDIA Quadro M5500": (2048, 1140, 8.0, 256, 211.0, "Maxwell", 2016, 1),
    "NVIDIA Quadro M5000": (2048, 1038, 8.0, 256, 211.0, "Maxwell", 2015, 0),
    "NVIDIA Quadro M5000M": (1536, 975, 8.0, 256, 160.0, "Maxwell", 2015, 1),
    "NVIDIA Quadro M4000": (1664, 773, 8.0, 256, 192.0, "Maxwell", 2015, 0),
    "NVIDIA Quadro M4000M": (1280, 975, 4.0, 256, 160.0, "Maxwell", 2015, 1),
    "NVIDIA Quadro M3000M": (1024, 1050, 4.0, 256, 160.0, "Maxwell", 2015, 1),
    "NVIDIA Quadro M2200": (1024, 1038, 4.0, 128, 88.0, "Maxwell", 2017, 1),
    "NVIDIA Quadro M2000": (768, 1163, 4.0, 128, 106.0, "Maxwell", 2016, 0),
    "NVIDIA Quadro M2000M": (640, 1137, 4.0, 128, 80.0, "Maxwell", 2015, 1),
    "NVIDIA Quadro M1200": (640, 1148, 4.0, 128, 80.0, "Maxwell", 2017, 1),
    "NVIDIA Quadro M1000M": (512, 1072, 2.0, 128, 80.0, "Maxwell", 2015, 1),
    "NVIDIA Quadro M620": (512, 1018, 2.0, 128, 80.0, "Maxwell", 2017, 1),
    "NVIDIA Quadro M600M": (384, 1097, 2.0, 128, 80.0, "Maxwell", 2015, 1),
    # NVIDIA Kepler (2012-2014)
    "NVIDIA Quadro K6000": (2880, 902, 12.0, 384, 288.0, "Kepler", 2013, 0),
    "NVIDIA Quadro K5200": (2304, 771, 8.0, 256, 192.0, "Kepler", 2014, 0),
    "NVIDIA Quadro K5100M": (1536, 771, 8.0, 256, 115.0, "Kepler", 2013, 1),
    "NVIDIA Quadro K5000": (1536, 706, 4.0, 256, 173.0, "Kepler", 2012, 0),
    "NVIDIA Quadro K4200": (1344, 784, 4.0, 256, 173.0, "Kepler", 2014, 0),
    "NVIDIA Quadro K4100M": (1152, 706, 4.0, 256, 102.0, "Kepler", 2013, 1),
    "NVIDIA Quadro K3100M": (768, 706, 4.0, 256, 102.0, "Kepler", 2013, 1),
    "NVIDIA Quadro K2200": (640, 1124, 4.0, 128, 80.0, "Kepler", 2014, 0),
    "NVIDIA Quadro K2100M": (576, 667, 2.0, 128, 48.0, "Kepler", 2013, 1),
    "NVIDIA Quadro K1200": (512, 1124, 4.0, 128, 80.0, "Kepler", 2015, 0),
    "NVIDIA Quadro K1100M": (384, 706, 2.0, 128, 45.0, "Kepler", 2013, 1),
    "NVIDIA Quadro K620": (384, 1124, 2.0, 128, 29.0, "Kepler", 2014, 0),
    # AMD RDNA 4 / RDNA 3.5 / RDNA 3 / RDNA 2 / Vega / Polaris / GCN
    "AMD Radeon AI PRO R9700": (4096, 2920, 32.0, 256, 640.0, "RDNA 4", 2025, 0),
    "AMD Radeon 8060S": (2560, 2900, 16.0, 256, 256.0, "RDNA 3.5", 2025, 1),
    "AMD Radeon PRO W7900": (6144, 2495, 48.0, 384, 864.0, "RDNA 3", 2023, 0),
    "AMD Radeon PRO W6800": (3840, 2320, 32.0, 256, 512.0, "RDNA 2", 2021, 0),
    "AMD Radeon PRO W6600": (1792, 2580, 8.0, 128, 224.0, "RDNA 2", 2021, 0),
    "AMD Radeon PRO W5700": (2304, 1930, 8.0, 256, 448.0, "RDNA 1", 2019, 0),
    "AMD Radeon PRO W5500": (1408, 1900, 8.0, 128, 224.0, "RDNA 1", 2020, 0),
    "AMD Radeon Pro WX 9100": (4096, 1500, 16.0, 2048, 483.8, "Vega", 2017, 0),
    "AMD Radeon Pro WX 7100": (2304, 1243, 8.0, 256, 224.0, "Polaris", 2016, 0),
    "AMD Radeon Pro WX 5100": (1792, 1086, 8.0, 256, 211.0, "Polaris", 2016, 0),
    "AMD Radeon Pro WX 4100": (1024, 1201, 4.0, 128, 96.0, "Polaris", 2016, 0),
    "AMD Radeon Pro WX 3100": (512, 1219, 4.0, 128, 96.0, "Polaris", 2017, 0),
    "AMD FirePro W9100": (2816, 930, 16.0, 512, 320.0, "GCN 2", 2014, 0),
    "AMD FirePro W8100": (2560, 824, 8.0, 512, 320.0, "GCN 2", 2014, 0),
    "AMD FirePro W7100": (1792, 920, 8.0, 256, 160.0, "GCN 3", 2014, 0),
    "AMD FirePro W5100": (768, 930, 4.0, 128, 96.0, "GCN 2", 2014, 0),
    "AMD FirePro W5130M": (512, 925, 2.0, 128, 64.0, "GCN 1", 2014, 1),
    "AMD FirePro W4170M": (384, 825, 2.0, 128, 72.0, "GCN 1", 2014, 1),
}

_GPU_MATCH_RULES: tuple[tuple[str, str], ...] = (
    (r"(NVIDIA\s+RTX\s+PRO\s+\d+\s*Blackwell)", "NVIDIA RTX PRO"),
    (r"(NVIDIA\s+RTX\s+\d+(?:\s+SFF)?\s+Ada)", "NVIDIA RTX Ada"),
    (r"(NVIDIA\s+GeForce\s+RTX\s+\d+(?:\s*Ti|\s*SUPER)?)", "NVIDIA GeForce RTX"),
    (r"(NVIDIA\s+RTX\s+A\d+)", "NVIDIA RTX A"),
    (r"(NVIDIA\s+Quadro\s+RTX\s+\d+)", "NVIDIA Quadro RTX"),
    (r"(NVIDIA\s+Quadro\s+GP100)", "NVIDIA Quadro GP100"),
    (r"(NVIDIA\s+Quadro\s+[PMKT]\d+M?)", "NVIDIA Quadro"),
    (r"(NVIDIA\s+T\d+)", "NVIDIA T"),
    (r"(AMD\s+Radeon\s+AI\s+PRO\s+R\d+)", "AMD Radeon AI PRO"),
    (r"(AMD\s+Radeon\s+PRO\s+W\d+X?)", "AMD Radeon PRO W"),
    (r"(AMD\s+Radeon\s+Pro\s+WX\s+\d+)", "AMD Radeon Pro WX"),
    (r"(AMD\s+Radeon\s+RX\s+\d+\s*(?:XT|XTX)?)", "AMD Radeon RX"),
    (r"(AMD\s+Radeon\s+8\d+S)", "AMD Radeon 8000S"),
    (r"(AMD\s+FirePro\s+W\d+M?)", "AMD FirePro"),
    (r"(Intel\s+Arc\s+(?:Pro\s+)?[AB]\d+)", "Intel Arc"),
)

_CANON_REMAP: dict[str, str] = {
    "NVIDIA Quadro K50001": "NVIDIA Quadro K5000",
    "NVIDIA Quadro K22001": "NVIDIA Quadro K2200",
    "NVIDIA Quadro M50001": "NVIDIA Quadro M5000",
}


def canon_gpu(raw: str) -> str:
    """Normalize a free-text SPEC GWPG GPU cell into a canonical GPU product name."""
    s = re.sub(r"\s+", " ", str(raw)).strip()
    s = re.sub(r"\(R\)|\(TM\)|®|™", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.match(r"^(?:Quadro|RTX|GeForce)\b", s, re.I):
        s = "NVIDIA " + s
    elif re.match(r"^(?:Radeon|FirePro)\b", s, re.I):
        s = "AMD " + s
    s = re.sub(r"FirePro\s+WX\s*(\d+)", r"Radeon Pro WX \1", s, flags=re.I)
    s = re.sub(r"Radeon\s+Pro\s+WX(\d+)", r"Radeon Pro WX \1", s, flags=re.I)
    for pat, _ in _GPU_MATCH_RULES:
        m = re.search(pat, s, re.I)
        if m:
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            val = re.sub(r"^nvidia\b", "NVIDIA", val, flags=re.I)
            val = re.sub(r"^amd\b", "AMD", val, flags=re.I)
            val = re.sub(r"radeon\s+pro\s+w(\d+)", r"Radeon PRO W\1", val, flags=re.I)
            val = re.sub(r"radeon\s+pro\s+wx", "Radeon Pro WX", val, flags=re.I)
            val = re.sub(r"\s+SFF\s+Ada", " Ada", val, flags=re.I)
            if ("Blackwell" in val or "Ada" in val) and "Laptop" in s:
                val = val + " Laptop"
            return _CANON_REMAP.get(val, val)
    return _CANON_REMAP.get(s[:45], s[:45])


def _slug(suite: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", suite.lower()).strip("_")


def collect(today: date | None = None) -> Path:
    """Fetch all 15 official SPEC GWPG summary HTML tables into data/raw/tech_value/gwpg/."""
    today = today or date.today()
    bundle: dict[str, dict[str, str]] = {}
    raw_dir = config.RAW_DIR / "gwpg"
    for suite, url in config.GWPG_ENDPOINTS.items():
        slug = _slug(suite)
        dest = raw_dir / f"{today.isoformat()}_{slug}.html"
        fetch(url, dest, timeout=120)
        bundle[suite] = {
            "url": url,
            "html": dest.read_text(encoding="utf-8", errors="replace"),
            "retrieved_at": read_meta(dest)["retrieved_at"],
        }
    manifest = raw_dir / f"{today.isoformat()}_gwpg_bundle.json"
    manifest.write_text(json.dumps(bundle), encoding="utf-8")
    return manifest


def parse_suite_html(suite: str, base_url: str, html: str) -> list[dict]:
    """Parse one SPEC GWPG HTML summary page into normalized disclosure records."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out: list[dict] = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if len(trs) < 3:
            continue
        header_idx = 0
        headers = [th.get_text(" ", strip=True) for th in trs[0].find_all(["th", "td"])]
        if (
            not any(
                "Graphic" in h or "GPU" in h or "System" in h or "Company" in h for h in headers
            )
            and len(trs) > 1
        ):
            headers_1 = [th.get_text(" ", strip=True) for th in trs[1].find_all(["th", "td"])]
            if any(
                "Graphic" in h or "GPU" in h or "System" in h or "Company" in h for h in headers_1
            ):
                headers = headers_1
                header_idx = 1
        if not any(
            "Graphic" in h or "GPU" in h or "System" in h or "Company" in h for h in headers
        ):
            continue
        for tr in trs[header_idx + 1 :]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 5:
                continue
            vals = [c.get_text(" ", strip=True) for c in cells]
            if vals[0] in headers:
                continue
            links = [a["href"] for a in tr.find_all("a", href=True)]
            html_link = next(
                (lnk for lnk in links if lnk.endswith(".html") or lnk.endswith(".pdf")),
                links[0] if links else None,
            )
            full_link = urljoin(base_url, html_link) if html_link else base_url
            row_dict = dict(zip(headers, vals, strict=False))

            gpu_raw = ""
            for k in headers:
                kl = k.lower().strip()
                if kl in (
                    "graphics",
                    "gpu",
                    "graphics device",
                    "graphic",
                    "graphics card",
                    "graphics accelerator",
                ):
                    gpu_raw = row_dict.get(k, "")
                    break
            if not gpu_raw:
                for k in headers:
                    if (
                        any(w in k.lower() for w in ("graphic", "gpu", "accelerator"))
                        and "composite" not in k.lower()
                    ):
                        gpu_raw = row_dict.get(k, "")
                        break
            if not gpu_raw:
                for k in headers:
                    if "company" in k.lower() or "system" in k.lower():
                        gpu_raw = row_dict.get(k, "")
                        break
            if not gpu_raw and len(vals) > 2:
                gpu_raw = vals[2]

            sys_raw = row_dict.get(
                "System",
                row_dict.get(
                    "Company/Product",
                    row_dict.get(
                        "Company/ Product",
                        row_dict.get(
                            "Submitter",
                            vals[0] + " " + vals[1] if len(vals) > 1 else vals[0],
                        ),
                    ),
                ),
            )
            vendor_oem = row_dict.get(
                "Submitter",
                row_dict.get("Submitted By", row_dict.get("Sponsor", vals[0].split()[0])),
            )
            cpu_raw = ""
            for k in headers:
                if (
                    "cpu" in k.lower() or "processor" in k.lower()
                ) and "composite" not in k.lower():
                    cpu_raw = row_dict.get(k, "")
                    break
            res_raw = "1920x1080"
            for k in headers:
                if "res" in k.lower():
                    res_raw = row_dict.get(k, "1920x1080")
                    break

            num_scores: list[tuple[str, float]] = []
            for k, v in row_dict.items():
                kl = k.lower().strip()
                if kl in (
                    "system",
                    "vendor",
                    "submitter",
                    "submitted by",
                    "sponsor",
                    "company/product",
                    "company/ product",
                    "cpu",
                    "cpu model",
                    "processor",
                    "graphics",
                    "gpu",
                    "graphics device",
                    "storage device",
                    "o/s",
                    "os",
                    "virtualized",
                    "version",
                    "test date",
                    "publication date",
                    "benchmark resolution",
                    "screen resolution",
                    "resolution",
                    "aa",
                    "anti-aliasing",
                    "cpu composite",
                    "cpu",
                    "storage",
                ):
                    continue
                if any(
                    w in kl
                    for w in (
                        "driver",
                        "date",
                        "mem",
                        "storage",
                        "display",
                        "report",
                        "link",
                        "#",
                        "threads",
                        "cores",
                        "ghz",
                        "mhz",
                        "cpu ",
                    )
                ):
                    continue
                m = re.search(r"^\d+(?:\.\d+)?$", v.strip())
                if m:
                    val = float(m.group(0))
                    if 0.1 < val < 5000:
                        num_scores.append((k, val))
            if not num_scores:
                continue

            geomean_score = float(np.exp(np.mean(np.log([s[1] for s in num_scores]))))
            max_score = float(np.max([s[1] for s in num_scores]))
            min_score = float(np.min([s[1] for s in num_scores]))
            best_workload = max(num_scores, key=lambda x: x[1])[0]

            gpu_canon = canon_gpu(gpu_raw)
            if gpu_canon not in GPU_SPECS:
                continue
            (
                shaders,
                boost_mhz,
                vram_gb,
                bus_bits,
                bw_gbps,
                arch,
                rel_year,
                is_mobile,
            ) = GPU_SPECS[gpu_canon]
            gpu_vendor = "NVIDIA" if "NVIDIA" in gpu_canon else "AMD"

            is_4k = (
                "3840" in res_raw
                or "4k" in res_raw.lower()
                or "3840" in gpu_raw
                or "2160" in res_raw
            )
            res_tier = "4K (3840x2160)" if is_4k else "FHD/QHD (1080p/1440p)"
            os_family = "Linux" if "Linux" in suite else "Windows"
            suite_base = suite.replace(" Linux", "")

            rows_out.append(
                {
                    "suite": suite,
                    "suite_base": suite_base,
                    "os_family": os_family,
                    "res_tier": res_tier,
                    "oem": vendor_oem.strip(),
                    "system": sys_raw.strip(),
                    "cpu": cpu_raw.strip(),
                    "gpu_raw": gpu_raw.strip(),
                    "processor": gpu_canon,
                    "vendor": gpu_vendor,
                    "family": arch,
                    "release_year": int(rel_year),
                    "is_mobile": int(is_mobile),
                    "cores": int(shaders),
                    "mhz": float(boost_mhz),
                    "vram_gb": float(vram_gb),
                    "bus_bits": int(bus_bits),
                    "bandwidth_gbps": float(bw_gbps),
                    "n_viewsets": int(len(num_scores)),
                    "composite_score": geomean_score,
                    "max_viewset_score": max_score,
                    "min_viewset_score": min_score,
                    "best_workload": best_workload,
                    "viewset_spread_ratio": max_score / max(geomean_score, 1e-6),
                    "disclosure_url": full_link,
                }
            )
    return rows_out


def parse(bundle: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Parse a bundle of SPEC GWPG HTML pages into a single tidy DataFrame."""
    all_rows: list[dict] = []
    retrieved_at = None
    for suite, item in bundle.items():
        rows = parse_suite_html(suite, item["url"], item["html"])
        retrieved_at = item.get("retrieved_at") or retrieved_at
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise ValueError("No SPEC GWPG GPU rows were parsed from the HTML bundle.")
    if retrieved_at:
        df["retrieved_at"] = retrieved_at
    return df.sort_values(["release_year", "processor", "composite_score"]).reset_index(drop=True)


def transform() -> pd.DataFrame:
    """Read the latest collected SPEC GWPG bundle and write spec_gwpg_results.parquet."""
    manifest = latest(config.RAW_DIR / "gwpg", "*_gwpg_bundle.json")
    bundle = json.loads(manifest.read_text(encoding="utf-8"))
    df = parse(bundle)
    config.GWPG_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.GWPG_TABLE, index=False)
    return df


# --- 3-View Multi-Model GPU Residual Consensus ------------------------------------------------
GPU_PLAIN_FEATURES = [
    "log_cores",
    "log_mhz",
    "log_vram",
    "log_bw",
    "is_mobile",
    "vendor_AMD",
]
GPU_STRUCTURAL_FEATURES = [
    *GPU_PLAIN_FEATURES,
    "log_cores_sq",
    "log_bw_sq",
    "cores_x_bw",
]
GPU_GBDT_FEATURES = [
    "log_cores",
    "log_mhz",
    "log_vram",
    "log_bw",
    "is_mobile",
    "vendor_AMD",
    "bus_bits",
]


def prepare_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log features and cohort-calibrated log score across SPEC GWPG suites.

    Different SPEC GWPG suites (e.g. SPECviewperf 15 FPS geomean vs SPECworkstation 4.0 index
    vs SPECapc application composite) and resolutions (1080p vs 4K) have different baseline
    intercepts and dynamic-range elasticities. Each suite-resolution cohort is aligned onto the
    SPECviewperf 2020 v3 FHD/QHD reference scale using the physical hardware throughput index
    (0.35 * log_cores + 0.35 * log_bw + 0.30 * log_mhz - 0.15 * is_mobile).
    """
    out = df.copy()
    out["log_score"] = np.log(out["composite_score"])
    out["log_cores"] = np.log(out["cores"])
    out["log_mhz"] = np.log(out["mhz"])
    out["log_vram"] = np.log(out["vram_gb"])
    out["log_bw"] = np.log(out["bandwidth_gbps"])
    out["vintage_years"] = (out["release_year"] - 2013).astype(float)
    out["vendor_AMD"] = (out["vendor"] == "AMD").astype(float)

    out["log_cores_sq"] = out["log_cores"] ** 2
    out["log_bw_sq"] = out["log_bw"] ** 2
    out["cores_x_bw"] = out["log_cores"] * out["log_bw"]

    out["cohort"] = out["suite_base"] + " | " + out["res_tier"]
    hw_index = (
        0.35 * out["log_cores"]
        + 0.35 * out["log_bw"]
        + 0.30 * out["log_mhz"]
        - 0.15 * out["is_mobile"]
    )
    ref_cohort = "SPECviewperf 2020 v3 | FHD/QHD (1080p/1440p)"
    ref_sub = out[out["cohort"] == ref_cohort]
    if len(ref_sub) >= 4 and float(hw_index.loc[ref_sub.index].std()) > 0.1:
        ref_hw = hw_index.loc[ref_sub.index]
        ref_b1 = float(np.cov(ref_sub["log_score"], ref_hw)[0, 1] / np.var(ref_hw))
        ref_hw_med = float(ref_hw.median())
        ref_y_med = float(ref_sub["log_score"].median())
    else:
        ref_b1 = 0.65
        ref_hw_med = float(hw_index.median())
        ref_y_med = float(out["log_score"].median())

    adj_scores = pd.Series(index=out.index, dtype=float)
    for _, g in out.groupby("cohort"):
        hw_g = hw_index.loc[g.index]
        y_g = g["log_score"]
        y_med = float(y_g.median())
        hw_med = float(hw_g.median())
        if len(g) >= 6 and float(hw_g.std()) > 0.15:
            b1 = float(np.cov(y_g, hw_g)[0, 1] / np.var(hw_g))
            b1 = float(np.clip(b1, 0.25, 1.5))
        else:
            b1 = ref_b1
        y_expected_on_ref = ref_y_med + ref_b1 * (hw_g - ref_hw_med)
        within_resid = (y_g - (y_med + b1 * (hw_g - hw_med))) * min(ref_b1 / b1, 1.25)
        adj_scores.loc[g.index] = y_expected_on_ref + within_resid

    out["log_score_adj"] = adj_scores
    out["adj_score"] = np.exp(out["log_score_adj"])
    return out


def _classify_gpu_consensus(
    ols_pct: float,
    struct_pct: float,
    gbdt_pct: float,
    shaders: float,
    bw_gbps: float,
) -> str:
    if ols_pct >= 12.0 and struct_pct >= 8.0 and gbdt_pct >= 5.0:
        return "True architectural standout"
    if ols_pct <= -12.0 and struct_pct <= -8.0 and gbdt_pct <= -5.0:
        return "Memory / scaling bottleneck"
    if abs(ols_pct - struct_pct) >= 12.0 or (
        shaders <= 1024 and abs(ols_pct) >= 12.0 and abs(struct_pct) < abs(ols_pct)
    ):
        return "Entry-shader linearity artifact"
    if ols_pct >= 10.0 and struct_pct >= 5.0:
        return "True architectural standout"
    if ols_pct <= -10.0 and struct_pct <= -5.0:
        return "Memory / scaling bottleneck"
    return "Near expected spec curve"


def out_of_fold_gpu_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-fold GroupKFold out-of-fold residuals for OLS, Structural L1, and GBDT."""
    prepared = prepare_gpu(df) if "log_score_adj" not in df.columns else df.copy()
    n_groups = prepared["processor"].nunique()
    n_splits = min(config.N_RESIDUAL_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof_ols = pd.Series(index=prepared.index, dtype=float)
    oof_struct = pd.Series(index=prepared.index, dtype=float)
    oof_gbdt = pd.Series(index=prepared.index, dtype=float)

    for train_idx, test_idx in gkf.split(prepared, groups=prepared["processor"]):
        tr = prepared.iloc[train_idx]
        te = prepared.iloc[test_idx]

        # 1. Plain OLS
        X_tr_ols = sm.add_constant(tr[GPU_PLAIN_FEATURES].astype(float), has_constant="add")
        X_te_ols = sm.add_constant(te[GPU_PLAIN_FEATURES].astype(float), has_constant="add")
        ols_res = sm.OLS(tr["log_score_adj"].astype(float), X_tr_ols).fit()
        oof_ols.iloc[test_idx] = ols_res.predict(X_te_ols).to_numpy()

        # 2. Structural Quadratic L1 (QuantReg median)
        X_tr_st = sm.add_constant(tr[GPU_STRUCTURAL_FEATURES].astype(float), has_constant="add")
        X_te_st = sm.add_constant(te[GPU_STRUCTURAL_FEATURES].astype(float), has_constant="add")
        try:
            st_res = sm.QuantReg(tr["log_score_adj"].astype(float), X_tr_st).fit(
                q=0.5, max_iter=1500
            )
            oof_struct.iloc[test_idx] = st_res.predict(X_te_st).to_numpy()
        except Exception:
            st_res = sm.OLS(tr["log_score_adj"].astype(float), X_tr_st).fit()
            oof_struct.iloc[test_idx] = st_res.predict(X_te_st).to_numpy()

        # 3. Nonlinear L1 GBDT
        gbdt = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=4,
            max_iter=120,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=0,
        )
        gbdt.fit(tr[GPU_GBDT_FEATURES], tr["log_score_adj"])
        oof_gbdt.iloc[test_idx] = gbdt.predict(te[GPU_GBDT_FEATURES])

    out = prepared.copy()
    out["predicted_log_score"] = oof_ols
    out["residual_log"] = out["log_score_adj"] - oof_ols
    out["residual_pct"] = 100.0 * (np.exp(out["residual_log"]) - 1.0)

    out["structural_residual_log"] = out["log_score_adj"] - oof_struct
    out["structural_residual_pct"] = 100.0 * (np.exp(out["structural_residual_log"]) - 1.0)

    out["gbdt_residual_log"] = out["log_score_adj"] - oof_gbdt
    out["gbdt_residual_pct"] = 100.0 * (np.exp(out["gbdt_residual_log"]) - 1.0)
    return out


def _suite_family(suite: str) -> str:
    if "viewperf 15" in suite.lower():
        return "SPECviewperf 15 (DX12 / Vulkan / OpenGL)"
    if "viewperf 2020" in suite.lower():
        return "SPECviewperf 2020 (OpenGL / DX12)"
    if "viewperf 13" in suite.lower() or "viewperf 12" in suite.lower():
        return "SPECviewperf 12.1 / 13 (Legacy OpenGL)"
    if "workstation" in suite.lower():
        return "SPECworkstation (Full System + GPU)"
    if "specapc" in suite.lower():
        return "SPECapc (Maya / Solidworks / Creo / 3ds Max)"
    return "Other SPEC GWPG"


def gpu_residual_ranking(
    with_residuals: pd.DataFrame,
    min_results: int = config.GWPG_MIN_RESULTS_FOR_RANK,
) -> pd.DataFrame:
    """Per-GPU multi-model residual consensus and OEM workstation tuning decomposition."""
    rows: list[dict] = []
    for proc, g in with_residuals.groupby("processor", sort=False):
        if len(g) < min_results:
            continue
        med_res = float(g["residual_log"].median())
        typical = g.iloc[(g["residual_log"] - med_res).abs().argsort().iloc[0]]

        ols_pct = float(100.0 * (np.exp(med_res) - 1.0))
        struct_med = float(g["structural_residual_log"].median())
        struct_pct = float(100.0 * (np.exp(struct_med) - 1.0))
        gbdt_med = float(g["gbdt_residual_log"].median())
        gbdt_pct = float(100.0 * (np.exp(gbdt_med) - 1.0))

        shaders = float(g["cores"].median())
        bw_gbps = float(g["bandwidth_gbps"].median())
        vram_gb = float(g["vram_gb"].median())
        med_score = float(g["adj_score"].median())

        # Same-GPU spread across OEM workstations / suites
        res_pct_min = float(g["residual_pct"].min())
        res_pct_max = float(g["residual_pct"].max())
        tuning_spread_pct = float(res_pct_max - res_pct_min)

        # Viewset specialization headroom: best viewset score vs composite geomean (%)
        peak_gain_pct = float(100.0 * (g["viewset_spread_ratio"].median() - 1.0))

        suite_label = str(typical["suite"])
        bus_w = int(typical["bus_bits"])
        mem_desc = f"{vram_gb:.0f} GB VRAM · {bw_gbps:.0f} GB/s ({bus_w}-bit)"
        rows.append(
            {
                "processor": proc,
                "vendor": typical["vendor"],
                "family": typical["family"],
                "first_avail": f"{int(typical['release_year'])}-01-01",
                "release_year": int(typical["release_year"]),
                "n_results": int(len(g)),
                "n_oems": int(g["oem"].nunique()),
                "cores_per_chip": shaders,
                "mhz": float(g["mhz"].median()),
                "vram_gb": vram_gb,
                "bus_bits": bus_w,
                "bandwidth_gbps": bw_gbps,
                "l3_mb_per_core": bw_gbps,
                "median_score": med_score,
                "score_per_core": float(med_score / max(shaders / 1000.0, 0.1)),
                "predicted_score": float(med_score / np.exp(med_res)),
                "residual_log": med_res,
                "residual_pct": ols_pct,
                "structural_residual_pct": struct_pct,
                "gbdt_residual_pct": gbdt_pct,
                "consensus_verdict": _classify_gpu_consensus(
                    ols_pct, struct_pct, gbdt_pct, shaders, bw_gbps
                ),
                "residual_pct_min": res_pct_min,
                "residual_pct_max": res_pct_max,
                "tuning_spread_pct": tuning_spread_pct,
                "peak_gain_pct": peak_gain_pct,
                "compiler_family": _suite_family(suite_label),
                "config_chips": 1,
                "config_cores": int(shaders),
                "config_threads_per_core": int(typical["n_viewsets"]),
                "config_memory": mem_desc,
                "config_compiler": f"{suite_label} ({typical['res_tier']})",
                "config_os": f"{typical['os_family']} · Peak: {typical['best_workload']}",
                "config_system": f"{typical['oem']} {typical['system']}".strip(),
                "config_test_date": str(int(typical["release_year"])),
                "disclosure_url": typical["disclosure_url"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("residual_log", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    path = collect()
    print(f"collected {path}")
    print(f"tidy GWPG rows: {len(transform())}")
