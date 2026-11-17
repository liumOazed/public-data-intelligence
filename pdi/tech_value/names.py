"""Turn free-text product and device names into one comparable model key.

"13th Gen Intel(R) Core(TM) i9-13900K" (a benchmark device) and "Intel - Core i9-13900K 13th Gen
24-Core Desktop Processor" (a retail listing) both become ("cpu", "intel core i9-13900k").
Names that are not a desktop CPU or a discrete desktop GPU return None, on purpose: an
unmatched product is shown as unmatched, never guessed.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Mapping, Sequence

from pdi.tech_value import config

_NOISE = re.compile(r"\((?:r|tm|c)\)|[®™©]", re.I)
_SPACES = re.compile(r"\s+")
_MOBILE = re.compile(r"\b(laptop|mobile|max-?q|notebook|server\s+edition|dc-\d+q)\b", re.I)

_RYZEN_TR = re.compile(r"\bthreadripper\s+(pro\s+)?(\d{4}(?:wx|x))\b")
_RYZEN = re.compile(r"\bryzen\s+([3579])\s+(?:pro\s+)?(\d{4}[a-z]?(?:3d)?[a-z]?)\b")
_CORE_ULTRA = re.compile(r"\bcore\s+ultra\s+([3579])\s+(?:processor\s+)?(\d{3}[a-z]{0,2})\b")
_CORE_I = re.compile(r"\b(i[3579])[-\s](\d{4,5}[a-z]{0,2})\b")

# Laptop chips: Core i ... H/HX/HK/HS/HQ/U/P/Y, Core Ultra ... V, and Ryzen ... H/HS/HX/U.
_MOBILE_CPU = re.compile(r"\d(?:h[xkqs]?|u|p|y|v)$")

_RTX_PRO = re.compile(r"\brtx\s+pro\s+(\d{4})(?:\s+blackwell)?\b")
_RTX_ADA = re.compile(r"\brtx\s+(\d{4})\s+(?:sff\s+)?ada\b")
_RTX_AMP = re.compile(r"\brtx\s+(a\d{3,4})\b")
_GEFORCE = re.compile(r"\b(?<!quadro\s)(rtx|gtx)\s*(\d{3,4})(\s*ti)?(\s*super)?\b")
_RADEON_PRO = re.compile(r"\bradeon\s+ai\s+pro\s+(r\d{4})\b")
_RADEON = re.compile(r"\brx\s*(\d{4})(?:\s*(xtx|xt|gre))?\b")
_ARC_PRO = re.compile(r"\barc\s+pro\s+([a-d]\d{2,3})\b")
_ARC = re.compile(r"\barc\s+(a310|a380|[a-d][579]\d0)\b")

# Fallback generation timelines used only when BLENDER_TABLE has not been built yet.
_FALLBACK_GENS: dict[str, list[int]] = {
    "amd_ryzen": [9, 8, 7, 5],
    "amd_threadripper": [9, 7],
    "intel_core": [15, 14, 13, 12],
    "nvidia_geforce": [50, 40, 30],
    "nvidia_ws": [3, 2, 1],
    "amd_radeon": [9, 7, 6],
    "amd_ws": [9],
    "intel_arc": [2, 1],
    "intel_arc_pro": [2, 1],
}


def product_line(kind: str, key: str) -> tuple[str, int, int] | None:
    """Extract (product_line, generation_rank, tier_number) from a normalised model_key."""
    if kind == "cpu":
        if m := re.match(r"^amd ryzen [3579] (\d{4,5})", key):
            num = int(m.group(1))
            return ("amd_ryzen", num // 1000, num % 1000)
        if m := re.match(r"^amd threadripper (\d{4,5})", key):
            num = int(m.group(1))
            return ("amd_threadripper", num // 1000, num % 1000)
        if m := re.match(r"^intel core ultra [3579] (\d{3,4})", key):
            num = int(m.group(1))
            return ("intel_core", 13 + num // 100, num % 100)
        if m := re.match(r"^intel core i[3579]-(\d{4,5})", key):
            num = int(m.group(1))
            return ("intel_core", num // 1000, num % 1000)
        return None
    if kind == "gpu":
        if m := re.match(r"^nvidia rtx pro (\d{4}) blackwell$", key):
            return ("nvidia_ws", 3, int(m.group(1)))
        if m := re.match(r"^nvidia rtx (\d{4}) ada$", key):
            return ("nvidia_ws", 2, int(m.group(1)))
        if m := re.match(r"^nvidia rtx a(\d{3,4})$", key):
            return ("nvidia_ws", 1, int(m.group(1)))
        if m := re.match(r"^nvidia (?:rtx|gtx) (\d{3,4})", key):
            num = int(m.group(1))
            return ("nvidia_geforce", num // 100, num % 100)
        if m := re.match(r"^amd radeon ai pro r(\d{4,5})", key):
            num = int(m.group(1))
            return ("amd_ws", num // 1000, num % 1000)
        if m := re.match(r"^amd rx (\d{4,5})", key):
            num = int(m.group(1))
            return ("amd_radeon", num // 1000, num % 1000)
        if m := re.match(r"^intel arc pro ([a-z])(\d{2,3})$", key):
            return ("intel_arc_pro", ord(m.group(1)) - ord("a") + 1, int(m.group(2)))
        if m := re.match(r"^intel arc ([a-z])(\d{3})$", key):
            if m.group(1) == "a" and int(m.group(2)) < 380:
                return None
            return ("intel_arc", ord(m.group(1)) - ord("a") + 1, int(m.group(2)))
    return None


def active_generations(models: Iterable[tuple[str, str]]) -> dict[str, list[int]]:
    """Distinct generations per product line in `models`, sorted newest-first."""
    by_line: dict[str, set[int]] = {}
    for kind, key in models:
        parsed = product_line(kind, key)
        if parsed:
            by_line.setdefault(parsed[0], set()).add(parsed[1])
    return {line: sorted(gens, reverse=True) for line, gens in by_line.items()}


def is_active_in_window(
    kind: str,
    key: str,
    line_gens: Mapping[str, Sequence[int]],
) -> bool:
    """True when (kind, key) belongs to the active generation window in `line_gens`."""
    parsed = product_line(kind, key)
    if not parsed:
        return False
    line, gen, tier = parsed
    gens = line_gens.get(line) or _FALLBACK_GENS.get(line, [])
    if not gens:
        return True
    if line in ("amd_ryzen", "intel_core"):
        return gen >= gens[min(3, len(gens) - 1)]
    if line == "amd_threadripper":
        return gen >= gens[min(1, len(gens) - 1)]
    if line == "nvidia_geforce":
        if gen >= gens[0]:
            return tier >= 50
        if len(gens) > 1 and gen == gens[1]:
            return tier >= 60
        if len(gens) > 2 and gen == gens[2]:
            return tier in (50, 60) and " ti" not in key and " super" not in key
        return False
    if line == "nvidia_ws":
        return gen >= gens[min(2, len(gens) - 1)]
    if line == "amd_radeon":
        if key == "amd rx 7650 gre":
            return False
        if gen >= gens[min(1, len(gens) - 1)]:
            return True
        if len(gens) > 2 and gen == gens[2]:
            return 600 <= tier <= 750
        return False
    return gen >= gens[min(1, len(gens) - 1)]


def active_keys(models: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """Dynamically select all (kind, key) in `models` belonging to current retail generations."""
    items = list(models)
    line_gens = active_generations(items)
    return {(kind, key) for kind, key in items if is_active_in_window(kind, key, line_gens)}


def newest_keys(models: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """Select (kind, key) in `models` belonging to the #1 newest generation of each product line."""
    items = list(models)
    line_gens = active_generations(items)
    out: set[tuple[str, str]] = set()
    for kind, key in items:
        parsed = product_line(kind, key)
        if not parsed or not is_active_in_window(kind, key, line_gens):
            continue
        line, gen, _ = parsed
        gens = line_gens.get(line)
        if gens and gen >= gens[0]:
            out.add((kind, key))
    return out


@functools.lru_cache(maxsize=1)
def _benchmark_generations() -> dict[str, list[int]]:
    if not config.BLENDER_TABLE.exists():
        return _FALLBACK_GENS
    import pandas as pd

    df = pd.read_parquet(config.BLENDER_TABLE, columns=["device_name"])
    pairs = [k for name in df["device_name"].unique() if (k := model_key(name)) is not None]
    return active_generations(pairs) or _FALLBACK_GENS


def retail_active(
    kind: str,
    key: str,
    reference_models: Iterable[tuple[str, str]] | None = None,
) -> bool:
    """True when a desktop CPU or discrete GPU belongs to a generation still sold new."""
    line_gens = (
        active_generations(reference_models)
        if reference_models is not None
        else _benchmark_generations()
    )
    return is_active_in_window(kind, key, line_gens)


def normalise(name: str) -> str:
    return _SPACES.sub(" ", _NOISE.sub(" ", name)).strip().lower()


def model_key(name: str | None) -> tuple[str, str] | None:
    """Return (kind, key) where kind is "cpu" or "gpu", or None if the name is not recognised."""
    if not name:
        return None
    text = normalise(name)
    if _MOBILE.search(text):
        return None

    # GPUs first: a prebuilt-PC listing mentions both, and those must not match as a bare CPU,
    # but a graphics-card listing never mentions a CPU.
    gpu = _gpu_key(text)
    cpu = _cpu_key(text)
    if gpu and cpu:
        return None  # a whole system, not a part
    if gpu:
        return ("gpu", gpu)
    if cpu and _MOBILE_CPU.search(cpu):
        return None
    if cpu:
        return ("cpu", cpu)
    return None


def _cpu_key(text: str) -> str | None:
    if m := _RYZEN_TR.search(text):
        return f"amd threadripper {'pro ' if m.group(1) else ''}{m.group(2)}"
    if m := _RYZEN.search(text):
        return f"amd ryzen {m.group(1)} {m.group(2)}"
    if m := _CORE_ULTRA.search(text):
        return f"intel core ultra {m.group(1)} {m.group(2)}"
    if "intel" in text or "core" in text:
        if m := _CORE_I.search(text):
            return f"intel core {m.group(1)}-{m.group(2)}"
    return None


def _gpu_key(text: str) -> str | None:
    if m := _RTX_PRO.search(text):
        return f"nvidia rtx pro {m.group(1)} blackwell"
    if m := _RTX_ADA.search(text):
        return f"nvidia rtx {m.group(1)} ada"
    if m := _RTX_AMP.search(text):
        return f"nvidia rtx {m.group(1)}"
    if "quadro" not in text and (m := _GEFORCE.search(text)) and not m.group(2).endswith("000"):
        ti = " ti" if m.group(3) else ""
        sup = " super" if m.group(4) else ""
        return f"nvidia {m.group(1)} {m.group(2)}{ti}{sup}"
    if m := _RADEON_PRO.search(text):
        return f"amd radeon ai pro {m.group(1)}"
    if m := _RADEON.search(text):
        suffix = f" {m.group(2)}" if m.group(2) else ""
        return f"amd rx {m.group(1)}{suffix}"
    if m := _ARC_PRO.search(text):
        return f"intel arc pro {m.group(1)}"
    if m := _ARC.search(text):
        return f"intel arc {m.group(1)}"
    return None
