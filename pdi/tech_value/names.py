"""Turn free-text product and device names into one comparable model key.

"13th Gen Intel(R) Core(TM) i9-13900K" (a benchmark device) and "Intel - Core i9-13900K 13th Gen
24-Core Desktop Processor" (a retail listing) both become ("cpu", "intel core i9-13900k").
Names that are not a desktop CPU or a discrete desktop GPU return None, on purpose: an
unmatched product is shown as unmatched, never guessed.
"""

from __future__ import annotations

import re

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
_RADEON = re.compile(r"\brx\s*(\d{4})(?:\s*(xtx|xt|gre)\b)?")
_ARC_PRO = re.compile(r"\barc\s+pro\s+([ab]\d{2,3})\b")
_ARC = re.compile(r"\barc\s+(a310|a380|a580|a750|a770|b570|b580)\b")

# Generations still sold new at US retail. Older parts remain eligible to be matched whenever a
# store lists them, but are not actively queried one by one.
_ACTIVE_CPU = re.compile(
    r"^(?:amd ryzen [3579] [5789]\d{3}|amd threadripper [79]\d{3}|"
    r"intel core i[3579]-1[234]\d{3}|intel core ultra [3579] 2\d{2})"
)
_ACTIVE_GPU = re.compile(
    r"^(?:nvidia rtx (?:50[56789]0|40[6789]0)(?:\s+ti)?(?:\s+super)?$|"
    r"nvidia rtx 30[56]0$|"
    r"nvidia rtx (?:pro \d{4} blackwell|\d{4} ada|a\d{3,4})$|"
    r"amd rx (?:90\d{2}|7\d{3}|6[67]00)(?:\s+(?:xtx|xt|gre))?$|"
    r"amd radeon ai pro r\d{4}$|"
    r"intel arc (?:pro [ab]\d{2,3}|a380|a580|a750|a770|b570|b580)$)"
)


def retail_active(kind: str, key: str) -> bool:
    """True when a desktop CPU or discrete GPU belongs to a generation still sold new."""
    if kind == "gpu" and key == "amd rx 7650 gre":
        return False
    pat = _ACTIVE_CPU if kind == "cpu" else _ACTIVE_GPU if kind == "gpu" else None
    return bool(pat and pat.search(key))


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
