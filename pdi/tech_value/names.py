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
_MOBILE = re.compile(r"\b(laptop|mobile|max-?q|notebook)\b", re.I)

_RYZEN_TR = re.compile(r"\bthreadripper\s+(pro\s+)?(\d{4}(?:wx|x))\b")
_RYZEN = re.compile(r"\bryzen\s+([3579])\s+(?:pro\s+)?(\d{4}[a-z]?(?:3d)?[a-z]?)\b")
_CORE_ULTRA = re.compile(r"\bcore\s+ultra\s+([3579])\s+(?:processor\s+)?(\d{3}[a-z]{0,2})\b")
_CORE_I = re.compile(r"\b(i[3579])[-\s](\d{4,5}[a-z]{0,2})\b")

_GEFORCE = re.compile(r"\b(rtx|gtx)\s*(\d{3,4})(\s*ti)?(\s*super)?\b")
_RADEON = re.compile(r"\brx\s*(\d{4})(?:\s*(xtx|xt|gre)\b)?")
_ARC = re.compile(r"\barc\s+(?:pro\s+)?([ab]\d{3})\b")


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
    if m := _GEFORCE.search(text):
        ti = " ti" if m.group(3) else ""
        sup = " super" if m.group(4) else ""
        return f"nvidia {m.group(1)} {m.group(2)}{ti}{sup}"
    if m := _RADEON.search(text):
        suffix = f" {m.group(2)}" if m.group(2) else ""
        return f"amd rx {m.group(1)}{suffix}"
    if m := _ARC.search(text):
        return f"intel arc {m.group(1)}"
    return None
