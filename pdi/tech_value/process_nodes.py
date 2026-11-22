"""Architecture and manufacturing process for every desktop CPU and GPU the efficiency page ranks.

Read from vendor product pages and launch material. The node number is the foundry's marketing
name, so it is a rough density class, not a measurement: Intel 7 is counted as 7 nm and TSMC 4N
as 5 nm, the classes they are usually compared with. AMD chiplet CPUs are listed by the node of
their compute dies.
"""

from __future__ import annotations

import re

# GPU architecture -> (process, nm class). A card on a different die from the rest of its
# architecture is listed in _GPU_DIE_OVERRIDES.
GPU_ARCH_NODE: dict[str, tuple[str, int]] = {
    "Kepler": ("TSMC 28 nm", 28),
    "Maxwell": ("TSMC 28 nm", 28),
    "Pascal": ("TSMC 16 nm", 16),
    "Turing": ("TSMC 12 nm", 12),
    "Ampere": ("Samsung 8 nm", 8),
    "Ada Lovelace": ("TSMC 4N", 5),
    "Blackwell": ("TSMC 4N", 5),
    "GCN 1": ("TSMC 28 nm", 28),
    "GCN 2": ("TSMC 28 nm", 28),
    "GCN 3": ("TSMC 28 nm", 28),
    "Polaris": ("GlobalFoundries 14 nm", 14),
    "Vega": ("GlobalFoundries 14 nm", 14),
    "RDNA 1": ("TSMC N7", 7),
    "RDNA 2": ("TSMC N7", 7),
    "RDNA 3": ("TSMC N5", 5),
    "RDNA 3.5": ("TSMC N4", 4),
    "RDNA 4": ("TSMC N4P", 4),
    "Alchemist": ("TSMC N6", 6),
    "Battlemage": ("TSMC N5", 5),
}
# Navi 33 is RDNA 3 on N6; GP107 and GP108 are Pascal on Samsung 14 nm.
_GPU_DIE_OVERRIDES = {
    re.compile(r"rx 7600|rx 7650|pro w7600|pro w7500"): ("TSMC N6", 6),
    re.compile(r"gtx 1050|quadro p1000|quadro p620|quadro p600|gt 1030"): ("Samsung 14 nm", 14),
}


def gpu_node(name: str, arch: str) -> tuple[str, int] | None:
    low = name.lower()
    for pattern, node in _GPU_DIE_OVERRIDES.items():
        if pattern.search(low):
            return node
    return GPU_ARCH_NODE.get(arch)


# CPU: (architecture, process, nm class) from the model number.
_ZEN = ("Zen", "GlobalFoundries 14 nm", 14)
_ZEN_PLUS = ("Zen+", "GlobalFoundries 12 nm", 12)
_ZEN2 = ("Zen 2", "TSMC N7", 7)
_ZEN3 = ("Zen 3", "TSMC N7", 7)
_ZEN4 = ("Zen 4", "TSMC N5", 5)
_ZEN4_APU = ("Zen 4 (Phoenix)", "TSMC N4", 4)
_ZEN5 = ("Zen 5", "TSMC N4P", 4)
_RYZEN = {"1": _ZEN, "2": _ZEN_PLUS, "3": _ZEN2, "4": _ZEN2, "5": _ZEN3, "7": _ZEN4,
          "8": _ZEN4_APU, "9": _ZEN5}  # fmt: skip
_THREADRIPPER = {"1": _ZEN, "2": _ZEN_PLUS, "3": _ZEN2, "5": _ZEN3, "7": _ZEN4, "9": _ZEN5}
_INTEL_4DIGIT = {
    "2": ("Sandy Bridge", "Intel 32 nm", 32),
    "3": ("Ivy Bridge", "Intel 22 nm", 22),
    "4": ("Haswell", "Intel 22 nm", 22),
    "5": ("Haswell-E", "Intel 22 nm", 22),
    "6": ("Skylake", "Intel 14 nm", 14),
    "7": ("Kaby Lake", "Intel 14 nm", 14),
    "8": ("Coffee Lake", "Intel 14 nm", 14),
    "9": ("Coffee Lake Refresh", "Intel 14 nm", 14),
}
_INTEL_5DIGIT = {
    "10": ("Comet Lake", "Intel 14 nm", 14),
    "11": ("Rocket Lake", "Intel 14 nm", 14),
    "12": ("Alder Lake", "Intel 7", 7),
    "13": ("Raptor Lake", "Intel 7", 7),
    "14": ("Raptor Lake Refresh", "Intel 7", 7),
}


def cpu_node(key: str) -> tuple[str, str, int] | None:
    """'amd ryzen 5 3600' -> ('Zen 2', 'TSMC N7', 7). None when the key is not recognised."""
    if m := re.match(r"amd threadripper (?:pro )?(\d)\d{3}", key):
        return _THREADRIPPER.get(m.group(1))
    if m := re.match(r"amd ryzen \d (\d)\d{3}(\w*)$", key):
        gen, suffix = m.group(1), m.group(2)
        if gen == "2" and suffix == "g":
            return _ZEN  # Raven Ridge
        if gen == "3" and suffix == "g":
            return _ZEN_PLUS  # Picasso
        return _RYZEN.get(gen)
    if key.startswith("intel core ultra"):
        return ("Arrow Lake", "TSMC N3B", 3)
    if m := re.match(r"intel core i\d-(\d{5})", key):
        if key.endswith("x") or key.endswith("xe"):
            return ("Cascade Lake-X", "Intel 14 nm", 14)
        return _INTEL_5DIGIT.get(m.group(1)[:2])
    if m := re.match(r"intel core i\d-(\d)\d{3}", key):
        gen = m.group(1)
        if gen == "6" and key.endswith("x"):
            return ("Broadwell-E", "Intel 14 nm", 14)
        if gen == "7" and key.endswith("x"):
            return ("Skylake-X", "Intel 14 nm", 14)
        return _INTEL_4DIGIT.get(gen)
    return None
