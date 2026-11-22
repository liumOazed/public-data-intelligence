"""Rated board power of desktop and workstation graphics cards, for the efficiency page.

Blender Open Data carries no power figure, so this table supplies one. Values are the
manufacturers' reference specifications, entered by hand:

  (rated board power W, launch year)

"Rated board power" is NVIDIA's total graphics power (TGP) or graphics card power, AMD's total
board power (TBP) and Intel's TBP, each for the manufacturer's own reference design. Partner
cards often ship with higher limits, and the power a card really draws in a given workload can
differ by 30% or more, so every figure here is shown as "rated", never as measured. Keys are
model keys from ``names.model_key``. A card missing here is not ranked on the efficiency page,
and ``NOT_ENTERED`` says why.
"""

from __future__ import annotations

GpuSpec = tuple[int, int]

GPU_SPECS: dict[str, GpuSpec] = {
    # AMD Radeon RX 5000 (RDNA, 2019-2020)
    "amd rx 5500 xt": (130, 2019),
    "amd rx 5600 xt": (150, 2020),
    "amd rx 5700 xt": (225, 2019),
    # AMD Radeon RX 6000 (RDNA 2, 2020-2022)
    "amd rx 6400": (53, 2022),
    "amd rx 6500 xt": (107, 2022),
    "amd rx 6600": (132, 2021),
    "amd rx 6600 xt": (160, 2021),
    "amd rx 6650 xt": (180, 2022),
    "amd rx 6700 xt": (230, 2021),
    "amd rx 6750 xt": (250, 2022),
    "amd rx 6800": (250, 2020),
    "amd rx 6800 xt": (300, 2020),
    "amd rx 6900 xt": (300, 2020),
    "amd rx 6950 xt": (335, 2022),
    # AMD Radeon RX 7000 (RDNA 3, 2022-2024)
    "amd rx 7600": (165, 2023),
    "amd rx 7600 xt": (190, 2024),
    "amd rx 7700 xt": (245, 2023),
    "amd rx 7800 xt": (263, 2023),
    "amd rx 7900 gre": (260, 2023),
    "amd rx 7900 xt": (315, 2022),
    "amd rx 7900 xtx": (355, 2022),
    # AMD Radeon RX 9000 and Radeon AI PRO (RDNA 4, 2025)
    "amd rx 9060 xt": (160, 2025),
    "amd rx 9070": (220, 2025),
    "amd rx 9070 gre": (220, 2025),
    "amd rx 9070 xt": (304, 2025),
    "amd radeon ai pro r9700": (300, 2025),
    # Intel Arc (Alchemist 2022-2023, Battlemage 2024-2025)
    "intel arc a380": (75, 2022),
    "intel arc a580": (185, 2023),
    "intel arc a750": (225, 2022),
    "intel arc a770": (225, 2022),
    "intel arc b570": (150, 2025),
    "intel arc b580": (190, 2024),
    "intel arc pro b50": (70, 2025),
    "intel arc pro b60": (200, 2025),
    # NVIDIA GeForce GTX 700 and 900 (Maxwell, 2014-2015)
    "nvidia gtx 750 ti": (60, 2014),
    "nvidia gtx 950": (90, 2015),
    "nvidia gtx 960": (120, 2015),
    "nvidia gtx 970": (145, 2014),
    "nvidia gtx 980": (165, 2014),
    "nvidia gtx 980 ti": (250, 2015),
    # NVIDIA GeForce GTX 10 (Pascal, 2016-2017)
    "nvidia gtx 1050": (75, 2016),
    "nvidia gtx 1050 ti": (75, 2016),
    "nvidia gtx 1060": (120, 2016),
    "nvidia gtx 1070": (150, 2016),
    "nvidia gtx 1070 ti": (180, 2017),
    "nvidia gtx 1080": (180, 2016),
    "nvidia gtx 1080 ti": (250, 2017),
    # NVIDIA GeForce GTX 16 (Turing, 2019)
    "nvidia gtx 1650": (75, 2019),
    "nvidia gtx 1650 super": (100, 2019),
    "nvidia gtx 1660": (120, 2019),
    "nvidia gtx 1660 super": (125, 2019),
    "nvidia gtx 1660 ti": (120, 2019),
    # NVIDIA GeForce RTX 20 (Turing, 2018-2019)
    "nvidia rtx 2060": (160, 2019),
    "nvidia rtx 2060 super": (175, 2019),
    "nvidia rtx 2070": (175, 2018),
    "nvidia rtx 2070 super": (215, 2019),
    "nvidia rtx 2080": (215, 2018),
    "nvidia rtx 2080 super": (250, 2019),
    "nvidia rtx 2080 ti": (250, 2018),
    # NVIDIA GeForce RTX 30 (Ampere, 2020-2022)
    "nvidia rtx 3050": (130, 2022),
    "nvidia rtx 3060": (170, 2021),
    "nvidia rtx 3060 ti": (200, 2020),
    "nvidia rtx 3070": (220, 2020),
    "nvidia rtx 3070 ti": (290, 2021),
    "nvidia rtx 3080": (320, 2020),
    "nvidia rtx 3080 ti": (350, 2021),
    "nvidia rtx 3090": (350, 2020),
    "nvidia rtx 3090 ti": (450, 2022),
    # NVIDIA GeForce RTX 40 (Ada Lovelace, 2022-2024)
    "nvidia rtx 4060": (115, 2023),
    "nvidia rtx 4060 ti": (160, 2023),
    "nvidia rtx 4070": (200, 2023),
    "nvidia rtx 4070 super": (220, 2024),
    "nvidia rtx 4070 ti": (285, 2023),
    "nvidia rtx 4070 ti super": (285, 2024),
    "nvidia rtx 4080": (320, 2022),
    "nvidia rtx 4080 super": (320, 2024),
    "nvidia rtx 4090": (450, 2022),
    # NVIDIA GeForce RTX 50 (Blackwell, 2025)
    "nvidia rtx 5050": (130, 2025),
    "nvidia rtx 5060": (145, 2025),
    "nvidia rtx 5060 ti": (180, 2025),
    "nvidia rtx 5070": (250, 2025),
    "nvidia rtx 5070 ti": (300, 2025),
    "nvidia rtx 5080": (360, 2025),
    "nvidia rtx 5090": (575, 2025),
    # NVIDIA workstation, Ampere (2020-2024)
    "nvidia rtx a400": (50, 2024),
    "nvidia rtx a1000": (50, 2024),
    "nvidia rtx a2000": (70, 2021),
    "nvidia rtx a4000": (140, 2021),
    "nvidia rtx a4500": (200, 2021),
    "nvidia rtx a5000": (230, 2021),
    "nvidia rtx a6000": (300, 2020),
    # NVIDIA workstation, Ada Lovelace (2022-2024)
    "nvidia rtx 2000 ada": (70, 2024),
    "nvidia rtx 4000 ada": (130, 2023),
    "nvidia rtx 4500 ada": (210, 2023),
    "nvidia rtx 5000 ada": (250, 2023),
    "nvidia rtx 6000 ada": (300, 2022),
    # NVIDIA workstation, Blackwell (2025)
    "nvidia rtx pro 4000 blackwell": (140, 2025),
    "nvidia rtx pro 4500 blackwell": (200, 2025),
    "nvidia rtx pro 5000 blackwell": (300, 2025),
}

# Benchmarked cards that are deliberately left out, with the reason the page shows.
NOT_ENTERED: dict[str, str] = {
    "nvidia gtx 1650 ti": "Laptop only: its power limit is set by each laptop maker",
    "nvidia rtx 2050": "Laptop only: its power limit is set by each laptop maker",
    "nvidia rtx pro 6000 blackwell": (
        "Sold at 600 W and 300 W (Max-Q) under one name; Blender cannot tell them apart"
    ),
    "amd rx 7650 gre": "No reference board power found on AMD's product pages yet",
}
