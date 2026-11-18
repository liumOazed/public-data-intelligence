import pytest

from pdi.tech_value.names import model_key, retail_active


@pytest.mark.parametrize(
    "name, expected",
    [
        ("AMD Ryzen 9 7950X 16-Core Processor", ("cpu", "amd ryzen 9 7950x")),
        (
            "AMD - Ryzen 9 7950X 16-Core - 32-Thread 4.5 GHz Socket AM5",
            ("cpu", "amd ryzen 9 7950x"),
        ),
        ("AMD Ryzen 7 9800X3D 8-Core Processor", ("cpu", "amd ryzen 7 9800x3d")),
        ("AMD Ryzen 5 5600 6-Core Processor", ("cpu", "amd ryzen 5 5600")),
        ("13th Gen Intel(R) Core(TM) i9-13900K", ("cpu", "intel core i9-13900k")),
        (
            "Intel - Core i9-13900K 13th Gen 24-Core Desktop Processor",
            ("cpu", "intel core i9-13900k"),
        ),
        ("Intel(R) Core(TM) Ultra 9 285K", ("cpu", "intel core ultra 9 285k")),
        (
            "Intel - Core Ultra 9 Processor 285K Desktop Processor",
            ("cpu", "intel core ultra 9 285k"),
        ),
        ("AMD Ryzen Threadripper PRO 7995WX 96-Cores", ("cpu", "amd threadripper pro 7995wx")),
        ("NVIDIA GeForce RTX 4070 Ti SUPER", ("gpu", "nvidia rtx 4070 ti super")),
        (
            "ASUS - TUF Gaming NVIDIA GeForce RTX 4070 Ti SUPER 16GB GDDR6X",
            ("gpu", "nvidia rtx 4070 ti super"),
        ),
        ("NVIDIA GeForce RTX 4070", ("gpu", "nvidia rtx 4070")),
        ("MSI - GeForce RTX 5070 Ti 16G VENTUS 3X OC", ("gpu", "nvidia rtx 5070 ti")),
        ("AMD Radeon RX 7900 XTX", ("gpu", "amd rx 7900 xtx")),
        ("XFX - SPEEDSTER AMD Radeon RX 7900 XT 20GB", ("gpu", "amd rx 7900 xt")),
        ("Intel(R) Arc(TM) B580 Graphics", ("gpu", "intel arc b580")),
    ],
)
def test_recognised(name, expected):
    assert model_key(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "NVIDIA GeForce RTX 4070 Laptop GPU",
        "Thermal Grizzly - Kryonaut Thermal Paste",
        "Apple M3 Max",
        # A whole PC names both a CPU and a GPU; it is not a part.
        "CyberPowerPC - Gamer Desktop - Intel Core i7-14700F - NVIDIA GeForce RTX 4060 Ti",
    ],
)
def test_not_recognised(name):
    assert model_key(name) is None


def test_ti_and_super_are_distinct_models():
    keys = {
        model_key(n) for n in ("RTX 4070", "RTX 4070 SUPER", "RTX 4070 Ti", "RTX 4070 Ti SUPER")
    }
    assert len(keys) == 4


def test_laptop_chips_are_not_desktop_parts():
    assert model_key("13th Gen Intel(R) Core(TM) i9-13900HX") is None
    assert model_key("Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz") is None
    assert model_key("Intel(R) Core(TM) Ultra 7 258V") is None
    assert model_key("AMD Ryzen 7 7840HS w/ Radeon 780M Graphics") is None
    assert model_key("AMD Ryzen 7 5800X3D") == ("cpu", "amd ryzen 7 5800x3d")
    assert model_key("Intel Core i5-12400F") == ("cpu", "intel core i5-12400f")


def test_retail_active_filters_discontinued_generations():
    assert retail_active("gpu", "nvidia rtx 5070 ti")
    assert retail_active("gpu", "nvidia rtx 4070 super")
    assert retail_active("gpu", "amd rx 9070 xt")
    assert retail_active("gpu", "amd rx 7800 xt")
    assert retail_active("gpu", "intel arc b580")
    assert not retail_active("gpu", "nvidia gtx 1080")
    assert not retail_active("gpu", "nvidia rtx 2080 ti")
    assert not retail_active("gpu", "amd rx 6800 xt")
    assert retail_active("cpu", "amd ryzen 7 9800x3d")
    assert retail_active("cpu", "amd ryzen 5 5600x")
    assert retail_active("cpu", "intel core i7-14700k")
    assert retail_active("cpu", "intel core ultra 9 285k")
    assert not retail_active("cpu", "amd ryzen 5 3600")
    assert not retail_active("cpu", "intel core i7-8700k")
    assert not retail_active("cpu", "amd threadripper pro 7995wx")
    assert retail_active("gpu", "nvidia rtx pro 6000 blackwell")
    assert retail_active("gpu", "nvidia rtx 4000 ada")
    assert retail_active("gpu", "nvidia rtx a4000")
    assert retail_active("gpu", "intel arc pro b50")


def test_workstation_gpus_are_distinct_from_consumer_and_quadro():
    assert model_key("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == (
        "gpu",
        "nvidia rtx pro 6000 blackwell",
    )
    assert model_key("NVIDIA RTX 4000 SFF Ada Generation") == ("gpu", "nvidia rtx 4000 ada")
    assert model_key("NVIDIA RTX A4000") == ("gpu", "nvidia rtx a4000")
    assert model_key("Intel(R) Arc(TM) Pro B50 Graphics") == ("gpu", "intel arc pro b50")
    assert model_key("AMD Radeon AI PRO R9700") == ("gpu", "amd radeon ai pro r9700")
    # Legacy Quadro, mobile Radeon M/S, and integrated laptop Arc chips do not collide with desktop GPUs.
    assert model_key("Quadro RTX 4000") is None
    assert model_key("Intel(R) Arc(TM) B390 GPU") is None
    assert model_key("NVIDIA RTX PRO 6000 Blackwell Server Edition") is None
    assert model_key("ONEXGPU 2 Egpu Dock with AMD Radeon RX 7800M, 12GB") is None
    assert model_key("AMD Radeon RX 7600S") is None
    assert model_key("XFX Speedster SWFT210 Radeon RX 7700XT Gaming Graphics Card") == (
        "gpu",
        "amd rx 7700 xt",
    )


def test_dynamic_generation_window_shifts_when_new_gen_launches():
    from pdi.tech_value.names import active_keys, newest_keys

    # Simulate a future benchmark snapshot where RTX 60-series, RX 10000, Ryzen 10000,
    # Core Ultra 300, and Arc C-series have appeared.
    future_models = [
        ("gpu", "nvidia rtx 6080"),
        ("gpu", "nvidia rtx 5080"),
        ("gpu", "nvidia rtx 4060"),
        ("gpu", "nvidia rtx 3060"),
        ("gpu", "amd rx 10070 xt"),
        ("gpu", "amd rx 9070 xt"),
        ("gpu", "amd rx 7700 xt"),
        ("gpu", "amd rx 6700 xt"),
        ("gpu", "intel arc c770"),
        ("gpu", "intel arc b580"),
        ("gpu", "intel arc a770"),
        ("cpu", "amd ryzen 7 10800x3d"),
        ("cpu", "amd ryzen 7 9800x3d"),
        ("cpu", "amd ryzen 7 8700g"),
        ("cpu", "amd ryzen 7 7800x3d"),
        ("cpu", "amd ryzen 7 5800x3d"),
        ("cpu", "intel core ultra 9 385k"),
        ("cpu", "intel core ultra 9 285k"),
        ("cpu", "intel core i9-14900k"),
        ("cpu", "intel core i9-13900k"),
        ("cpu", "intel core i9-12900k"),
    ]
    active = active_keys(future_models)
    newest = newest_keys(future_models)

    # Newest generations are automatically discovered.
    assert ("gpu", "nvidia rtx 6080") in newest
    assert ("gpu", "amd rx 10070 xt") in newest
    assert ("gpu", "intel arc c770") in newest
    assert ("cpu", "amd ryzen 7 10800x3d") in newest
    assert ("cpu", "intel core ultra 9 385k") in newest

    # And the oldest generations in each product line automatically age out.
    assert ("gpu", "nvidia rtx 4060") in active
    assert ("gpu", "nvidia rtx 3060") not in active
    assert ("gpu", "amd rx 7700 xt") in active
    assert ("gpu", "amd rx 6700 xt") not in active
    assert ("gpu", "intel arc b580") in active
    assert ("gpu", "intel arc a770") not in active
    assert ("cpu", "amd ryzen 7 7800x3d") in active
    assert ("cpu", "amd ryzen 7 5800x3d") not in active
    assert ("cpu", "intel core i9-13900k") in active
    assert ("cpu", "intel core i9-12900k") not in active
