from pdi.tech_value import retail


def _listings(fixtures):
    html = (fixtures / "magento_search_sample.html").read_text()
    return retail.parse_magento_search(html, "https://www.example.invalid/search?q=x")


def test_parse_magento_search(fixtures):
    items = _listings(fixtures)
    assert [x.sku for x in items] == ["111", "222", "333", "444"]
    first = items[0]
    assert first.name == "SAMPLE Brand GeForce RTX 4070 Ti SUPER 16GB Video Card"
    assert first.price == 799.99 and first.in_stock
    # Relative links are resolved, and the out-of-stock marker is read.
    assert items[1].url == "https://www.example.invalid/sample-rtx-4070.html"
    assert not items[1].in_stock


def test_only_the_exact_part_is_kept(fixtures):
    items = _listings(fixtures)
    kept = [x.sku for x in items if retail.keep(x, "gpu", "nvidia rtx 4070 ti super")]
    # Not the plain 4070, not the bracket, not the whole PC.
    assert kept == ["111"]


def test_search_term_drops_the_vendor():
    assert retail.search_term("amd ryzen 7 9800x3d") == "ryzen 7 9800x3d"
    assert retail.search_term("nvidia rtx 4070 ti super") == "rtx 4070 ti super"


def test_no_robots_means_no_crawling():
    class Dead:
        def get(self, *a, **k):
            raise retail.requests.ConnectionError

    assert not retail.Robots(Dead()).allowed("https://www.example.invalid/search?q=x")


def test_whole_computers_are_not_parts():
    # Both were kept as CPU listings on the first live run.
    mini = retail.Listing(
        sku="a",
        name="MINISFORUM Mini PC NAB9S Intel Core i9-13900HX 14 Cores 32GB RAM DDR4 1TB PCIe4.0 SSD Mini Desktop Computer",
        price=799.99,
        in_stock=True,
        url="https://www.example.invalid/a",
    )
    tower = retail.Listing(
        sku="b",
        name="HP Z2 G1i Workstation - Intel Core Ultra 7 265K - 32 GB - 1 TB SSD - Small Form Factor - Windows 11 Pro",
        price=3818.99,
        in_stock=True,
        url="https://www.example.invalid/b",
    )
    part = retail.Listing(
        sku="c",
        name="Intel Core Ultra 7 265K Desktop Processor 20 Cores 5.5GHz Turbo 125W TDP Boxed",
        price=309.99,
        in_stock=True,
        url="https://www.example.invalid/c",
    )
    assert not retail.keep(mini, "cpu", "intel core i9-13900hx")
    assert not retail.keep(tower, "cpu", "intel core ultra 7 265k")
    assert retail.keep(part, "cpu", "intel core ultra 7 265k")


def test_gaming_desktop_processor_and_gpu_metal_backplate_are_kept():
    intel_box = retail.Listing(
        sku="i7",
        name="Intel Core i7-14700K Gaming Desktop Processor 20 cores (8 P-cores + 12 E-cores)",
        price=328.95,
        in_stock=True,
        url="https://www.example.invalid/i7",
    )
    gpu_backplate = retail.Listing(
        sku="rx",
        name="ASRock Radeon RX 9060 XT Challenger 8GB GDDR6 PCI Express 5.0 x16 Graphics Card | Metal Backplate",
        price=349.99,
        in_stock=True,
        url="https://www.example.invalid/rx",
    )
    waterblock = retail.Listing(
        sku="wb",
        name="Alphacool Core Geforce RTX 5090 Reference with Backplate",
        price=219.99,
        in_stock=True,
        url="https://www.example.invalid/wb",
    )
    assert retail.keep(intel_box, "cpu", "intel core i7-14700k")
    assert retail.keep(gpu_backplate, "gpu", "amd rx 9060 xt")
    assert not retail.keep(waterblock, "gpu", "nvidia rtx 5090")


def test_match_wanted_harvests_any_benchmarked_model_on_page(fixtures):
    items = _listings(fixtures)
    acceptable = {("gpu", "nvidia rtx 4070 ti super"), ("gpu", "nvidia rtx 4070")}
    matched = [(x.sku, retail.match_wanted(x, acceptable)) for x in items]
    assert matched == [
        ("111", ("gpu", "nvidia rtx 4070 ti super")),
        ("222", ("gpu", "nvidia rtx 4070")),
        ("333", None),
        ("444", None),
    ]


def test_workstation_edition_gpu_is_kept_and_next_page_is_parsed():
    pro_gpu = retail.Listing(
        sku="p6000",
        name="PNY NVIDIA RTX PRO 6000 Blackwell Workstation Edition 96GB GDDR7 Graphics Card",
        price=8499.99,
        in_stock=True,
        url="https://www.example.invalid/p6000",
    )
    assert retail.keep(pro_gpu, "gpu", "nvidia rtx pro 6000 blackwell")
    html = '<div class="pages"><li class="item pages-item-next"><a class="action next" href="/video-cards.html?p=2">Next</a></li></div>'
    assert (
        retail.parse_magento_next_page(html, "https://www.centralcomputer.com/video-cards.html")
        == "https://www.centralcomputer.com/video-cards.html?p=2"
    )
    assert (
        retail.parse_magento_next_page("<div>No next</div>", "https://www.centralcomputer.com")
        is None
    )
