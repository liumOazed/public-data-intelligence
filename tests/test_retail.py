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
