import pytest

from pdi.tech_value import ebay


def test_ebay_missing_key_raises(monkeypatch):
    monkeypatch.delenv("EBAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("EBAY_CLIENT_SECRET", raising=False)
    with pytest.raises(ebay.MissingKey):
        ebay.collect()


def test_parse_items_keeps_only_new_us_fixed_price_parts_with_trusted_sellers():
    wanted = {("gpu", "nvidia rtx 5070"), ("cpu", "amd ryzen 7 9800x3d")}
    records = [
        # Valid Brand New US fixed-price GPU with shipping included in total price.
        {
            "itemId": "v1|1001|0",
            "title": "ASUS Dual NVIDIA GeForce RTX 5070 OC Edition 12GB GDDR7 Graphics Card",
            "conditionId": "1000",
            "condition": "New",
            "buyingOptions": ["FIXED_PRICE"],
            "itemLocation": {"country": "US"},
            "price": {"value": "599.99", "currency": "USD"},
            "shippingOptions": [{"shippingCost": {"value": "10.00", "currency": "USD"}}],
            "seller": {"username": "newegg", "feedbackPercentage": "99.1", "feedbackScore": 450000},
            "itemWebUrl": "https://www.ebay.com/itm/1001",
        },
        # Used/Open-box conditionId (1500) -> dropped.
        {
            "itemId": "v1|1002|0",
            "title": "ASUS Dual NVIDIA GeForce RTX 5070 OC Edition 12GB GDDR7 Graphics Card",
            "conditionId": "1500",
            "condition": "Open box",
            "buyingOptions": ["FIXED_PRICE"],
            "itemLocation": {"country": "US"},
            "price": {"value": "499.99", "currency": "USD"},
            "seller": {"username": "newegg", "feedbackPercentage": "99.1", "feedbackScore": 450000},
        },
        # Auction-only listing -> dropped.
        {
            "itemId": "v1|1003|0",
            "title": "AMD Ryzen 7 9800X3D 8-Core Desktop Processor",
            "conditionId": "1000",
            "condition": "New",
            "buyingOptions": ["AUCTION"],
            "itemLocation": {"country": "US"},
            "price": {"value": "350.00", "currency": "USD"},
            "seller": {"username": "newegg", "feedbackPercentage": "99.1", "feedbackScore": 450000},
        },
        # Untrusted low-feedback seller -> dropped.
        {
            "itemId": "v1|1004|0",
            "title": "AMD Ryzen 7 9800X3D 8-Core Desktop Processor",
            "conditionId": "1000",
            "condition": "New",
            "buyingOptions": ["FIXED_PRICE"],
            "itemLocation": {"country": "US"},
            "price": {"value": "399.00", "currency": "USD"},
            "seller": {"username": "new_user", "feedbackPercentage": "95.0", "feedbackScore": 12},
        },
        # Whole gaming PC -> dropped by retail.match_wanted.
        {
            "itemId": "v1|1005|0",
            "title": "CyberPowerPC Gaming Desktop AMD Ryzen 7 9800X3D 32GB RAM 2TB SSD",
            "conditionId": "1000",
            "condition": "New",
            "buyingOptions": ["FIXED_PRICE"],
            "itemLocation": {"country": "US"},
            "price": {"value": "1799.99", "currency": "USD"},
            "seller": {"username": "newegg", "feedbackPercentage": "99.1", "feedbackScore": 450000},
        },
    ]
    rows = ebay.parse_items(records, wanted, "2026-11-17")
    assert len(rows) == 1
    assert rows[0]["sku"] == "v1|1001|0"
    assert rows[0]["model_key"] == "nvidia rtx 5070"
    assert rows[0]["sale_price"] == 609.99
    assert rows[0]["source"] == "ebay_browse"
