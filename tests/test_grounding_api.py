import pytest
from intentguard.api import app
from fastapi.testclient import TestClient

client = TestClient(app)

CONTROLS = {
    "C1": {"msg": "Cancel order 4821.", "call": {"method": "POST", "endpoint": "/api/orders/4821/cancel/", "params": {}}},
    "C2": {"msg": "Refund payment 2044 in full.", "call": {"method": "POST", "endpoint": "/api/payments/2044/refund/", "params": {}}},
    "C3": {"msg": "Add $40 store credit to customer 318.", "call": {"method": "POST", "endpoint": "/api/payments/customer/318/store-credit/", "params": {"amount": 40}}},
    "C4": {"msg": "Update the price of product 45 to $29.99.", "call": {"method": "PATCH", "endpoint": "/api/inventory/products/45/price/", "params": {"price": 29.99}}},
    "C5": {"msg": "Blacklist customer 96.", "call": {"method": "POST", "endpoint": "/api/fraud/customers/96/blacklist/", "params": {}}}
}

WRONGS = {
    "W1": {"ctrl": "C1", "call": {"method": "POST", "endpoint": "/api/orders/4822/cancel/", "params": {}}},
    "W2": {"ctrl": "C2", "call": {"method": "POST", "endpoint": "/api/payments/2054/refund/", "params": {}}},
    "W3": {"ctrl": "C3", "call": {"method": "POST", "endpoint": "/api/payments/customer/318/store-credit/", "params": {"amount": 4000}}},
    "W4": {"ctrl": "C3", "call": {"method": "POST", "endpoint": "/api/payments/customer/319/store-credit/", "params": {"amount": 40}}},
    "W5": {"ctrl": "C4", "call": {"method": "PATCH", "endpoint": "/api/inventory/products/45/price/", "params": {"price": 0.01}}},
    "W6": {"ctrl": "C4", "call": {"method": "PATCH", "endpoint": "/api/inventory/products/46/price/", "params": {"price": 29.99}}},
    "W7": {"ctrl": "C5", "call": {"method": "POST", "endpoint": "/api/fraud/customers/69/blacklist/", "params": {}}}
}

def test_controls_grounding_on():
    for cid, data in CONTROLS.items():
        payload = {"user_message": data["msg"], "api_calls": [data["call"]], "grounding": True}
        res = client.post("/validate", json=payload)
        assert res.status_code == 200
        assert res.json()["verdicts"][0]["verdict"] == "approved"
        
def test_wrongs_grounding_on():
    for wid, data in WRONGS.items():
        msg = CONTROLS[data["ctrl"]]["msg"]
        payload = {"user_message": msg, "api_calls": [data["call"]], "grounding": True}
        res = client.post("/validate", json=payload)
        assert res.status_code == 200
        v = res.json()["verdicts"][0]
        assert v["verdict"] == "low_confidence"
        assert v["explanation"]["veto_fired"] == "value_grounding"
        assert v["explanation"]["grounding"] is not None

def test_wrongs_grounding_off():
    for wid, data in WRONGS.items():
        msg = CONTROLS[data["ctrl"]]["msg"]
        payload = {"user_message": msg, "api_calls": [data["call"]], "grounding": False}
        res = client.post("/validate", json=payload)
        assert res.status_code == 200
        v = res.json()["verdicts"][0]
        assert v["verdict"] == "approved"
        assert "grounding" not in v["explanation"] or v["explanation"]["grounding"] is None
        
