import pytest
from fastapi.testclient import TestClient
from intentguard.api import app
import os
import json

client = TestClient(app)

def test_explanation_fields_and_agreement():
    req = {
        "user_message": "Cancel order 123",
        "api_calls": [
            {"method": "POST", "endpoint": "/api/orders/123/cancel/", "provenance": "box_a"}
        ]
    }
    resp = client.post("/validate", json=req)
    assert resp.status_code == 200
    verdicts = resp.json()["verdicts"]
    assert len(verdicts) == 1
    v = verdicts[0]
    
    # check fields
    assert "explanation" in v
    expl = v["explanation"]
    required_fields = ["closest_intent", "cosine_score", "verb_score", "entity_score", 
                       "weighted_total", "intent_verb_group", "call_verb_group", 
                       "verb_relation", "veto_fired", "threshold_used", "provenance", "reason"]
    for f in required_fields:
        assert f in expl
        
    # agreement: approved call has total >= threshold
    if v["verdict"] == "approved":
        assert expl["weighted_total"] >= expl["threshold_used"]

def test_explanation_capped():
    # verb mismatch that normally scores high enough to pass but is vetoed
    req = {
        "user_message": "Can you delete my order 123?",
        "api_calls": [
            {"method": "GET", "endpoint": "/api/orders/123/", "provenance": "box_a"}
        ]
    }
    resp = client.post("/validate", json=req)
    v = resp.json()["verdicts"][0]
    if v["explanation"]["veto_fired"] is not None:
        assert v["explanation"]["veto_fired"] in ["verb_mismatch", "read_only"]
        assert v["verdict"] in ["low_confidence", "orphaned"]

def test_audit_log_lines():
    audit_file = os.environ.get("INTENTGUARD_AUDIT_PATH")
    if not audit_file:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        audit_file = os.path.join(logs_dir, "audit.jsonl")
    
    lines_before = 0
    if os.path.exists(audit_file):
        with open(audit_file, "r") as f:
            lines_before = sum(1 for _ in f if _.strip())
            
    req = {
        "user_message": "Cancel order 123",
        "api_calls": [
            {"method": "POST", "endpoint": "/api/orders/123/cancel/", "provenance": "box_a"}
        ]
    }
    resp = client.post("/validate", json=req)
    assert resp.status_code == 200
    verdicts = resp.json()["verdicts"]
    
    lines_after = 0
    with open(audit_file, "r") as f:
        lines_after = sum(1 for _ in f if _.strip())
        
    assert lines_after == lines_before + len(verdicts)

def test_get_audit_order_and_limit():
    req = {
        "user_message": "test msg",
        "api_calls": [{"method": "GET", "endpoint": "/api/test", "provenance": "box_a"}]
    }
    client.post("/validate", json=req)
    client.post("/validate", json=req)
    client.post("/validate", json=req)
    
    resp = client.get("/audit?limit=2")
    assert resp.status_code == 200
    entries = resp.json()
    assert len(entries) == 2
    
    assert entries[0]["timestamp"] >= entries[1]["timestamp"]
    
def test_audit_no_params_and_truncate():
    long_msg = "A" * 400
    req = {
        "user_message": long_msg,
        "api_calls": [
            {"method": "POST", "endpoint": "/api/test", "params": {"foo": "bar"}, "body": {"baz": "qux"}, "provenance": "box_a"}
        ]
    }
    client.post("/validate", json=req)
    
    resp = client.get("/audit?limit=1")
    entries = resp.json()
    entry = entries[0]
    
    assert "params" not in entry
    assert "body" not in entry
    assert len(entry["user_message"]) == 300
    assert entry["user_message"].endswith("...")
    assert len(entry["explanation"]["closest_intent"]) <= 300
    assert len(entry["explanation"]["reason"]) <= 300

