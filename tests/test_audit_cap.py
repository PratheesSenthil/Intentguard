import os
import json
from fastapi.testclient import TestClient
from intentguard.api import app

client = TestClient(app)

def test_audit_cap():
    audit_path = os.environ.get("INTENTGUARD_AUDIT_PATH")
    if audit_path and os.path.exists(audit_path):
        os.remove(audit_path)
        
    long_msg = "A" * 400
    payload = {
        "user_message": long_msg,
        "api_calls": [{"method": "GET", "endpoint": "/a"}],
        "audit": True
    }
    client.post("/validate", json=payload)
    
    with open(audit_path, "r", encoding="utf-8") as f:
        line = f.readline()
    entry = json.loads(line)
    
    assert len(entry["user_message"]) == 300
    assert entry["user_message"].endswith("...")
    assert len(entry["explanation"]["closest_intent"]) <= 300
    assert len(entry["explanation"]["reason"]) <= 300
