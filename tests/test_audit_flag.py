import os
import json
from fastapi.testclient import TestClient
from intentguard.api import app

client = TestClient(app)

def test_audit_flag():
    # Write nothing initially
    audit_path = os.environ.get("INTENTGUARD_AUDIT_PATH")
    if audit_path and os.path.exists(audit_path):
        os.remove(audit_path)
        
    payload = {
        "user_message": "test",
        "api_calls": [
            {"method": "GET", "endpoint": "/a"},
            {"method": "POST", "endpoint": "/b"}
        ],
        "audit": True
    }
    client.post("/validate", json=payload)
    
    lines = []
    if audit_path and os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0]).get("mode") == "live"
    
    # Second call with audit=False
    payload["audit"] = False
    client.post("/validate", json=payload)
    
    lines2 = []
    if audit_path and os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            lines2 = f.readlines()
    assert len(lines2) == 2 # still 2
