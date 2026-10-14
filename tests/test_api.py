"""
tests.test_api
==============
FastAPI integration tests for intentguard's /validate and /health endpoints.

Uses FastAPI's synchronous TestClient (backed by httpx) so no async test
runner is needed — pytest-asyncio is not required for these tests.

Test categories
---------------
1. GET /health  – basic liveness
2. POST /validate – happy-path with all four verdict types
3. POST /validate – 422 validation errors (missing fields, bad types)
4. POST /validate – mixed realistic e-commerce scenario
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intentguard.api import app

# Module-level TestClient — shared across all tests in this file.
# Raises exceptions by default (raise_server_exceptions=True).
client = TestClient(app, raise_server_exceptions=True)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

CART_ADD_CALL = {
    "method": "POST",
    "endpoint": "/cart/items",
    "body": {"product": "shoes", "quantity": 1},
    "provenance": "box_a",
}

UPDATE_EMAIL_CALL = {
    "method": "PUT",
    "endpoint": "/profile/email",
    "body": {"email": "user@example.com"},
    "provenance": "box_a",
}

APPLY_COUPON_CALL = {
    "method": "POST",
    "endpoint": "/cart/apply-coupon",
    "params": {"code": "SAVE10"},
    "provenance": "box_a",
}

INJECTED_CALL = {
    "method": "DELETE",
    "endpoint": "/account",
    "provenance": "box_b",
}

UNRELATED_BOX_A_CALL = {
    "method": "GET",
    "endpoint": "/admin/system/logs",
    "provenance": "box_a",
}


# ===========================================================================
# 1. GET /health
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health should return HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_keys(self):
        """Response must contain 'status' and 'model' keys."""
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "model" in body

    def test_health_status_ok(self):
        """'status' field must be 'ok'."""
        assert client.get("/health").json()["status"] == "ok"

    def test_health_model_name(self):
        """'model' field must report 'all-MiniLM-L6-v2'."""
        assert client.get("/health").json()["model"] == "all-MiniLM-L6-v2"


# ===========================================================================
# 2. POST /validate — response structure
# ===========================================================================

class TestValidateResponseStructure:
    def test_returns_200_for_valid_request(self):
        """A well-formed request must return HTTP 200."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
        })
        assert resp.status_code == 200

    def test_response_has_top_level_keys(self):
        """Response body must have 'verdicts', 'intents_extracted', 'threshold'."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
        })
        body = resp.json()
        assert "verdicts"          in body
        assert "intents_extracted" in body
        assert "threshold"         in body

    def test_verdicts_is_list(self):
        """'verdicts' must be a list."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
        })
        assert isinstance(resp.json()["verdicts"], list)

    def test_verdicts_length_matches_calls(self):
        """One verdict entry per api_call."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL, INJECTED_CALL],
        })
        assert len(resp.json()["verdicts"]) == 2

    def test_verdict_entry_keys(self):
        """Each verdict dict must have call_index, verdict, matched_intent, score, provenance."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
        })
        v = resp.json()["verdicts"][0]
        for key in ("call_index", "verdict", "matched_intent", "score", "provenance"):
            assert key in v, f"Missing key {key!r} in verdict entry"

    def test_verdict_strings_are_valid(self):
        """Every verdict string must be one of the four valid values."""
        valid = {"approved", "orphaned", "injected", "low_confidence"}
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL, INJECTED_CALL, UNRELATED_BOX_A_CALL],
        })
        for v in resp.json()["verdicts"]:
            assert v["verdict"] in valid, f"Unexpected verdict: {v['verdict']!r}"

    def test_intents_extracted_is_list_of_strings(self):
        """'intents_extracted' must be a non-empty list of strings."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
        })
        extracted = resp.json()["intents_extracted"]
        assert isinstance(extracted, list)
        assert len(extracted) >= 1
        assert all(isinstance(i, str) for i in extracted)

    def test_threshold_echoed_in_response(self):
        """The threshold used should appear in the response."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [CART_ADD_CALL],
            "threshold": 0.42,
        })
        assert resp.json()["threshold"] == pytest.approx(0.42, abs=1e-6)

    def test_call_index_is_sequential(self):
        """call_index in verdicts must be sequential 0, 1, 2, …"""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [CART_ADD_CALL, INJECTED_CALL, UNRELATED_BOX_A_CALL],
        })
        for i, v in enumerate(resp.json()["verdicts"]):
            assert v["call_index"] == i


# ===========================================================================
# 3. POST /validate — individual verdicts
# ===========================================================================

class TestValidateVerdicts:
    def test_approved_verdict(self):
        """A closely-matching box_a call should return 'approved'."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my shopping cart",
            "api_calls": [CART_ADD_CALL],
            "threshold": 0.3,
            "grounding": False,
        })
        assert resp.json()["verdicts"][0]["verdict"] == "approved"

    def test_injected_verdict(self):
        """A box_b call with no intent match should return 'injected'."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [INJECTED_CALL],
            "threshold": 0.9,
        })
        assert resp.json()["verdicts"][0]["verdict"] == "injected"

    def test_orphaned_verdict(self):
        """An unrelated box_a call with score below low_conf_floor should return 'orphaned'."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [UNRELATED_BOX_A_CALL],
            "threshold": 0.99,
            "low_conf_floor": 0.99,
        })
        assert resp.json()["verdicts"][0]["verdict"] == "orphaned"

    def test_low_confidence_verdict(self):
        """An unmatched box_a call with middling score should return 'low_confidence'."""
        resp = client.post("/validate", json={
            "user_message": "Update my profile",
            "api_calls": [{"method": "PUT", "endpoint": "/profile", "provenance": "box_a"}],
            "threshold": 0.99,
            "low_conf_floor": 0.0,
        })
        verdict = resp.json()["verdicts"][0]["verdict"]
        assert verdict in ("low_confidence", "approved"), (
            f"Expected low_confidence (or approved), got {verdict!r}"
        )

    def test_approved_has_matched_intent(self):
        """An approved verdict must have a non-null matched_intent."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to shopping cart",
            "api_calls": [CART_ADD_CALL],
            "threshold": 0.3,
            "grounding": False,
        })
        assert resp.json()["verdicts"][0]["matched_intent"] is not None

    def test_injected_has_null_matched_intent(self):
        """An injected verdict must have matched_intent == null."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [INJECTED_CALL],
            "threshold": 0.9,
        })
        assert resp.json()["verdicts"][0]["matched_intent"] is None

    def test_box_a_unmatched_is_not_injected(self):
        """A box_a unmatched call must never be 'injected'."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to my cart",
            "api_calls": [UNRELATED_BOX_A_CALL],
            "threshold": 0.99,
        })
        assert resp.json()["verdicts"][0]["verdict"] != "injected"


# ===========================================================================
# 4. POST /validate — 422 validation errors
# ===========================================================================

class TestValidateErrors:
    def test_missing_user_message_returns_422(self):
        """Omitting user_message entirely should return 422."""
        resp = client.post("/validate", json={
            "api_calls": [CART_ADD_CALL],
        })
        assert resp.status_code == 422

    def test_empty_user_message_returns_422(self):
        """Empty string user_message should return 422."""
        resp = client.post("/validate", json={
            "user_message": "",
            "api_calls": [CART_ADD_CALL],
        })
        assert resp.status_code == 422

    def test_whitespace_user_message_returns_422(self):
        """Whitespace-only user_message should return 422."""
        resp = client.post("/validate", json={
            "user_message": "   ",
            "api_calls": [CART_ADD_CALL],
        })
        assert resp.status_code == 422

    def test_missing_api_calls_returns_422(self):
        """Omitting api_calls entirely should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
        })
        assert resp.status_code == 422

    def test_empty_api_calls_returns_422(self):
        """Empty api_calls list should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [],
        })
        assert resp.status_code == 422

    def test_call_missing_endpoint_returns_422(self):
        """An api_call dict missing 'endpoint' should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [{"method": "POST"}],   # endpoint missing
        })
        assert resp.status_code == 422

    def test_call_missing_method_returns_422(self):
        """An api_call dict missing 'method' should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [{"endpoint": "/cart/items"}],   # method missing
        })
        assert resp.status_code == 422

    def test_invalid_provenance_value_returns_422(self):
        """A provenance value other than 'box_a'/'box_b' should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [{"method": "POST", "endpoint": "/cart", "provenance": "evil"}],
        })
        assert resp.status_code == 422

    def test_threshold_out_of_range_returns_422(self):
        """threshold > 1.0 should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [CART_ADD_CALL],
            "threshold": 1.5,
        })
        assert resp.status_code == 422

    def test_low_conf_floor_negative_returns_422(self):
        """low_conf_floor < 0.0 should return 422."""
        resp = client.post("/validate", json={
            "user_message": "Add shoes to cart",
            "api_calls": [CART_ADD_CALL],
            "low_conf_floor": -0.1,
        })
        assert resp.status_code == 422


# ===========================================================================
# 5. Realistic mixed e-commerce scenario
# ===========================================================================

class TestRealisticScenario:
    """
    User message: "Update my email and apply coupon SAVE10 to my cart"
    Expected intents extracted: ["Update my email", "apply coupon SAVE10 to my cart"]

    API calls:
      0: PUT /profile/email  (box_a)   → should be approved (matches "update email")
      1: POST /cart/apply-coupon?code=SAVE10 (box_a) → should be approved (matches "apply coupon")
      2: DELETE /account (box_b)        → should be injected
    """

    USER_MSG = "Update my email and apply coupon SAVE10 to my cart"
    API_CALLS = [
        {
            "method": "PUT",
            "endpoint": "/profile/email",
            "body": {"email": "user@example.com"},
            "provenance": "box_a",
        },
        {
            "method": "POST",
            "endpoint": "/cart/apply-coupon",
            "params": {"code": "SAVE10"},
            "provenance": "box_a",
        },
        {
            "method": "DELETE",
            "endpoint": "/account",
            "provenance": "box_b",
        },
    ]

    def _get_result(self, threshold: float = 0.3) -> dict:
        resp = client.post("/validate", json={
            "user_message": self.USER_MSG,
            "api_calls": self.API_CALLS,
            "threshold": threshold,
        })
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
        return resp.json()

    def test_returns_200(self):
        """The realistic scenario request must return HTTP 200."""
        assert client.post("/validate", json={
            "user_message": self.USER_MSG,
            "api_calls": self.API_CALLS,
        }).status_code == 200

    def test_three_verdicts_returned(self):
        """Must return exactly 3 verdicts for 3 calls."""
        assert len(self._get_result()["verdicts"]) == 3

    def test_two_intents_extracted(self):
        """'Update my email AND apply coupon…' should yield 2 intents."""
        result = self._get_result()
        assert len(result["intents_extracted"]) == 2

    def test_delete_account_is_injected(self):
        """Call 2 (DELETE /account, box_b) must be 'injected'."""
        # Use high threshold so it cannot accidentally be approved.
        result = self._get_result(threshold=0.9)
        assert result["verdicts"][2]["verdict"] == "injected"

    def test_injected_call_provenance_is_box_b(self):
        """Injected call's provenance field must be 'box_b'."""
        result = self._get_result(threshold=0.9)
        assert result["verdicts"][2]["provenance"] == "box_b"

    def test_box_a_calls_are_not_injected(self):
        """Calls 0 and 1 (box_a) must never be 'injected'."""
        result = self._get_result()
        assert result["verdicts"][0]["verdict"] != "injected"
        assert result["verdicts"][1]["verdict"] != "injected"

    def test_email_call_verdict_valid(self):
        """Call 0 (PUT /profile/email) must have a valid verdict string."""
        valid = {"approved", "orphaned", "injected", "low_confidence"}
        result = self._get_result()
        assert result["verdicts"][0]["verdict"] in valid

    def test_coupon_call_verdict_valid(self):
        """Call 1 (POST /cart/apply-coupon) must have a valid verdict string."""
        valid = {"approved", "orphaned", "injected", "low_confidence"}
        result = self._get_result()
        assert result["verdicts"][1]["verdict"] in valid

    def test_score_is_float_in_range(self):
        """All score values must be floats in [0.0, 1.0]."""
        result = self._get_result()
        for v in result["verdicts"]:
            s = v["score"]
            assert isinstance(s, float), f"score is {type(s)}, expected float"
            assert 0.0 <= s <= 1.0, f"score {s} out of range"
