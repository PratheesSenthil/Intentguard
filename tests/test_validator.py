"""
tests.test_validator
====================
Test suite for intentguard.

Activation status
-----------------
[ACTIVE]  TestEmbedder   – embedder.py is fully implemented
[SKIP]    TestScorer     – scorer.py not yet implemented
[SKIP]    TestMatcher    – matcher.py not yet implemented
[SKIP]    TestValidator  – validator.py not yet implemented
[SKIP]    TestEdgeCases  – depends on full pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

from intentguard.embedder import encode, encode_batch


# ===========================================================================
# 1. Embedder tests  [ACTIVE]
# ===========================================================================

class TestEmbedder:
    # ------------------------------------------------------------------
    # Shape and dtype
    # ------------------------------------------------------------------

    def test_encode_returns_correct_shape(self):
        """encode() should return a 1-D numpy array of length 384."""
        vec = encode("Book a flight to Paris")
        assert isinstance(vec, np.ndarray), "Expected np.ndarray"
        assert vec.ndim == 1, f"Expected 1-D array, got ndim={vec.ndim}"
        assert vec.shape == (384,), f"Expected shape (384,), got {vec.shape}"

    def test_encode_returns_float32(self):
        """encode() should return dtype float32."""
        vec = encode("Some text")
        assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"

    def test_encode_is_normalised(self):
        """encode() output should be an L2-unit vector (norm ≈ 1.0)."""
        vec = encode("Normalisation check")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ≈ 1.0, got {norm}"

    def test_encode_different_texts_differ(self):
        """Two semantically different texts should not produce identical vectors."""
        vec_a = encode("Book a flight to Tokyo")
        vec_b = encode("Delete the user account")
        assert not np.allclose(vec_a, vec_b), (
            "Expected different embeddings for unrelated texts"
        )

    def test_encode_same_text_reproducible(self):
        """Encoding the same text twice should return identical vectors."""
        text = "Reproducibility check"
        assert np.array_equal(encode(text), encode(text)), (
            "encode() should be deterministic for the same input"
        )

    # ------------------------------------------------------------------
    # encode_batch – shape, dtype, normalisation
    # ------------------------------------------------------------------

    def test_encode_batch_shape(self):
        """encode_batch(texts) should return shape (len(texts), 384)."""
        texts = ["hello", "world", "intentguard"]
        vecs = encode_batch(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (3, 384), f"Expected (3, 384), got {vecs.shape}"

    def test_encode_batch_dtype(self):
        """encode_batch() should return dtype float32."""
        vecs = encode_batch(["test"])
        assert vecs.dtype == np.float32, f"Expected float32, got {vecs.dtype}"

    def test_encode_batch_rows_are_normalised(self):
        """Every row of encode_batch() output should be L2-unit-normalised."""
        texts = ["First sentence", "Second sentence", "Third one"]
        vecs = encode_batch(texts)
        for i, row in enumerate(vecs):
            norm = float(np.linalg.norm(row))
            assert abs(norm - 1.0) < 1e-5, (
                f"Row {i} has L2 norm {norm}, expected ≈ 1.0"
            )

    def test_encode_batch_matches_encode(self):
        """Each row of encode_batch([t]) should be numerically equal to encode(t)."""
        texts = ["Match check alpha", "Match check beta"]
        batch = encode_batch(texts)
        for i, text in enumerate(texts):
            single = encode(text)
            assert np.allclose(batch[i], single, atol=1e-6), (
                f"encode_batch row {i} differs from encode() for {text!r}"
            )

    def test_encode_batch_single_element(self):
        """encode_batch() with one element should return shape (1, 384)."""
        vecs = encode_batch(["solo"])
        assert vecs.shape == (1, 384), f"Expected (1, 384), got {vecs.shape}"

    # ------------------------------------------------------------------
    # Singleton / caching behaviour
    # ------------------------------------------------------------------

    def test_model_is_cached(self):
        """_get_model() should return the same object on repeated calls (singleton)."""
        from intentguard.embedder import _get_model
        m1 = _get_model()
        m2 = _get_model()
        assert m1 is m2, "Expected the same SentenceTransformer instance (singleton)"

    # ------------------------------------------------------------------
    # Input validation – encode()
    # ------------------------------------------------------------------

    def test_encode_none_raises(self):
        """encode(None) should raise ValueError."""
        with pytest.raises(ValueError, match="None"):
            encode(None)  # type: ignore[arg-type]

    def test_encode_empty_string_raises(self):
        """encode('') should raise ValueError."""
        with pytest.raises(ValueError):
            encode("")

    def test_encode_whitespace_only_raises(self):
        """encode('   ') should raise ValueError (whitespace-only treated as empty)."""
        with pytest.raises(ValueError):
            encode("   ")

    def test_encode_wrong_type_raises(self):
        """encode(123) should raise ValueError."""
        with pytest.raises(ValueError):
            encode(123)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Input validation – encode_batch()
    # ------------------------------------------------------------------

    def test_encode_batch_empty_list_raises(self):
        """encode_batch([]) should raise ValueError."""
        with pytest.raises(ValueError):
            encode_batch([])

    def test_encode_batch_none_raises(self):
        """encode_batch(None) should raise ValueError."""
        with pytest.raises(ValueError):
            encode_batch(None)  # type: ignore[arg-type]

    def test_encode_batch_contains_empty_string_raises(self):
        """encode_batch(['ok', '']) should raise ValueError for the empty element."""
        with pytest.raises(ValueError):
            encode_batch(["ok", ""])

    def test_encode_batch_contains_none_raises(self):
        """encode_batch(['ok', None]) should raise ValueError."""
        with pytest.raises(ValueError):
            encode_batch(["ok", None])  # type: ignore[list-item]

    def test_encode_batch_contains_whitespace_raises(self):
        """encode_batch(['ok', '   ']) should raise ValueError."""
        with pytest.raises(ValueError):
            encode_batch(["ok", "   "])

    def test_encode_batch_wrong_type_raises(self):
        """encode_batch('not a list') should raise ValueError."""
        with pytest.raises(ValueError):
            encode_batch("not a list")  # type: ignore[arg-type]


# ===========================================================================
# 2. Scorer tests  [ACTIVE]
# ===========================================================================

from intentguard.scorer import (
    score_pair,
    _cosine_score,
    _verb_score,
    _entity_score,
    ALPHA, BETA, GAMMA,
)


class TestScorer:
    # ------------------------------------------------------------------
    # Weight sanity
    # ------------------------------------------------------------------

    def test_weights_sum_to_one(self):
        """ALPHA + BETA + GAMMA must equal 1.0 (enforced at module load)."""
        assert abs(ALPHA + BETA + GAMMA - 1.0) < 1e-9, (
            f"Weights sum to {ALPHA + BETA + GAMMA}, expected 1.0"
        )

    # ------------------------------------------------------------------
    # score_pair() — range and type
    # ------------------------------------------------------------------

    def test_score_pair_returns_float(self):
        """score_pair() must return a Python float."""
        s = score_pair("Update my cart", "update_cart item=shoes")
        assert isinstance(s, float), f"Expected float, got {type(s)}"

    def test_score_pair_range(self):
        """score_pair() must return a value in [0.0, 1.0]."""
        pairs = [
            ("Book a flight to Paris", "book_flight destination=Paris"),
            ("Cancel my order",        "delete_order order_id=42"),
            ("Send me an invoice",     "generate_report type=pdf"),
        ]
        for intent, call in pairs:
            s = score_pair(intent, call)
            assert 0.0 <= s <= 1.0, (
                f"score_pair({intent!r}, {call!r}) = {s} — out of [0,1]"
            )

    def test_score_pair_identical_texts_high(self):
        """Identical intent and call text should score >= 0.85."""
        text = "update the user profile"
        s = score_pair(text, text)
        assert s >= 0.85, f"Identical texts scored {s}, expected >= 0.85"

    def test_score_pair_semantically_related_high(self):
        """Semantically close intent and call should score >= 0.5."""
        s = score_pair(
            "Add item to my shopping cart",
            "add_to_cart product_id=123",
        )
        assert s >= 0.5, f"Semantically related pair scored {s}, expected >= 0.5"

    def test_score_pair_unrelated_texts_low(self):
        """Completely unrelated texts should score < 0.5."""
        s = score_pair(
            "Book a flight to Tokyo",
            "delete_account user_id=99",
        )
        assert s < 0.5, (
            f"Unrelated pair scored {s}, expected < 0.5"
        )

    def test_verb_mismatch_lowers_score(self):
        """'Cancel order' vs 'create_order' should score lower than 'Create order' vs 'create_order'."""
        s_match    = score_pair("Create an order for shoes",  "create_order item=shoes")
        s_mismatch = score_pair("Cancel my order for shoes",  "create_order item=shoes")
        assert s_match > s_mismatch, (
            f"Verb-match score ({s_match}) should exceed verb-mismatch score ({s_mismatch})"
        )

    def test_entity_overlap_raises_score(self):
        """Shared entity 'cart' should give a higher score than no shared entity."""
        s_overlap   = score_pair("Update my cart",   "update_cart item=shoes")
        s_nooverlap = score_pair("Update something", "update_cart item=shoes")
        assert s_overlap >= s_nooverlap, (
            f"Entity-overlap score ({s_overlap}) should be >= no-overlap ({s_nooverlap})"
        )

    # ------------------------------------------------------------------
    # score_pair() — input validation
    # ------------------------------------------------------------------

    def test_score_pair_none_intent_raises(self):
        """score_pair(None, ...) should raise ValueError."""
        with pytest.raises(ValueError):
            score_pair(None, "update_cart")  # type: ignore[arg-type]

    def test_score_pair_empty_intent_raises(self):
        """score_pair('', ...) should raise ValueError."""
        with pytest.raises(ValueError):
            score_pair("", "update_cart")

    def test_score_pair_none_call_raises(self):
        """score_pair(..., None) should raise ValueError."""
        with pytest.raises(ValueError):
            score_pair("Update my cart", None)  # type: ignore[arg-type]

    def test_score_pair_empty_call_raises(self):
        """score_pair(..., '') should raise ValueError."""
        with pytest.raises(ValueError):
            score_pair("Update my cart", "")

    # ------------------------------------------------------------------
    # _cosine_score() — unit tests
    # ------------------------------------------------------------------

    def test_cosine_score_unit_vectors(self):
        """_cosine_score of a unit vector with itself should be ~1.0."""
        v = np.zeros(384, dtype=np.float32)
        v[0] = 1.0
        assert abs(_cosine_score(v, v) - 1.0) < 1e-5

    def test_cosine_score_orthogonal(self):
        """_cosine_score of orthogonal unit vectors should be ~0.0."""
        v1 = np.zeros(384, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(384, dtype=np.float32); v2[1] = 1.0
        assert abs(_cosine_score(v1, v2)) < 1e-5

    def test_cosine_score_rejects_unnormalised(self):
        """_cosine_score should raise ValueError if a vector is not unit-normalised."""
        v_norm = np.zeros(384, dtype=np.float32); v_norm[0] = 1.0
        v_bad  = np.ones(384, dtype=np.float32)  # norm >> 1
        with pytest.raises(ValueError, match="normalised"):
            _cosine_score(v_bad, v_norm)

    def test_cosine_score_output_clamped(self):
        """_cosine_score output must be in [-1.0, 1.0]."""
        v = np.zeros(384, dtype=np.float32); v[0] = 1.0
        noisy = (v + 1e-7).astype(np.float32)
        noisy /= np.linalg.norm(noisy)
        result = _cosine_score(v, noisy)
        assert -1.0 <= result <= 1.0

    # ------------------------------------------------------------------
    # _verb_score() — unit tests
    # ------------------------------------------------------------------

    def test_verb_score_synonym_match(self):
        """'Modify' and 'update' should score 1.0 (same canonical group)."""
        assert _verb_score("Modify my profile", "update_profile") == 1.0

    def test_verb_score_different_groups(self):
        """'Delete' and 'create' are in different groups — should score 0.0."""
        assert _verb_score("Delete my account", "create_account") == 0.0

    def test_verb_score_no_verb_in_either(self):
        """When neither text has a recognisable verb, return 1.0 (tie/no-signal)."""
        assert _verb_score("Paris London", "Tokyo Berlin") == 1.0

    def test_verb_score_underscore_split(self):
        """API call names like 'cancel_order' should be tokenised correctly."""
        assert _verb_score("Cancel my order", "cancel_order order_id=1") == 1.0

    # ------------------------------------------------------------------
    # _entity_score() — unit tests
    # ------------------------------------------------------------------

    def test_entity_score_no_entities(self):
        """_entity_score returns 0.0 when neither text contains domain entities."""
        assert _entity_score("do something nice", "perform an action") == 0.0

    def test_entity_score_full_overlap(self):
        """Identical entity sets should give 1.0."""
        assert _entity_score("update cart items", "add cart items") == 1.0

    def test_entity_score_partial_overlap(self):
        """Partially overlapping entity sets should give a value in (0, 1)."""
        s = _entity_score("update cart and order", "fetch cart details")
        assert 0.0 < s < 1.0, f"Expected partial overlap score, got {s}"

    def test_entity_score_no_overlap(self):
        """Disjoint entity sets — cart vs payment — should score 0.0."""
        assert _entity_score("update cart", "fetch payment") == 0.0


# ===========================================================================
# 3. Matcher tests  [ACTIVE]
# ===========================================================================

from intentguard.matcher import match, _build_score_matrix, _resolve_conflicts


# ---------------------------------------------------------------------------
# Helpers — inject synthetic score matrices for speed + determinism
# ---------------------------------------------------------------------------

def _fake_intents(n: int) -> list[str]:
    return [f"intent number {i}" for i in range(n)]

def _fake_calls(n: int) -> list[str]:
    return [f"call number {i}" for i in range(n)]


class TestMatcher:
    # ------------------------------------------------------------------
    # Return structure
    # ------------------------------------------------------------------

    def test_return_keys_present(self):
        """match() must return a dict with 'assignments', 'score_matrix', 'threshold'."""
        result = match(
            ["update my cart"],
            ["update_cart"],
            score_matrix=np.array([[0.9]], dtype=np.float32),
        )
        assert "assignments"  in result
        assert "score_matrix" in result
        assert "threshold"    in result

    def test_assignments_covers_all_calls(self):
        """assignments dict must have an entry for every call index 0..N-1."""
        S = np.array([[0.9, 0.2, 0.8]], dtype=np.float32)  # 1 intent, 3 calls
        result = match(_fake_intents(1), _fake_calls(3), score_matrix=S)
        assert set(result["assignments"].keys()) == {0, 1, 2}

    # ------------------------------------------------------------------
    # Edge case 1: one intent, one call, clear match
    # ------------------------------------------------------------------

    def test_one_intent_one_call_approved(self):
        """1 intent x 1 call above threshold -> assigned to intent 0."""
        S = np.array([[0.85]], dtype=np.float32)
        result = match(_fake_intents(1), _fake_calls(1), threshold=0.5, score_matrix=S)
        assert result["assignments"][0] == 0

    def test_one_intent_one_call_unassigned(self):
        """1 intent x 1 call below threshold -> unassigned (None)."""
        S = np.array([[0.3]], dtype=np.float32)
        result = match(_fake_intents(1), _fake_calls(1), threshold=0.5, score_matrix=S)
        assert result["assignments"][0] is None

    # ------------------------------------------------------------------
    # Edge case 2: one intent, many calls -- 1:many (no Hungarian)
    # ------------------------------------------------------------------

    def test_one_intent_many_calls_all_approved(self):
        """1 intent claiming 3 calls above threshold -- all three approved, no conflict."""
        S = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        result = match(_fake_intents(1), _fake_calls(3), threshold=0.5, score_matrix=S)
        a = result["assignments"]
        assert a[0] == 0, "call 0 should be assigned to intent 0"
        assert a[1] == 0, "call 1 should be assigned to intent 0"
        assert a[2] == 0, "call 2 should be assigned to intent 0"

    def test_one_intent_many_calls_mixed(self):
        """1 intent: some calls above threshold (approved), some below (None)."""
        S = np.array([[0.9, 0.2, 0.8, 0.1]], dtype=np.float32)
        result = match(_fake_intents(1), _fake_calls(4), threshold=0.5, score_matrix=S)
        a = result["assignments"]
        assert a[0] == 0
        assert a[1] is None
        assert a[2] == 0
        assert a[3] is None

    # ------------------------------------------------------------------
    # Edge case 3: two intents contesting one call (true conflict -> Hungarian)
    # ------------------------------------------------------------------

    def test_conflict_two_intents_one_call_winner(self):
        """Two intents both above threshold for same call -- higher scorer wins."""
        #           call0
        # intent0:  0.75
        # intent1:  0.90  <- should win
        S = np.array([[0.75], [0.90]], dtype=np.float32)
        result = match(_fake_intents(2), _fake_calls(1), threshold=0.5, score_matrix=S)
        assert result["assignments"][0] == 1, (
            "Intent 1 (score 0.90) should beat intent 0 (score 0.75)"
        )

    def test_conflict_resolution_assigns_one_intent(self):
        """After conflict resolution exactly one intent wins per contested call."""
        S = np.array([[0.75], [0.90]], dtype=np.float32)
        result = match(_fake_intents(2), _fake_calls(1), threshold=0.5, score_matrix=S)
        assigned = result["assignments"][0]
        assert assigned in {0, 1}, f"Expected 0 or 1, got {assigned}"

    def test_conflict_winner_below_threshold_becomes_none(self):
        """If threshold is above all scores, contested call is unassigned."""
        S = np.array([[0.6], [0.7]], dtype=np.float32)
        result = match(_fake_intents(2), _fake_calls(1), threshold=0.8, score_matrix=S)
        # threshold=0.8: no claimants above it -> not contested -> unassigned
        assert result["assignments"][0] is None

    # ------------------------------------------------------------------
    # Edge case 4: zero intents above threshold
    # ------------------------------------------------------------------

    def test_zero_claimants_unassigned(self):
        """A call with no intents above threshold returns None."""
        S = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        result = match(_fake_intents(3), _fake_calls(1), threshold=0.5, score_matrix=S)
        assert result["assignments"][0] is None

    # ------------------------------------------------------------------
    # Edge case 5: three intents, two calls -- mixed conflict + direct
    # ------------------------------------------------------------------

    def test_mixed_conflict_and_direct(self):
        """
        3 intents x 2 calls:
          call0: contested by intent0(0.85) and intent1(0.90) -> Hungarian -> intent1 wins
          call1: only claimed by intent2(0.80) -> direct, no conflict
        """
        S = np.array([
            [0.85, 0.30],   # intent0
            [0.90, 0.20],   # intent1
            [0.10, 0.80],   # intent2
        ], dtype=np.float32)
        result = match(_fake_intents(3), _fake_calls(2), threshold=0.5, score_matrix=S)
        a = result["assignments"]
        assert a[0] == 1, f"call0 -> intent1 (0.90 > 0.85), got {a[0]}"
        assert a[1] == 2, f"call1 -> intent2 (direct, 0.80), got {a[1]}"

    def test_hungarian_globally_optimal(self):
        """
        2 intents x 2 calls both contested -- Hungarian finds global optimum.
          intent0 scores: call0=0.90, call1=0.70
          intent1 scores: call0=0.60, call1=0.95
        Optimal: intent0->call0 (0.90) + intent1->call1 (0.95) = 1.85
        vs alternative: intent0->call1 + intent1->call0 = 0.70+0.60 = 1.30
        """
        S = np.array([
            [0.90, 0.70],
            [0.60, 0.95],
        ], dtype=np.float32)
        result = match(_fake_intents(2), _fake_calls(2), threshold=0.5, score_matrix=S)
        a = result["assignments"]
        assert a[0] == 0, f"call0 -> intent0 (globally optimal), got {a[0]}"
        assert a[1] == 1, f"call1 -> intent1 (globally optimal), got {a[1]}"

    # ------------------------------------------------------------------
    # Pre-computed matrix passthrough
    # ------------------------------------------------------------------

    def test_precomputed_score_matrix_accepted(self):
        """match() with a valid pre-computed matrix skips _build_score_matrix."""
        S = np.array([[0.8, 0.1]], dtype=np.float32)
        result = match(_fake_intents(1), _fake_calls(2), threshold=0.5, score_matrix=S)
        assert result["assignments"][0] == 0
        assert result["assignments"][1] is None

    def test_precomputed_wrong_shape_raises(self):
        """A pre-computed matrix with wrong shape should raise ValueError."""
        S = np.array([[0.8, 0.1, 0.5]], dtype=np.float32)  # (1,3) but need (1,2)
        with pytest.raises(ValueError, match="shape"):
            match(_fake_intents(1), _fake_calls(2), score_matrix=S)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_empty_intents_raises(self):
        """match() should raise ValueError when intents list is empty."""
        with pytest.raises(ValueError):
            match([], ["call"])

    def test_empty_calls_raises(self):
        """match() should raise ValueError when calls list is empty."""
        with pytest.raises(ValueError):
            match(["intent"], [])

    def test_none_intents_raises(self):
        """match() should raise ValueError when intents is None."""
        with pytest.raises(ValueError):
            match(None, ["call"])  # type: ignore[arg-type]

    def test_whitespace_intent_raises(self):
        """match() should raise ValueError for whitespace-only intent element."""
        with pytest.raises(ValueError):
            match(["   "], ["call"])

    # ------------------------------------------------------------------
    # _build_score_matrix integration (uses real model)
    # ------------------------------------------------------------------

    def test_build_score_matrix_shape(self):
        """_build_score_matrix should return shape (M, N) float32."""
        intents = ["update profile", "delete account"]
        calls   = ["update_profile", "remove_account", "get_cart"]
        S = _build_score_matrix(intents, calls)
        assert S.shape == (2, 3), f"Expected (2, 3), got {S.shape}"
        assert S.dtype == np.float32

    def test_build_score_matrix_values_in_range(self):
        """All entries of _build_score_matrix should be in [0, 1]."""
        S = _build_score_matrix(["add to cart"], ["add_to_cart", "delete_order"])
        assert np.all(S >= 0.0) and np.all(S <= 1.0)


# ===========================================================================
# 4. Validator end-to-end tests  [ACTIVE]
# ===========================================================================

from intentguard.validator import (
    validate,
    _extract_intents,
    _serialise_call,
    _detect_injection,
    DEFAULT_THRESHOLD,
    LOW_CONF_FLOOR,
)

# Shared call fixtures
_CART_CALL = {
    "method": "POST", "endpoint": "/cart/items",
    "body": {"product": "shoes"}, "provenance": "box_a",
}
_INJECTED_CALL = {
    "method": "DELETE", "endpoint": "/account",
    "provenance": "box_b",
}


class TestValidator:
    # ------------------------------------------------------------------
    # Return structure
    # ------------------------------------------------------------------

    def test_return_structure(self):
        """validate() must return dict with 'verdicts', 'intents_extracted', 'threshold'."""
        result = validate("Add shoes to my cart", [_CART_CALL])
        assert "verdicts"          in result
        assert "intents_extracted" in result
        assert "threshold"         in result

    def test_verdicts_length_matches_calls(self):
        """verdicts list must have exactly len(api_calls) entries."""
        result = validate("Add shoes to cart", [_CART_CALL, _INJECTED_CALL])
        assert len(result["verdicts"]) == 2

    def test_verdict_entry_has_required_keys(self):
        """Each verdict entry must contain call_index, verdict, matched_intent, score, provenance."""
        result = validate("Add shoes to cart", [_CART_CALL])
        v = result["verdicts"][0]
        for key in ("call_index", "verdict", "matched_intent", "score", "provenance"):
            assert key in v, f"Missing key {key!r} in verdict"

    def test_verdict_call_index_sequential(self):
        """call_index in each verdict should equal its position in the list."""
        result = validate("Add shoes to cart", [_CART_CALL, _INJECTED_CALL])
        for i, v in enumerate(result["verdicts"]):
            assert v["call_index"] == i

    # ------------------------------------------------------------------
    # Approved verdict
    # ------------------------------------------------------------------

    def test_approved_verdict(self):
        """A call closely matching the user message should receive 'approved'."""
        result = validate(
            "Add shoes to my shopping cart",
            [{"method": "POST", "endpoint": "/cart/items",
              "body": {"product": "shoes"}, "provenance": "box_a"}],
            threshold=0.3,
        )
        assert result["verdicts"][0]["verdict"] == "approved"

    def test_approved_has_matched_intent(self):
        """An approved verdict must have a non-None matched_intent string."""
        result = validate(
            "Add shoes to my shopping cart",
            [{"method": "POST", "endpoint": "/cart/items",
              "body": {"product": "shoes"}, "provenance": "box_a"}],
            threshold=0.3,
        )
        assert result["verdicts"][0]["matched_intent"] is not None

    # ------------------------------------------------------------------
    # Injected verdict
    # ------------------------------------------------------------------

    def test_injected_verdict(self):
        """A box_b call with no intent match should receive 'injected'."""
        result = validate(
            "Add shoes to my cart",
            [{"method": "DELETE", "endpoint": "/account", "provenance": "box_b"}],
            threshold=0.9,
        )
        assert result["verdicts"][0]["verdict"] == "injected"

    def test_injected_has_no_matched_intent(self):
        """An injected verdict must have matched_intent == None."""
        result = validate(
            "Add shoes to my cart",
            [{"method": "DELETE", "endpoint": "/account", "provenance": "box_b"}],
            threshold=0.9,
        )
        assert result["verdicts"][0]["matched_intent"] is None

    def test_box_a_unmatched_is_not_injected(self):
        """An unmatched box_a call must NOT be 'injected'."""
        result = validate(
            "Add shoes to my cart",
            [{"method": "GET", "endpoint": "/admin/logs", "provenance": "box_a"}],
            threshold=0.99,
        )
        verdict = result["verdicts"][0]["verdict"]
        assert verdict != "injected", (
            f"box_a unmatched call should be orphaned/low_confidence, got {verdict!r}"
        )

    # ------------------------------------------------------------------
    # Low-confidence verdict
    # ------------------------------------------------------------------

    def test_low_confidence_verdict(self):
        """An unmatched box_a call with middling score should be 'low_confidence'."""
        result = validate(
            "Update my profile",
            [{"method": "PUT", "endpoint": "/profile", "provenance": "box_a"}],
            threshold=0.99,
            low_conf_floor=0.0,
        )
        verdict = result["verdicts"][0]["verdict"]
        assert verdict in ("low_confidence", "approved"), (
            f"Expected low_confidence (or approved), got {verdict!r}"
        )

    # ------------------------------------------------------------------
    # Orphaned verdict
    # ------------------------------------------------------------------

    def test_orphaned_verdict(self):
        """A box_a call scoring below low_conf_floor should be 'orphaned'."""
        result = validate(
            "Add shoes to my cart",
            [{"method": "GET", "endpoint": "/admin/system/logs",
              "provenance": "box_a"}],
            threshold=0.99,
            low_conf_floor=0.99,
        )
        assert result["verdicts"][0]["verdict"] == "orphaned"

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_empty_user_message_raises(self):
        """validate() should raise ValueError for an empty user_message."""
        with pytest.raises(ValueError):
            validate("", [_CART_CALL])

    def test_whitespace_message_raises(self):
        """validate() should raise ValueError for whitespace-only user_message."""
        with pytest.raises(ValueError):
            validate("   ", [_CART_CALL])

    def test_none_message_raises(self):
        """validate() should raise ValueError when user_message is None."""
        with pytest.raises(ValueError):
            validate(None, [_CART_CALL])  # type: ignore[arg-type]

    def test_empty_api_calls_raises(self):
        """validate() should raise ValueError for an empty api_calls list."""
        with pytest.raises(ValueError):
            validate("Add shoes to cart", [])

    def test_none_api_calls_raises(self):
        """validate() should raise ValueError when api_calls is None."""
        with pytest.raises(ValueError):
            validate("Add shoes to cart", None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # _extract_intents unit tests
    # ------------------------------------------------------------------

    def test_extract_single_intent(self):
        """A simple message without conjunctions should yield one intent."""
        intents = _extract_intents("Delete my account")
        assert len(intents) == 1

    def test_extract_compound_and(self):
        """'X and Y' should yield two intents."""
        intents = _extract_intents("Add shoes to cart and apply coupon SAVE10")
        assert len(intents) == 2

    def test_extract_semicolon(self):
        """Semicolon-separated phrases should split into multiple intents."""
        intents = _extract_intents("get order; cancel shipment")
        assert len(intents) >= 2

    def test_extract_fallback_short_message(self):
        """If splitting yields nothing >=4 chars, return the whole message."""
        msg = "Go"
        intents = _extract_intents(msg)
        assert intents == [msg]

    # ------------------------------------------------------------------
    # _serialise_call unit tests
    # ------------------------------------------------------------------

    def test_serialise_basic_endpoint(self):
        """Simple endpoint should produce space-separated tokens."""
        s, _ = _serialise_call({"method": "GET", "endpoint": "/orders"})
        assert "get" in s and "orders" in s

    def test_serialise_hyphenated_endpoint(self):
        """Hyphens in endpoint paths should become spaces."""
        s, _ = _serialise_call({"method": "POST", "endpoint": "/cart/apply-coupon"})
        assert "apply" in s and "coupon" in s

    def test_serialise_includes_param_value(self):
        """Meaningful param values should appear in the serialised string."""
        s, _ = _serialise_call({
            "method": "POST", "endpoint": "/cart/apply-coupon",
            "params": {"code": "SAVE10"},
        })
        assert "SAVE10" in s

    def test_serialise_missing_method_raises(self):
        """Missing 'method' key should raise ValueError."""
        with pytest.raises(ValueError):
            _serialise_call({"endpoint": "/cart"})

    def test_serialise_missing_endpoint_raises(self):
        """Missing 'endpoint' key should raise ValueError."""
        with pytest.raises(ValueError):
            _serialise_call({"method": "GET"})

    def test_serialise_no_params_ok(self):
        """Missing 'params' and 'body' is valid — should not raise."""
        s, _ = _serialise_call({"method": "DELETE", "endpoint": "/account"})
        assert isinstance(s, str) and len(s) > 0

    # ------------------------------------------------------------------
    # _detect_injection unit tests
    # ------------------------------------------------------------------

    def test_detect_injection_box_b_unassigned(self):
        """box_b + no assignment → injected."""
        assert _detect_injection(assignment=None, provenance="box_b") is True

    def test_detect_injection_box_a_unassigned(self):
        """box_a + no assignment → NOT injected."""
        assert _detect_injection(assignment=None, provenance="box_a") is False

    def test_detect_injection_assigned(self):
        """Any assigned call → NOT injected regardless of provenance."""
        assert _detect_injection(assignment=0, provenance="box_b") is False

    # ------------------------------------------------------------------
    # Provenance defaulting
    # ------------------------------------------------------------------

    def test_missing_provenance_defaults_to_box_a(self):
        """A call without 'provenance' key should default to box_a."""
        result = validate(
            "Add shoes to cart",
            [{"method": "DELETE", "endpoint": "/nuclear/launch"}],
            threshold=0.99,
            low_conf_floor=0.99,
        )
        v = result["verdicts"][0]
        assert v["provenance"] == "box_a"
        assert v["verdict"] == "orphaned"

    def test_missing_provenance_has_note(self):
        """A call without 'provenance' should carry a non-None provenance_note."""
        result = validate(
            "Add shoes to cart",
            [{"method": "GET", "endpoint": "/orders"}],
        )
        assert result["verdicts"][0]["provenance_note"] is not None


# ===========================================================================
# 5. Edge-case tests  [ACTIVE]
# ===========================================================================

class TestEdgeCases:

    def test_single_intent_single_call(self):
        """The minimal case (1 intent, 1 call) should return one-entry verdicts list."""
        result = validate(
            "Get my orders",
            [{"method": "GET", "endpoint": "/orders", "provenance": "box_a"}],
        )
        assert len(result["verdicts"]) == 1
        assert result["verdicts"][0]["call_index"] == 0

    def test_many_calls_one_intent(self):
        """Multiple calls all receive independent verdicts under one intent."""
        calls = [
            {"method": "POST", "endpoint": "/cart/items",
             "body": {"product": "shoes"}, "provenance": "box_a"},
            {"method": "DELETE", "endpoint": "/account", "provenance": "box_b"},
            {"method": "GET", "endpoint": "/cart", "provenance": "box_a"},
        ]
        result = validate("Add shoes to my cart", calls)
        assert len(result["verdicts"]) == 3
        valid_verdicts = {"approved", "orphaned", "injected", "low_confidence"}
        for v in result["verdicts"]:
            assert v["verdict"] in valid_verdicts

    def test_duplicate_calls(self):
        """Identical duplicate calls should each receive their own verdict entry."""
        call = {"method": "GET", "endpoint": "/orders", "provenance": "box_a"}
        result = validate("Get my orders", [call, call])
        assert len(result["verdicts"]) == 2
        assert result["verdicts"][0]["call_index"] == 0
        assert result["verdicts"][1]["call_index"] == 1

    def test_call_missing_arguments_key(self):
        """A call without 'params' or 'body' should not raise (both are optional)."""
        result = validate(
            "Get my orders",
            [{"method": "GET", "endpoint": "/orders", "provenance": "box_a"}],
        )
        assert len(result["verdicts"]) == 1

    def test_unicode_text(self):
        """Non-ASCII user messages should not raise — pipeline must handle unicode."""
        try:
            result = validate(
                "カートに靴を追加",   # Japanese: "Add shoes to cart"
                [{"method": "POST", "endpoint": "/cart/items",
                  "body": {"product": "shoes"}, "provenance": "box_a"}],
            )
            assert len(result["verdicts"]) == 1
        except Exception as exc:
            pytest.fail(f"Unicode user message raised unexpectedly: {exc}")

    def test_compound_intent_extracted(self):
        """'X and Y' message should produce two entries in intents_extracted."""
        result = validate(
            "Add shoes to my cart and apply coupon SAVE10",
            [{"method": "POST", "endpoint": "/cart/items",
              "body": {"product": "shoes"}, "provenance": "box_a"}],
        )
        assert len(result["intents_extracted"]) == 2

    def test_intents_extracted_in_result(self):
        """intents_extracted field should be a non-empty list of strings."""
        result = validate(
            "Get my order history",
            [{"method": "GET", "endpoint": "/orders", "provenance": "box_a"}],
        )
        extracted = result["intents_extracted"]
        assert isinstance(extracted, list)
        assert len(extracted) >= 1
        assert all(isinstance(i, str) for i in extracted)
