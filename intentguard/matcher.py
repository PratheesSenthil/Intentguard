"""
intentguard.matcher
===================
Implements 1-to-many intent-to-call matching with conflict resolution via the
Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).

Algorithm in detail
-------------------
Given M intents and N API calls:

Step 1 — Score matrix
    Build an (M × N) float32 matrix S where S[i, j] = score_pair(intent_i, call_j).

Step 2 — Threshold pass (column-wise)
    For every call column j, collect the set of intent rows that score >= threshold.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  claimants(j) = { i : S[i, j] >= threshold }                           │
    │                                                                         │
    │  |claimants(j)| == 0  →  call j is UNASSIGNED  (no intent qualifies)   │
    │  |claimants(j)| == 1  →  call j goes directly to that intent            │
    │                          (1:many allowed — no Hungarian needed)         │
    │  |claimants(j)| >= 2  →  call j is CONTESTED — added to conflict set   │
    └─────────────────────────────────────────────────────────────────────────┘

    Critically: an intent can sponsor many calls simultaneously in the
    single-claimant case.  Hungarian is invoked ONLY for contested calls.

Step 3 — Conflict resolution (Hungarian, contested calls only)
    Collect the union of all contested intents I_c and contested calls J_c.
    Extract the sub-matrix S[I_c, :][:, J_c].

    Run scipy.optimize.linear_sum_assignment(-sub_matrix) to maximise total
    alignment (the function minimises by default, so we negate).

    The algorithm returns a 1:1 pairing of (intent, call) across the sub-matrix.
    Each contested call receives its highest-scoring assigned intent.

    Any contested call whose assigned score is still below threshold is
    demoted to UNASSIGNED (edge-case: threshold may eliminate all candidates
    after the global optimum assignment).

Step 4 — Return
    {
        "assignments": { call_index → intent_index | None },
        "score_matrix": the full M×N numpy array,
        "threshold":    the threshold value used,
    }

    Verdict labels ("approved" / "orphaned" / "injected" / "low_confidence")
    are NOT assigned here — that is validator.py's responsibility.

Design notes
------------
* _build_score_matrix() caches scores per unique (intent, call) text pair
  using a simple dict so repeated strings don't trigger redundant model calls.
* _resolve_conflicts() is a pure function — it takes the full matrix plus
  index lists and returns only the conflict assignments.
* match() is the single public entry-point; it validates inputs, delegates to
  the helpers, and assembles the structured return dict.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_string_list(lst: object, name: str) -> list[str]:
    """Validate and return a non-empty list of non-empty strings.

    Parameters
    ----------
    lst : object
        The value to validate.
    name : str
        Parameter name to include in error messages.

    Returns
    -------
    list[str]
        The validated list (same object, not a copy).

    Raises
    ------
    ValueError
        On any validation failure.
    """
    if lst is None:
        raise ValueError(f"match(): {name} must not be None.")
    if not isinstance(lst, list):
        raise ValueError(
            f"match(): {name} must be a list, got {type(lst).__name__!r}."
        )
    if len(lst) == 0:
        raise ValueError(
            f"match(): {name} must not be empty — provide at least one string."
        )
    for idx, item in enumerate(lst):
        if not isinstance(item, (str, dict)):
            raise ValueError(
                f"match(): {name}[{idx}] must be a str or dict, got {type(item).__name__!r}."
            )
        if isinstance(item, str) and not item.strip():
            raise ValueError(
                f"match(): {name}[{idx}] must not be empty or whitespace-only."
            )
    return lst  # type: ignore[return-value]


def _build_score_matrix(intents: list[str], calls: list[str], domain_config: dict | None = None) -> np.ndarray:
    """Construct the (M × N) score matrix using ``scorer.score_pair``.

    Uses a per-call cache keyed on ``(intent_text, call_text)`` to avoid
    re-encoding the same text pair more than once (relevant when any intent
    or call string appears multiple times in the input lists).

    Parameters
    ----------
    intents : list[str]
        List of M intent strings.  Assumed already validated.
    calls : list[str]
        List of N call-description strings.  Assumed already validated.
    domain_config : dict, optional
        Loaded domain configuration dictionary.

    Returns
    -------
    np.ndarray
        Shape ``(M, N)``, dtype ``float32``.
        Entry ``[i, j]`` is the composite alignment score between intent *i*
        and call *j*.
    """
    from intentguard.scorer import score_pair  # local import avoids circular dep

    M, N = len(intents), len(calls)
    matrix = np.zeros((M, N), dtype=np.float32)

    # Simple memoisation: cache score_pair results for unique text pairs.
    # This is O(unique_pairs) model calls rather than O(M*N) in the worst case.
    _cache: dict[tuple[str, str], float] = {}

    for i, intent in enumerate(intents):
        for j, call in enumerate(calls):
            call_key = str(call) if isinstance(call, dict) else call
            key = (intent, call_key)
            if key not in _cache:
                _cache[key] = score_pair(intent, call, domain_config=domain_config)
            matrix[i, j] = _cache[key]

    return matrix


def _resolve_conflicts(
    score_matrix: np.ndarray,
    contested_intents: list[int],
    contested_calls: list[int],
    threshold: float,
) -> dict[int, int | None]:
    """Run the Hungarian algorithm on the contested sub-matrix.

    Called ONLY when at least one call is claimed by two or more intents.

    The sub-matrix is formed by rows ``contested_intents`` and columns
    ``contested_calls`` of ``score_matrix``.  ``linear_sum_assignment`` is run
    on the **negated** sub-matrix to turn the minimisation into a maximisation.

    The algorithm returns a 1:1 pairing — each contested call gets at most one
    intent.  Calls whose winning score is still below *threshold* are demoted
    to unassigned (``None``).

    Parameters
    ----------
    score_matrix : np.ndarray
        Full (M × N) score matrix.
    contested_intents : list[int]
        Sorted list of row indices involved in at least one conflict.
    contested_calls : list[int]
        Sorted list of column indices that are contested (≥2 claimants).
    threshold : float
        Minimum score for a winning assignment to count as "approved".
        Contested calls whose winning score falls below this become unassigned.

    Returns
    -------
    dict[int, int | None]
        Maps each contested call index → winning intent index (or None if the
        best possible assignment still scores below threshold).

    Notes
    -----
    Why only contested calls go through Hungarian
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The 1:many requirement means one intent can validly approve many calls.
    Running the full matrix through Hungarian would impose a 1:1 constraint
    globally, wrongly demoting legitimately multiply-approved calls.
    By scoping Hungarian to genuinely contested columns only, we preserve
    the 1:many semantics while still resolving ambiguous assignments optimally.
    """
    # Build the sub-matrix: rows = contested intents, cols = contested calls
    # Shape: (len(contested_intents), len(contested_calls))
    sub = score_matrix[np.ix_(contested_intents, contested_calls)]

    # linear_sum_assignment minimises — negate to maximise alignment.
    row_ind, col_ind = linear_sum_assignment(-sub)

    # Invert local indices back to global indices.
    # row_ind[k] → index into contested_intents → global intent index
    # col_ind[k] → index into contested_calls   → global call index
    results: dict[int, int | None] = {}

    for local_row, local_col in zip(row_ind, col_ind):
        global_call   = contested_calls[local_col]
        global_intent = contested_intents[local_row]
        winning_score = float(sub[local_row, local_col])

        if winning_score >= threshold:
            results[global_call] = global_intent
        else:
            # Even the best possible assignment falls below threshold.
            results[global_call] = None

    # Any contested call NOT covered by the assignment (can happen when
    # len(contested_intents) < len(contested_calls) — the algorithm can only
    # make min(rows, cols) assignments) → mark as unassigned.
    for global_call in contested_calls:
        if global_call not in results:
            results[global_call] = None

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match(
    intents: list[str],
    calls: list[str],
    call_verbs: list[str | None] | None = None,
    threshold: float = 0.55,
    score_matrix: np.ndarray | None = None,
    domain_config: dict | None = None,
) -> dict:
    """Match each API call to at most one intent, resolving conflicts via Hungarian.

    Implements 1-to-many matching: a single intent can approve many calls, but
    each call is assigned to at most one intent.  The Hungarian algorithm is
    used **only** to break ties when a single call is claimed by two or more
    intents simultaneously.

    Parameters
    ----------
    intents : list[str]
        M natural-language intent strings.  Must be non-empty; each element
        must be a non-empty, non-whitespace-only string.
    calls : list[str]
        N API-call description strings to match.  Same constraints as intents.
    threshold : float, optional
        Minimum composite score required for an intent to "claim" a call.
        Default: ``0.50``.  Pass a higher value (e.g. 0.60) for stricter
        matching; lower for more permissive matching.
    score_matrix : np.ndarray or None, optional
        Pre-computed (M × N) score matrix.  If supplied, ``_build_score_matrix``
        is skipped entirely — useful when validator.py has already computed
        embeddings and wants to avoid redundant encoding.
        Must have shape ``(len(intents), len(calls))`` and dtype float32.
    domain_config : dict, optional
        Loaded domain configuration dictionary.

    Returns
    -------
    dict with three keys:

        "assignments" : dict[int, int | None]
            Maps call index (0 … N-1) to the index of its winning intent, or
            ``None`` if no intent scored at or above *threshold* for that call.

        "score_matrix" : np.ndarray
            The full (M × N) score matrix (float32).  Passed through so
            validator.py can inspect individual scores for the
            "low_confidence" vs "orphaned" distinction without recomputing.

        "threshold" : float
            The threshold value that was applied, for downstream reference.

    Raises
    ------
    ValueError
        If *intents* or *calls* fails validation (empty list, None, contains
        empty/whitespace-only strings, or non-string elements).
    ValueError
        If a pre-computed *score_matrix* has the wrong shape.

    Examples
    --------
    >>> result = match(["Book a flight"], ["book_flight dest=Paris"])
    >>> result["assignments"][0]  # intent index 0, or None
    0
    """
    # --- Input validation ---------------------------------------------------
    _validate_string_list(intents, "intents")
    _validate_string_list(calls,   "calls")

    M, N = len(intents), len(calls)

    # --- Score matrix -------------------------------------------------------
    if score_matrix is None:
        S = _build_score_matrix(intents, calls, domain_config=domain_config)
    else:
        S = np.asarray(score_matrix, dtype=np.float32)
        if S.shape != (M, N):
            raise ValueError(
                f"Pre-computed score_matrix has shape {S.shape}, "
                f"expected ({M}, {N}) for {M} intents × {N} calls."
            )

    # --- Threshold pass (column-wise) ---------------------------------------
    # assignments[j] = winning intent index, or None
    assignments: dict[int, int | None] = {}

    # Track which calls are contested (≥2 claimants above threshold)
    contested_calls: list[int]    = []
    contested_intents_set: set[int] = set()

    for j in range(N):
        claimants = [i for i in range(M) if S[i, j] >= threshold]

        if len(claimants) == 0:
            # No intent qualifies — unassigned.
            assignments[j] = None

        elif len(claimants) == 1:
            # Unambiguous 1:many assignment — no conflict resolution needed.
            # One intent can own many such calls independently.
            assignments[j] = claimants[0]

        else:
            # Genuine conflict: two or more intents compete for this call.
            contested_calls.append(j)
            contested_intents_set.update(claimants)
            # assignments[j] set later by _resolve_conflicts

    # --- Conflict resolution (Hungarian, contested only) --------------------
    if contested_calls:
        contested_intents = sorted(contested_intents_set)

        conflict_assignments = _resolve_conflicts(
            score_matrix=S,
            contested_intents=contested_intents,
            contested_calls=contested_calls,
            threshold=threshold,
        )
        assignments.update(conflict_assignments)

    # --- Assemble result ----------------------------------------------------
    return {
        "assignments":  assignments,
        "score_matrix": S,
        "threshold":    threshold,
    }
