"""
intentguard.validator
=====================
Top-level orchestrator that wires together embedder → scorer → matcher and
returns a per-call verdict for each API call produced by an LLM agent.

Pipeline
--------
::

    user_message (str)
        │
        ▼  _extract_intents()
        │  Split compound requests into discrete intent strings
        │  using lightweight conjunction/punctuation heuristics.
        │
        ▼  _serialise_call()  (per call)
        │  Convert structured API-call dict (method + endpoint + params/body)
        │  into a clean natural-language-ish string suitable for embedding.
        │
        ▼  matcher.match(intents, serialised_calls, threshold)
        │  Builds the score matrix, runs threshold pass + Hungarian conflict
        │  resolution, returns {assignments, score_matrix, threshold}.
        │
        ▼  _detect_injection()  (per unassigned call)
        │  Applies provenance-based injection detection:
        │  assignment is None AND provenance == "box_b" → "injected"
        │
        ▼  Verdict assembly
           For each call:
             assigned                    → "approved"
             unassigned + box_b          → "injected"
             unassigned + box_a + score ≥ low_conf_floor  → "low_confidence"
             unassigned + box_a + score < low_conf_floor  → "orphaned"

Verdict semantics
-----------------
"approved"
    The call is semantically sanctioned by the user's stated intent and
    scored above *threshold*.

"low_confidence"
    No intent scored above *threshold*, but the best score is ≥ *low_conf_floor*.
    The call may be legitimate — flag for human review.

"orphaned"
    No intent scored at or above *low_conf_floor*.  The call has no visible
    connection to the user's request; likely an over-eager agent action.

"injected"
    No intent matched AND the call originated from external/retrieved content
    (provenance == "box_b").  Treated as a likely prompt-injection artefact.

Provenance model
----------------
Each API call dict may carry a "provenance" key:
    "box_a"  – originated from the user's own request context (default)
    "box_b"  – originated from external / retrieved / tool-output context
               (e.g. a RAG document, function output, plugin response)

If "provenance" is absent the call is treated as "box_a" — but a warning
string is recorded in the verdict's "provenance_note" field so the absence
is visible to callers rather than silent.

Constants
---------
DEFAULT_THRESHOLD : float = 0.50
    Minimum composite score for an intent to "approve" a call.
LOW_CONF_FLOOR : float = 0.35
    Scores in [LOW_CONF_FLOOR, threshold) → "low_confidence".
    Scores below LOW_CONF_FLOOR → "orphaned".
"""

from __future__ import annotations

import re
import warnings

# ---------------------------------------------------------------------------
# Thresholds (public so callers can inspect / override)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.55
LOW_CONF_FLOOR:    float = 0.25

# ---------------------------------------------------------------------------
# Patterns used by _serialise_call
# ---------------------------------------------------------------------------

# Matches UUIDs, numeric IDs, hex strings — strip these from the text because
# they add embedding noise without semantic content.
_ID_PATTERN = re.compile(
    r"\b(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # UUID
    r"|[0-9a-f]{24,}"   # long hex (Mongo ObjectId etc.)
    r"|[0-9]{6,}"       # long numeric ID
    r")\b",
    re.IGNORECASE,
)

# Characters that act as word separators in endpoint paths / param names.
_SEP_PATTERN = re.compile(r"[/_\-]+")

# Leading/trailing path slashes.
_SLASH_STRIP = re.compile(r"^/+|/+$")


def _clean_token(token: str) -> str:
    """Lowercase a token and drop ID-like substrings."""
    token = _ID_PATTERN.sub("", token).strip()
    return token.lower()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_intents(user_message: str) -> list[str]:
    """Decompose a user message into a list of discrete intent strings.

    Strategy (heuristic, v1)
    ------------------------
    Split on conjunctions and sentence-boundary signals that typically
    separate distinct actions in natural language:

        - " and "       (most common in compound requests)
        - " also "      ("I'd like to also …")
        - " as well as "
        - ". " / ".\n"  (sentence boundary)
        - ";"           (explicit list separator)
        - ", and " / ", then "

    Each fragment is stripped and must be at least 4 characters long
    (filters out empty/trivial splits like "I" or "a").
    If splitting produces zero usable fragments, the whole message is
    returned as a single intent — this is the guaranteed safe fallback.

    Parameters
    ----------
    user_message : str
        Raw user input.  Must be a non-empty, non-whitespace string
        (caller is responsible for this validation).

    Returns
    -------
    list[str]
        One or more intent strings, each ≥ 4 characters.

    Examples
    --------
    >>> _extract_intents("Update my cart and apply coupon SAVE10")
    ['Update my cart', 'apply coupon SAVE10']
    >>> _extract_intents("Delete my account")
    ['Delete my account']
    """
    # Split on common compound-intent separators.
    # Order matters: longer/more-specific patterns first.
    split_pattern = re.compile(
        r"\s+and\s+|\s+also\s+|\s+as\s+well\s+as\s+"
        r"|[;]\s*"
        r"|[.]\s+"              # ". " sentence boundary
        r"|,\s+and\s+|,\s+then\s+",
        re.IGNORECASE,
    )
    raw_fragments = split_pattern.split(user_message)
    fragments = [f.strip() for f in raw_fragments if f.strip() and len(f.strip()) >= 4]

    # Guaranteed fallback.
    if not fragments:
        return [user_message.strip()]

    return fragments


def _serialise_call(api_call: dict, domain_config: dict | None = None) -> tuple[str, str | None]:
    for key in ("method", "endpoint"):
        if key not in api_call:
            raise ValueError(
                f"_serialise_call(): api_call is missing required key {key!r}. "
                f"Got keys: {list(api_call.keys())}"
            )

    method   = str(api_call["method"]).lower()
    endpoint = str(api_call["endpoint"])

    endpoint = _SLASH_STRIP.sub("", endpoint)
    
    url_segments = endpoint.split('/')
    last_url_segment = url_segments[-1] if url_segments else ""
    
    ep_tokens = _SEP_PATTERN.split(endpoint)
    ep_tokens = [_clean_token(t) for t in ep_tokens if _clean_token(t)]

    
    derived_verb = None
    if ep_tokens and ep_tokens[0] == 'api':
        ep_tokens = ep_tokens[1:]

    if ep_tokens and domain_config and last_url_segment:
        from intentguard.scorer import _normalise_entity
        
        last_raw = last_url_segment
        v_can = domain_config.get('_verb_canonical', {})
        e_voc = set(domain_config.get('entity_vocab', []))
        
        last_segment = _normalise_entity(last_raw.replace('-', ' '))
        last_lower = last_raw.lower()
        
        if last_lower in v_can:
            derived_verb = v_can[last_lower]
        elif last_segment in e_voc:
            if method == 'post': derived_verb = 'create'
            elif method == 'get': derived_verb = 'get'
            elif method == 'delete': derived_verb = 'delete'
            elif method in ('patch', 'put'): derived_verb = 'update'
            else: derived_verb = method
        else:
            words = last_raw.split('-')
            found_verb = None
            for w in words:
                norm_w = _normalise_entity(w)
                if norm_w in v_can:
                    found_verb = v_can[norm_w]
                    break
                    
            if found_verb: 
                derived_verb = found_verb
            elif last_segment in v_can: 
                derived_verb = v_can[last_segment]
            elif method in ('get', 'delete'): 
                derived_verb = method
            elif method in ('patch', 'put'): 
                derived_verb = 'update'
            else:
                derived_verb = None
            
    parts: list[str] = []
    if derived_verb: parts.append(derived_verb)
    elif method != 'post': parts.append(method)
    parts.extend(ep_tokens)

    for bag_key in ("params", "body"):
        bag = api_call.get(bag_key)
        if not bag or not isinstance(bag, dict):
            continue
        for v in bag.values():
            v_str = str(v).strip()
            if not v_str or _ID_PATTERN.fullmatch(v_str):
                continue
            # Keep only if number or single word of at most 20 chars
            if v_str.isdigit() or (" " not in v_str and len(v_str) <= 20):
                parts.append(v_str)

    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip(), derived_verb

def _detect_injection(
    assignment: int | None,
    provenance: str,
) -> bool:
    """Determine whether an unassigned call should be labelled "injected".

    The intentguard injection model is provenance-based rather than
    purely heuristic:

    - If ``assignment is None`` (no intent approved the call) **AND**
      ``provenance == "box_b"`` (call originated from external/retrieved
      content, not the user's own request): the call is flagged as injected.

    - If ``assignment is None`` **AND** ``provenance == "box_a"``: the call
      is user-originated but unclaimed — this is "orphaned" or
      "low_confidence", NOT injected.  An agent might legitimately make
      calls the user vaguely implied; conflating these with injections
      would produce too many false positives.

    Parameters
    ----------
    assignment : int | None
        The matched intent index, or None if no intent approved this call.
    provenance : str
        "box_a" (user-request context) or "box_b" (external/retrieved context).

    Returns
    -------
    bool
        True iff the call should receive the "injected" verdict.
    """
    return assignment is None and provenance == "box_b"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    user_message: str,
    api_calls: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    low_conf_floor: float = LOW_CONF_FLOOR,
    domain: str | dict = "ecommerce",
    trusted_values: list[str] = None,
    grounding: bool = True,
) -> dict:
    """Validate whether each API call is semantically sanctioned by the user.

    Orchestrates the full intentguard pipeline:
    intent extraction → call serialisation → scoring → matching → verdict.

    Parameters
    ----------
    user_message : str
        The raw message the user sent to the LLM agent.
        Must be a non-empty, non-whitespace-only string.
    api_calls : list[dict]
        API call dicts produced by the LLM agent.  Each must contain at
        minimum "method" and "endpoint" keys.  An optional "provenance"
        key ("box_a" | "box_b") controls injection detection; defaults to
        "box_a" with a note recorded in the verdict if absent.
    threshold : float, optional
        Minimum composite score for a call to be approved.
        Default: ``DEFAULT_THRESHOLD`` (0.50).
    low_conf_floor : float, optional
        Scores in [low_conf_floor, threshold) → "low_confidence".
        Scores below low_conf_floor → "orphaned".
        Default: ``LOW_CONF_FLOOR`` (0.35).

    Returns
    -------
    dict with three keys:

        "verdicts" : list[dict]
            One entry per api_call, in input order::

                {
                    "call_index":     int,
                    "verdict":        str,   # "approved"|"orphaned"|
                                            # "injected"|"low_confidence"
                    "matched_intent": str | None,   # text of winning intent
                    "score":          float,         # best composite score
                    "provenance":     str,           # "box_a" or "box_b"
                    "provenance_note": str | None,   # warning if key was absent
                    "serialised_call": str,          # what was embedded
                }

        "intents_extracted" : list[str]
            The intent strings derived from *user_message*.

        "threshold" : float
            The threshold value that was applied.

    Raises
    ------
    ValueError
        If *user_message* is None, empty, or whitespace-only.
    ValueError
        If *api_calls* is None or an empty list.
    ValueError
        If any call dict is missing required keys (propagated from
        ``_serialise_call()``).

    Examples
    --------
    >>> result = validate(
    ...     user_message="Add shoes to my cart",
    ...     api_calls=[
    ...         {"method": "POST", "endpoint": "/cart/items",
    ...          "body": {"product": "shoes"}, "provenance": "box_a"},
    ...         {"method": "DELETE", "endpoint": "/account",
    ...          "provenance": "box_b"},
    ...     ],
    ... )
    >>> result["verdicts"][0]["verdict"]
    'approved'
    >>> result["verdicts"][1]["verdict"]
    'injected'
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if user_message is None:
        raise ValueError("validate(): user_message must not be None.")
    if not isinstance(user_message, str):
        raise ValueError(
            f"validate(): user_message must be a str, got {type(user_message).__name__!r}."
        )
    if not user_message.strip():
        raise ValueError(
            "validate(): user_message must not be empty or whitespace-only."
        )
    if api_calls is None:
        raise ValueError("validate(): api_calls must not be None.")
    if not isinstance(api_calls, list):
        raise ValueError(
            f"validate(): api_calls must be a list, got {type(api_calls).__name__!r}."
        )
    if len(api_calls) == 0:
        raise ValueError(
            "validate(): api_calls must not be empty — provide at least one call."
        )

    # ------------------------------------------------------------------
    # 2. Extract intents
    # ------------------------------------------------------------------
    intents = _extract_intents(user_message)

    # ------------------------------------------------------------------
    # 2.5. Load Domain Configuration (moved up for serialise_call)
    # ------------------------------------------------------------------
    if isinstance(domain, dict):
        domain_config = domain
    else:
        from intentguard.domain_config import load_domain_config
        domain_config = load_domain_config(domain)
        
    # Context carry-over
    import re
    from intentguard.scorer import _extract_entities, _tokenise
    evoc = domain_config.get('_entity_set', frozenset())
    intent_contexts = []
    if len(intents) > 1:
        fragment_ctxs = []
        for intent in intents:
            toks = _tokenise(intent)
            ents = list(_extract_entities(toks, intent, evoc))
            words = re.findall(r'\b[A-Za-z0-9]+\b', intent)
            codes = [w for w in words if any(c.isdigit() for c in w)]
            fragment_ctxs.append(set(ents + codes))
            
        for i, intent in enumerate(intents):
            other_ctx = set()
            for j, ctx in enumerate(fragment_ctxs):
                if i != j:
                    other_ctx.update(ctx)
            other_ctx = other_ctx - fragment_ctxs[i]
            if other_ctx:
                intent_contexts.append(intent + " " + " ".join(sorted(other_ctx)))
            else:
                intent_contexts.append(intent)
    else:
        intent_contexts = list(intents)

    # ------------------------------------------------------------------
    # 3. Serialise calls + extract provenance
    # ------------------------------------------------------------------
    serialised: list[str] = []
    call_verbs: list[str | None] = []
    provenances: list[str] = []
    provenance_notes: list[str | None] = []

    for idx, call in enumerate(api_calls):
        call_str, c_verb = _serialise_call(call, domain_config=domain_config)
        serialised.append(call_str)
        call_verbs.append(c_verb)

        raw_prov = call.get("provenance")
        if raw_prov is None:
            provenances.append("box_a")
            provenance_notes.append(
                f"call[{idx}]: 'provenance' key absent — defaulted to 'box_a'."
            )
        elif raw_prov not in ("box_a", "box_b"):
            # Unknown provenance — treat conservatively as box_a.
            provenances.append("box_a")
            provenance_notes.append(
                f"call[{idx}]: unrecognised provenance {raw_prov!r} — defaulted to 'box_a'."
            )
        else:
            provenances.append(raw_prov)
            provenance_notes.append(None)

    # ------------------------------------------------------------------
    # 4. Match
    # ------------------------------------------------------------------
    from intentguard.matcher import match  # local import avoids circular dep
    
    import numpy as np
    from intentguard.embedder import encode
    from intentguard.scorer import _cosine_score, _verb_score, _entity_score, ALPHA, BETA, GAMMA
    
    M, N = len(intents), len(serialised)
    S = np.zeros((M, N), dtype=np.float32)
    for i, intent in enumerate(intents):
        v_intent = encode(intent_contexts[i])
        for j, call_str in enumerate(serialised):
            v_call = encode(call_str)
            b_score = _cosine_score(v_intent, v_call)
            v_score = _verb_score(intent, call_str, domain_config=domain_config, extracted_call_verb=call_verbs[j])
            e_score = _entity_score(intent, call_str, domain_config=domain_config)
            S[i, j] = ALPHA * b_score + BETA * v_score + GAMMA * e_score

    match_result = match(
        intents, 
        serialised, 
        call_verbs=call_verbs,
        threshold=threshold, 
        score_matrix=S,
        domain_config=domain_config
    )
    assignments:  dict[int, int | None] = match_result["assignments"]
    score_matrix = match_result["score_matrix"]   # shape (M, N)

    # ------------------------------------------------------------------
    # 5. Assemble verdicts
    # ------------------------------------------------------------------
    verdicts: list[dict] = []

    for j, call in enumerate(api_calls):
        assignment   = assignments[j]
        provenance   = provenances[j]
        prov_note    = provenance_notes[j]

        # Best score for this call across all intents.
        best_score = float(score_matrix[:, j].max())

        # Determine verdict.
        veto_fired = None
        if assignment is not None:
            verdict = "approved"
            matched_intent = intents[assignment]
            
            from intentguard.scorer import _extract_verb, _tokenise
            v_can = domain_config.get('_verb_canonical', {})
            intent_verb = _extract_verb(_tokenise(matched_intent), v_can)
            call_verb = call_verbs[j]
            call_method = str(call.get("method", "")).lower()
            
            compatible_groups = domain_config.get('compatible', [])
            def is_compatible(v1, v2):
                if v1 == v2: return True
                for grp in compatible_groups:
                    if v1 in grp and v2 in grp: return True
                return False
                
            def in_compatible_list(v1, v2):
                for grp in compatible_groups:
                    if v1 in grp and v2 in grp and v1 != v2: return True
                return False

            # STEP 1 - Verb-mismatch veto
            if intent_verb is not None and call_verb is not None and not is_compatible(intent_verb, call_verb):
                verdict = "low_confidence" if best_score >= low_conf_floor else "orphaned"
                veto_fired = "verb_mismatch"
                
            # STEP 2 - Read-only rule
            elif intent_verb == 'get' and call_method != 'get' and not in_compatible_list('get', call_verb):
                verdict = "low_confidence" if best_score >= low_conf_floor else "orphaned"
                veto_fired = "read_only"
        elif _detect_injection(assignment, provenance):
            verdict = "injected"
            matched_intent = None
        elif best_score >= low_conf_floor:
            verdict = "low_confidence"
            matched_intent = None
        else:
            verdict = "orphaned"
            matched_intent = None

        best_i = assignment if assignment is not None else int(np.argmax(score_matrix[:, j]))
        expl_intent = intents[best_i]
        
        # recompute component scores for the explanation
        v_intent_emb = encode(intent_contexts[best_i])
        v_call_emb = encode(serialised[j])
        b_score = float(_cosine_score(v_intent_emb, v_call_emb))
        c_verb = call_verbs[j]
        v_score = float(_verb_score(expl_intent, serialised[j], domain_config=domain_config, extracted_call_verb=c_verb))
        e_score = float(_entity_score(expl_intent, serialised[j], domain_config=domain_config))
        
        from intentguard.scorer import _extract_verb, _tokenise
        v_can = domain_config.get('_verb_canonical', {})
        intent_verb_grp = _extract_verb(_tokenise(expl_intent), v_can)
        call_verb_grp = c_verb
        
        reason = "Unknown reason."
        if verdict == "approved":
            reason = f"Call matched intent '{expl_intent}' with score {best_score:.3f} >= threshold {threshold}."
        elif verdict == "injected":
            reason = f"Call came from unverified context (provenance {provenance}) without strong intent support."
        elif veto_fired == "verb_mismatch":
            reason = f"Call had sufficient score ({best_score:.3f}) but was vetoed due to incompatible verbs: '{intent_verb_grp}' vs '{call_verb_grp}'."
        elif veto_fired == "read_only":
            reason = f"Call had sufficient score ({best_score:.3f}) but was vetoed because intent verb is 'get' and call verb is not read-only."
        elif verdict == "low_confidence":
            reason = f"Call score {best_score:.3f} was below threshold {threshold} but above low confidence floor {low_conf_floor}."
        elif verdict == "orphaned":
            reason = f"Call score {best_score:.3f} was below the lowest confidence floor {low_conf_floor}."

        verb_relation = "unknown"
        if intent_verb_grp is not None and call_verb_grp is not None:
            if intent_verb_grp == call_verb_grp:
                verb_relation = "same_group"
            else:
                comp = False
                for grp in domain_config.get('compatible', []):
                    if intent_verb_grp in grp and call_verb_grp in grp:
                        comp = True
                        break
                verb_relation = "compatible" if comp else "different"

        if grounding:
            from intentguard.grounding import process_grounding
            all_grounded, grounding_objs = process_grounding(user_message, call, trusted_values)
            if verdict == "approved" and not all_grounded:
                verdict = "low_confidence"
                veto_fired = "value_grounding"
                bad_val = next((o["value"] for o in grounding_objs if not o["grounded"]), None)
                reason = f"Value {bad_val} in the call does not appear in the request."
                matched_intent = None
        else:
            grounding_objs = None

        explanation = {
            "closest_intent": expl_intent,
            "cosine_score": round(b_score, 3),
            "verb_score": round(v_score, 3),
            "entity_score": round(e_score, 3),
            "weighted_total": round(best_score, 6),
            "intent_verb_group": intent_verb_grp,
            "call_verb_group": call_verb_grp,
            "verb_relation": verb_relation,
            "veto_fired": veto_fired,
            "threshold_used": threshold,
            "provenance": provenance,
            "reason": reason
        }
        
        if grounding:
            explanation["grounding"] = grounding_objs

        verdicts.append({
            "call_index":      j,
            "verdict":         verdict,
            "matched_intent":  matched_intent,
            "score":           round(best_score, 6),
            "provenance":      provenance,
            "provenance_note": prov_note,
            "serialised_call": serialised[j],
            "explanation":     explanation
        })

    # ------------------------------------------------------------------
    # 6. Return structured result
    # ------------------------------------------------------------------
    return {
        "verdicts":          verdicts,
        "intents_extracted": intents,
        "threshold":         threshold,
    }
