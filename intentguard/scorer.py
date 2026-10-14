"""
intentguard.scorer
==================
Computes a composite semantic similarity score between a user-intent string
and an API-call description string.

Scoring formula
---------------
The final score *s* is a weighted combination of three sub-scores:

    s = α · cosine_sim + β · verb_match + γ · entity_overlap

where:

    cosine_sim    (float in [-1, 1] → clamped to [0, 1])
        Dot product of the two L2-normalised embedding vectors produced by
        ``embedder.encode()``.  Captures broad semantic relatedness.

    verb_match    (float in {0.0, 1.0})
        Whether the dominant action verb in *intent_text* aligns with the
        dominant verb found in *call_text*, using a hardcoded synonym map.
        Guards against cosine being fooled by topically related but
        opposite-action calls (e.g. "cancel order" vs "create_order").

    entity_overlap (float in [0, 1])
        Jaccard similarity of domain-relevant entities found in both texts.
        Tuned for an e-commerce domain but easy to extend.
        Returns 0.0 — not 1.0 — when both entity sets are empty.

Weight defaults (must sum to 1.0 — enforced by assertion at module load):
    alpha (α) = 0.60
    beta  (β) = 0.25
    gamma (γ) = 0.15

Design notes
------------
* All private helpers are pure functions (no side effects, no global state).
* _cosine_score() takes pre-computed numpy vectors (not raw strings) so it
  can be called efficiently from matcher._build_score_matrix() without
  re-encoding the same text repeatedly.
* score_pair() is the convenient public entry point that handles encoding.
"""

from __future__ import annotations

import re
import string

import numpy as np

# ---------------------------------------------------------------------------
# Weight hyperparameters
# ---------------------------------------------------------------------------
ALPHA: float = 0.60   # weight for cosine similarity
BETA: float  = 0.25   # weight for verb-match bonus
GAMMA: float = 0.15   # weight for entity-overlap bonus

# Enforce at import time — changing one weight without the others will be
# caught immediately rather than silently producing wrong scores.
assert abs(ALPHA + BETA + GAMMA - 1.0) < 1e-9, (
    f"ALPHA + BETA + GAMMA must equal 1.0, got {ALPHA + BETA + GAMMA}"
)

# ---------------------------------------------------------------------------
# Stop-words (domain agnostic)
# ---------------------------------------------------------------------------
# Stop-words to skip even if they appear in the entity vocab check
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "but", "not", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "i", "my", "me", "you", "your",
    "it", "its", "we", "our", "they", "their", "this", "that", "all",
    "also", "just", "please", "need", "want", "like", "get", "make",
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace and underscores."""
    # Replace underscores (common in API call names like "book_flight") with spaces
    text = text.replace("_", " ")
    # Strip punctuation except hyphens (preserve "e-commerce")
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.ASCII)
    return [tok.lower() for tok in text.split() if tok]


def _cosine_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two pre-computed L2-normalised vectors.

    Because embedder.encode() guarantees unit-norm output, cosine similarity
    reduces to a dot product.  This function additionally verifies that
    both input vectors are in fact normalised (to within float32 tolerance),
    so a future caller cannot silently pass in un-normalised embeddings and
    get wrong scores.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Shape ``(D,)``, dtype float32.  Must be L2-unit-normalised
        (norm within 1e-3 of 1.0).

    Returns
    -------
    float
        Cosine similarity clamped to ``[-1.0, 1.0]``.

    Raises
    ------
    ValueError
        If either vector has a norm that deviates from 1.0 by more than 1e-3.
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    tol = 1e-3  # generous tolerance for float32 precision loss
    if abs(norm_a - 1.0) > tol:
        raise ValueError(
            f"vec_a is not L2-normalised: norm={norm_a:.6f} (expected ~1.0). "
            "Pass the output of embedder.encode() or call encode(..., normalize_embeddings=True)."
        )
    if abs(norm_b - 1.0) > tol:
        raise ValueError(
            f"vec_b is not L2-normalised: norm={norm_b:.6f} (expected ~1.0). "
            "Pass the output of embedder.encode() or call encode(..., normalize_embeddings=True)."
        )
    # Dot product == cosine similarity for unit vectors.
    # Clamp to [-1, 1] to absorb float32 drift (e.g. 1.0000001).
    raw = float(np.dot(vec_a, vec_b))
    return float(np.clip(raw, -1.0, 1.0))


def _extract_verb(tokens: list[str], verb_canonical: dict[str, str]) -> str | None:
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in verb_canonical: return verb_canonical[bigram]
    for tok in tokens:
        if tok in verb_canonical: return verb_canonical[tok]
    return None



def _verb_score(intent_text: str, call_text: str, domain_config: dict | None = None, extracted_call_verb: str | None = None, call_verb_provided: bool = False) -> float:
    """Binary verb-alignment score between intent and call texts.

    Strategy
    --------
    1. Tokenise both strings (lowercase, strip punctuation, split on
       whitespace and underscores so ``"book_flight"`` → ``["book", "flight"]``).
    2. Scan each token list for the first token that appears in the verb
       synonym map and return its canonical form.
    3. Return 1.0 if both canonical forms are the same (or are both None
       and therefore "no recognisable verb in either text" — a safe tie),
       else 0.0.

    Parameters
    ----------
    intent_text, call_text : str
    domain_config : dict, optional
        Loaded domain configuration mapping containing '_verb_canonical'.

    Returns
    -------
    float
        1.0  — verbs match (including both-unknown tie)
        0.0  — one verb known, the other unknown or a different canonical group
    """
    if domain_config is None:
        from intentguard.domain_config import load_domain_config
        domain_config = load_domain_config("ecommerce")

    verb_canonical = domain_config["_verb_canonical"]

    intent_tokens = _tokenise(intent_text)
    call_tokens   = _tokenise(call_text)
    intent_verb   = _extract_verb(intent_tokens, verb_canonical)
    call_verb     = _extract_verb(call_tokens, verb_canonical)

    if intent_verb is None and call_verb is None:
        # Neither text contains a recognisable action verb — tie (no signal).
        return 1.0
    return 1.0 if intent_verb == call_verb else 0.0


def _normalise_entity(tok: str) -> str:
    """Normalise token: lowercase, strip possessives, reduce basic plurals to singular."""
    tok = tok.lower()
    if tok.endswith("'s"):
        tok = tok[:-2]
    elif tok.endswith("s'"):
        tok = tok[:-2]

    plurals_map = {
        "customers": "customer",
        "products": "product",
        "shipments": "shipment",
        "tickets": "ticket",
        "codes": "code",
        "addresses": "address",
        "orders": "order",
        "items": "item",
        "payments": "payment",
        "coupons": "coupon"
    }
    return plurals_map.get(tok, tok)

def _extract_entities(tokens: list[str], raw_text: str, entity_vocab: frozenset[str]) -> set[str]:
    import string
    entities = set()
    for tok in tokens:
        norm = _normalise_entity(tok)
        if norm in entity_vocab and norm not in _STOP_WORDS: entities.add(norm)
    raw_tokens = raw_text.split()
    for i, raw_tok in enumerate(raw_tokens):
        clean = raw_tok.strip(string.punctuation)
        if len(clean) >= 3 and clean[0].isupper() and clean.lower() not in _STOP_WORDS and i > 0:
            entities.add(_normalise_entity(clean))
    return entities

def _entity_score(intent_text: str, call_text: str, domain_config: dict | None = None) -> float:
    """Jaccard similarity of domain-relevant entities found in both texts.

    Parameters
    ----------
    intent_text, call_text : str
    domain_config : dict, optional
        Loaded domain configuration mapping containing '_entity_set'.

    Returns
    -------
    float
        |intersection| / |union| of entity sets.
        Returns 0.0 if both entity sets are empty — an empty-match is a
        genuine zero-overlap situation, not a perfect match.
    """
    if domain_config is None:
        from intentguard.domain_config import load_domain_config
        domain_config = load_domain_config("ecommerce")

    entity_vocab = domain_config["_entity_set"]

    intent_tokens = _tokenise(intent_text)
    call_tokens   = _tokenise(call_text)

    intent_entities = _extract_entities(intent_tokens, intent_text, entity_vocab)
    call_entities   = _extract_entities(call_tokens,   call_text, entity_vocab)

    union = intent_entities | call_entities
    if not union:
        return 0.0

    intersection = intent_entities & call_entities
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_pair(intent_text: str, call_text: str, domain_config: dict | None = None, extracted_call_verb: str | None = None, call_verb_provided: bool = False) -> float:
    if not isinstance(intent_text, str):
        raise ValueError(f"score_pair(): intent_text must be a str, got {type(intent_text).__name__!r}.")
    if not isinstance(call_text, str):
        raise ValueError(f"score_pair(): call_text must be a str, got {type(call_text).__name__!r}.")

    from intentguard.embedder import encode
    v_intent = encode(intent_text)
    v_call   = encode(call_text)

    base_score = _cosine_score(v_intent, v_call)
    verb_score = _verb_score(intent_text, call_text, domain_config=domain_config, extracted_call_verb=extracted_call_verb, call_verb_provided=call_verb_provided)
    ent_score  = _entity_score(intent_text, call_text, domain_config=domain_config)

    final_score = (ALPHA * base_score) + (BETA * verb_score) + (GAMMA * ent_score)
    return float(final_score)
