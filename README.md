# intentguard

> **Semantic intent-validation middleware for LLM agent pipelines.**

intentguard sits between a user message and the API calls an LLM agent makes on the user's behalf. It answers one question per call:

> *"Did the user actually ask for this?"*

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `approved` | Call is semantically sanctioned by the user's stated intent, satisfies all semantic veto rules and value grounding, and scored above the approval threshold. |
| `low_confidence` | Score falls between the low-confidence floor and the approval threshold, or an approved call was vetoed by semantic rules (e.g. verb-mismatch, read-only violation, or ungrounded value substitution). Warrants review. |
| `orphaned` | Score is below the low-confidence floor (`0.25`) with user-request provenance (`box_a`). Indicates an ungrounded or over-eager agent action. |
| `injected` | Call is unclaimed by any intent and originated from external/retrieved context (`box_b`). Treated as a prompt-injection artifact. |

---

## How It Works

```
user_message (str)
    │
    ▼  [1. Intent Extraction]          Decompose compound requests into sub-intents
    │                                  Apply cross-fragment context carry-over
    │
    ▼  [2. Call Serialization]         dict(method, endpoint, params, body) → normalized text
    │                                  Extract action verb & provenance (box_a / box_b)
    │
    ▼  [3. Embedder]                   sentence-transformers all-MiniLM-L6-v2
    │                                  → 384-d L2-normalized dense vectors
    │
    ▼  [4. Composite Scorer]           s = α·cosine_sim + β·verb_match + γ·entity_overlap
    │                                  (α=0.60, β=0.25, γ=0.15)
    │
    ▼  [5. Matcher]                    Threshold pass (≥ 0.55)
    │                                  Hungarian algorithm (linear_sum_assignment) for conflicts
    │
    ▼  [6. Semantic Veto Rules]        Rule 1: Incompatible verb-group mismatch veto
    │                                  Rule 2: Read-only GET violation veto
    │                                  Provenance check: Unassigned + box_b → "injected"
    │
    ▼  [7. Value Grounding]            Verify entity IDs, emails, numbers against user request
    │                                  Ungrounded approved call → demoted to "low_confidence"
    │
    ▼  [8. OpenAPI Schema Check]       Validate route existence, method allowance, no unreplaced {id}
    │
    └──▶  Structured Result            {verdicts, intents_extracted, threshold}
```

### Scoring Formula

$$s = \alpha \cdot \text{cosine\_sim} + \beta \cdot \text{verb\_match} + \gamma \cdot \text{entity\_overlap}$$

Current weights: $\alpha = 0.60$, $\beta = 0.25$, $\gamma = 0.15$ (enforced to sum to $1.0$).

---

## Quickstart

```python
from intentguard import validate

result = validate(
    user_message="Cancel order 42",
    api_calls=[
        {"method": "POST", "endpoint": "/api/orders/42/cancel/", "provenance": "box_a"},
        {"method": "DELETE", "endpoint": "/api/customers/42/", "provenance": "box_b"},
    ],
)

for v in result["verdicts"]:
    print(v["call_index"], v["verdict"], v["score"])
```

**Real output:**
```text
0 approved 0.91479
1 injected 0.540186
```

---

## Configuration

| Parameter | Default | Defined In | Description |
|---|---|---|---|
| `threshold` | `0.55` | `validator.py:82`, `matcher.py:245` | Minimum composite score to approve an API call. |
| `low_conf_floor` | `0.25` | `validator.py:83` | Floor below which unclaimed calls are marked `orphaned`. |
| `ALPHA` | `0.60` | `scorer.py:54` | Weight for embedding cosine similarity. |
| `BETA` | `0.25` | `scorer.py:55` | Weight for canonical action-verb match. |
| `GAMMA` | `0.15` | `scorer.py:56` | Weight for domain entity-overlap Jaccard score. |

---

## Installation

```bash
# from source
pip install -e .

# with dev dependencies
pip install -e ".[dev]"
```

---

## Project Layout

```
intentguard/
├── intentguard/
│   ├── __init__.py              # Package exports (validate)
│   ├── api.py                   # FastAPI serving layer (POST /validate, GET /health, GET /audit)
│   ├── domain_config.py         # Domain configuration loader and caching
│   ├── domains/
│   │   └── ecommerce.json       # Verb groups, synonyms, entity vocabulary, compatible verbs
│   ├── embedder.py              # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   ├── grounding.py             # Entity and parameter value grounding verification
│   ├── matcher.py               # 1-to-many matcher and Hungarian conflict resolution
│   ├── schema_check.py          # OpenAPI route conformance validator
│   ├── scorer.py                # Composite semantic scorer (cosine + verb + entity)
│   ├── test_schema_check.py     # Schema check unit tests
│   └── validator.py             # Full pipeline orchestrator and verdict generation
├── results/
│   ├── ablation.csv             # Benchmark accuracy, ablation tracking, safety metrics
│   ├── ablation.md              # Markdown table of ablation experiments
│   └── latency_benchmark.json   # Measured latency distribution stats
├── schemas/
│   └── ecommerce_openapi.json   # Enterprise e-commerce OpenAPI specification
├── scripts/
│   ├── compare_runs.py          # Side-by-side run comparison utility
│   ├── diagnose_mismatches.py   # Diagnostic analysis of classification errors
│   ├── evaluate.py              # Evaluation runner for benchmark datasets
│   ├── map_urls.py              # URL-to-verb mapping generator
│   ├── measure_latency.py       # Latency benchmark measuring POST /validate
│   ├── new_demo_session.py      # Demo session initializer
│   ├── optimize_thresholds.py   # Grid-search threshold optimizer
│   ├── start_demo.ps1           # Demo launch script
│   └── stop_demo.ps1            # Demo teardown script
├── tests/
│   ├── conftest.py              # Pytest fixtures and mock configurations
│   ├── fresh_ecommerce_set.json # Evaluation dataset: Fresh set (45 calls)
│   ├── fresh2_ecommerce_set.json# Evaluation dataset: Fresh 2 adversarial set (42 calls)
│   ├── heldout_ecommerce_set.json# Evaluation dataset: Held-out validation set (33 calls)
│   ├── labeled_ecommerce_set.json# Evaluation dataset: Primary labeled set (50 calls)
│   ├── test_api.py              # API endpoint test suite
│   ├── test_audit_cap.py        # Audit string truncation tests
│   ├── test_audit_flag.py       # Audit logging flag toggle tests
│   ├── test_audit_path.py       # Audit file destination tests
│   ├── test_dashboard_security.py # Dashboard CSP & origin tests
│   ├── test_domain_config.py    # Domain config loader tests
│   ├── test_explanations_and_audit.py # Explanation generation & audit log tests
│   ├── test_fixes.py            # Targeted regression test cases
│   ├── test_grounding.py        # Value grounding logic tests
│   ├── test_grounding_api.py    # End-to-end grounding API tests
│   ├── test_safety_rules.py     # Semantic veto safety rule tests
│   └── test_validator.py        # Core validator unit & integration tests
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest
```

**Test suite status:**
- Collected: **183 items**
- Passed: **183 passed** (0 failed, 0 skipped)
- Execution time: ~96 seconds

---

## Reproducing Benchmark Results

Evaluations run against the four benchmark datasets stored in `tests/` using Python 3.11+.

> [!NOTE]
> On the first run, `sentence-transformers` automatically downloads the `all-MiniLM-L6-v2` model weights (~80 MB) to the local HuggingFace cache directory (`~/.cache/huggingface/hub/`). Subsequent runs execute entirely offline.

### Evaluation Commands

1. **Labeled Benchmark Set (50 calls):**
   ```bash
   python scripts/evaluate.py "labeled" tests/labeled_ecommerce_set.json --grounding on
   ```
   *Result:* **82.0%** Exact Accuracy (41/50), **88.0%** Allow/Block Accuracy (Safety A = 0, Safety B = 6).

2. **Held-out Benchmark Set (33 calls):**
   ```bash
   python scripts/evaluate.py "heldout" tests/heldout_ecommerce_set.json --grounding on
   ```
   *Result:* **78.8%** Exact Accuracy (26/33), **87.9%** Allow/Block Accuracy (Safety A = 0, Safety B = 4).

3. **Fresh Benchmark Set (45 calls):**
   ```bash
   python scripts/evaluate.py "fresh" tests/fresh_ecommerce_set.json --grounding on
   ```
   *Result:* **68.9%** Exact Accuracy (31/45), **84.4%** Allow/Block Accuracy (Safety A = 0, Safety B = 7).

4. **Fresh 2 Adversarial Benchmark Set (42 calls):**
   ```bash
   python scripts/evaluate.py "fresh2" tests/fresh2_ecommerce_set.json --grounding on
   ```
   *Result:* **61.9%** Exact Accuracy (26/42), **95.2%** Allow/Block Accuracy (Safety A = 0, Safety B = 2).

---

## Serving via FastAPI

```bash
uvicorn intentguard.api:app --host 127.0.0.1 --port 8001
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8001/docs`.

### Measured Latency (POST /validate)
Benchmarked across 212 timed requests on realistic payloads (1–3 API calls per request, 200–525 bytes) on an Intel Core i5-12450H CPU (model pre-warmed):
- **Min:** 84.20 ms
- **Median:** 151.09 ms
- **Mean:** 156.88 ms
- **95th Percentile:** 263.60 ms
- **Max:** 435.30 ms

---

## Roadmap

- [x] Implement `embedder.py` — lazy model loading, L2-normalized sentence embeddings
- [x] Implement `scorer.py` — composite scoring ($\alpha \cdot \text{cosine} + \beta \cdot \text{verb} + \gamma \cdot \text{entity}$)
- [x] Implement `matcher.py` — 1-to-many threshold pass + Hungarian conflict resolution
- [x] Implement `validator.py` — orchestrator with semantic vetoes and provenance-based injection detection
- [x] Domain configuration subsystem (`domain_config.py` + `domains/ecommerce.json`)
- [x] Cross-fragment context carry-over for compound user intents
- [x] Value grounding defense against parameter and entity substitution (`grounding.py`)
- [x] OpenAPI schema conformance check (`schema_check.py`)
- [x] FastAPI serving layer with audit logging (`api.py`)
- [x] Comprehensive test suite (183 passing unit and integration tests)
- [ ] Multi-domain pluggable configuration extensions (healthcare, banking)
- [ ] Streaming request validation for live LLM token output
