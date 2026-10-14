# intentguard

> **Semantic intent-validation middleware for LLM agent pipelines.**

intentguard sits between a user message and the API calls an LLM agent makes on
the user's behalf.  It answers one question per call:

> *"Did the user actually ask for this?"*

---

## Verdicts

| Verdict | Meaning |
|---|---|
| `approved` | Call is semantically sanctioned by the user's stated intent and scored above the approval threshold. |
| `low_confidence` | Best score is between the soft floor and the approval threshold — call *may* be legitimate but warrants review. |
| `orphaned` | No intent scored high enough to claim this call.  May indicate an over-eager agent. |
| `injected` | Call is unclaimed **and** exhibits characteristics of a prompt-injection artefact. |

---

## How it works

```
user_message
    │
    ▼  [intent extraction]     split compound requests into sub-intents
    │
    ▼  [call serialisation]    dict → "name key=value …" strings
    │
    ▼  [embedder]              sentence-transformers all-MiniLM-L6-v2
    │                          → 384-d L2-normalised vectors
    │
    ▼  [scorer]                α·cosine + β·verb_match + γ·entity_overlap
    │
    ▼  [matcher]               1:many threshold pass
    │                          Hungarian algorithm for conflict resolution
    │
    ▼  [validator]             maps matcher output → 4-way verdict
    │
    └──▶  {call_index: {status, intent_index, score, call}}
```

### Scoring formula

$$s = \alpha \cdot \text{cosine\_sim} + \beta \cdot \text{verb\_match} + \gamma \cdot \text{entity\_overlap}$$

Default weights: α = 0.70, β = 0.20, γ = 0.10 *(empirically tuned — subject to change)*.

---

## Quickstart

```python
from intentguard import validate

result = validate(
    user_message="Book me a flight to Tokyo",
    api_calls=[
        {"name": "search_flights",  "arguments": {"destination": "Tokyo"}},
        {"name": "delete_account",  "arguments": {}},
    ],
)

for idx, verdict in result.items():
    print(idx, verdict["status"], verdict["score"])
# 0  approved        0.92
# 1  injected        0.04
```

---

## Installation

```bash
# from source
pip install -e ".[dev]"

# with FastAPI serving layer
pip install -e ".[api]"
```

---

## Project layout

```
intentguard/
├── intentguard/
│   ├── __init__.py      re-exports validate()
│   ├── embedder.py      sentence-transformers wrapper
│   ├── scorer.py        composite similarity scorer
│   ├── matcher.py       1:many matching + Hungarian conflict resolution
│   └── validator.py     pipeline orchestrator → verdict dict
├── tests/
│   └── test_validator.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Running tests

```bash
pytest -v
```

All tests are currently **skipped** (skeleton phase).  They will be activated
as each module is implemented.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.60` | Minimum score to approve a call. |
| `LOW_CONF_FLOOR` | `0.35` | Floor below which calls are orphaned rather than low-confidence. |
| `ALPHA` | `0.70` | Weight for cosine similarity. |
| `BETA` | `0.20` | Weight for verb-match bonus. |
| `GAMMA` | `0.10` | Weight for entity-overlap bonus. |

---

## Roadmap

- [ ] Implement `embedder.py` — lazy model loading, encode / encode_batch
- [ ] Implement `scorer.py` — cosine + verb + entity sub-scores
- [ ] Implement `matcher.py` — threshold pass + Hungarian resolution
- [ ] Implement `validator.py` — full pipeline + injection heuristics
- [ ] Activate & pass all 25 unit tests
- [ ] FastAPI serving layer (`/validate` endpoint)
- [ ] Empirical weight tuning on benchmark dataset
- [ ] Multi-intent decomposition in `_extract_intents()`

---

## License

MIT
