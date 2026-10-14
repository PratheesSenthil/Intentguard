"""
intentguard.api
===============
FastAPI serving layer for the intentguard semantic intent-validation pipeline.

Endpoints
---------
POST /validate
    Accepts a user message + list of API calls, returns per-call verdicts.

GET /health
    Lightweight liveness probe; also confirms which model is loaded.

Usage
-----
Run locally::

    python -m intentguard.api          # via __main__ block
    uvicorn intentguard.api:app        # directly with uvicorn

The app instance is importable for use with FastAPI's TestClient::

    from intentguard.api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from intentguard.embedder import _MODEL_NAME
from intentguard.validator import validate, DEFAULT_THRESHOLD, LOW_CONF_FLOOR

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class APICallModel(BaseModel):
    """Represents a single API call produced by an LLM agent."""

    method: str = Field(
        ...,
        description="HTTP method (GET, POST, PUT, DELETE, PATCH, …).",
        examples=["POST"],
    )
    endpoint: str = Field(
        ...,
        description="API endpoint path, e.g. '/cart/items'.",
        examples=["/cart/items"],
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional query parameters.",
        examples=[{"code": "SAVE10"}],
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description="Optional request body.",
        examples=[{"product": "shoes", "quantity": 1}],
    )
    provenance: Literal["box_a", "box_b"] | None = Field(
        default=None,
        description=(
            "'box_a' = originated from user-request context (default). "
            "'box_b' = originated from external/retrieved content "
            "(triggers injection detection if unmatched)."
        ),
    )

    @field_validator("method")
    @classmethod
    def method_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'method' must not be empty or whitespace-only.")
        return v

    @field_validator("endpoint")
    @classmethod
    def endpoint_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'endpoint' must not be empty or whitespace-only.")
        return v


class ValidateRequest(BaseModel):
    """Request body for POST /validate."""

    user_message: str = Field(
        ...,
        description="The raw message the user sent to the LLM agent.",
        examples=["Update my email and apply coupon SAVE10 to my cart"],
        min_length=1,
    )
    api_calls: list[APICallModel] = Field(
        ...,
        description="List of API call dicts produced by the LLM agent.",
        min_length=1,
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        description=(
            "Minimum composite score [0, 1] for an intent to approve a call. "
            f"Default: {DEFAULT_THRESHOLD}."
        ),
        ge=0.0,
        le=1.0,
    )
    low_conf_floor: float = Field(
        default=LOW_CONF_FLOOR,
        description=(
            "Scores in [low_conf_floor, threshold) yield 'low_confidence'. "
            f"Default: {LOW_CONF_FLOOR}."
        ),
        ge=0.0,
        le=1.0,
    )
    domain: str = Field(
        default="ecommerce",
        description="Domain configuration name (e.g. 'ecommerce') or absolute path to JSON file.",
    )
    audit: bool = Field(
        default=True,
        description="Whether to record these calls in the audit log.",
    )
    trusted_values: list[str] = Field(
        default=[],
        description="Values the calling application itself vouches for."
    )
    grounding: bool = Field(
        default=True,
        description="Whether to perform value grounding on approved calls."
    )

    @field_validator("user_message")
    @classmethod
    def message_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "'user_message' must not be empty or whitespace-only."
            )
        return v


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="intentguard",
    description=(
        "Semantic intent-validation middleware for LLM agent pipelines. "
        "Checks whether each API call made by an LLM agent is sanctioned "
        "by the user's stated intent."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Exception handler — convert intentguard ValueErrors to 422 with detail
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Surface intentguard ValueError as a 422 Unprocessable Entity."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.exception_handler(FileNotFoundError)
async def not_found_error_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Surface intentguard FileNotFoundError (e.g. domain config missing) as a 422."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["ops"],
)
async def health() -> HealthResponse:
    """Return service health status and the name of the loaded embedding model.

    Does **not** load the model — just reports the configured model name.
    Use this as a lightweight Kubernetes/Docker liveness probe.
    """
    return HealthResponse(status="ok", model=_MODEL_NAME)


@app.post(
    "/validate",
    summary="Validate API calls against a user intent",
    tags=["core"],
    response_description=(
        "Per-call verdicts ('approved' | 'orphaned' | 'injected' | 'low_confidence') "
        "plus extracted intents and the threshold used."
    ),
)
async def validate_calls(request: ValidateRequest) -> dict:
    """Run the full intentguard pipeline on the supplied API calls.

    **Pipeline summary**

    1. Decompose `user_message` into intent fragments (splits on "and", ";", etc.).
    2. Serialise each `api_call` dict into an embeddable string.
    3. Score every (intent, call) pair with the composite scorer
       (cosine + verb-match + entity-overlap).
    4. Run 1-to-many matching with Hungarian conflict resolution.
    5. Assign each call one of four verdicts:

    | Verdict | Condition |
    |---|---|
    | `approved` | Score ≥ threshold, intent claimed the call |
    | `low_confidence` | No intent approved; score ∈ [low_conf_floor, threshold) |
    | `orphaned` | No intent approved; score < low_conf_floor |
    | `injected` | No intent approved AND provenance == `box_b` |

    **Provenance note**: calls without a `provenance` key default to `box_a`
    and a `provenance_note` is included in their verdict entry.
    """
    # Convert Pydantic models to plain dicts so validator.validate() can
    # consume them unchanged (it expects dict keys "method", "endpoint", etc.)
    raw_calls: list[dict] = [call.model_dump(exclude_none=False) for call in request.api_calls]

    # Pydantic sets Optional fields to None when absent; validator._serialise_call
    # gracefully ignores None params/body. Provenance None → validator defaults to box_a.

    result = validate(
        user_message=request.user_message,
        api_calls=raw_calls,
        threshold=request.threshold,
        low_conf_floor=request.low_conf_floor,
        domain=request.domain,
        trusted_values=request.trusted_values,
        grounding=request.grounding,
    )
    
    from intentguard.schema_check import check_call
    import time
    import json
    import os
    
    audit_file = os.environ.get("INTENTGUARD_AUDIT_PATH")
    if not audit_file:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        audit_file = os.path.join(logs_dir, "audit.jsonl")
    
    for v in result.get("verdicts", []):
        idx = v.get("call_index")
        if idx is not None and idx < len(raw_calls):
            call_obj = raw_calls[idx]
            schema_res = check_call(call_obj.get("method", ""), call_obj.get("endpoint", ""))
            v["schema"] = schema_res
            
            def cap(text: str | None) -> str | None:
                if text is None: return text
                return text if len(text) <= 300 else text[:297] + "..."
            
            user_msg = cap(request.user_message)
            expl_copy = v.get("explanation", {}).copy()
            if "closest_intent" in expl_copy:
                expl_copy["closest_intent"] = cap(expl_copy["closest_intent"])
            if "reason" in expl_copy:
                expl_copy["reason"] = cap(expl_copy["reason"])
                
            if request.audit:
                audit_entry = {
                    "timestamp": time.time(),
                    "user_message": user_msg,
                    "method": call_obj.get("method", ""),
                    "endpoint": call_obj.get("endpoint", ""),
                    "provenance": v.get("provenance"),
                    "verdict": v.get("verdict"),
                    "score": v.get("score"),
                    "schema": schema_res,
                    "explanation": expl_copy,
                    "mode": "live"
                }
                with open(audit_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(audit_entry) + "\n")
            
    return result


@app.get("/audit", summary="Get audit logs", tags=["ops"])
async def get_audit(limit: int = 50) -> list[dict]:
    import os
    import json
    audit_file = os.environ.get("INTENTGUARD_AUDIT_PATH")
    if not audit_file:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        audit_file = os.path.join(logs_dir, "audit.jsonl")
    if not os.path.exists(audit_file):
        return []
    
    entries = []
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    entries.reverse()
    return entries[:limit]


# ---------------------------------------------------------------------------
# Local dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "intentguard.api:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info",
    )
