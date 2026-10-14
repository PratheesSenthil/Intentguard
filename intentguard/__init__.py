"""
intentguard
===========
Semantic intent-validation middleware / SDK.

Provides a pipeline that checks whether API calls made by a language-model
agent are semantically sanctioned by the user's stated intent:

    user_message  →  embedder  →  scorer  →  matcher  →  validator  →  verdict

Public API
----------
>>> from intentguard import validate
>>> result = validate(user_message="Book a flight", api_calls=[...])

Each call in *api_calls* receives one of four verdicts:
    "approved"        – cosine + verb/entity score ≥ threshold, unambiguous match
    "low_confidence"  – best score is above a soft floor but below threshold
    "orphaned"        – no intent scored high enough to claim this call
    "injected"        – call was flagged by the conflict-resolution step as a
                        likely prompt-injection artefact
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("intentguard")
except PackageNotFoundError:  # running from source without installation
    __version__ = "0.0.0.dev0"

# Convenience re-export so callers can do `from intentguard import validate`
from intentguard.validator import validate  # noqa: E402

__all__ = ["validate", "__version__"]
