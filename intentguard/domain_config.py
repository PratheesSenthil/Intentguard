"""
intentguard.domain_config
=========================
Loads and caches domain configurations for intentguard.
"""

import json
from pathlib import Path

_CACHE: dict[str, dict] = {}

def load_domain_config(name_or_path: str) -> dict:
    """Load a domain configuration by name (bundled) or path.
    
    If name_or_path matches a bundled domain in intentguard/domains/ (e.g. "ecommerce"),
    it loads that bundled JSON. Otherwise it attempts to load from the given path.
    
    The returned dict will contain "_verb_canonical" and "_entity_set" keys
    pre-computed for fast lookup by scorer.py.
    """
    if name_or_path in _CACHE:
        return _CACHE[name_or_path]

    base_dir = Path(__file__).parent
    bundled_path = base_dir / "domains" / f"{name_or_path}.json"
    
    if bundled_path.exists():
        target_path = bundled_path
    else:
        target_path = Path(name_or_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Domain config not found: {name_or_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from {target_path}: {e}")

    if not isinstance(data, dict):
        raise ValueError("Domain config must be a JSON object.")
    
    if "verb_groups" not in data:
        raise ValueError("Domain config missing required key: 'verb_groups'")
    
    if "entity_vocab" not in data:
        raise ValueError("Domain config missing required key: 'entity_vocab'")

    # Build canonical lookup cache for fast scorer execution
    verb_canonical = {}
    for canonical, synonyms in data["verb_groups"].items():
        for syn in synonyms:
            verb_canonical[syn] = canonical
    
    data["_verb_canonical"] = verb_canonical
    data["_entity_set"] = frozenset(data["entity_vocab"])

    _CACHE[name_or_path] = data
    return data
