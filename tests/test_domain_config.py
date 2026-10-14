"""
tests.test_domain_config
========================
Tests for intentguard domain configuration loading and error handling.
"""

import json
import pytest
from pathlib import Path

from intentguard.domain_config import load_domain_config

def test_load_bundled_ecommerce_domain():
    """Loading 'ecommerce' should resolve to the bundled ecommerce.json and pre-compute sets."""
    config = load_domain_config("ecommerce")
    
    # Check structure
    assert config["domain"] == "ecommerce"
    assert "verb_groups" in config
    assert "entity_vocab" in config
    
    # Check pre-computed sets
    assert "_verb_canonical" in config
    assert "_entity_set" in config
    assert isinstance(config["_entity_set"], frozenset)

def test_load_custom_domain_file(tmp_path: Path):
    """Providing a file path should load that JSON file as domain config."""
    custom_json = tmp_path / "custom_domain.json"
    custom_data = {
        "domain": "healthcare",
        "verb_groups": {
            "schedule": ["schedule", "book", "arrange"]
        },
        "entity_vocab": ["appointment", "doctor", "patient"]
    }
    custom_json.write_text(json.dumps(custom_data), encoding="utf-8")
    
    config = load_domain_config(str(custom_json))
    assert config["domain"] == "healthcare"
    assert "schedule" in config["verb_groups"]
    assert "patient" in config["_entity_set"]
    assert config["_verb_canonical"]["book"] == "schedule"

def test_load_nonexistent_domain_raises():
    """Loading a non-existent bundled name or file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Domain config not found: nope_not_a_domain"):
        load_domain_config("nope_not_a_domain")

def test_load_malformed_json(tmp_path: Path):
    """Loading a file that is not valid JSON raises ValueError."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not json {", encoding="utf-8")
    
    with pytest.raises(ValueError, match="Failed to parse JSON from"):
        load_domain_config(str(bad_json))

def test_load_missing_required_keys(tmp_path: Path):
    """Loading JSON without verb_groups or entity_vocab raises ValueError."""
    bad_config = tmp_path / "missing_keys.json"
    
    # Missing verb_groups
    bad_config.write_text(json.dumps({"domain": "x", "entity_vocab": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key: 'verb_groups'"):
        load_domain_config(str(bad_config))
        
    # Missing entity_vocab
    bad_config.write_text(json.dumps({"domain": "x", "verb_groups": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key: 'entity_vocab'"):
        load_domain_config(str(bad_config))
