import pytest

def sanitize_metadata(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    cleaned = {}
    for k, v in meta.items():
        if isinstance(k, str) and k.strip():
            cleaned[k.strip()] = v
    return cleaned

def test_sanitize_metadata_valid():
    data = {"env": "prod", "  version ": "1.2.0", "active": True}
    assert sanitize_metadata(data) == {"env": "prod", "version": "1.2.0", "active": True}

def test_sanitize_metadata_empty_keys():
    data = {"": "val", "   ": "val2", "valid": 100}
    assert sanitize_metadata(data) == {"valid": 100}
