"""Test Unicode character handling in serialization."""

from langfuse._client.attributes import _serialize

def test_mixed_unicode_preserved():
    """Test that mixed Unicode content is preserved."""
    data = {
        "japanese": "こんにちは",
        "chinese": "你好",
        "korean": "안녕하세요",
        "arabic": "مرحبا",
        "russian": "Привет",
        "emoji": "Hello, 🌍!",
    }
    serialized = _serialize(data)
    assert serialized is not None

    assert "\\u" not in serialized, "Should not contain Unicode escapes"
    for value in data.values():
        assert value in serialized, f"Should contain {value}"
