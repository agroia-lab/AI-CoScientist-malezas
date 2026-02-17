"""Tests for AIScientistFramework._safely_parse_json.

This is the most critical utility in the framework -- it must handle
well-formed JSON, markdown-wrapped JSON, JSON embedded in prose,
and various malformed inputs without ever raising an exception.
"""

import json


# -- Valid inputs ---------------------------------------------------------


def test_valid_json_string(framework):
    """Plain valid JSON string returns the parsed dict."""
    result = framework._safely_parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_json_code_fence(framework):
    """JSON wrapped in ```json ... ``` code fences is stripped and parsed."""
    raw = '```json\n{"key": "value"}\n```'
    result = framework._safely_parse_json(raw)
    assert result == {"key": "value"}


def test_json_bare_code_fence(framework):
    """JSON wrapped in bare ``` fences (no language tag) is still parsed."""
    raw = '```\n{"a": 1}\n```'
    result = framework._safely_parse_json(raw)
    assert result == {"a": 1}


def test_nested_json(framework):
    """Deeply nested JSON is fully extracted."""
    nested = {"a": {"b": {"c": 1}}}
    result = framework._safely_parse_json(json.dumps(nested))
    assert result == nested


def test_json_preceded_by_text(framework):
    """JSON preceded by prose like 'Here is the result: {...}' is extracted."""
    raw = 'Here is the result: {"answer": 42}'
    result = framework._safely_parse_json(raw)
    assert result == {"answer": 42}


def test_json_with_escaped_quotes(framework):
    """JSON containing escaped quotes inside string values parses correctly."""
    raw = '{"text": "He said \\"hello\\" to me"}'
    result = framework._safely_parse_json(raw)
    assert result["text"] == 'He said "hello" to me'


def test_multiple_json_objects_extracts_first(framework):
    """When input contains multiple JSON objects, the first valid one is returned."""
    raw = 'prefix {"first": 1} middle {"second": 2} end'
    result = framework._safely_parse_json(raw)
    assert result == {"first": 1}


def test_complex_hypotheses_payload(framework):
    """Realistic generation-agent payload parses correctly."""
    payload = {
        "hypotheses": [
            {"text": "Hypothesis 1", "justification": "Reason 1"},
            {"text": "Hypothesis 2", "justification": "Reason 2"},
        ]
    }
    result = framework._safely_parse_json(json.dumps(payload))
    assert len(result["hypotheses"]) == 2
    assert result["hypotheses"][0]["text"] == "Hypothesis 1"


def test_json_with_array_values(framework):
    """JSON containing array values parses correctly."""
    raw = '{"items": [1, 2, 3], "name": "test"}'
    result = framework._safely_parse_json(raw)
    assert result == {"items": [1, 2, 3], "name": "test"}


# -- Error / edge-case inputs --------------------------------------------


def test_empty_string_returns_error(framework):
    """Empty string returns a dict with an 'error' key."""
    result = framework._safely_parse_json("")
    assert "error" in result


def test_whitespace_only_returns_error(framework):
    """Whitespace-only string returns a dict with an 'error' key."""
    result = framework._safely_parse_json("   \n\t  ")
    assert "error" in result


def test_none_returns_error(framework):
    """None input returns a dict with an 'error' key."""
    result = framework._safely_parse_json(None)
    assert "error" in result


def test_non_string_int_returns_error(framework):
    """Integer input returns a dict with an 'error' key."""
    result = framework._safely_parse_json(12345)
    assert "error" in result


def test_non_string_list_returns_error(framework):
    """List input returns a dict with an 'error' key."""
    result = framework._safely_parse_json([1, 2, 3])
    assert "error" in result


def test_string_with_no_json(framework):
    """Plain text with no JSON returns a dict with an 'error' key."""
    result = framework._safely_parse_json(
        "This is just plain text with no braces."
    )
    assert "error" in result


def test_incomplete_json(framework):
    """Incomplete/truncated JSON returns a dict with an 'error' key."""
    result = framework._safely_parse_json('{"key": "val')
    assert "error" in result
