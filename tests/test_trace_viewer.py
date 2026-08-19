"""Tests for the trace viewer CLI utility."""

import json
import tempfile
from pathlib import Path

from mcp_research_agent_system.trace_viewer import (
    format_event_plain,
    format_event_rich,
    format_timestamp,
    load_trace,
)


def test_load_trace_valid_jsonl():
    """Test loading valid JSONL trace file."""
    events = [
        {"timestamp": "2024-01-01T12:00:00.000Z", "event_type": "node_entry", "payload": {"node_name": "planner"}},
        {"timestamp": "2024-01-01T12:00:01.000Z", "event_type": "node_exit", "payload": {"node_name": "planner", "duration_ms": 500.0}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
        temp_path = f.name

    try:
        loaded = load_trace(temp_path)
        assert len(loaded) == 2
        assert loaded[0]["event_type"] == "node_entry"
        assert loaded[1]["event_type"] == "node_exit"
    finally:
        Path(temp_path).unlink()


def test_load_trace_skips_empty_lines():
    """Test that empty lines in JSONL are skipped."""
    events = [
        {"timestamp": "2024-01-01T12:00:00.000Z", "event_type": "node_entry", "payload": {}},
        "",  # Empty line
        {"timestamp": "2024-01-01T12:00:01.000Z", "event_type": "node_exit", "payload": {}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for event in events:
            if isinstance(event, dict):
                f.write(json.dumps(event) + "\n")
            else:
                f.write("\n")
        temp_path = f.name

    try:
        loaded = load_trace(temp_path)
        assert len(loaded) == 2
    finally:
        Path(temp_path).unlink()


def test_load_trace_handles_invalid_json():
    """Test that invalid JSON lines are skipped with warning."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"valid": "json"}\n')
        f.write('not valid json\n')
        f.write('{"also": "valid"}\n')
        temp_path = f.name

    try:
        loaded = load_trace(temp_path)
        assert len(loaded) == 2
        assert loaded[0]["valid"] == "json"
        assert loaded[1]["also"] == "valid"
    finally:
        Path(temp_path).unlink()


def test_format_timestamp_iso():
    """Test formatting ISO timestamps."""
    ts = "2024-01-01T12:30:45.123456Z"
    result = format_timestamp(ts)
    assert result == "12:30:45.123"


def test_format_timestamp_with_timezone():
    """Test formatting timestamps with timezone offset."""
    ts = "2024-01-01T12:30:45.123456+00:00"
    result = format_timestamp(ts)
    assert result == "12:30:45.123"


def test_format_timestamp_invalid():
    """Test formatting invalid timestamps returns as-is."""
    ts = "invalid-timestamp"
    result = format_timestamp(ts)
    assert result == "invalid-timestamp"


def test_format_event_plain_node_entry():
    """Test formatting node_entry event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "event_type": "node_entry",
        "payload": {
            "node_name": "planner",
            "state_snapshot": {"research_goal": "test goal", "sub_queries": 3},
        },
    }
    result = format_event_plain(event)
    assert "node_entry" in result
    assert "planner" in result
    assert "research_goal=test goal" in result
    assert "sub_queries=3" in result


def test_format_event_plain_node_exit():
    """Test formatting node_exit event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:01.000Z",
        "event_type": "node_exit",
        "payload": {
            "node_name": "planner",
            "status": "success",
            "duration_ms": 500.5,
            "state_snapshot": {"sub_queries": 3},
        },
    }
    result = format_event_plain(event)
    assert "node_exit" in result
    assert "planner" in result
    assert "[success]" in result
    assert "500.5ms" in result


def test_format_event_plain_node_error():
    """Test formatting node_error event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "event_type": "node_error",
        "payload": {
            "node_name": "researcher",
            "error": "Connection timeout",
            "state_snapshot": {"current_query_index": 0},
        },
    }
    result = format_event_plain(event)
    assert "node_error" in result
    assert "researcher" in result
    assert "Connection timeout" in result


def test_format_event_plain_tool_call():
    """Test formatting tool_call event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "event_type": "tool_call",
        "payload": {
            "tool_name": "search_papers",
            "input": {"query": "machine learning"},
        },
    }
    result = format_event_plain(event)
    assert "tool_call" in result
    assert "search_papers" in result


def test_format_event_plain_tool_result():
    """Test formatting tool_result event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:01.000Z",
        "event_type": "tool_result",
        "payload": {
            "tool_name": "search_papers",
            "duration_ms": 1200.0,
        },
    }
    result = format_event_plain(event)
    assert "tool_result" in result
    assert "search_papers" in result
    assert "1200.0ms" in result


def test_format_event_plain_tool_error():
    """Test formatting tool_error event in plain text."""
    event = {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "event_type": "tool_error",
        "payload": {
            "tool_name": "search_papers",
            "error": "API rate limit exceeded",
        },
    }
    result = format_event_plain(event)
    assert "tool_error" in result
    assert "search_papers" in result
    assert "API rate limit exceeded" in result


def test_format_event_rich_returns_tuple():
    """Test that format_event_rich returns a 6-tuple."""
    event = {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "event_type": "node_entry",
        "payload": {"node_name": "planner", "state_snapshot": {}},
    }
    result = format_event_rich(event)
    assert isinstance(result, tuple)
    assert len(result) == 6
    timestamp, event_type, name, status, duration, details = result
    assert timestamp == "12:00:00.000"
    assert event_type == "node_entry"
    assert name == "planner"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
