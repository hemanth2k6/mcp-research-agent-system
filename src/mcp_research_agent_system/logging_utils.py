"""Structured JSONL logging utility for MCP server."""

import json
from datetime import UTC, datetime
from pathlib import Path


def log_event(event_type: str, payload: dict) -> None:
    """Append a structured JSON log event to logs/trace.jsonl.

    Args:
        event_type: Type of event (e.g., "tool_call", "tool_result", "error")
        payload: Dictionary of event data to log
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "trace.jsonl"

    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def log_tool_call(tool_name: str, input_data: dict) -> None:
    """Log a tool call event."""
    log_event(
        "tool_call",
        {
            "tool_name": tool_name,
            "input": input_data,
        },
    )


def log_tool_result(tool_name: str, output_summary: dict, duration_ms: float) -> None:
    """Log a tool result event."""
    log_event(
        "tool_result",
        {
            "tool_name": tool_name,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
        },
    )


def log_tool_error(tool_name: str, error: str, input_data: dict) -> None:
    """Log a tool error event."""
    log_event(
        "tool_error",
        {
            "tool_name": tool_name,
            "error": error,
            "input": input_data,
        },
    )
