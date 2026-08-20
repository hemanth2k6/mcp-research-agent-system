"""Structured JSONL logging utility for MCP server and agent nodes."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def configure_logging() -> None:
    """Configure logging - ensures log directory exists."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)


def get_log_dir() -> Path:
    """Get the log directory path."""
    return Path("logs")


def _get_log_file() -> Path:
    """Get the log file path, creating directory if needed."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "trace.jsonl"


def _write_event(event: dict[str, Any]) -> None:
    """Write a single event to the JSONL log file."""
    with _get_log_file().open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def log_event(event_type: str, payload: dict[str, Any]) -> None:
    """Append a structured JSON log event to logs/trace.jsonl.

    Args:
        event_type: Type of event (e.g., "tool_call", "tool_result", "error")
        payload: Dictionary of event data to log
    """
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    _write_event(event)


def log_tool_call(tool_name: str, input_data: dict[str, Any]) -> None:
    """Log a tool call event."""
    log_event(
        "tool_call",
        {
            "tool_name": tool_name,
            "input": input_data,
        },
    )


def log_tool_result(tool_name: str, output_summary: dict[str, Any], duration_ms: float) -> None:
    """Log a tool result event."""
    log_event(
        "tool_result",
        {
            "tool_name": tool_name,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
        },
    )


def log_tool_error(tool_name: str, error: str, input_data: dict[str, Any]) -> None:
    """Log a tool error event."""
    log_event(
        "tool_error",
        {
            "tool_name": tool_name,
            "error": error,
            "input": input_data,
        },
    )


# --- Agent node logging helpers ---

def log_node_entry(node_name: str, state_snapshot: dict[str, Any]) -> float:
    """Log node entry event and return start time for duration calculation.

    Args:
        node_name: Name of the node (planner, researcher, validator, synthesizer)
        state_snapshot: Safe subset of relevant state fields

    Returns:
        Start time (monotonic) for use with log_node_exit
    """
    start_time = time.monotonic()
    log_event(
        "node_entry",
        {
            "node_name": node_name,
            "state_snapshot": state_snapshot,
            "start_time": start_time,
        },
    )
    return start_time


def log_node_exit(node_name: str, state_snapshot: dict[str, Any], start_time: float, status: str = "success") -> None:
    """Log node exit event with duration.

    Args:
        node_name: Name of the node
        state_snapshot: Safe subset of relevant state fields on exit
        start_time: Monotonic start time from log_node_entry
        status: "success" or "error"
    """
    duration_ms = (time.monotonic() - start_time) * 1000
    log_event(
        "node_exit",
        {
            "node_name": node_name,
            "state_snapshot": state_snapshot,
            "duration_ms": duration_ms,
            "status": status,
        },
    )


def log_node_error(node_name: str, error: str, state_snapshot: dict[str, Any], start_time: float) -> None:
    """Log node error event with duration."""
    duration_ms = (time.monotonic() - start_time) * 1000
    log_event(
        "node_error",
        {
            "node_name": node_name,
            "error": error,
            "state_snapshot": state_snapshot,
            "duration_ms": duration_ms,
        },
    )


def safe_state_snapshot(state: dict[str, Any], include_keys: list[str] | None = None) -> dict[str, Any]:
    """Extract a safe subset of state for logging (avoids huge objects).

    Args:
        state: Full state dict
        include_keys: Specific keys to include (if None, uses default safe keys)

    Returns:
        Dict with safe, serializable state fields
    """
    if include_keys is None:
        include_keys = [
            "research_goal",
            "sub_queries",
            "current_query_index",
            "researcher_attempts",
            "validation_status",
            "validated_findings_count",
            "error",
        ]

    snapshot: dict[str, Any] = {}
    for key in include_keys:
        value = state.get(key)
        if isinstance(value, list):
            snapshot[key] = len(value)  # Log count instead of full list
        else:
            snapshot[key] = value
    return snapshot
