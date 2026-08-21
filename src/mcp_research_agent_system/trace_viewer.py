"""Trace viewer CLI utility for reading and displaying JSONL trace logs.

Usage:
    python -m mcp_research_agent_system.trace_viewer logs/trace.jsonl
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_trace(file_path: str) -> list[dict[str, Any]]:
    """Load trace events from a JSONL file."""
    events = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)  # noqa: T201
    return events


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S.%f")[:-3]
    except Exception:
        return ts


def format_event_plain(event: dict[str, Any]) -> str:
    """Format a single event for plain text output."""
    timestamp = format_timestamp(event.get("timestamp", ""))
    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})

    if event_type in ("node_entry", "node_exit", "node_error"):
        node_name = payload.get("node_name", "unknown")
        state_snapshot = payload.get("state_snapshot", {})
        duration = payload.get("duration_ms")
        status = payload.get("status", "")
        error = payload.get("error", "")

        parts = [f"{timestamp} | {event_type:12} | {node_name:12}"]
        if status:
            parts.append(f"[{status}]")
        if duration is not None:
            parts.append(f"{duration:.1f}ms")
        if error:
            parts.append(f"ERROR: {error}")
        if state_snapshot:
            state_str = ", ".join(f"{k}={v}" for k, v in state_snapshot.items())
            parts.append(f"({state_str})")
        return " ".join(parts)

    elif event_type in ("tool_call", "tool_result", "tool_error"):
        tool_name = payload.get("tool_name", "unknown")
        parts = [f"{timestamp} | {event_type:12} | {tool_name:12}"]
        if event_type == "tool_result":
            duration = payload.get("duration_ms")
            if duration is not None:
                parts.append(f"{duration:.1f}ms")
        elif event_type == "tool_error":
            error = payload.get("error", "")
            parts.append(f"ERROR: {error}")
        return " ".join(parts)

    else:
        # Generic event
        return f"{timestamp} | {event_type:12} | {json.dumps(payload)[:80]}"


def format_event_rich(event: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Format a single event for rich table output."""
    timestamp = format_timestamp(event.get("timestamp", ""))
    event_type = event.get("event_type", "unknown")
    payload = event.get("payload", {})

    if event_type in ("node_entry", "node_exit", "node_error"):
        node_name = payload.get("node_name", "unknown")
        state_snapshot = payload.get("state_snapshot", {})
        duration = payload.get("duration_ms")
        status = payload.get("status", "")

        state_str = (
            ", ".join(f"{k}={v}" for k, v in state_snapshot.items()) if state_snapshot else ""
        )
        duration_str = f"{duration:.1f}ms" if duration is not None else ""
        return timestamp, event_type, node_name, status, duration_str, state_str

    elif event_type in ("tool_call", "tool_result", "tool_error"):
        tool_name = payload.get("tool_name", "unknown")
        duration = payload.get("duration_ms")
        error = payload.get("error", "")

        duration_str = f"{duration:.1f}ms" if duration is not None else ""
        error_str = error if event_type == "tool_error" else ""
        return timestamp, event_type, tool_name, "", duration_str, error_str

    else:
        payload_str = json.dumps(payload)[:80]
        return timestamp, event_type, "", "", "", payload_str


def print_trace_plain(events: list[dict[str, Any]]) -> None:
    """Print trace events in plain text format."""
    print(
        f"{'TIMESTAMP':<14} | {'EVENT_TYPE':<12} | {'NODE/TOOL':<12} | {'STATUS':<12} | {'DURATION':<10} | DETAILS"
    )  # noqa: T201
    print("-" * 120)  # noqa: T201
    for event in events:
        print(format_event_plain(event))  # noqa: T201


def print_trace_rich(events: list[dict[str, Any]]) -> None:
    """Print trace events using rich table."""
    console = Console()

    table = Table(title="MCP Research Agent Trace", show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", style="cyan", width=14)
    table.add_column("Event Type", style="yellow", width=12)
    table.add_column("Node/Tool", style="green", width=14)
    table.add_column("Status", style="blue", width=10)
    table.add_column("Duration", style="magenta", width=10)
    table.add_column("Details", style="white", ratio=1)

    for event in events:
        timestamp, event_type, name, status, duration, details = format_event_rich(event)

        # Color-code by event type
        if event_type == "node_entry":
            event_type_display = "[green]ENTRY[/green]"
        elif event_type == "node_exit":
            event_type_display = "[blue]EXIT[/blue]"
        elif event_type == "node_error":
            event_type_display = "[red]ERROR[/red]"
        elif event_type == "tool_call":
            event_type_display = "[cyan]TOOL_CALL[/cyan]"
        elif event_type == "tool_result":
            event_type_display = "[green]TOOL_OK[/green]"
        elif event_type == "tool_error":
            event_type_display = "[red]TOOL_ERR[/red]"
        else:
            event_type_display = event_type

        table.add_row(timestamp, event_type_display, name, status, duration, details)

    console.print(table)


def main() -> int:
    """Main entry point for the trace viewer CLI."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m mcp_research_agent_system.trace_viewer <trace_file.jsonl>",
            file=sys.stderr,
        )  # noqa: T201
        return 1

    trace_file = sys.argv[1]
    if not Path(trace_file).exists():
        print(f"Error: File not found: {trace_file}", file=sys.stderr)  # noqa: T201
        return 1

    events = load_trace(trace_file)
    if not events:
        print("No events found in trace file.", file=sys.stderr)  # noqa: T201
        return 0

    if RICH_AVAILABLE:
        print_trace_rich(events)
    else:
        print_trace_plain(events)

    return 0


if __name__ == "__main__":
    sys.exit(main())
