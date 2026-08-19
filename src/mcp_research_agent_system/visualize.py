"""Graph visualization utilities for exporting state machine diagrams."""

from pathlib import Path
from typing import Any

from .agents.graph import build_graph


def export_graph_diagram(output_path: str | Path = "docs/graph_diagram.mmd") -> str:
    """Export the research agent state graph as a Mermaid diagram.

    Uses LangGraph's built-in get_graph().draw_mermaid() to generate
    a Mermaid flowchart representation of the state machine.

    Args:
        output_path: Path to write the Mermaid diagram file.
                    Defaults to "docs/graph_diagram.mmd"

    Returns:
        The Mermaid diagram source code as a string.
    """
    graph = build_graph()
    langgraph_graph = graph.get_graph()
    mermaid_code: str = langgraph_graph.draw_mermaid()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mermaid_code, encoding="utf-8")

    return mermaid_code


def get_graph_structure() -> dict[str, Any]:
    """Get the graph structure as a dictionary for programmatic inspection.

    Returns:
        Dictionary with nodes, edges, and entry/exit points.
    """
    graph = build_graph()
    langgraph_graph = graph.get_graph()

    return {
        "nodes": [node.name for node in langgraph_graph.nodes.values()],
        "edges": [(edge.source, edge.target) for edge in langgraph_graph.edges],
        "entry_point": langgraph_graph.entry_point,
        "exit_points": langgraph_graph.exit_points,
    }


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else "docs/graph_diagram.mmd"
    mermaid = export_graph_diagram(output_file)
    print(f"Graph diagram exported to {output_file}")  # noqa: T201
    print("\n--- Mermaid Diagram ---")  # noqa: T201
    print(mermaid)  # noqa: T201
