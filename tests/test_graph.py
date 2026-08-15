"""Tests for the LangGraph state machine skeleton."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_research_agent_system.agents.graph import build_graph
from mcp_research_agent_system.agents.state import (
    ResearchState,
    create_initial_state,
)


@pytest.fixture
def empty_state() -> ResearchState:
    """Minimal state with a research goal but no sub-queries."""
    return create_initial_state("Test research goal")


@pytest.fixture
def single_query_state() -> ResearchState:
    """State with one sub-query and valid status (should go straight to synthesizer)."""
    state = create_initial_state("Test research goal")
    state["sub_queries"] = ["What are transformers?"]
    state["validation_status"] = "valid"
    return state


@pytest.fixture
def multi_query_state() -> ResearchState:
    """State with multiple sub-queries, valid status."""
    state = create_initial_state("Test research goal")
    state["sub_queries"] = ["Query A", "Query B", "Query C"]
    state["validation_status"] = "valid"
    return state


@pytest.fixture
def failed_query_state() -> ResearchState:
    """State with invalid status and attempts >= MAX (should go to synthesizer)."""
    state = create_initial_state("Test research goal")
    state["sub_queries"] = ["Hard query"]
    state["validation_status"] = "invalid"
    state["researcher_attempts"] = 3
    return state


def _run_graph_and_get_logs(state: ResearchState) -> tuple[dict, list]:
    """Run the graph and return (final_state, node_names_logged)."""
    with patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log:
        mock_log.log_event = MagicMock()
        graph = build_graph()
        result = graph.invoke(state)
        # Collect node names from log calls
        logged_nodes = []
        for call in mock_log.log_event.call_args_list:
            args = call.args
            if args and len(args) >= 2:
                payload = args[1]
                if isinstance(payload, dict) and "node" in payload:
                    logged_nodes.append(payload["node"])
        return result, logged_nodes


def test_build_graph_returns_compiled_graph():
    """Test that build_graph returns a compiled graph instance."""
    graph = build_graph()
    assert graph is not None
    # Compiled graphs expose invoke
    assert hasattr(graph, "invoke")


def test_graph_runs_empty_state_to_end(empty_state):
    """Test graph terminates on empty state (no sub-queries -> synthesizer -> END)."""
    result, logged_nodes = _run_graph_and_get_logs(empty_state)
    # planner, researcher, validator, synthesizer should all run
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes
    # Result should be a dict (state)
    assert isinstance(result, dict)


def test_graph_runs_single_query(single_query_state):
    """Test graph with one valid query routes planner->researcher->validator->synthesizer."""
    result, logged_nodes = _run_graph_and_get_logs(single_query_state)
    # Validator sees valid + no more queries -> synthesizer
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes
    assert logged_nodes.index("planner") < logged_nodes.index("researcher")
    assert logged_nodes.index("researcher") < logged_nodes.index("synthesizer") or \
           logged_nodes[-1] == "synthesizer"


def test_graph_runs_multi_query(multi_query_state):
    """Test graph with multiple valid queries advances through all then synthesizes."""
    result, logged_nodes = _run_graph_and_get_logs(multi_query_state)
    # Should hit researcher at least once, validator, and synthesizer
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes
    # First node is planner
    assert logged_nodes[0] == "planner"


def test_graph_handles_failed_query(failed_query_state):
    """Test invalid + max attempts -> synthesizer without infinite loop."""
    result, logged_nodes = _run_graph_and_get_logs(failed_query_state)
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes


def test_graph_does_not_infinite_loop_on_invalid_retry():
    """Test that invalid + attempts < MAX retries but eventually exhausts."""
    state = create_initial_state("Test")
    state["sub_queries"] = ["q1"]
    state["validation_status"] = "invalid"
    state["researcher_attempts"] = 1  # < MAX, should retry once then exhaust

    result, logged_nodes = _run_graph_and_get_logs(state)
    # Should not loop forever — eventually reaches synthesizer
    assert "synthesizer" in logged_nodes
    # Researcher should appear at most MAX+1 times (initial + retries)
    researcher_count = logged_nodes.count("researcher")
    assert researcher_count <= 4  # 1 initial + 3 retries max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
