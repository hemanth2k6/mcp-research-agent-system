"""Tests for the LangGraph state machine skeleton."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_research_agent_system.agents.graph import build_graph
from mcp_research_agent_system.agents.planner import PlannerDecomposition
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


async def _run_graph_and_get_logs(state: ResearchState) -> tuple[dict, list]:
    """Run the graph and return (final_state, node_names_logged)."""
    from mcp_research_agent_system.agents.planner import ValidationOutcome
    from mcp_research_agent_system.agents.synthesizer import SynthesizedReport

    # Create a MagicMock for the LLM (not AsyncMock) so with_structured_output works as a regular method
    mock_llm = MagicMock()

    # For synthesizer - structured output
    mock_synthesizer_structured = AsyncMock()
    mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
        report="# Test Report\n\n## Overview\nTest\n\n## Key Themes\nTheme 1\n\n## Notable Papers\nPaper 1\n\n## Gaps / Open Questions\nGap 1"
    ))

    # For planner - need to mock invoke for fallback
    mock_planner_response = MagicMock()
    mock_planner_response.content = '{"sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}'
    mock_llm.invoke = MagicMock(return_value=mock_planner_response)

    # For validator - mock LLM judge structured output
    mock_validator_structured = AsyncMock()
    mock_validator_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
        is_valid=True,
        reason="Papers are relevant to the sub-query",
        revised_query=None,
    ))

    def with_structured_output_side_effect(model_class):
        if model_class.__name__ == "SynthesizedReport":
            return mock_synthesizer_structured
        elif model_class.__name__ == "ValidationOutcome":
            return mock_validator_structured
        return AsyncMock()

    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output_side_effect)

    with (
        patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log,
        patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose,
        patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm),
        patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm),
    ):
        mock_log.log_event = MagicMock()
        mock_log.log_node_entry = MagicMock(return_value=0.0)
        mock_log.log_node_exit = MagicMock()
        mock_log.log_node_error = MagicMock()
        mock_log.safe_state_snapshot = MagicMock(return_value={})
        mock_decompose.return_value = PlannerDecomposition(
            sub_queries=["sub-query 1", "sub-query 2", "sub-query 3"]
        )
        graph = build_graph()
        result = await graph.ainvoke(state)
        # Collect node names from log calls - new format uses node_entry/node_exit/node_error events
        logged_nodes = []
        for call in mock_log.log_event.call_args_list:
            args = call.args
            if args and len(args) >= 2:
                payload = args[1]
                if isinstance(payload, dict) and "node_name" in payload:
                    logged_nodes.append(payload["node_name"])
        # Also collect from log_node_entry calls
        for call in mock_log.log_node_entry.call_args_list:
            args = call.args
            if args and len(args) >= 1:
                logged_nodes.append(args[0])
        return result, logged_nodes


@pytest.fixture
def mock_graph_setup():
    """Complete mock setup for graph tests with all LLM mocks."""
    from mcp_research_agent_system.agents.planner import ValidationOutcome
    from mcp_research_agent_system.agents.synthesizer import SynthesizedReport

    mock_llm = MagicMock()
    mock_synthesizer_structured = AsyncMock()
    mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
        report="# Test Report\n\n## Overview\nTest\n\n## Key Themes\nTheme 1\n\n## Notable Papers\nPaper 1\n\n## Gaps / Open Questions\nGap 1"
    ))

    mock_planner_response = MagicMock()
    mock_planner_response.content = '{"sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}'
    mock_llm.invoke = MagicMock(return_value=mock_planner_response)

    mock_validator_structured = AsyncMock()
    mock_validator_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
        is_valid=True,
        reason="Papers are relevant to the sub-query",
        revised_query=None,
    ))

    def with_structured_output_side_effect(model_class):
        if model_class.__name__ == "SynthesizedReport":
            return mock_synthesizer_structured
        elif model_class.__name__ == "ValidationOutcome":
            return mock_validator_structured
        return AsyncMock()

    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output_side_effect)

    with (
        patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log,
        patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose,
        patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm),
        patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm),
    ):
        mock_log.log_event = MagicMock()
        mock_log.log_node_entry = MagicMock(return_value=0.0)
        mock_log.log_node_exit = MagicMock()
        mock_log.log_node_error = MagicMock()
        mock_log.safe_state_snapshot = MagicMock(return_value={})
        mock_decompose.return_value = PlannerDecomposition(
            sub_queries=["sub-query 1", "sub-query 2", "sub-query 3"]
        )
        graph = build_graph()
        yield graph, mock_log


async def test_researcher_node_no_sub_query_error(mock_graph_setup):
    """Test researcher_node handles missing sub-query at index (line 90-101)."""
    graph, mock_log = mock_graph_setup

    # State with invalid current_query_index (beyond sub_queries length)
    state = create_initial_state("Test goal")
    state["sub_queries"] = ["only one query"]
    state["current_query_index"] = 5  # Beyond length
    state["validation_status"] = "pending"

    result = await graph.ainvoke(state)

    # Should log error and proceed
    error_logged = any(
        call.args[0] == "node_error" and "No sub-query available" in str(call.args[1])
        for call in mock_log.log_event.call_args_list
    )
    assert error_logged or result.get("error") is not None


async def test_researcher_node_handles_researcher_error(mock_graph_setup):
    """Test researcher_node handles ResearcherError from run_research (lines 126-133)."""
    from mcp_research_agent_system.agents.researcher import ResearcherError

    graph, mock_log = mock_graph_setup

    state = create_initial_state("Test goal")
    state["sub_queries"] = ["valid query"]
    state["current_query_index"] = 0
    state["validation_status"] = "pending"

    with patch("mcp_research_agent_system.agents.graph.run_research") as mock_run:
        mock_run.side_effect = ResearcherError("MCP server failed")

        result = await graph.ainvoke(state)

    # Should log error and continue to validator
    assert "error" in result or result.get("researcher_output") == []


async def test_validator_node_no_sub_query_error(mock_graph_setup):
    """Test validator_node handles missing sub-query at index (lines 175-185)."""
    graph, mock_log = mock_graph_setup

    state = create_initial_state("Test goal")
    state["sub_queries"] = ["only one"]
    state["current_query_index"] = 5  # Beyond length
    state["validation_status"] = "pending"
    state["researcher_output"] = []

    result = await graph.ainvoke(state)

    # Should log error and set validation_status to invalid
    assert result.get("validation_status") == "invalid"


async def test_validator_node_exhausted_attempts():
    """Test validator_node exhausted attempts path (lines 278-292)."""
    from mcp_research_agent_system.agents.graph import build_graph
    from mcp_research_agent_system.agents.planner import ValidationOutcome
    from mcp_research_agent_system.agents.state import create_initial_state
    from mcp_research_agent_system.agents.synthesizer import SynthesizedReport

    mock_llm = MagicMock()
    mock_synthesizer_structured = AsyncMock()
    mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
        report="# Test Report\n\n## Overview\nTest\n\n## Key Themes\nTheme 1\n\n## Notable Papers\nPaper 1\n\n## Gaps / Open Questions\nGap 1"
    ))

    mock_planner_response = MagicMock()
    mock_planner_response.content = '{"sub_queries": ["hard query"]}'
    mock_llm.invoke = MagicMock(return_value=mock_planner_response)

    mock_validator_structured = AsyncMock()
    mock_validator_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
        is_valid=False,
        reason="Papers are not relevant",
        revised_query="revised query",
    ))

    def with_structured_output_side_effect(model_class):
        if model_class.__name__ == "SynthesizedReport":
            return mock_synthesizer_structured
        elif model_class.__name__ == "ValidationOutcome":
            return mock_validator_structured
        return AsyncMock()

    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output_side_effect)

    with (
        patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log,
        patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose,
        patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm),
        patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm),
        patch("mcp_research_agent_system.agents.graph.run_research") as mock_run_research,
    ):
        mock_log.log_event = MagicMock()
        mock_log.log_node_entry = MagicMock(return_value=0.0)
        mock_log.log_node_exit = MagicMock()
        mock_log.log_node_error = MagicMock()
        mock_log.safe_state_snapshot = MagicMock(return_value={})
        mock_decompose.return_value = MagicMock(sub_queries=["hard query"])
        mock_run_research.return_value = MagicMock(papers=[])

        graph = build_graph()

        state = create_initial_state("Test goal")
        state["sub_queries"] = ["hard query"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 3  # At MAX
        state["validation_status"] = "pending"
        state["researcher_output"] = []

        result = await graph.ainvoke(state)

        # Should set error and validation_status invalid, proceed to synthesizer
        assert result.get("validation_status") == "invalid"
        assert "error" in result
        logged_nodes = [call.args[0] for call in mock_log.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes


async def test_synthesizer_node_error_handling():
    """Test synthesizer_node error handling (lines 331-338)."""
    from mcp_research_agent_system.agents.graph import build_graph
    from mcp_research_agent_system.agents.planner import ValidationOutcome
    from mcp_research_agent_system.agents.state import create_initial_state

    mock_llm = MagicMock()
    mock_synthesizer_structured = AsyncMock()
    mock_synthesizer_structured.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))
    # Also mock the fallback invoke
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

    mock_planner_response = MagicMock()
    mock_planner_response.content = '{"sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"]}'
    mock_llm.invoke = MagicMock(return_value=mock_planner_response)

    mock_validator_structured = AsyncMock()
    mock_validator_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
        is_valid=True,
        reason="Papers are relevant to the sub-query",
        revised_query=None,
    ))

    def with_structured_output_side_effect(model_class):
        if model_class.__name__ == "SynthesizedReport":
            return mock_synthesizer_structured
        elif model_class.__name__ == "ValidationOutcome":
            return mock_validator_structured
        return AsyncMock()

    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output_side_effect)

    with (
        patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log,
        patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose,
        patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm),
        patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm),
    ):
        mock_log.log_event = MagicMock()
        mock_log.log_node_entry = MagicMock(return_value=0.0)
        mock_log.log_node_exit = MagicMock()
        mock_log.log_node_error = MagicMock()
        mock_log.safe_state_snapshot = MagicMock(return_value={})
        mock_decompose.return_value = PlannerDecomposition(
            sub_queries=["sub-query 1", "sub-query 2", "sub-query 3"]
        )
        graph = build_graph()

        state = create_initial_state("Test goal")
        state["validated_findings"] = [{"title": "test"}]
        state["validation_status"] = "valid"
        state["sub_queries"] = ["q1", "q2", "q3"]
        state["current_query_index"] = 3

        # Graph should raise the error
        try:
            await graph.ainvoke(state)
            raise AssertionError("Expected exception")
        except Exception as e:
            assert "LLM failed" in str(e) or "synthesize" in str(e).lower()

        # Error should be logged via log_node_error (which calls log_event internally)
        # Check log_node_error was called with "synthesizer" and the error
        error_logged = any(
            call.args[0] == "synthesizer" and ("LLM failed" in str(call.args[1]) or "synthesize" in str(call.args[1]).lower())
            for call in mock_log.log_node_error.call_args_list
        )
        assert error_logged


async def test_router_invalid_exhausted_goes_to_synthesizer(mock_graph_setup):
    """Test router routes exhausted invalid to synthesizer (line 360)."""
    graph, mock_log = mock_graph_setup

    state = create_initial_state("Test goal")
    state["sub_queries"] = ["query"]
    state["current_query_index"] = 0
    state["researcher_attempts"] = 3  # At MAX
    state["validation_status"] = "invalid"
    state["researcher_output"] = []

    await graph.ainvoke(state)

    # Should go to synthesizer, not loop
    logged_nodes = [call.args[0] for call in mock_log.log_node_entry.call_args_list]
    assert "synthesizer" in logged_nodes
    # Validator should not be called again after exhaustion
    validator_count = logged_nodes.count("validator")
    # Validator runs once, then router sends to synthesizer
    assert validator_count == 1


def test_build_graph_returns_compiled_graph():
    """Test that build_graph returns a compiled graph instance."""
    graph = build_graph()
    assert graph is not None
    # Compiled graphs expose invoke
    assert hasattr(graph, "invoke")


def test_graph_runs_empty_state_to_end(empty_state):
    """Test graph terminates on empty state (no sub-queries -> synthesizer -> END)."""
    result, logged_nodes = asyncio.run(_run_graph_and_get_logs(empty_state))
    # planner, researcher, validator, synthesizer should all run
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes
    # Result should be a dict (state)
    assert isinstance(result, dict)


def test_graph_runs_single_query(single_query_state):
    """Test graph with one valid query routes planner->researcher->validator->synthesizer."""
    result, logged_nodes = asyncio.run(_run_graph_and_get_logs(single_query_state))
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
    result, logged_nodes = asyncio.run(_run_graph_and_get_logs(multi_query_state))
    # Should hit researcher at least once, validator, and synthesizer
    assert "planner" in logged_nodes
    assert "researcher" in logged_nodes
    assert "validator" in logged_nodes
    assert "synthesizer" in logged_nodes
    # First node is planner
    assert logged_nodes[0] == "planner"


def test_graph_handles_failed_query(failed_query_state):
    """Test invalid + max attempts -> synthesizer without infinite loop."""
    result, logged_nodes = asyncio.run(_run_graph_and_get_logs(failed_query_state))
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

    result, logged_nodes = asyncio.run(_run_graph_and_get_logs(state))
    # Should not loop forever — eventually reaches synthesizer
    assert "synthesizer" in logged_nodes
    # Researcher should appear at most MAX+1 times (initial + retries)
    researcher_count = logged_nodes.count("researcher")
    assert researcher_count <= 4  # 1 initial + 3 retries max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
