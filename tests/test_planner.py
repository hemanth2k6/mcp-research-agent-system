"""Tests for the planner agent goal decomposition."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from mcp_research_agent_system.agents.planner import (
    PlannerDecomposition,
    decompose_goal,
)
from mcp_research_agent_system.errors import PlannerError


class TestPlannerDecomposition:
    """Tests for the PlannerDecomposition Pydantic model."""

    def test_valid_decomposition(self):
        """Test valid 3-5 sub-queries."""
        decomp = PlannerDecomposition(sub_queries=["q1", "q2", "q3"])
        assert len(decomp.sub_queries) == 3

    def test_min_three_queries(self):
        """Test minimum 3 queries enforced."""
        with pytest.raises(ValidationError):
            PlannerDecomposition(sub_queries=["q1", "q2"])

    def test_max_five_queries(self):
        """Test maximum 5 queries enforced."""
        with pytest.raises(ValidationError):
            PlannerDecomposition(sub_queries=["q1", "q2", "q3", "q4", "q5", "q6"])


class TestDecomposeGoal:
    """Tests for the decompose_goal function."""

    def test_successful_structured_output(self):
        """Test successful decomposition via structured output on first attempt."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = PlannerDecomposition(
            sub_queries=["transformer NLP", "attention mechanism", "BERT benchmarks"]
        )
        mock_llm.with_structured_output.return_value = mock_structured

        result = decompose_goal("Test goal", llm=mock_llm)

        assert isinstance(result, PlannerDecomposition)
        assert len(result.sub_queries) == 3
        mock_llm.with_structured_output.assert_called_once_with(PlannerDecomposition)
        mock_structured.invoke.assert_called_once()

    def test_fallback_manual_parse_success(self):
        """Test fallback manual JSON parsing when structured output fails."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output not supported")
        mock_llm.with_structured_output.return_value = mock_structured

        # Mock the fallback invoke to return valid JSON
        mock_response = MagicMock()
        mock_response.content = '{"sub_queries": ["query A", "query B", "query C"]}'
        mock_llm.invoke.return_value = mock_response

        result = decompose_goal("Test goal", llm=mock_llm)

        assert isinstance(result, PlannerDecomposition)
        assert result.sub_queries == ["query A", "query B", "query C"]
        assert mock_structured.invoke.call_count == 1
        assert mock_llm.invoke.call_count == 1

    def test_fallback_extracts_json_from_markdown(self):
        """Test fallback extracts JSON from markdown code blocks."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # Response wrapped in markdown
        mock_response = MagicMock()
        mock_response.content = '```json\n{"sub_queries": ["q1", "q2", "q3"]}\n```'
        mock_llm.invoke.return_value = mock_response

        result = decompose_goal("Test goal", llm=mock_llm)

        assert result.sub_queries == ["q1", "q2", "q3"]

    def test_fallback_extracts_json_from_surrounding_text(self):
        """Test fallback extracts JSON from surrounding explanatory text."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = 'Here is the result: {"sub_queries": ["a", "b", "c"]} end.'
        mock_llm.invoke.return_value = mock_response

        result = decompose_goal("Test goal", llm=mock_llm)

        assert result.sub_queries == ["a", "b", "c"]

    def test_fallback_retry_on_first_failure(self):
        """Test fallback retries once when first attempt fails."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # First fallback attempt returns malformed JSON, second succeeds
        mock_response1 = MagicMock()
        mock_response1.content = "not json at all"
        mock_response2 = MagicMock()
        mock_response2.content = '{"sub_queries": ["q1", "q2", "q3"]}'
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]

        result = decompose_goal("Test goal", llm=mock_llm)

        assert result.sub_queries == ["q1", "q2", "q3"]
        assert mock_llm.invoke.call_count == 2

    def test_total_failure_raises_planner_error(self):
        """Test that PlannerError is raised after all retries exhausted."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # Both fallback attempts fail
        mock_response = MagicMock()
        mock_response.content = "completely invalid"
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(PlannerError) as exc_info:
            decompose_goal("Test goal", llm=mock_llm)

        assert "Failed to decompose research goal" in str(exc_info.value)
        assert mock_llm.invoke.call_count == 2  # max_retries = 2

    def test_planner_error_chains_original_exception(self):
        """Test PlannerError chains the last underlying exception."""
        mock_llm = MagicMock(spec=ChatOpenAI)
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = "invalid"
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(PlannerError) as exc_info:
            decompose_goal("Test goal", llm=mock_llm)

        assert exc_info.value.__cause__ is not None


class TestPlannerNodeIntegration:
    """Integration tests for planner_node in the graph."""

    @patch("mcp_research_agent_system.agents.graph.decompose_goal")
    @patch("mcp_research_agent_system.agents.graph.logging_utils")
    def test_planner_node_populates_sub_queries(self, mock_log, mock_decompose):
        """Test planner_node calls decompose_goal and populates state."""
        from mcp_research_agent_system.agents.graph import planner_node
        from mcp_research_agent_system.agents.state import create_initial_state

        mock_decompose.return_value = PlannerDecomposition(
            sub_queries=["query 1", "query 2", "query 3"]
        )

        state = create_initial_state("Research goal")
        result = planner_node(state)

        assert "sub_queries" in result
        assert len(result["sub_queries"]) == 3
        assert result["current_query_index"] == 0
        assert result["researcher_attempts"] == 0
        assert result["validation_status"] == "pending"
        mock_decompose.assert_called_once_with("Research goal")

    @patch("mcp_research_agent_system.agents.graph.decompose_goal")
    @patch("mcp_research_agent_system.agents.graph.logging_utils")
    def test_planner_node_passthrough_when_sub_queries_exist(self, mock_log, mock_decompose):
        """Test planner_node does not re-decompose when sub_queries already present."""
        from mcp_research_agent_system.agents.graph import planner_node
        from mcp_research_agent_system.agents.state import ResearchState

        state = ResearchState(
            research_goal="Goal",
            sub_queries=["existing"],
            current_query_index=0,
            researcher_attempts=0,
            researcher_output=[],
            validated_findings=[],
            validation_status="pending",
            final_report=None,
            error=None,
        )
        result = planner_node(state)

        assert result["sub_queries"] == ["existing"]
        mock_decompose.assert_not_called()

    @patch("mcp_research_agent_system.agents.graph.decompose_goal")
    @patch("mcp_research_agent_system.agents.graph.logging_utils")
    def test_planner_node_propagates_error(self, mock_log, mock_decompose):
        """Test planner_node propagates PlannerError."""
        from mcp_research_agent_system.agents.graph import planner_node
        from mcp_research_agent_system.agents.state import create_initial_state
        from mcp_research_agent_system.errors import PlannerError

        mock_decompose.side_effect = PlannerError("Decomposition failed")

        state = create_initial_state("Goal")

        with pytest.raises(PlannerError):
            planner_node(state)

        # Should have logged entry and error
        assert mock_log.log_event.call_count == 2
        entry_call = mock_log.log_event.call_args_list[0]
        error_call = mock_log.log_event.call_args_list[1]
        assert entry_call.args[1]["phase"] == "entry"
        assert error_call.args[1]["phase"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
