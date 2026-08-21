"""Tests for the planner agent goal decomposition."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from mcp_research_agent_system.agents.planner import (
    PlannerDecomposition,
    _parse_llm_json_response,
    _token_overlap,
    decompose_goal,
)
from mcp_research_agent_system.agents.researcher import PaperResult, ResearchResult
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


class TestParseLlmJsonResponse:
    """Tests for _parse_llm_json_response error paths."""

    def test_fallback_json_extraction_failure_raises_planner_error(self):
        """Test that JSON-looking-but-invalid content raises PlannerError (lines 72-73)."""
        # Contains a {.*} match but the extracted JSON is invalid
        with pytest.raises(PlannerError) as exc_info:
            _parse_llm_json_response('{"sub_queries": ["q1", "q2", "q3"')  # truncated JSON

        assert "Failed to parse LLM response" in str(exc_info.value)

    def test_no_braces_raises_planner_error(self):
        """Test that content without any braces raises PlannerError."""
        with pytest.raises(PlannerError):
            _parse_llm_json_response("no json here at all")


class TestTokenOverlap:
    """Tests for _token_overlap edge cases."""

    def test_empty_query_returns_zero(self):
        """Test empty query returns 0.0 (line 84)."""
        assert _token_overlap("", "some text") == 0.0

    def test_empty_text_returns_zero(self):
        """Test empty text returns 0.0 (line 84)."""
        assert _token_overlap("some query", "") == 0.0

    def test_whitespace_only_query_returns_zero(self):
        """Test whitespace-only query yields no tokens -> 0.0 (line 90)."""
        # Punctuation-only strings produce zero \w+ tokens
        assert _token_overlap("!!! ???", "some text") == 0.0

    def test_full_overlap_returns_one(self):
        """Test identical token sets return 1.0."""
        assert _token_overlap("hello world", "world hello") == 1.0

    def test_partial_overlap(self):
        """Test partial overlap returns ratio."""
        assert _token_overlap("hello world foo", "hello world") == pytest.approx(2 / 3)


class TestSuggestRevisedQuery:
    """Tests for _suggest_revised_query fallback branch."""

    def test_unknown_failure_reason_appends_paper(self):
        """Test unknown failure reason appends ' paper' (line 162)."""
        from mcp_research_agent_system.agents.planner import _suggest_revised_query

        result = _suggest_revised_query("quantum computing", "validation error")
        assert result == "quantum computing paper"

    def test_short_query_no_results_appends_recent(self):
        """Test short query with 'no results' appends ' recent'."""
        from mcp_research_agent_system.agents.planner import _suggest_revised_query

        result = _suggest_revised_query("quantum", "no results")
        assert result == "quantum recent"

    def test_long_query_no_results_truncates_to_three_words(self):
        """Test long query with 'no results' keeps first three words."""
        from mcp_research_agent_system.agents.planner import _suggest_revised_query

        result = _suggest_revised_query("quantum error correction surface codes", "no results")
        assert result == "quantum error correction"

    def test_off_topic_appends_survey(self):
        """Test 'off-topic results' appends ' survey'."""
        from mcp_research_agent_system.agents.planner import _suggest_revised_query

        result = _suggest_revised_query("quantum computing", "off-topic results")
        assert result == "quantum computing survey"


class TestLLMJudgeFallbackParsing:
    """Tests for the LLM judge's manual-parse fallback path (lines 224-236)."""

    @pytest.fixture
    def ambiguous_result(self) -> ResearchResult:
        """Create a research result with ambiguous overlap to trigger LLM judge.

        Uses low-overlap papers that will pass empty check but fail high-overlap check,
        falling into the ambiguous zone (0.1 <= overlap < 0.3) where LLM judge is called.
        """
        papers = [
            PaperResult(
                arxiv_id="2401.00001v1",
                title="Graph Neural Networks: A Review",
                authors=["Author A"],
                abstract="We review graph neural network architectures and their applications in computer vision.",
                category="cs.LG",
                published_date="2024-01-01T00:00:00+00:00",
                updated_date="2024-01-02T00:00:00+00:00",
                pdf_url="http://arxiv.org/pdf/2401.00001v1",
            ),
        ]
        return ResearchResult(
            sub_query="graph neural networks for molecular property prediction drug discovery",
            papers=papers,
            cached_summaries=[],
            raw_tool_calls=[],
        )

    async def test_structured_output_fails_then_fallback_parses_direct_json(
        self, ambiguous_result
    ):
        """Structured output raises; fallback llm.ainvoke returns valid JSON (lines 224-230)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = '{"is_valid": true, "reason": "relevant", "revised_query": null}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm
        ):
            from mcp_research_agent_system.agents.planner import validate_research_output

            outcome = await validate_research_output(
                "graph neural networks for molecular property prediction drug discovery", ambiguous_result
            )

        assert outcome.is_valid is True
        assert mock_llm.ainvoke.call_count == 1

    async def test_fallback_extracts_json_from_surrounding_text(self, ambiguous_result):
        """Fallback response wraps JSON in prose; regex extraction succeeds (lines 233-236)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = (
            'My judgment: {"is_valid": false, "reason": "tangential", '
            '"revised_query": "gnn molecular property prediction"} hope that helps'
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm
        ):
            from mcp_research_agent_system.agents.planner import validate_research_output

            outcome = await validate_research_output(
                "graph neural networks for molecular property prediction drug discovery", ambiguous_result
            )

        assert outcome.is_valid is False
        assert outcome.revised_query == "gnn molecular property prediction"

    async def test_all_fallback_attempts_fail_defaults_invalid(self, ambiguous_result):
        """Both fallback attempts fail -> default invalid ValidationOutcome (lines 243-248)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = MagicMock()
        mock_response.content = "completely unparseable"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm
        ):
            from mcp_research_agent_system.agents.planner import validate_research_output

            outcome = await validate_research_output(
                "graph neural networks for molecular property prediction drug discovery", ambiguous_result
            )

        # Defaults to invalid with a generic revised query after retries exhausted
        assert outcome.is_valid is False
        assert "failed after retries" in outcome.reason.lower()
        assert outcome.revised_query is not None
        assert mock_llm.ainvoke.call_count == 2  # max_retries = 2


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

        # Should have logged entry and error using new logging helpers
        assert mock_log.log_node_entry.call_count == 1
        assert mock_log.log_node_error.call_count == 1
        entry_call = mock_log.log_node_entry.call_args_list[0]
        error_call = mock_log.log_node_error.call_args_list[0]
        assert entry_call.args[0] == "planner"
        assert error_call.args[0] == "planner"
        assert "Decomposition failed" in error_call.args[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
