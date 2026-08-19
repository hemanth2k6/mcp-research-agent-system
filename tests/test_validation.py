"""Tests for planner validation with retry loop.

Tests cover:
1. Heuristic catches empty result -> invalid, no LLM call made
2. Heuristic catches clearly off-topic result -> invalid
3. Ambiguous case -> falls through to mocked LLM-judge, test both judge verdicts
4. Full graph integration: force researcher to return bad results twice, then good
   results on 3rd attempt - assert graph retried correctly, attempts capped at 3.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from mcp_research_agent_system.agents.graph import build_graph, validator_node
from mcp_research_agent_system.agents.planner import (
    ValidationOutcome,
    validate_research_output,
)
from mcp_research_agent_system.agents.researcher import PaperResult, ResearchResult
from mcp_research_agent_system.agents.state import (
    ResearchState,
    create_initial_state,
)


def _make_paper(title: str, abstract: str) -> PaperResult:
    """Helper to create PaperResult with required fields."""
    return PaperResult(
        arxiv_id="2401.12345v1",
        title=title,
        authors=["Author A"],
        abstract=abstract,
        category="cs.AI",
        published_date="2024-01-09T00:00:00+00:00",
        updated_date="2024-01-10T00:00:00+00:00",
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )


def _make_research_result(sub_query: str, papers: list[PaperResult]) -> ResearchResult:
    """Helper to create ResearchResult with required fields."""
    return ResearchResult(
        sub_query=sub_query,
        papers=papers,
        cached_summaries=[],
        raw_tool_calls=[],
    )


class TestHeuristicValidation:
    """Tests for heuristic-first validation (no LLM call)."""

    async def test_empty_result_invalid_no_llm_call(self):
        """Test empty papers -> invalid without invoking LLM."""
        sub_query = "transformer attention mechanisms nlp"
        result = _make_research_result(sub_query, [])

        with patch("mcp_research_agent_system.agents.planner.get_llm") as mock_get_llm:
            outcome = await validate_research_output(sub_query, result)
            # get_llm should NOT be called for empty result (heuristic catches it)
            mock_get_llm.assert_not_called()

        assert outcome.is_valid is False
        assert "zero papers" in outcome.reason.lower()
        assert outcome.revised_query is not None
        assert outcome.revised_query != sub_query

    async def test_off_topic_result_invalid(self):
        """Test clearly off-topic papers -> invalid via heuristic."""
        sub_query = "quantum computing error correction"
        papers = [
            _make_paper(
                "Deep Learning for Image Classification",
                "This paper surveys convolutional neural networks for computer vision tasks.",
            ),
            _make_paper(
                "Reinforcement Learning in Robotics",
                "We present an approach using policy gradients for robotic control.",
            ),
        ]
        result = _make_research_result(sub_query, papers)

        with patch("mcp_research_agent_system.agents.planner.get_llm") as mock_get_llm:
            outcome = await validate_research_output(sub_query, result)
            mock_get_llm.assert_not_called()

        assert outcome.is_valid is False
        assert outcome.revised_query is not None

    async def test_relevant_result_valid_heuristic(self):
        """Test relevant papers -> valid via heuristic."""
        sub_query = "transformer attention mechanism NLP"
        papers = [
            _make_paper(
                "Attention Is All You Need",
                "We propose the Transformer, a model architecture relying entirely on attention mechanisms.",
            ),
            _make_paper(
                "Efficient Transformers for NLP",
                "This work surveys transformer-based architectures for natural language processing.",
            ),
        ]
        result = _make_research_result(sub_query, papers)

        with patch("mcp_research_agent_system.agents.planner.get_llm") as mock_get_llm:
            outcome = await validate_research_output(sub_query, result)
            mock_get_llm.assert_not_called()

        assert outcome.is_valid is True
        assert outcome.revised_query is None


class TestLLMJudgeFallback:
    """Tests for LLM-judge fallback in ambiguous cases."""

    async def test_ambiguous_falls_through_to_llm_judge_valid(self):
        """Test ambiguous case -> LLM judge called, returns valid."""
        sub_query = "graph neural networks for molecular property prediction"
        papers = [
            _make_paper(
                "Graph Neural Networks: A Review",
                "We review graph neural network architectures and their applications.",
            ),
            _make_paper(
                "Machine Learning for Chemistry",
                "This survey covers machine learning methods applied to chemical problems.",
            ),
        ]
        result = _make_research_result(sub_query, papers)

        # Mock the LLM judge to return valid
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
            is_valid=True,
            reason="Papers are relevant to graph neural networks and chemistry",
            revised_query=None,
        ))
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm
        ):
            outcome = await validate_research_output(sub_query, result)

        assert outcome.is_valid is True
        mock_structured.ainvoke.assert_called()

    async def test_ambiguous_falls_through_to_llm_judge_invalid(self):
        """Test ambiguous case -> LLM judge called, returns invalid with revised query."""
        sub_query = "graph neural networks for molecular property prediction"
        papers = [
            _make_paper(
                "Graph Neural Networks: A Review",
                "We review graph neural network architectures and their applications.",
            ),
            _make_paper(
                "Machine Learning for Chemistry",
                "This survey covers machine learning methods applied to chemical problems.",
            ),
        ]
        result = _make_research_result(sub_query, papers)

        # Mock the LLM judge to return invalid
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=ValidationOutcome(
            is_valid=False,
            reason="Papers are tangentially related but not directly about molecular property prediction",
            revised_query="graph neural networks molecular property prediction drug discovery",
        ))
        mock_llm.with_structured_output.return_value = mock_structured

        with patch(
            "mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm
        ):
            outcome = await validate_research_output(sub_query, result)

        assert outcome.is_valid is False
        assert outcome.revised_query is not None
        assert outcome.revised_query != sub_query
        mock_structured.ainvoke.assert_called()


class TestValidatorNodeIntegration:
    """Integration tests for validator_node with retry loop in graph."""

    async def test_retry_loop_then_success(self):
        """Test graph retries with bad results, then succeeds."""
        # Setup: single sub-query, will start with attempts=0
        state = create_initial_state("Research goal")
        state["sub_queries"] = ["quantum error correction"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"

        # Side effect sequence:
        # 1st call: returns bad (empty) result -> validator says invalid, retry
        # 2nd call: returns bad (off-topic) result -> validator says invalid, retry
        # 3rd call: returns good result -> validator says valid
        valid_paper = _make_paper(
            "Quantum Error Correction with Surface Codes",
            "We present surface code implementations for fault-tolerant quantum computing.",
        )

        bad_empty = _make_research_result("quantum error correction", [])
        bad_offtopic = _make_research_result(
            "quantum error correction",
            [_make_paper("Cooking Recipes", "A book about cooking.")],
        )
        good_result = _make_research_result("quantum error correction", [valid_paper])

        results_sequence = [bad_empty, bad_offtopic, good_result]

        run_research_mock = AsyncMock(side_effect=results_sequence)

        # Mock synthesizer LLM
        from mcp_research_agent_system.agents.synthesizer import SynthesizedReport
        mock_synthesizer_structured = AsyncMock()
        mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
            report="# Test Report\n\n## Overview\nTest\n\n## Key Themes\nTheme 1\n\n## Notable Papers\nPaper 1\n\n## Gaps / Open Questions\nGap 1"
        ))

        def synthesizer_get_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output = MagicMock(return_value=mock_synthesizer_structured)
            return mock_llm

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ), patch(
            "mcp_research_agent_system.agents.graph.decompose_goal"
        ) as mock_decompose, patch(
            "mcp_research_agent_system.agents.graph.logging_utils"
        ) as mock_log, patch(
            "mcp_research_agent_system.agents.synthesizer.get_llm", side_effect=synthesizer_get_llm
        ):
            mock_log.log_event = MagicMock()
            mock_log.log_node_entry = MagicMock(return_value=0.0)
            mock_log.log_node_exit = MagicMock()
            mock_log.log_node_error = MagicMock()
            mock_log.safe_state_snapshot = MagicMock(return_value={})
            mock_decompose.return_value = MagicMock(sub_queries=["quantum error correction"])

            graph = build_graph()
            result = await graph.ainvoke(state)

        # Researcher called 3 times (initial + 2 retries before success on 3rd)
        assert run_research_mock.call_count == 3
        assert result["validation_status"] == "valid"
        # Check logged_nodes from log_node_entry calls
        logged_nodes = [call.args[0] for call in mock_log.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes
        assert len(result["validated_findings"]) >= 1

    async def test_attempts_capped_at_max_then_exhausted(self):
        """Test that attempts capped at MAX_RESEARCHER_ATTEMPTS and graph proceeds to synthesizer."""
        from mcp_research_agent_system.agents.graph import MAX_RESEARCHER_ATTEMPTS

        state = create_initial_state("Research goal")
        state["sub_queries"] = ["impossible query xyz"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"

        # All results are bad (empty)
        bad_result = _make_research_result("impossible query xyz", [])

        run_research_mock = AsyncMock(return_value=bad_result)

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ), patch(
            "mcp_research_agent_system.agents.graph.decompose_goal"
        ) as mock_decompose, patch(
            "mcp_research_agent_system.agents.graph.logging_utils"
        ) as mock_log:
            mock_log.log_event = MagicMock()
            mock_log.log_node_entry = MagicMock(return_value=0.0)
            mock_log.log_node_exit = MagicMock()
            mock_log.log_node_error = MagicMock()
            mock_log.safe_state_snapshot = MagicMock(return_value={})
            mock_decompose.return_value = MagicMock(sub_queries=["impossible query xyz"])

            graph = build_graph()
            result = await graph.ainvoke(state)

        # Researcher called MAX_RESEARCHER_ATTEMPTS times (initial + MAX-1 retries before exhausting)
        # After MAX_RESEARCHER_ATTEMPTS calls, validator exhausts and goes to synthesizer
        assert run_research_mock.call_count == MAX_RESEARCHER_ATTEMPTS
        assert result["researcher_attempts"] == MAX_RESEARCHER_ATTEMPTS
        assert result["validation_status"] == "invalid"
        # Check logged_nodes from log_node_entry calls
        logged_nodes = [call.args[0] for call in mock_log.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes

        # The router handles exhausted case without calling validator again,
        # so "invalid_exhausted" is not logged. Verify graph proceeds to synthesizer.
        # The validator logs "invalid_retry" for the final retry, then router goes to synthesizer
        validator_exits = [
            call.kwargs.get("status") for call in mock_log.log_node_exit.call_args_list
            if call.args[0] == "validator"
        ]
        # All validator exits should be "invalid_retry" (never "invalid_exhausted" in current flow)
        assert all(status == "invalid_retry" for status in validator_exits)
        # Synthesizer should be called
        assert "synthesizer" in logged_nodes

    async def test_validator_node_updates_sub_query_on_retry(self):
        """Test that validator_node updates sub_queries[idx] to revised query on retry."""
        state: ResearchState = create_initial_state("Research goal")
        state["sub_queries"] = ["original query"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        state["researcher_output"] = []  # empty -> heuristic invalid

        result = await validator_node(state)

        assert result["validation_status"] == "invalid"
        assert result["researcher_attempts"] == 1
        # The sub_query at index 0 should be changed to the revised query
        assert result["sub_queries"][0] != "original query"
        assert result["sub_queries"][0] is not None

    async def test_validator_node_valid_advances(self):
        """Test that validator_node advances to next query when valid and more remain."""
        state: ResearchState = create_initial_state("Research goal")
        state["sub_queries"] = ["query A", "query B"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        state["researcher_output"] = [
            {
                "arxiv_id": "2401.12345v1",
                "title": "Query A Related Paper",
                "authors": ["Author"],
                "abstract": "This paper discusses query A topics extensively.",
                "category": "cs.AI",
                "published_date": "2024-01-09T00:00:00+00:00",
                "updated_date": "2024-01-10T00:00:00+00:00",
                "pdf_url": "http://arxiv.org/pdf/2401.12345v1",
            }
        ]

        result = await validator_node(state)

        assert result["validation_status"] == "pending"  # reset for next query
        assert result["current_query_index"] == 1
        assert result["researcher_attempts"] == 0
        assert len(result["validated_findings"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
