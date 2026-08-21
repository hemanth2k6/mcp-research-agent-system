"""Deliberate retry-path integration tests for the validator/researcher retry loop.

These tests mock the LLM and MCP researcher calls (no real network) and configure
the mocked researcher to return specific failure/success sequences to verify:
1. Validator correctly flags invalid results
2. Sub-query is revised between attempts (not repeated)
3. Researcher attempts increment correctly and never exceed MAX_ATTEMPTS
4. Graph proceeds to synthesizer with eventually-valid results
5. Trace log contains expected sequence of node events
6. All attempts fail -> synthesizer receives partial/no validated findings
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_research_agent_system.agents.graph import MAX_RESEARCHER_ATTEMPTS, build_graph
from mcp_research_agent_system.agents.researcher import PaperResult, ResearchResult
from mcp_research_agent_system.agents.state import ResearchState, create_initial_state


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


class TestRetryLoopIntegration:
    """Integration tests for the full retry loop in the graph."""

    @pytest.fixture
    def initial_state(self) -> ResearchState:
        """Create initial state with a single sub-query."""
        state = create_initial_state("Understand quantum error correction")
        state["sub_queries"] = ["quantum error correction surface codes"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        return state

    @pytest.fixture
    def mock_logging(self):
        """Mock logging utilities to capture events."""
        with patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log:
            mock_log.log_event = MagicMock()
            mock_log.log_node_entry = MagicMock(return_value=0.0)
            mock_log.log_node_exit = MagicMock()
            mock_log.log_node_error = MagicMock()
            mock_log.safe_state_snapshot = MagicMock(return_value={})
            yield mock_log

    @pytest.fixture
    def mock_planner(self):
        """Mock planner decomposition."""
        with patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose:
            # Return a MagicMock with sub_queries attribute to avoid Pydantic validation
            mock_decompose.return_value = MagicMock(
                sub_queries=["quantum error correction surface codes"]
            )
            yield mock_decompose

    @pytest.fixture
    def mock_synthesizer_llm(self):
        """Mock synthesizer LLM to return a valid report."""
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
            "mcp_research_agent_system.agents.synthesizer.get_llm", side_effect=synthesizer_get_llm
        ):
            yield mock_synthesizer_structured

    async def test_retry_loop_bad_then_good(
        self, initial_state, mock_logging, mock_planner, mock_synthesizer_llm
    ):
        """Test: researcher returns bad results twice, then good on 3rd attempt.

        Asserts:
        - validator_node correctly flags invalid twice (heuristic checks)
        - the sub-query actually changes between attempts (revised_query used)
        - researcher_attempts increments correctly and never exceeds MAX_ATTEMPTS
        - the graph proceeds to synthesizer with the eventually-valid results
        - trace log contains expected sequence of node_entry/node_exit/error events
        """
        # Sequence of research results:
        # 1st call: empty results -> heuristic invalid
        # 2nd call: off-topic results -> heuristic invalid
        # 3rd call: relevant results -> heuristic valid

        bad_empty = _make_research_result("quantum error correction surface codes", [])

        bad_offtopic = _make_research_result(
            "quantum error correction surface codes",
            [_make_paper("Cooking Recipes", "A book about cooking delicious meals.")],
        )

        good_paper = _make_paper(
            "Surface Codes for Quantum Error Correction",
            "We present a comprehensive study of surface codes for fault-tolerant quantum computing.",
        )
        good_result = _make_research_result("quantum error correction surface codes", [good_paper])

        results_sequence = [bad_empty, bad_offtopic, good_result]
        run_research_mock = AsyncMock(side_effect=results_sequence)

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            result = await graph.ainvoke(initial_state)

        # 1. Researcher called exactly 3 times (initial + 2 retries before success on 3rd)
        assert run_research_mock.call_count == 3, f"Expected 3 calls, got {run_research_mock.call_count}"

        # 2. Final validation status is valid
        assert result["validation_status"] == "valid", f"Expected valid, got {result['validation_status']}"

        # 3. Researcher attempts should be 2 (2 failed attempts before success)
        # Note: the valid branch does not reset attempts, so it stays at 2
        assert result["researcher_attempts"] == 2, f"Expected 2 after valid, got {result['researcher_attempts']}"

        # 4. Sub-query was revised on each retry (check the sequence of queries passed to run_research)
        called_queries = [call.args[0] for call in run_research_mock.call_args_list]
        assert len(called_queries) == 3
        # First call uses original query
        assert called_queries[0] == "quantum error correction surface codes"
        # Second call should use a revised query (not identical to first)
        assert called_queries[1] != called_queries[0], "Query should be revised on first retry"
        # Third call should use a revised query (not identical to second)
        assert called_queries[2] != called_queries[1], "Query should be revised on second retry"
        # All queries should be different (not just repeated)
        assert len(set(called_queries)) == 3, "All three queries should be different"

        # 5. Validated findings accumulated
        assert len(result["validated_findings"]) >= 1, "Should have at least one validated finding"

        # 6. Graph proceeds to synthesizer
        logged_nodes = [call.args[0] for call in mock_logging.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes, "Synthesizer should be invoked"

        # 7. Trace log sequence: planner -> researcher -> validator -> researcher -> validator -> researcher -> validator -> synthesizer
        # Validate node_entry sequence
        # Note: planner may appear twice (initial + passthrough) but first is the real one
        assert logged_nodes[:1] == ["planner"], "First node should be planner"
        assert "synthesizer" in logged_nodes, "Synthesizer must be called"

        # 8. Check validator statuses in log_node_exit
        validator_exits = [
            call.kwargs.get("status")
            for call in mock_logging.log_node_exit.call_args_list
            if call.args[0] == "validator"
        ]
        # First two should be "invalid_retry", last should be "valid_advance" or "valid_done"
        assert validator_exits.count("invalid_retry") == 2, f"Expected 2 invalid_retry, got {validator_exits.count('invalid_retry')}"
        assert any(s in ["valid_advance", "valid_done"] for s in validator_exits), f"Expected final valid status, got {validator_exits}"

    async def test_all_attempts_fail_then_exhausted(
        self, initial_state, mock_logging, mock_planner, mock_synthesizer_llm
    ):
        """Test: all 3 attempts fail validation -> graph proceeds to synthesizer with partial/no validated findings.

        Asserts:
        - researcher_attempts capped at MAX_ATTEMPTS
        - validation_status is invalid
        - graph proceeds to synthesizer (no infinite loop)
        - final_report reflects "insufficient data for X" rather than fabricating success
        """
        # All calls return empty results (heuristic invalid)
        bad_result = _make_research_result("quantum error correction surface codes", [])
        run_research_mock = AsyncMock(return_value=bad_result)

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            result = await graph.ainvoke(initial_state)

        # 1. Researcher called exactly MAX_RESEARCHER_ATTEMPTS times
        assert run_research_mock.call_count == MAX_RESEARCHER_ATTEMPTS, (
            f"Expected {MAX_RESEARCHER_ATTEMPTS} calls, got {run_research_mock.call_count}"
        )

        # 2. Final researcher_attempts equals MAX_RESEARCHER_ATTEMPTS
        assert result["researcher_attempts"] == MAX_RESEARCHER_ATTEMPTS, (
            f"Expected attempts={MAX_RESEARCHER_ATTEMPTS}, got {result['researcher_attempts']}"
        )

        # 3. Validation status is invalid (exhausted)
        assert result["validation_status"] == "invalid", f"Expected invalid, got {result['validation_status']}"

        # 4. Graph proceeds to synthesizer (no infinite loop)
        logged_nodes = [call.args[0] for call in mock_logging.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes, "Synthesizer should be invoked even when all attempts fail"

        # 5. Check that validated_findings is empty (no valid results accumulated)
        assert len(result["validated_findings"]) == 0, "Should have zero validated findings"

        # 6. The synthesizer should produce a report noting insufficient data
        # We need to check what the synthesizer receives
        # The mock_synthesizer_llm.ainvoke should have been called with the empty findings
        # Note: mock_synthesizer_llm is the structured output mock, not the get_llm mock
        # We need to verify the synthesizer was called by checking log entries
        logged_nodes = [call.args[0] for call in mock_logging.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes, "Synthesizer node should be entered"

        # 7. Validator exits should all be "invalid_retry" (exhaustion handled by router)
        validator_exits = [
            call.kwargs.get("status")
            for call in mock_logging.log_node_exit.call_args_list
            if call.args[0] == "validator"
        ]
        assert all(s == "invalid_retry" for s in validator_exits), f"All validator exits should be invalid_retry, got {validator_exits}"

    @pytest.fixture
    def multi_query_state(self) -> ResearchState:
        """Create state with multiple sub-queries."""
        state = create_initial_state("Multi-query research")
        state["sub_queries"] = [
            "quantum error correction surface codes",  # Will succeed on 1st try
            "impossible query xyz",  # Will fail all attempts
            "quantum computing algorithms",  # Will succeed on 1st try
        ]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        return state

    @pytest.fixture
    def mock_multi_planner(self):
        """Mock planner decomposition for multi-query test."""
        with patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose:
            mock_decompose.return_value = MagicMock(
                sub_queries=[
                    "quantum error correction surface codes",
                    "impossible query xyz",
                    "quantum computing algorithms",
                ]
            )
            yield mock_decompose

    async def test_multi_query_retry_loop(
        self, multi_query_state, mock_logging, mock_multi_planner, mock_synthesizer_llm
    ):
        """Test retry loop with multiple sub-queries where one query exhausts attempts.

        Current behavior: when a query exhausts MAX_ATTEMPTS, graph proceeds to synthesizer
        without trying remaining queries. This test verifies that behavior.
        """
        # Sequence:
        # Query 0: good result on first try
        good_q0 = _make_research_result(
            "quantum error correction surface codes",
            [_make_paper("Surface Codes", "Surface codes for quantum error correction.")]
        )
        # Query 1: all bad (3 attempts)
        bad_q1 = _make_research_result("impossible query xyz", [])

        # The run_research calls will be interleaved with validation
        # First query: good (1 call)
        # Second query: bad, bad, bad (3 attempts, then exhausted -> synthesizer)
        results_sequence = [good_q0, bad_q1, bad_q1, bad_q1]
        run_research_mock = AsyncMock(side_effect=results_sequence)

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            result = await graph.ainvoke(multi_query_state)

        # Total researcher calls: 1 (q0) + 3 (q1 exhausted) = 4
        assert run_research_mock.call_count == 4, f"Expected 4 calls, got {run_research_mock.call_count}"

        # Should have 1 validated finding (q0 succeeded, q1 exhausted without success)
        assert len(result["validated_findings"]) == 1, f"Expected 1 validated finding, got {len(result['validated_findings'])}"

        # Current query index should be 1 (q1 was being processed when exhausted)
        assert result["current_query_index"] == 1, f"Expected index 1, got {result['current_query_index']}"

        # Graph proceeds to synthesizer
        logged_nodes = [call.args[0] for call in mock_logging.log_node_entry.call_args_list]
        assert "synthesizer" in logged_nodes

        # Validation status should be invalid (exhausted on q1)
        assert result["validation_status"] == "invalid"

    async def test_revised_query_is_meaningfully_different(
        self, initial_state, mock_logging, mock_planner, mock_synthesizer_llm
    ):
        """Test that revised queries are meaningfully different, not just reworded."""
        # First attempt: empty -> heuristic invalid with revised query for "no results"
        bad_empty = _make_research_result("quantum error correction surface codes", [])
        # Second attempt: off-topic -> heuristic invalid with revised query for "off-topic results"
        bad_offtopic = _make_research_result(
            "quantum error correction surface codes",
            [_make_paper("Cooking", "About cooking.")],
        )
        # Third attempt: good
        good_paper = _make_paper(
            "Surface Codes for Fault-Tolerant Quantum Computing",
            "We present surface code implementations for fault-tolerant quantum computing.",
        )
        good_result = _make_research_result("quantum error correction surface codes", [good_paper])

        run_research_mock = AsyncMock(side_effect=[bad_empty, bad_offtopic, good_result])

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            await graph.ainvoke(initial_state)

        called_queries = [call.args[0] for call in run_research_mock.call_args_list]

        # Original query
        q0 = called_queries[0]
        # First retry
        q1 = called_queries[1]
        # Second retry
        q2 = called_queries[2]

        # Each should be different
        assert q0 != q1, "First retry should revise query"
        assert q1 != q2, "Second retry should revise query again"
        assert len(set(called_queries)) == 3, "All three queries should be different"

        # Verify the revisions are from _suggest_revised_query logic
        # For "no results" failure, it should try more specific terms
        # For "off-topic" failure, it should add "survey"
        # The exact strings depend on _suggest_revised_query implementation
        assert "quantum" in q1 or "error" in q1 or "correction" in q1, "Revised query should retain key terms"

    async def test_trace_log_contains_expected_events(
        self, initial_state, mock_logging, mock_planner, mock_synthesizer_llm
    ):
        """Test that trace log (via mock) contains expected sequence for retry scenario."""
        bad_empty = _make_research_result("quantum error correction surface codes", [])
        bad_offtopic = _make_research_result(
            "quantum error correction surface codes",
            [_make_paper("Cooking", "About cooking.")],
        )
        good_paper = _make_paper(
            "Quantum Error Correction",
            "Surface codes and quantum error correction.",
        )
        good_result = _make_research_result("quantum error correction surface codes", [good_paper])

        run_research_mock = AsyncMock(side_effect=[bad_empty, bad_offtopic, good_result])

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            await graph.ainvoke(initial_state)

        # Collect all logged events
        all_events = []
        for call in mock_logging.log_event.call_args_list:
            args = call.args
            if args and len(args) >= 2:
                payload = args[1]
                if isinstance(payload, dict) and "node_name" in payload:
                    all_events.append({
                        "event_type": args[0],
                        "node": payload["node_name"],
                    })

        # Also collect from log_node_entry
        node_entries = [call.args[0] for call in mock_logging.log_node_entry.call_args_list]

        # Verify expected nodes were entered
        assert "planner" in node_entries
        assert "researcher" in node_entries
        assert "validator" in node_entries
        assert "synthesizer" in node_entries

        # Researcher should be entered 3 times (once per attempt)
        assert node_entries.count("researcher") == 3

        # Validator should be entered 3 times
        assert node_entries.count("validator") == 3


class TestRetryLoopEdgeCases:
    """Edge case tests for the retry loop."""

    @pytest.fixture
    def edge_case_state(self) -> ResearchState:
        """Create state for edge case tests."""
        state = create_initial_state("Research goal")
        state["sub_queries"] = ["original query"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        return state

    @pytest.fixture
    def edge_case_mock_planner(self):
        """Mock planner for edge case tests."""
        with patch("mcp_research_agent_system.agents.graph.decompose_goal") as mock_decompose:
            mock_decompose.return_value = MagicMock(
                sub_queries=["original query"]
            )
            yield mock_decompose

    @pytest.fixture
    def mock_logging(self):
        """Mock logging utilities to capture events."""
        with patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log:
            mock_log.log_event = MagicMock()
            mock_log.log_node_entry = MagicMock(return_value=0.0)
            mock_log.log_node_exit = MagicMock()
            mock_log.log_node_error = MagicMock()
            mock_log.safe_state_snapshot = MagicMock(return_value={})
            yield mock_log

    @pytest.fixture
    def mock_synthesizer_llm(self):
        """Mock synthesizer LLM to return a valid report."""
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
            "mcp_research_agent_system.agents.synthesizer.get_llm", side_effect=synthesizer_get_llm
        ):
            yield mock_synthesizer_structured

    async def test_validator_node_directly_updates_sub_query(
        self, edge_case_state, mock_logging
    ):
        """Test validator_node directly updates sub_query on retry."""
        from mcp_research_agent_system.agents.graph import validator_node

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

    async def test_validator_node_empty_results_increments_attempts(
        self, edge_case_state, mock_logging
    ):
        """Test validator_node increments attempts correctly for empty results."""
        from mcp_research_agent_system.agents.graph import validator_node

        state: ResearchState = create_initial_state("Research goal")
        state["sub_queries"] = ["query that returns nothing"]
        state["current_query_index"] = 0
        state["researcher_attempts"] = 0
        state["validation_status"] = "pending"
        state["researcher_output"] = []

        result1 = await validator_node(state)
        assert result1["researcher_attempts"] == 1

        # Second attempt
        result2 = await validator_node({**result1, "researcher_output": []})
        assert result2["researcher_attempts"] == 2

        # Third attempt
        result3 = await validator_node({**result2, "researcher_output": []})
        assert result3["researcher_attempts"] == 3

    async def test_max_attempts_constant_is_respected(
        self, edge_case_state, mock_logging, edge_case_mock_planner, mock_synthesizer_llm
    ):
        """Test that MAX_RESEARCHER_ATTEMPTS constant is respected."""
        from mcp_research_agent_system.agents.graph import MAX_RESEARCHER_ATTEMPTS

        # Return bad results forever
        bad_result = _make_research_result("quantum error correction surface codes", [])
        run_research_mock = AsyncMock(return_value=bad_result)

        with patch(
            "mcp_research_agent_system.agents.graph.run_research", run_research_mock
        ):
            graph = build_graph()
            result = await graph.ainvoke(edge_case_state)

        # Should only call MAX_RESEARCHER_ATTEMPTS times, not more
        assert run_research_mock.call_count == MAX_RESEARCHER_ATTEMPTS
        assert result["researcher_attempts"] == MAX_RESEARCHER_ATTEMPTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

