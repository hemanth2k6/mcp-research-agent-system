"""Tests for the synthesizer agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from mcp_research_agent_system.agents.graph import build_graph
from mcp_research_agent_system.agents.state import create_initial_state
from mcp_research_agent_system.agents.synthesizer import (
    SynthesizedReport,
    _build_insufficient_data_report,
    _parse_llm_json_response,
    synthesize_report,
)
from mcp_research_agent_system.errors import SynthesizerError

# Sample paper data for testing
SAMPLE_FINDINGS = [
    {
        "arxiv_id": "2401.12345v1",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
        "abstract": "We propose the Transformer, a model architecture relying entirely on attention mechanisms.",
        "category": "cs.CL",
        "published_date": "2017-06-12T00:00:00+00:00",
        "updated_date": "2017-06-12T00:00:00+00:00",
        "pdf_url": "http://arxiv.org/pdf/1706.03762v5",
    },
    {
        "arxiv_id": "2401.67890v1",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M.-W.", "Lee, K.", "Toutanova, K."],
        "abstract": "We introduce BERT, a method for pre-training deep bidirectional representations from unlabeled text.",
        "category": "cs.CL",
        "published_date": "2018-10-11T00:00:00+00:00",
        "updated_date": "2019-05-24T00:00:00+00:00",
        "pdf_url": "http://arxiv.org/pdf/1810.04805v2",
    },
    {
        "arxiv_id": "2401.54321v1",
        "title": "Efficient Transformers: A Survey",
        "authors": ["Tay, Y.", "Dehghani, M.", "Bahri, D.", "Metzler, D."],
        "abstract": "This survey provides a comprehensive overview of efficient transformer architectures.",
        "category": "cs.CL",
        "published_date": "2020-09-23T00:00:00+00:00",
        "updated_date": "2022-03-15T00:00:00+00:00",
        "pdf_url": "http://arxiv.org/pdf/2009.06732v1",
    },
]


class TestSynthesizedReportModel:
    """Tests for the SynthesizedReport Pydantic model."""

    def test_valid_report(self):
        """Test valid report structure."""
        report = SynthesizedReport(report="# Test Report\n\nContent here.")
        assert report.report == "# Test Report\n\nContent here."

    def test_empty_report_allowed(self):
        """Test that empty report string is allowed (will be caught by business logic)."""
        report = SynthesizedReport(report="")
        assert report.report == ""


class TestInsufficientDataReport:
    """Tests for the _build_insufficient_data_report helper."""

    def test_builds_report_with_empty_findings(self):
        """Test report generation with zero findings."""
        research_goal = "Test research goal"
        findings = []

        report = _build_insufficient_data_report(research_goal, findings)

        assert "Test research goal" in report
        assert "Insufficient Data" in report
        assert "No papers were retrieved or validated" in report
        assert "Recommendations" in report
        assert "Refine the research goal" in report

    def test_builds_report_with_findings_that_failed_validation(self):
        """Test report generation when findings exist but failed validation."""
        research_goal = "Test research goal"
        findings = SAMPLE_FINDINGS

        report = _build_insufficient_data_report(research_goal, findings)

        assert "Test research goal" in report
        assert "Insufficient Data" in report
        assert "**Validated findings collected**: 3 papers" in report
        assert "Recommendations" in report


class TestSynthesizeReport:
    """Tests for the synthesize_report function."""

    @pytest.mark.asyncio
    async def test_successful_structured_output(self):
        """Test successful report generation via structured output on first attempt."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
            report="# Research Report\n\n## Overview\nTest\n\n## Key Themes\nTheme 1\n\n## Notable Papers\nPaper 1\n\n## Gaps / Open Questions\nGap 1"
        ))
        mock_llm.with_structured_output.return_value = mock_structured

        result = await synthesize_report("Test research goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert isinstance(result, str)
        assert "Research Report" in result
        assert "Overview" in result
        assert "Key Themes" in result
        assert "Notable Papers" in result
        assert "Gaps" in result
        mock_llm.with_structured_output.assert_called_once_with(SynthesizedReport)
        mock_structured.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_manual_parse_success(self):
        """Test fallback manual JSON parsing when structured output fails."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output not supported")
        mock_llm.with_structured_output.return_value = mock_structured

        # Mock the fallback ainvoke to return valid JSON
        mock_response = AsyncMock()
        mock_response.content = '{"report": "# Test Report\\n\\n## Overview\\nTest content"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert isinstance(result, str)
        assert "Test Report" in result
        assert "Overview" in result
        assert mock_structured.ainvoke.call_count == 1
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_extracts_json_from_markdown(self):
        """Test fallback extracts JSON from markdown code blocks."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # Response wrapped in markdown
        mock_response = AsyncMock()
        mock_response.content = '```json\n{"report": "# Markdown Report\\n\\nContent"}\n```'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert "Markdown Report" in result

    @pytest.mark.asyncio
    async def test_fallback_extracts_json_from_surrounding_text(self):
        """Test fallback extracts JSON from surrounding explanatory text."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = 'Here is the report: {"report": "# Surrounded Report\\n\\nContent"} end.'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert "Surrounded Report" in result

    @pytest.mark.asyncio
    async def test_fallback_retry_on_first_failure(self):
        """Test fallback retries once when first attempt fails."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # First fallback attempt returns malformed JSON, second succeeds
        mock_response1 = AsyncMock()
        mock_response1.content = "not json at all"
        mock_response2 = AsyncMock()
        mock_response2.content = '{"report": "# Retry Report\\n\\nContent"}'
        mock_llm.ainvoke = AsyncMock(side_effect=[mock_response1, mock_response2])

        result = await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert "Retry Report" in result
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_total_failure_raises_synthesizer_error(self):
        """Test that SynthesizerError is raised after all retries exhausted."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # Both fallback attempts fail
        mock_response = AsyncMock()
        mock_response.content = "completely invalid"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with pytest.raises(SynthesizerError) as exc_info:
            await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert "Failed to synthesize report" in str(exc_info.value)
        assert mock_llm.ainvoke.call_count == 2  # max_retries = 2

    @pytest.mark.asyncio
    async def test_synthesizer_error_chains_original_exception(self):
        """Test SynthesizerError chains the last underlying exception."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = "invalid"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with pytest.raises(SynthesizerError) as exc_info:
            await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_empty_findings_returns_insufficient_data_report(self):
        """Test that empty findings triggers the insufficient data path without LLM call."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(report="Should not be called"))
        mock_llm.with_structured_output.return_value = mock_structured

        result = await synthesize_report("Test goal", [], llm=mock_llm)

        # Should NOT call LLM for empty findings
        mock_structured.ainvoke.assert_not_called()
        mock_llm.ainvoke.assert_not_called()

        # Should return the insufficient data report
        assert "Insufficient Data" in result
        assert "Test goal" in result
        assert "No papers were retrieved or validated" in result
        assert "Recommendations" in result

    @pytest.mark.asyncio
    async def test_logs_input_and_output(self):
        """Test that synthesizer logs input and output events."""
        with patch("mcp_research_agent_system.agents.synthesizer.logging_utils") as mock_log:
            mock_log.log_event = MagicMock()

            mock_llm = AsyncMock(spec=ChatOpenAI)
            mock_structured = AsyncMock()
            mock_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
                report="# Test Report\n\n## Overview\nTest content"
            ))
            mock_llm.with_structured_output.return_value = mock_structured

            await synthesize_report("Test goal", SAMPLE_FINDINGS, llm=mock_llm)

            # Check log calls
            assert mock_log.log_event.call_count == 2
            input_call = mock_log.log_event.call_args_list[0]
            output_call = mock_log.log_event.call_args_list[1]

            assert input_call.args[0] == "synthesizer_input"
            assert input_call.args[1]["research_goal"] == "Test goal"
            assert input_call.args[1]["findings_count"] == 3

            assert output_call.args[0] == "synthesizer_output"
            assert output_call.args[1]["status"] == "success"
            assert "report_length" in output_call.args[1]


class TestSynthesizerNodeIntegration:
    """Integration tests for synthesizer_node in the graph."""

    @pytest.mark.asyncio
    async def test_full_graph_run_with_mocked_llm_and_researcher(self):
        """Test full run from planner through synthesizer with mocked LLM and researcher."""
        # Create initial state
        state = create_initial_state("Research: transformer architectures for NLP")

        # Mock the planner decomposition
        from mcp_research_agent_system.agents.planner import PlannerDecomposition
        mock_decomposition = PlannerDecomposition(
            sub_queries=[
                "transformer architecture NLP",
                "attention mechanism improvements",
                "efficient transformer variants"
            ]
        )

        # Mock researcher to return sample findings for each sub-query
        researcher_results = [
            {
                "arxiv_id": "2401.11111v1",
                "title": "Transformer Architecture Paper 1",
                "authors": ["Author A"],
                "abstract": "This paper presents a novel transformer architecture for NLP tasks.",
                "category": "cs.CL",
                "published_date": "2024-01-01T00:00:00+00:00",
                "updated_date": "2024-01-01T00:00:00+00:00",
                "pdf_url": "http://arxiv.org/pdf/2401.11111v1",
            },
            {
                "arxiv_id": "2401.22222v1",
                "title": "Attention Mechanism Improvements",
                "authors": ["Author B"],
                "abstract": "We propose improvements to the attention mechanism for better performance.",
                "category": "cs.CL",
                "published_date": "2024-01-02T00:00:00+00:00",
                "updated_date": "2024-01-02T00:00:00+00:00",
                "pdf_url": "http://arxiv.org/pdf/2401.22222v1",
            },
            {
                "arxiv_id": "2401.33333v1",
                "title": "Efficient Transformer Variants Survey",
                "authors": ["Author C"],
                "abstract": "A comprehensive survey of efficient transformer architectures.",
                "category": "cs.CL",
                "published_date": "2024-01-03T00:00:00+00:00",
                "updated_date": "2024-01-03T00:00:00+00:00",
                "pdf_url": "http://arxiv.org/pdf/2401.33333v1",
            },
        ]

        # Build mock LLM that handles both planner and synthesizer
        mock_llm = AsyncMock(spec=ChatOpenAI)

        # Mock synchronous invoke for planner fallback path
        mock_planner_response = MagicMock()
        mock_planner_response.content = json.dumps({"sub_queries": [
            "transformer architecture NLP",
            "attention mechanism improvements",
            "efficient transformer variants"
        ]})
        mock_llm.invoke = MagicMock(return_value=mock_planner_response)

        # For planner - structured output
        mock_planner_structured = AsyncMock()
        mock_planner_structured.ainvoke = AsyncMock(return_value=mock_decomposition)

        # For validator - not called if heuristic passes (we'll make sure it does)
        # For synthesizer - structured output
        mock_synthesizer_structured = AsyncMock()
        mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(
            report="""# Research Report: Transformer Architectures for NLP

## Overview
This report examines transformer architectures for natural language processing.

## Key Themes
### Theme 1: Novel Transformer Architectures
Papers exploring new architectural variants.

### Theme 2: Attention Mechanism Improvements
Work on optimizing and improving attention.

### Theme 3: Efficiency Optimizations
Surveys and papers on efficient transformer variants.

## Notable Papers
- **Transformer Architecture Paper 1** (arXiv:2401.11111) - Author A: This paper presents a novel transformer architecture for NLP tasks.
- **Attention Mechanism Improvements** (arXiv:2401.22222) - Author B: We propose improvements to the attention mechanism for better performance.
- **Efficient Transformer Variants Survey** (arXiv:2401.33333) - Author C: A comprehensive survey of efficient transformer architectures.

## Gaps / Open Questions
- Long-context handling in efficient transformers remains an open challenge.
- Multilingual transfer in novel architectures needs more exploration.
"""
        ))

        # Switch between planner and synthesizer structured outputs
        call_count = {"planner": 0, "synthesizer": 0}

        def with_structured_output_side_effect(model_class):
            if model_class.__name__ == "PlannerDecomposition":
                call_count["planner"] += 1
                return mock_planner_structured
            elif model_class.__name__ == "SynthesizedReport":
                call_count["synthesizer"] += 1
                return mock_synthesizer_structured
            return AsyncMock()

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        # Mock run_research to return our sample findings
        async def mock_run_research(sub_query, settings=None):
            # Return different findings based on query
            if "architecture" in sub_query:
                papers = [researcher_results[0]]
            elif "attention" in sub_query:
                papers = [researcher_results[1]]
            else:
                papers = [researcher_results[2]]

            # Create ResearchResult-like object
            from mcp_research_agent_system.agents.researcher import PaperResult, ResearchResult
            paper_objects = [
                PaperResult(
                    arxiv_id=p["arxiv_id"],
                    title=p["title"],
                    authors=p["authors"],
                    abstract=p["abstract"],
                    category=p["category"],
                    published_date=p["published_date"],
                    updated_date=p["updated_date"],
                    pdf_url=p["pdf_url"],
                ) for p in papers
            ]
            return ResearchResult(
                sub_query=sub_query,
                papers=paper_objects,
                cached_summaries=[],
                raw_tool_calls=[],
            )

        # Patch get_llm in the modules that use it (planner, validator, synthesizer)
        with patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm), \
             patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm), \
             patch("mcp_research_agent_system.agents.graph.run_research", mock_run_research), \
             patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log:
            mock_log.log_event = MagicMock()

            graph = build_graph()
            result = await graph.ainvoke(state)

        # Verify final report is populated and non-empty
        assert "final_report" in result
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 0
        assert "Research Report" in result["final_report"]
        assert "Overview" in result["final_report"]
        assert "Key Themes" in result["final_report"]
        assert "Notable Papers" in result["final_report"]
        assert "Gaps" in result["final_report"]

        # Verify all 3 sub-queries were processed
        assert len(result["validated_findings"]) == 3

        # Verify synthesizer was called
        assert call_count["synthesizer"] == 1

    @pytest.mark.asyncio
    async def test_full_graph_with_empty_findings_produces_insufficient_data_report(self):
        """Test full graph produces insufficient data report when all sub-queries return empty."""
        state = create_initial_state("Research: impossible topic xyz123")

        from mcp_research_agent_system.agents.planner import PlannerDecomposition
        mock_decomposition = PlannerDecomposition(
            sub_queries=[
                "impossible topic xyz123 variant 1",
                "impossible topic xyz123 variant 2",
                "impossible topic xyz123 variant 3",
            ]
        )

        mock_llm = AsyncMock(spec=ChatOpenAI)

        # Mock synchronous invoke for planner fallback path
        mock_planner_response = MagicMock()
        mock_planner_response.content = json.dumps({"sub_queries": [
            "impossible topic xyz123 variant 1",
            "impossible topic xyz123 variant 2",
            "impossible topic xyz123 variant 3",
        ]})
        mock_llm.invoke = MagicMock(return_value=mock_planner_response)

        mock_planner_structured = AsyncMock()
        mock_planner_structured.ainvoke = AsyncMock(return_value=mock_decomposition)

        mock_synthesizer_structured = AsyncMock()
        # Synthesizer should NOT be called with structured output for empty findings
        # but we'll set it up anyway in case
        mock_synthesizer_structured.ainvoke = AsyncMock(return_value=SynthesizedReport(report="Should not be used"))

        def with_structured_output_side_effect(model_class):
            if model_class.__name__ == "PlannerDecomposition":
                return mock_planner_structured
            elif model_class.__name__ == "SynthesizedReport":
                return mock_synthesizer_structured
            return AsyncMock()

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        # Mock run_research to return empty findings
        async def mock_run_research(sub_query, settings=None):
            from mcp_research_agent_system.agents.researcher import ResearchResult
            return ResearchResult(
                sub_query=sub_query,
                papers=[],
                cached_summaries=[],
                raw_tool_calls=[],
            )

        # Patch get_llm in the modules that use it
        with patch("mcp_research_agent_system.agents.planner.get_llm", return_value=mock_llm), \
             patch("mcp_research_agent_system.agents.synthesizer.get_llm", return_value=mock_llm), \
             patch("mcp_research_agent_system.agents.graph.run_research", mock_run_research), \
             patch("mcp_research_agent_system.agents.graph.logging_utils") as mock_log:
            mock_log.log_event = MagicMock()

            graph = build_graph()
            result = await graph.ainvoke(state)

        # Verify final report is the insufficient data report
        assert "final_report" in result
        assert result["final_report"] is not None
        assert "Insufficient Data" in result["final_report"]
        assert "impossible topic xyz123" in result["final_report"]
        assert "Recommendations" in result["final_report"]
        assert "Refine the research goal" in result["final_report"]


class TestParseLlmJsonResponse:
    """Tests for _parse_llm_json_response error paths (lines 52, 62)."""

    def test_fallback_json_extraction_failure_raises_synthesizer_error(self):
        """Test that JSON-looking-but-invalid content raises SynthesizerError (lines 55-62)."""
        with pytest.raises(SynthesizerError) as exc_info:
            _parse_llm_json_response('{"report": "truncated')  # invalid JSON

        assert "Failed to parse LLM response" in str(exc_info.value)

    def test_no_braces_raises_synthesizer_error(self):
        """Test that content without any braces raises SynthesizerError."""
        with pytest.raises(SynthesizerError):
            _parse_llm_json_response("no json here at all")


class TestSynthesizeReportFallbackPaths:
    """Tests for fallback parsing paths in synthesize_report (lines 178-209)."""

    @pytest.fixture
    def sample_findings(self):
        """Sample findings for testing."""
        return [
            {
                "arxiv_id": "2401.12345v1",
                "title": "Test Paper",
                "authors": ["Author A"],
                "abstract": "Test abstract",
                "category": "cs.AI",
                "published_date": "2024-01-01T00:00:00+00:00",
                "updated_date": "2024-01-01T00:00:00+00:00",
                "pdf_url": "http://arxiv.org/pdf/2401.12345v1",
            }
        ]

    @pytest.mark.asyncio
    async def test_fallback_direct_json_parse(self, sample_findings):
        """Test fallback directly parses valid JSON (lines 189-191)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = '{"report": "# Fallback Report\\n\\n## Overview\\nDirect JSON parse"}'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert "Fallback Report" in result
        assert "Direct JSON parse" in result
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_extracts_json_from_markdown(self, sample_findings):
        """Test fallback extracts JSON from markdown code block (lines 200-208)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = '```json\n{"report": "# Markdown Report\\n\\nContent from markdown"}\n```'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert "Markdown Report" in result
        assert "Content from markdown" in result

    @pytest.mark.asyncio
    async def test_fallback_extracts_json_from_surrounding_text(self, sample_findings):
        """Test fallback extracts JSON from surrounding text (lines 200-208)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = 'Here is your report: {"report": "# Surrounded Report\\n\\nFrom surrounding text"} done.'
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert "Surrounded Report" in result
        assert "From surrounding text" in result

    @pytest.mark.asyncio
    async def test_fallback_retry_on_first_failure(self, sample_findings):
        """Test fallback retries when first attempt fails (lines 182, 211)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        # First fallback attempt returns malformed JSON, second succeeds
        mock_response1 = AsyncMock()
        mock_response1.content = "not json at all"
        mock_response2 = AsyncMock()
        mock_response2.content = '{"report": "# Retry Report\\n\\nSecond attempt works"}'
        mock_llm.ainvoke = AsyncMock(side_effect=[mock_response1, mock_response2])

        result = await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert "Retry Report" in result
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_both_attempts_fail_raises_synthesizer_error(self, sample_findings):
        """Test both fallback attempts fail -> raises SynthesizerError (lines 211-222)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = "completely invalid"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with pytest.raises(SynthesizerError) as exc_info:
            await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert "Failed to synthesize report" in str(exc_info.value)
        assert mock_llm.ainvoke.call_count == 2  # max_retries = 2

    @pytest.mark.asyncio
    async def test_fallback_error_chains_original_exception(self, sample_findings):
        """Test SynthesizerError chains the last underlying exception (line 222)."""
        mock_llm = AsyncMock(spec=ChatOpenAI)
        mock_structured = AsyncMock()
        mock_structured.ainvoke.side_effect = Exception("Structured output failed")
        mock_llm.with_structured_output.return_value = mock_structured

        mock_response = AsyncMock()
        mock_response.content = "invalid"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with pytest.raises(SynthesizerError) as exc_info:
            await synthesize_report("Test goal", sample_findings, llm=mock_llm)

        assert exc_info.value.__cause__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
