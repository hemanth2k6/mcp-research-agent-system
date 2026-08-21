"""Tests for the CLI entrypoint."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcp_research_agent_system.cli import main, parse_args, run_pipeline


class TestParseArgs:
    """Tests for argument parsing."""

    def test_basic_goal(self) -> None:
        """Test parsing a basic research goal."""
        with patch.object(sys, "argv", ["research-agent", "test goal"]):
            args = parse_args()
            assert args.goal == "test goal"
            assert args.verbose is False
            assert args.output is None

    def test_verbose_flag(self) -> None:
        """Test parsing with verbose flag."""
        with patch.object(sys, "argv", ["research-agent", "test goal", "-v"]):
            args = parse_args()
            assert args.verbose is True

    def test_verbose_long_flag(self) -> None:
        """Test parsing with --verbose long flag."""
        with patch.object(sys, "argv", ["research-agent", "test goal", "--verbose"]):
            args = parse_args()
            assert args.verbose is True

    def test_output_flag(self) -> None:
        """Test parsing with output flag."""
        with patch.object(sys, "argv", ["research-agent", "test goal", "-o", "report.md"]):
            args = parse_args()
            assert args.output == Path("report.md")

    def test_output_long_flag(self) -> None:
        """Test parsing with --output long flag."""
        with patch.object(sys, "argv", ["research-agent", "test goal", "--output", "report.md"]):
            args = parse_args()
            assert args.output == Path("report.md")

    def test_combined_flags(self) -> None:
        """Test parsing with both verbose and output."""
        with patch.object(sys, "argv", ["research-agent", "test goal", "-v", "-o", "report.md"]):
            args = parse_args()
            assert args.verbose is True
            assert args.output == Path("report.md")


class TestRunPipeline:
    """Tests for the run_pipeline async function."""

    @pytest.mark.asyncio
    async def test_success_path(self) -> None:
        """Test successful pipeline execution returns report."""
        mock_graph = AsyncMock()
        mock_final_state = {
            "final_report": "# Test Report\n\nContent here.",
            "error": None,
        }
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    report = await run_pipeline("test goal", verbose=False)

        assert report == "# Test Report\n\nContent here."
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_path_verbose(self) -> None:
        """Test successful pipeline with verbose streaming."""
        mock_graph = AsyncMock()

        # Simulate streaming events
        async def astream(state, stream_mode):
            events = [
                {"planner": {"sub_queries": ["q1", "q2"]}},
                {"researcher": {"current_query_index": 0, "researcher_output": [{"paper": "p1"}]}},
                {
                    "validator": {
                        "validation_status": "valid",
                        "validated_findings": [{"paper": "p1"}],
                    }
                },
                {"synthesizer": {"final_report": "# Report"}},
            ]
            for e in events:
                yield e

        mock_graph.astream = astream
        mock_graph.ainvoke = AsyncMock(
            return_value={"final_report": "# Final Report", "error": None}
        )

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    report = await run_pipeline("test goal", verbose=True)

        assert report == "# Final Report"

    @pytest.mark.asyncio
    async def test_planner_error(self) -> None:
        """Test PlannerError is caught and re-raised as RuntimeError."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Planner failed"))

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    with pytest.raises(RuntimeError, match="Pipeline execution failed"):
                        await run_pipeline("test goal", verbose=False)

    @pytest.mark.asyncio
    async def test_researcher_error(self) -> None:
        """Test ResearcherError is caught and re-raised."""
        from mcp_research_agent_system.errors import ResearcherError

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=ResearcherError("MCP server failed"))

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    with pytest.raises(RuntimeError, match="Researcher error"):
                        await run_pipeline("test goal", verbose=False)

    @pytest.mark.asyncio
    async def test_synthesizer_error(self) -> None:
        """Test SynthesizerError is caught and re-raised."""
        from mcp_research_agent_system.errors import SynthesizerError

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(side_effect=SynthesizerError("LLM failed"))

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    with pytest.raises(RuntimeError, match="Synthesizer error"):
                        await run_pipeline("test goal", verbose=False)

    @pytest.mark.asyncio
    async def test_no_final_report(self) -> None:
        """Test error when pipeline completes but no final report."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={"final_report": None, "error": None})

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    with pytest.raises(RuntimeError, match="no final report was generated"):
                        await run_pipeline("test goal", verbose=False)

    @pytest.mark.asyncio
    async def test_error_in_state(self) -> None:
        """Test error propagated from state."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"final_report": None, "error": "Validation failed"}
        )

        with patch("mcp_research_agent_system.cli.build_graph", return_value=mock_graph):
            with patch("mcp_research_agent_system.cli.configure_logging"):
                with patch("mcp_research_agent_system.cli.get_log_dir", return_value=Path("logs")):
                    with pytest.raises(RuntimeError, match="Pipeline failed: Validation failed"):
                        await run_pipeline("test goal", verbose=False)


class TestMain:
    """Tests for the main CLI entrypoint."""

    def test_success_exits_zero(self, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
        """Test successful run exits with code 0 and prints report."""
        mock_report = "# Test Report\n\nContent here."

        with patch(
            "mcp_research_agent_system.cli.run_pipeline", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_report

            with patch.object(sys, "argv", ["research-agent", "test goal"]):
                exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Test Report" in captured.out

    def test_success_with_output_file(self, tmp_path: Path) -> None:
        """Test successful run with --output writes file."""
        mock_report = "# Test Report\n\nContent here."
        output_file = tmp_path / "report.md"

        with patch(
            "mcp_research_agent_system.cli.run_pipeline", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_report

            with patch.object(sys, "argv", ["research-agent", "test goal", "-o", str(output_file)]):
                exit_code = main()

        assert exit_code == 0
        assert output_file.exists()
        assert output_file.read_text() == mock_report

    def test_error_exits_nonzero(self, capsys: pytest.CaptureFixture) -> None:
        """Test error exits with non-zero code and prints error."""
        with patch(
            "mcp_research_agent_system.cli.run_pipeline", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = RuntimeError("Pipeline failed: test error")

            with patch.object(sys, "argv", ["research-agent", "test goal"]):
                exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "test error" in captured.out

    def test_keyboard_interrupt_exits_130(self, capsys: pytest.CaptureFixture) -> None:
        """Test KeyboardInterrupt exits with code 130."""
        with patch(
            "mcp_research_agent_system.cli.run_pipeline", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            with patch.object(sys, "argv", ["research-agent", "test goal"]):
                exit_code = main()

        assert exit_code == 130
        captured = capsys.readouterr()
        assert "Interrupted by user" in captured.out

    def test_verbose_mode_prints_progress(self, capsys: pytest.CaptureFixture) -> None:
        """Test verbose mode shows progress output."""
        mock_report = "# Report"

        with patch(
            "mcp_research_agent_system.cli.run_pipeline", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_report

            with patch.object(sys, "argv", ["research-agent", "test goal", "-v"]):
                exit_code = main()

        assert exit_code == 0
        # run_pipeline should be called with verbose=True (second positional arg)
        mock_run.assert_called_once()
        assert mock_run.call_args[0][1] is True
