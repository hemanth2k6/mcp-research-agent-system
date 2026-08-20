"""CLI entrypoint for the MCP Research Agent System.

Provides a single command to run the full multi-agent research pipeline on a
user-provided research goal.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .agents.graph import build_graph
from .agents.state import create_initial_state
from .errors import PlannerError, ResearcherError, SynthesizerError
from .logging_utils import configure_logging, get_log_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Run the multi-agent research pipeline on a research goal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  research-agent \"What are the latest advances in transformer architectures?\"\n"
            "  research-agent \"Impact of quantum computing on cryptography\" --output report.md\n"
            "  research-agent \"LLM alignment techniques\" --verbose\n"
        ),
    )
    parser.add_argument(
        "goal",
        type=str,
        help="The research goal/question to investigate",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print node-by-node progress as the graph runs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        help="Write final report to a markdown file (in addition to stdout)",
    )
    return parser.parse_args()


def print_verbose_step(console: Console, node: str, status: str, details: str = "") -> None:
    """Print a verbose step for the CLI."""
    if status == "success":
        style = "green"
        symbol = "✓"
    elif status == "error":
        style = "red"
        symbol = "✗"
    elif status == "retry":
        style = "yellow"
        symbol = "⟳"
    else:
        style = "blue"
        symbol = "→"

    console.print(f"  [{style}]{symbol}[/{style}] {node}: {details}")


async def run_pipeline(goal: str, verbose: bool = False) -> str:
    """Run the research pipeline and return the final report."""
    # Configure logging first
    configure_logging()
    log_dir = get_log_dir()

    # Build the graph
    graph = build_graph()

    # Create initial state
    initial_state = create_initial_state(goal)

    console = Console()

    if verbose:
        console.print(f"[bold cyan]Starting research pipeline for:[/bold cyan] {goal}")
        console.print(f"[dim]Trace logs will be written to: {log_dir}[/dim]\n")

    # Invoke the graph
    try:
        # Use astream for verbose progress tracking
        if verbose:
            # Stream through the graph to show progress
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "planner":
                        sub_queries = node_output.get("sub_queries", [])
                        print_verbose_step(
                            console,
                            "planner",
                            "success",
                            f"Generated {len(sub_queries)} sub-queries: {', '.join(sub_queries[:3])}{'...' if len(sub_queries) > 3 else ''}",
                        )
                    elif node_name == "researcher":
                        idx = node_output.get("current_query_index", 0)
                        query = node_output.get("sub_queries", [""])[idx] if idx < len(node_output.get("sub_queries", [])) else ""
                        attempts = node_output.get("researcher_attempts", 0)
                        researcher_output = node_output.get("researcher_output", [])
                        print_verbose_step(
                            console,
                            "researcher",
                            "success" if researcher_output else "retry",
                            f"Query {idx + 1}: {query[:60]}... (attempt {attempts + 1}, {len(researcher_output)} papers)",
                        )
                    elif node_name == "validator":
                        status = node_output.get("validation_status", "pending")
                        attempts = node_output.get("researcher_attempts", 0)
                        if status == "valid":
                            validated_count = len(node_output.get("validated_findings", []))
                            print_verbose_step(
                                console,
                                "validator",
                                "success",
                                f"Validation passed (total validated: {validated_count})",
                            )
                        elif status == "invalid":
                            if attempts < 3:
                                print_verbose_step(
                                    console,
                                    "validator",
                                    "retry",
                                    f"Validation failed, retrying (attempt {attempts + 1}/3)",
                                )
                            else:
                                print_verbose_step(
                                    console,
                                    "validator",
                                    "error",
                                    f"Validation exhausted after {attempts} attempts",
                                )
                    elif node_name == "synthesizer":
                        synth_report = node_output.get("final_report")
                        print_verbose_step(
                            console,
                            "synthesizer",
                            "success",
                                f"Report generated ({len(synth_report)} chars)" if synth_report else "No report generated",
                        )
            # Get final state after streaming
            final_state = await graph.ainvoke(initial_state)
        else:
            # Simple invoke without verbose progress
            final_state = await graph.ainvoke(initial_state)

        final_report: str | None = final_state.get("final_report")
        error = final_state.get("error")

        if error and not final_report:
            raise RuntimeError(f"Pipeline failed: {error}")

        if not final_report:
            raise RuntimeError("Pipeline completed but no final report was generated")

        return final_report

    except PlannerError as e:
        raise RuntimeError(f"Planner error: {e}") from e
    except ResearcherError as e:
        raise RuntimeError(f"Researcher error: {e}") from e
    except SynthesizerError as e:
        raise RuntimeError(f"Synthesizer error: {e}") from e
    except Exception as e:
        # Re-raise with context
        raise RuntimeError(f"Pipeline execution failed: {e}") from e


def main() -> int:
    """Main CLI entrypoint. Returns exit code."""
    args = parse_args()
    console = Console()

    try:
        # Run the async pipeline
        final_report = asyncio.run(run_pipeline(args.goal, args.verbose))

        # Write to output file if specified
        if args.output:
            args.output.write_text(final_report, encoding="utf-8")
            console.print(f"\n[green]Report written to:[/green] {args.output}")

        # Print report to stdout
        console.print("\n" + "=" * 60)
        console.print(Markdown(final_report))

        # Print log location
        log_dir = get_log_dir()
        console.print(f"\n[dim]Full trace log written to: {log_dir}[/dim]")

        return 0

    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
