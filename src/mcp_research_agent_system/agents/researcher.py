"""Researcher agent — calls the MCP server over stdio to search arXiv papers."""

import sys
from datetime import datetime
from typing import Any

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, ValidationError

from .. import logging_utils
from ..config import Settings


class ResearcherError(Exception):
    """Raised when the researcher agent encounters an error (server startup, tool call, timeout, etc.)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class PaperResult(BaseModel):
    """A paper returned from the MCP server."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    category: str
    published_date: str
    updated_date: str
    pdf_url: str


class SearchPapersResult(BaseModel):
    """Result of search_papers tool call."""

    papers: list[PaperResult]


class CachedSummaryEntry(BaseModel):
    """A cached search summary entry."""

    query_text: str
    category: str | None
    max_results: int
    date_from: str | None
    arxiv_ids: list[str]
    fetched_at: str
    ttl_expires_at: str


class GetCachedSummaryResult(BaseModel):
    """Result of get_cached_summary tool call."""

    cached_summaries: list[CachedSummaryEntry]


class ResearchResult(BaseModel):
    """Complete research result for a sub-query."""

    sub_query: str
    papers: list[PaperResult]
    cached_summaries: list[CachedSummaryEntry]
    raw_tool_calls: list[dict[str, Any]]  # raw MCP tool call/response for logging/debugging
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


async def run_research(sub_query: str, settings: Settings | None = None) -> ResearchResult:
    """Run research for a single sub-query by connecting to the MCP server over stdio.

    This function:
    1. Spawns the MCP server as a subprocess (stdio transport)
    2. Initializes an MCP client session
    3. Calls search_papers with the sub_query
    4. Calls get_cached_summary with the sub_query (to find related prior work)
    5. Parses results into typed models
    6. Closes the session cleanly

    Design note: We spawn a fresh MCP server subprocess per sub-query. This is acceptable
    for our scale (a handful of sub-queries per run) because:
    - Each sub-query is independent and the server is stateless (no persistent connections to arXiv)
    - Simpler than managing a long-lived session across the graph (no reconnection logic, no
      resource leaks if the graph is interrupted)
    - The overhead is low (Python process startup ~50-100ms, MCP handshake ~10ms)
    - If scale grows, we could refactor to hold one session in the graph state and reuse it

    Args:
        sub_query: The sub-query string to research
        settings: Optional Settings instance (uses defaults if not provided)

    Returns:
        ResearchResult with papers, cached summaries, and raw tool call logs

    Raises:
        ResearcherError: If server fails to start, tool call times out, or tool returns error
    """
    settings = settings or Settings()

    # Configure MCP server subprocess parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_research_agent_system.mcp_server"],
        env={
            "DATABASE_PATH": settings.database_path,
            "CACHE_TTL_HOURS": str(settings.cache_ttl_hours),
            "ARXIV_RATE_LIMIT_DELAY": str(settings.arxiv_rate_limit_delay),
        },
    )

    raw_tool_calls: list[dict[str, Any]] = []

    try:
        # Connect to MCP server over stdio
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the MCP session
                await session.initialize()

                # Call search_papers tool
                search_args = {"query": sub_query, "max_results": 10}
                logging_utils.log_event(
                    "researcher_tool_call",
                    {"tool": "search_papers", "input": search_args, "sub_query": sub_query},
                )

                try:
                    search_result = await session.call_tool("search_papers", search_args)
                except Exception as e:
                    raise ResearcherError(
                        f"search_papers tool call failed: {e}",
                        {"sub_query": sub_query, "tool": "search_papers", "error": str(e)},
                    ) from e

                raw_tool_calls.append(
                    {
                        "tool": "search_papers",
                        "input": search_args,
                        "result": {
                            "is_error": search_result.is_error,
                            "structured_content": search_result.structured_content,
                            "content": [
                                c.model_dump(mode="json") if hasattr(c, "model_dump") else str(c)
                                for c in search_result.content
                            ],
                        },
                    }
                )

                if search_result.is_error:
                    error_text = search_result.content[0].text if search_result.content else "Unknown error"
                    raise ResearcherError(
                        f"search_papers returned error: {error_text}",
                        {"sub_query": sub_query, "tool": "search_papers", "error": error_text},
                    )

                if not search_result.structured_content:
                    raise ResearcherError(
                        "search_papers returned no structured content",
                        {"sub_query": sub_query, "tool": "search_papers"},
                    )

                # Parse search results
                try:
                    search_output = SearchPapersResult(**search_result.structured_content)
                except ValidationError as e:
                    raise ResearcherError(
                        f"Failed to parse search_papers result: {e}",
                        {"sub_query": sub_query, "tool": "search_papers", "raw": search_result.structured_content},
                    ) from e

                # Call get_cached_summary tool (to find related prior cached research)
                cached_args = {"topic": sub_query}
                logging_utils.log_event(
                    "researcher_tool_call",
                    {"tool": "get_cached_summary", "input": cached_args, "sub_query": sub_query},
                )

                try:
                    cached_result = await session.call_tool("get_cached_summary", cached_args)
                except Exception as e:
                    # Don't fail the whole research if cached summary fails — log and continue
                    logging_utils.log_event(
                        "researcher_tool_error",
                        {"tool": "get_cached_summary", "sub_query": sub_query, "error": str(e)},
                    )
                    cached_result = None

                cached_summaries: list[CachedSummaryEntry] = []
                if cached_result is not None:
                    raw_tool_calls.append(
                        {
                            "tool": "get_cached_summary",
                            "input": cached_args,
                            "result": {
                                "is_error": cached_result.is_error,
                                "structured_content": cached_result.structured_content,
                                "content": [
                                    c.model_dump(mode="json") if hasattr(c, "model_dump") else str(c)
                                    for c in cached_result.content
                                ],
                            },
                        }
                    )

                    if not cached_result.is_error and cached_result.structured_content:
                        try:
                            cached_output = GetCachedSummaryResult(**cached_result.structured_content)
                            cached_summaries = cached_output.cached_summaries
                        except ValidationError as e:
                            logging_utils.log_event(
                                "researcher_tool_error",
                                {
                                    "tool": "get_cached_summary",
                                    "sub_query": sub_query,
                                    "error": f"Failed to parse: {e}",
                                },
                            )

                return ResearchResult(
                    sub_query=sub_query,
                    papers=search_output.papers,
                    cached_summaries=cached_summaries,
                    raw_tool_calls=raw_tool_calls,
                )

    except ResearcherError:
        # Re-raise our own errors
        raise
    except FileNotFoundError as e:
        raise ResearcherError(
            f"MCP server command not found: {e}",
            {"command": server_params.command, "args": server_params.args},
        ) from e
    except PermissionError as e:
        raise ResearcherError(
            f"Permission denied starting MCP server: {e}",
            {"command": server_params.command, "args": server_params.args},
        ) from e
    except Exception as e:
        # Catch-all for unexpected errors (connection issues, timeouts, etc.)
        raise ResearcherError(
            f"Unexpected error during research: {e}",
            {"sub_query": sub_query, "error_type": type(e).__name__, "error": str(e)},
        ) from e
