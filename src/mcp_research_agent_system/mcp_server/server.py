"""MCP Server exposing arXiv search and cache tools."""

import json
import time
from datetime import datetime
from typing import Any

from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from pydantic import BaseModel, Field, ValidationError

from .. import logging_utils
from ..arxiv_client import Paper
from ..cache import cached_search
from ..config import Settings
from ..db.connection import get_connection, init_db


class SearchPapersInput(BaseModel):
    """Input schema for search_papers tool."""

    query: str = Field(..., description="Search query string")
    category: str | None = Field(None, description="arXiv category filter (e.g., 'cs.AI')")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results")
    date_from: str | None = Field(
        None, description="Only return papers after this date (ISO 8601 format)"
    )


class GetPaperDetailsInput(BaseModel):
    """Input schema for get_paper_details tool."""

    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2401.12345v1')")


class GetCachedSummaryInput(BaseModel):
    """Input schema for get_cached_summary tool."""

    topic: str = Field(..., description="Topic to search cached summaries for")


class PaperOutput(BaseModel):
    """Output schema for a paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    category: str
    published_date: str
    updated_date: str
    pdf_url: str


class SearchPapersOutput(BaseModel):
    """Output schema for search_papers tool."""

    papers: list[PaperOutput]


class GetPaperDetailsOutput(BaseModel):
    """Output schema for get_paper_details tool."""

    paper: PaperOutput | None = None


class CachedSummaryEntry(BaseModel):
    """A cached search summary entry."""

    query_text: str
    category: str | None
    max_results: int
    date_from: str | None
    arxiv_ids: list[str]
    fetched_at: str
    ttl_expires_at: str


class GetCachedSummaryOutput(BaseModel):
    """Output schema for get_cached_summary tool."""

    cached_summaries: list[CachedSummaryEntry]


def _paper_to_output(paper: Paper) -> PaperOutput:
    """Convert a Paper model to output schema."""
    return PaperOutput(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        category=paper.category,
        published_date=paper.published_date.isoformat(),
        updated_date=paper.updated_date.isoformat(),
        pdf_url=paper.pdf_url,
    )


def _create_error_result(message: str) -> types.CallToolResult:
    """Create a structured error result."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        is_error=True,
    )


def _create_success_result(structured_content: dict[str, Any]) -> types.CallToolResult:
    """Create a structured success result."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="")],
        structured_content=structured_content,
        is_error=False,
    )


async def _handle_search_papers(arguments: dict[str, Any]) -> types.CallToolResult:
    """Handle search_papers tool call."""
    start_time = time.time()
    logging_utils.log_tool_call("search_papers", arguments)

    try:
        input_data = SearchPapersInput(**arguments)
    except ValidationError as e:
        error_msg = f"Invalid input: {e}"
        logging_utils.log_tool_error("search_papers", error_msg, arguments)
        return _create_error_result(error_msg)

    try:
        settings = Settings()
        init_db(settings)

        # Parse date_from if provided
        date_from = None
        if input_data.date_from:
            try:
                date_from = datetime.fromisoformat(input_data.date_from.replace("Z", "+00:00"))
            except ValueError as e:
                error_msg = f"Invalid date_from format: {e}"
                logging_utils.log_tool_error("search_papers", error_msg, arguments)
                return _create_error_result(error_msg)

        papers = await cached_search(
            query=input_data.query,
            settings=settings,
            category=input_data.category,
            max_results=input_data.max_results,
            date_from=date_from,
        )

        output = SearchPapersOutput(papers=[_paper_to_output(p) for p in papers])

        duration_ms = (time.time() - start_time) * 1000
        logging_utils.log_tool_result(
            "search_papers",
            {"paper_count": len(papers)},
            duration_ms,
        )

        return _create_success_result(output.model_dump(mode="json"))

    except Exception as e:
        error_msg = f"Search failed: {e}"
        logging_utils.log_tool_error("search_papers", error_msg, arguments)
        return _create_error_result(error_msg)


async def _handle_get_paper_details(arguments: dict[str, Any]) -> types.CallToolResult:
    """Handle get_paper_details tool call."""
    start_time = time.time()
    logging_utils.log_tool_call("get_paper_details", arguments)

    try:
        input_data = GetPaperDetailsInput(**arguments)
    except ValidationError as e:
        error_msg = f"Invalid input: {e}"
        logging_utils.log_tool_error("get_paper_details", error_msg, arguments)
        return _create_error_result(error_msg)

    try:
        settings = Settings()
        init_db(settings)

        with get_connection(settings) as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?",
                (input_data.arxiv_id,),
            ).fetchone()

        if row is None:
            output = GetPaperDetailsOutput(paper=None)
            duration_ms = (time.time() - start_time) * 1000
            logging_utils.log_tool_result(
                "get_paper_details",
                {"found": False},
                duration_ms,
            )
            return _create_success_result(output.model_dump(mode="json"))

        # Convert row to Paper object
        from ..cache import _db_row_to_paper

        paper = _db_row_to_paper(row)
        paper_output = _paper_to_output(paper)

        output = GetPaperDetailsOutput(paper=paper_output)

        duration_ms = (time.time() - start_time) * 1000
        logging_utils.log_tool_result(
            "get_paper_details",
            {"found": True, "arxiv_id": input_data.arxiv_id},
            duration_ms,
        )

        return _create_success_result(output.model_dump(mode="json"))

    except Exception as e:
        error_msg = f"Get paper details failed: {e}"
        logging_utils.log_tool_error("get_paper_details", error_msg, arguments)
        return _create_error_result(error_msg)


async def _handle_get_cached_summary(arguments: dict[str, Any]) -> types.CallToolResult:
    """Handle get_cached_summary tool call."""
    start_time = time.time()
    logging_utils.log_tool_call("get_cached_summary", arguments)

    try:
        input_data = GetCachedSummaryInput(**arguments)
    except ValidationError as e:
        error_msg = f"Invalid input: {e}"
        logging_utils.log_tool_error("get_cached_summary", error_msg, arguments)
        return _create_error_result(error_msg)

    try:
        settings = Settings()
        init_db(settings)

        # Search for cached entries matching the topic loosely
        # We use LIKE on query_text for loose matching
        topic_lower = input_data.topic.lower()
        with get_connection(settings) as conn:
            rows = conn.execute(
                """
                SELECT * FROM search_cache
                WHERE LOWER(query_text) LIKE ?
                ORDER BY fetched_at DESC
                LIMIT 20
                """,
                (f"%{topic_lower}%",),
            ).fetchall()

        summaries = []
        for row in rows:
            summaries.append(
                CachedSummaryEntry(
                    query_text=row["query_text"],
                    category=row["category"],
                    max_results=row["max_results"],
                    date_from=row["date_from"],
                    arxiv_ids=json.loads(row["arxiv_ids"]),
                    fetched_at=row["fetched_at"],
                    ttl_expires_at=row["ttl_expires_at"],
                )
            )

        output = GetCachedSummaryOutput(cached_summaries=summaries)

        duration_ms = (time.time() - start_time) * 1000
        logging_utils.log_tool_result(
            "get_cached_summary",
            {"match_count": len(summaries)},
            duration_ms,
        )

        return _create_success_result(output.model_dump(mode="json"))

    except Exception as e:
        error_msg = f"Get cached summary failed: {e}"
        logging_utils.log_tool_error("get_cached_summary", error_msg, arguments)
        return _create_error_result(error_msg)


TOOL_HANDLERS = {
    "search_papers": _handle_search_papers,
    "get_paper_details": _handle_get_paper_details,
    "get_cached_summary": _handle_get_cached_summary,
}

TOOL_DEFINITIONS = [
    types.Tool(
        name="search_papers",
        description="Search arXiv for papers with caching. Returns a list of papers with title, authors, abstract snippet, published date, and PDF URL.",
        input_schema=SearchPapersInput.model_json_schema(by_alias=True),
    ),
    types.Tool(
        name="get_paper_details",
        description="Get full paper details from the local cache by arXiv ID. Returns the complete paper record including full abstract.",
        input_schema=GetPaperDetailsInput.model_json_schema(by_alias=True),
    ),
    types.Tool(
        name="get_cached_summary",
        description="Check if we have already researched a topic by searching cached search results. Returns matching cached entries with query text, category, arxiv IDs, and TTL info.",
        input_schema=GetCachedSummaryInput.model_json_schema(by_alias=True),
    ),
]


async def _list_tools_handler(
    context: ServerRequestContext,
    params: types.PaginatedRequestParams | None,
) -> types.ListToolsResult:
    """Handle tools/list request."""
    return types.ListToolsResult(tools=TOOL_DEFINITIONS)


async def _call_tool_handler(
    context: ServerRequestContext,
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    """Handle tools/call request."""
    handler = TOOL_HANDLERS.get(params.name)
    if handler is None:
        return _create_error_result(f"Unknown tool: {params.name}")

    arguments = params.arguments or {}
    return await handler(arguments)


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server(
        name="mcp-research-agent-system",
        version="0.1.0",
        title="arXiv Research MCP Server",
        description="MCP server exposing arXiv search and cached paper tools",
    )

    # Register handlers
    server.add_request_handler(
        "tools/list",
        types.PaginatedRequestParams,
        _list_tools_handler,
    )
    server.add_request_handler(
        "tools/call",
        types.CallToolRequestParams,
        _call_tool_handler,
    )

    return server
