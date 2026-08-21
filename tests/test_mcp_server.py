"""Tests for the MCP server tools."""

import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import StdioServerParameters, types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from mcp_research_agent_system.arxiv_client import Paper
from mcp_research_agent_system.cache import _upsert_paper, cached_search
from mcp_research_agent_system.config import Settings
from mcp_research_agent_system.db.connection import get_connection, init_db
from mcp_research_agent_system.mcp_server import server as server_module
from mcp_research_agent_system.mcp_server.server import (
    TOOL_HANDLERS,
    GetCachedSummaryInput,
    GetPaperDetailsInput,
    SearchPapersInput,
    create_server,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def test_settings(temp_db_path):
    """Create test settings with temp database."""
    return Settings(
        database_path=str(temp_db_path),
        cache_ttl_hours=24,
        arxiv_rate_limit_delay=0.0,
    )


@pytest.fixture
def sample_papers():
    """Create sample Paper objects for testing."""
    return [
        Paper(
            arxiv_id="2401.12345v1",
            title="Paper One: Machine Learning Advances",
            authors=["Author A"],
            abstract="Abstract one about machine learning.",
            category="cs.AI",
            published_date=datetime(2024, 1, 9, tzinfo=UTC),
            updated_date=datetime(2024, 1, 10, tzinfo=UTC),
            pdf_url="http://arxiv.org/pdf/2401.12345v1",
        ),
        Paper(
            arxiv_id="2401.67890v2",
            title="Paper Two: Deep Learning Survey",
            authors=["Author B", "Author C"],
            abstract="Abstract two about deep learning.",
            category="cs.LG",
            published_date=datetime(2024, 1, 11, tzinfo=UTC),
            updated_date=datetime(2024, 1, 12, tzinfo=UTC),
            pdf_url="http://arxiv.org/pdf/2401.67890v2",
        ),
    ]


@pytest.fixture
def mock_arxiv_client(sample_papers):
    """Create a mock ArxivClient that returns sample papers."""
    client = MagicMock()
    client.search = AsyncMock(return_value=sample_papers)
    client.close = AsyncMock()
    return client


@pytest.fixture
def server_params(test_settings):
    """Create StdioServerParameters for the test server."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_research_agent_system.mcp_server"],
        env={
            "DATABASE_PATH": test_settings.database_path,
            "CACHE_TTL_HOURS": "24",
            "ARXIV_RATE_LIMIT_DELAY": "0.0",
        },
    )


async def _initialize_session(session: ClientSession) -> None:
    """Initialize the MCP session."""
    await session.initialize()


# --- Subprocess + stdio client tests (list tools, integration smoke) ---


@pytest.mark.asyncio
async def test_list_tools(server_params):
    """Test that the server lists all three tools."""
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await _initialize_session(session)
            result = await session.list_tools()

    assert len(result.tools) == 3
    tool_names = {tool.name for tool in result.tools}
    assert tool_names == {"search_papers", "get_paper_details", "get_cached_summary"}

    # Verify each tool has inputSchema
    for tool in result.tools:
        assert tool.input_schema is not None
        assert tool.input_schema.get("type") == "object"


# --- In-process handler-level tests (no network, controlled mocks) ---


@pytest.mark.asyncio
async def test_search_papers_success(test_settings, sample_papers):
    """Test search_papers tool returns papers when cached_search succeeds."""
    with (
        patch.object(server_module, "Settings", return_value=test_settings),
        patch.object(server_module, "cached_search", new=AsyncMock(return_value=sample_papers)),
    ):
        result = await TOOL_HANDLERS["search_papers"](
            {
                "query": "machine learning",
                "category": "cs.AI",
                "max_results": 10,
            }
        )

    assert isinstance(result, types.CallToolResult)
    assert not result.is_error
    assert result.structured_content is not None
    papers = result.structured_content["papers"]
    assert len(papers) == 2
    assert papers[0]["arxiv_id"] == "2401.12345v1"
    assert papers[0]["title"] == "Paper One: Machine Learning Advances"
    assert papers[1]["arxiv_id"] == "2401.67890v2"


@pytest.mark.asyncio
async def test_search_papers_empty_result(test_settings):
    """Test search_papers returns empty list when no papers found."""
    with (
        patch.object(server_module, "Settings", return_value=test_settings),
        patch.object(server_module, "cached_search", new=AsyncMock(return_value=[])),
    ):
        result = await TOOL_HANDLERS["search_papers"]({"query": "nonexistent topic xyz123"})

    assert not result.is_error
    assert result.structured_content is not None
    papers = result.structured_content["papers"]
    assert papers == []


@pytest.mark.asyncio
async def test_search_papers_invalid_input():
    """Test search_papers returns error for missing required query field."""
    result = await TOOL_HANDLERS["search_papers"]({})

    assert result.is_error
    assert "Invalid input" in result.content[0].text


@pytest.mark.asyncio
async def test_search_papers_invalid_date():
    """Test search_papers returns error for invalid date format."""
    result = await TOOL_HANDLERS["search_papers"](
        {"query": "machine learning", "date_from": "not-a-date"}
    )

    assert result.is_error
    assert "Invalid date_from format" in result.content[0].text


@pytest.mark.asyncio
async def test_search_papers_max_results_validation():
    """Test search_papers validates max_results bounds."""
    # Too high
    result = await TOOL_HANDLERS["search_papers"]({"query": "test", "max_results": 200})
    assert result.is_error
    assert "Invalid input" in result.content[0].text

    # Too low
    result = await TOOL_HANDLERS["search_papers"]({"query": "test", "max_results": 0})
    assert result.is_error
    assert "Invalid input" in result.content[0].text


@pytest.mark.asyncio
async def test_search_papers_exception_handling(test_settings):
    """Test search_papers handles exceptions gracefully."""

    async def failing_search(*args, **kwargs):
        raise Exception("API connection failed")

    with (
        patch.object(server_module, "Settings", return_value=test_settings),
        patch.object(server_module, "cached_search", new=failing_search),
    ):
        result = await TOOL_HANDLERS["search_papers"]({"query": "test"})

    assert result.is_error
    assert "Search failed" in result.content[0].text


@pytest.mark.asyncio
async def test_get_paper_details_found(test_settings, sample_papers):
    """Test get_paper_details returns paper when found in cache."""
    # Populate the cache
    init_db(test_settings)
    with get_connection(test_settings) as conn:
        for paper in sample_papers:
            _upsert_paper(conn, paper, datetime.now(UTC).isoformat())

    with patch.object(server_module, "Settings", return_value=test_settings):
        result = await TOOL_HANDLERS["get_paper_details"]({"arxiv_id": "2401.12345v1"})

    assert not result.is_error
    assert result.structured_content is not None
    paper = result.structured_content["paper"]
    assert paper is not None
    assert paper["arxiv_id"] == "2401.12345v1"
    assert paper["title"] == "Paper One: Machine Learning Advances"


@pytest.mark.asyncio
async def test_get_paper_details_not_found(test_settings):
    """Test get_paper_details returns null paper when not in cache."""
    init_db(test_settings)

    with patch.object(server_module, "Settings", return_value=test_settings):
        result = await TOOL_HANDLERS["get_paper_details"]({"arxiv_id": "9999.99999v1"})

    assert not result.is_error
    assert result.structured_content is not None
    paper = result.structured_content["paper"]
    assert paper is None


@pytest.mark.asyncio
async def test_get_paper_details_invalid_input():
    """Test get_paper_details returns error for missing arxiv_id."""
    result = await TOOL_HANDLERS["get_paper_details"]({})

    assert result.is_error
    assert "Invalid input" in result.content[0].text


@pytest.mark.asyncio
async def test_get_cached_summary_matches(test_settings, sample_papers, mock_arxiv_client):
    """Test get_cached_summary returns matching cached entries."""
    # Populate cache with some entries
    await cached_search(
        query="machine learning basics",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )
    await cached_search(
        query="deep learning fundamentals",
        settings=test_settings,
        category="cs.LG",
        max_results=5,
        client=mock_arxiv_client,
    )

    with patch.object(server_module, "Settings", return_value=test_settings):
        result = await TOOL_HANDLERS["get_cached_summary"]({"topic": "machine learning"})

    assert not result.is_error
    assert result.structured_content is not None
    summaries = result.structured_content["cached_summaries"]
    assert len(summaries) >= 1
    # Should match "machine learning basics"
    assert any("machine learning" in s["query_text"].lower() for s in summaries)


@pytest.mark.asyncio
async def test_get_cached_summary_no_matches(test_settings):
    """Test get_cached_summary returns empty list when no matches."""
    init_db(test_settings)

    with patch.object(server_module, "Settings", return_value=test_settings):
        result = await TOOL_HANDLERS["get_cached_summary"]({"topic": "nonexistent topic xyz123"})

    assert not result.is_error
    assert result.structured_content is not None
    summaries = result.structured_content["cached_summaries"]
    assert summaries == []


@pytest.mark.asyncio
async def test_get_cached_summary_invalid_input():
    """Test get_cached_summary returns error for missing topic."""
    result = await TOOL_HANDLERS["get_cached_summary"]({})

    assert result.is_error
    assert "Invalid input" in result.content[0].text


@pytest.mark.asyncio
async def test_tool_input_schemas():
    """Test that tool input schemas are valid JSON schemas."""
    # Test search_papers schema
    schema = SearchPapersInput.model_json_schema()
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]

    # Test get_paper_details schema
    schema = GetPaperDetailsInput.model_json_schema()
    assert schema["type"] == "object"
    assert "arxiv_id" in schema["properties"]
    assert schema["required"] == ["arxiv_id"]

    # Test get_cached_summary schema
    schema = GetCachedSummaryInput.model_json_schema()
    assert schema["type"] == "object"
    assert "topic" in schema["properties"]
    assert schema["required"] == ["topic"]


@pytest.mark.asyncio
async def test_create_server():
    """Test create_server returns a configured Server instance."""
    server = create_server()
    assert server is not None
    assert server.server_info.name == "mcp-research-agent-system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
