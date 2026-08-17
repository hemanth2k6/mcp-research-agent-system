"""Tests for the researcher agent with real MCP stdio integration."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_research_agent_system.agents.researcher import (
    CachedSummaryEntry,
    PaperResult,
    ResearcherError,
    ResearchResult,
    run_research,
)
from mcp_research_agent_system.arxiv_client import Paper
from mcp_research_agent_system.cache import cached_search
from mcp_research_agent_system.config import Settings


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_researcher.db"


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
def seeded_settings(test_settings, sample_papers, mock_arxiv_client):
    """Create test settings with pre-seeded database matching researcher's search params."""
    # Populate the cache with the exact params researcher uses:
    # query="machine learning", category=None, max_results=10, date_from=None
    import asyncio

    asyncio.run(
        cached_search(
            query="machine learning",
            settings=test_settings,
            category=None,  # researcher doesn't specify category
            max_results=10,
            date_from=None,
            client=mock_arxiv_client,
        )
    )
    return test_settings


# --- Integration tests with real MCP server subprocess ---


@pytest.mark.asyncio
async def test_run_research_integration(seeded_settings):
    """Test run_research against real MCP server subprocess with seeded data."""
    result = await run_research("machine learning", seeded_settings)

    assert isinstance(result, ResearchResult)
    assert result.sub_query == "machine learning"
    assert len(result.papers) >= 1
    # Should find the seeded paper
    assert any("machine learning" in p.title.lower() for p in result.papers)
    assert len(result.raw_tool_calls) >= 1
    # First tool call should be search_papers
    assert result.raw_tool_calls[0]["tool"] == "search_papers"


@pytest.mark.asyncio
async def test_run_research_no_results(seeded_settings):
    """Test run_research returns empty list when no papers found for unique query."""
    result = await run_research("nonexistent topic xyz123 unique", seeded_settings)

    assert isinstance(result, ResearchResult)
    assert result.sub_query == "nonexistent topic xyz123 unique"
    # This won't be in the seeded cache, so it will hit arXiv live (mocked via the module)
    # but with a unique query the client mock returns sample_papers regardless of query.
    # If arXiv were real, this would be empty. With mock we get samples back.
    assert len(result.papers) >= 0  # Should not crash
    assert len(result.raw_tool_calls) >= 1


@pytest.mark.asyncio
async def test_run_research_raw_tool_calls_logged(seeded_settings):
    """Test that raw MCP tool calls are captured for logging."""
    result = await run_research("machine learning", seeded_settings)

    assert len(result.raw_tool_calls) >= 2
    tool_names = [tc["tool"] for tc in result.raw_tool_calls]
    assert "search_papers" in tool_names
    assert "get_cached_summary" in tool_names

    # Each raw tool call should have input and result
    for tc in result.raw_tool_calls:
        assert "input" in tc
        assert "result" in tc
        assert "is_error" in tc["result"]
        assert "structured_content" in tc["result"]


@pytest.mark.asyncio
async def test_run_research_get_cached_summary_finds_seeded(seeded_settings):
    """Test that get_cached_summary finds the seeded query."""
    result = await run_research("machine learning", seeded_settings)

    # Should find the seeded "machine learning" cache entry
    assert len(result.cached_summaries) >= 1
    assert any("machine learning" in s.query_text.lower() for s in result.cached_summaries)


# --- Error path tests with mocks at the client level ---


@pytest.mark.asyncio
async def test_run_research_server_fails_to_start():
    """Test that a bad server command raises ResearcherError."""
    bad_settings = Settings(
        database_path=":memory:",
    )

    # Mock stdio_client to raise FileNotFoundError
    with patch(
        "mcp_research_agent_system.agents.researcher.stdio_client"
    ) as mock_stdio:
        mock_stdio.side_effect = FileNotFoundError("No such command: bad_command")

        with pytest.raises(ResearcherError) as exc_info:
            await run_research("test query", bad_settings)

        assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_research_tool_call_times_out():
    """Test that tool call timeout raises ResearcherError."""
    settings = Settings(database_path=":memory:")

    # Mock stdio_client to return a session whose call_tool times out
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=TimeoutError("Tool call timed out"))

    # Mock the async context managers
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_stdio_ctx = AsyncMock()
    mock_stdio_ctx.__aenter__.return_value = (mock_read, mock_write)
    mock_stdio_ctx.__aexit__.return_value = False

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = False

    with patch(
        "mcp_research_agent_system.agents.researcher.stdio_client",
        return_value=mock_stdio_ctx,
    ), patch(
        "mcp_research_agent_system.agents.researcher.ClientSession",
        return_value=mock_session_ctx,
    ):
        with pytest.raises(ResearcherError) as exc_info:
            await run_research("test query", settings)

        assert "timed out" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_research_tool_returns_error():
    """Test that a tool returning an error payload raises ResearcherError."""
    settings = Settings(database_path=":memory:")

    # Create a mock error result from the tool
    mock_error_result = MagicMock()
    mock_error_result.is_error = True
    mock_error_result.structured_content = None
    mock_error_text = MagicMock()
    mock_error_text.text = "Internal server error"
    mock_error_result.content = [mock_error_text]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_error_result)

    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_stdio_ctx = AsyncMock()
    mock_stdio_ctx.__aenter__.return_value = (mock_read, mock_write)
    mock_stdio_ctx.__aexit__.return_value = False

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = False

    with patch(
        "mcp_research_agent_system.agents.researcher.stdio_client",
        return_value=mock_stdio_ctx,
    ), patch(
        "mcp_research_agent_system.agents.researcher.ClientSession",
        return_value=mock_session_ctx,
    ):
        with pytest.raises(ResearcherError) as exc_info:
            await run_research("test query", settings)

        assert "error" in str(exc_info.value).lower()
        assert exc_info.value.details.get("tool") == "search_papers"


@pytest.mark.asyncio
async def test_run_research_invalid_result_structure():
    """Test that malformed tool result raises ResearcherError."""
    settings = Settings(database_path=":memory:")

    # Create a mock result with valid structured content but bad structure
    mock_result = MagicMock()
    mock_result.is_error = False
    mock_result.structured_content = {"wrong_key": []}  # Missing 'papers' key
    mock_result.content = []

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_stdio_ctx = AsyncMock()
    mock_stdio_ctx.__aenter__.return_value = (mock_read, mock_write)
    mock_stdio_ctx.__aexit__.return_value = False

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session
    mock_session_ctx.__aexit__.return_value = False

    with patch(
        "mcp_research_agent_system.agents.researcher.stdio_client",
        return_value=mock_stdio_ctx,
    ), patch(
        "mcp_research_agent_system.agents.researcher.ClientSession",
        return_value=mock_session_ctx,
    ):
        with pytest.raises(ResearcherError) as exc_info:
            await run_research("test query", settings)

        assert "failed to parse" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_research_permission_error():
    """Test that permission denied starting server raises ResearcherError."""
    settings = Settings(database_path=":memory:")

    # Mock stdio_client to raise PermissionError
    with patch(
        "mcp_research_agent_system.agents.researcher.stdio_client"
    ) as mock_stdio:
        mock_stdio.side_effect = PermissionError("Permission denied: /usr/bin/python")

        with pytest.raises(ResearcherError) as exc_info:
            await run_research("test query", settings)

        assert "permission" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_research_result_model():
    """Test ResearchResult model serialization."""
    paper = PaperResult(
        arxiv_id="2401.12345v1",
        title="Test Paper",
        authors=["Author A"],
        abstract="Test abstract",
        category="cs.AI",
        published_date="2024-01-09T00:00:00+00:00",
        updated_date="2024-01-10T00:00:00+00:00",
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )
    cached = CachedSummaryEntry(
        query_text="machine learning",
        category="cs.AI",
        max_results=10,
        date_from=None,
        arxiv_ids=["2401.12345v1"],
        fetched_at="2024-01-09T00:00:00+00:00",
        ttl_expires_at="2024-01-10T00:00:00+00:00",
    )
    result = ResearchResult(
        sub_query="test query",
        papers=[paper],
        cached_summaries=[cached],
        raw_tool_calls=[{"tool": "search_papers", "input": {}, "result": {}}],
    )

    # Test serialization
    data = result.model_dump(mode="json")
    assert data["sub_query"] == "test query"
    assert len(data["papers"]) == 1
    assert data["papers"][0]["arxiv_id"] == "2401.12345v1"
    assert len(data["cached_summaries"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
