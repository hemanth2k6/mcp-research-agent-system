"""Tests for the arXiv client."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_research_agent_system.arxiv_client import ArxivClient, Paper


@pytest.fixture
def sample_atom_xml() -> str:
    """Load the sample Atom XML fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_atom_response.xml"
    return fixture_path.read_text()


@pytest.fixture
def mock_httpx_response(sample_atom_xml: str) -> httpx.Response:
    """Create a mock httpx Response with the sample XML."""
    response = MagicMock(spec=httpx.Response)
    response.text = sample_atom_xml
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def settings():
    """Create test settings with short rate limit."""
    from mcp_research_agent_system.config import Settings

    return Settings(
        arxiv_rate_limit_delay=0.0,
        database_path=":memory:",
    )


@pytest.mark.asyncio
async def test_search_parses_papers_correctly(settings, mock_httpx_response):
    """Test that search correctly parses the Atom XML response."""
    client = ArxivClient(settings)

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response

        papers = await client.search(query="machine learning", category="cs.AI", max_results=10)

    assert len(papers) == 2

    # Check first paper
    p1 = papers[0]
    assert p1.arxiv_id == "2401.12345v1"
    assert p1.title == "Deep Learning for Robotics: A Survey"
    assert p1.authors == ["Alice Johnson", "Bob Smith"]
    assert "deep learning applied to robotics" in p1.abstract.lower()
    assert p1.category == "cs.AI"
    assert p1.published_date == datetime(2024, 1, 9, 12, 0, 0, tzinfo=UTC)
    assert p1.updated_date == datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
    assert p1.pdf_url == "http://arxiv.org/pdf/2401.12345v1"

    # Check second paper
    p2 = papers[1]
    assert p2.arxiv_id == "2401.67890v2"
    assert p2.title == "Reinforcement Learning in Large State Spaces"
    assert p2.authors == ["Carol Williams"]
    assert p2.category == "cs.LG"


@pytest.mark.asyncio
async def test_search_filters_by_date_from(settings, mock_httpx_response):
    """Test that date_from filter works correctly."""
    client = ArxivClient(settings)

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response

        # First paper is 2024-01-09, second is 2024-01-11
        date_from = datetime(2024, 1, 10)
        papers = await client.search(
            query="machine learning",
            category="cs.AI",
            max_results=10,
            date_from=date_from,
        )

    # Only second paper should be returned
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.67890v2"


@pytest.mark.asyncio
async def test_search_builds_correct_query_params(settings, mock_httpx_response):
    """Test that search builds correct query parameters."""
    client = ArxivClient(settings)

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response

        await client.search(
            query="machine learning",
            category="cs.AI",
            max_results=5,
            date_from=datetime(2024, 1, 1),
        )

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    params = call_args[1]["params"]

    assert params["search_query"] == "cat:cs.AI AND (machine learning)"
    assert params["max_results"] == 5
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


@pytest.mark.asyncio
async def test_search_without_category(settings, mock_httpx_response):
    """Test search without category filter."""
    client = ArxivClient(settings)

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response

        await client.search(query="machine learning", max_results=10)

    call_args = mock_get.call_args
    params = call_args[1]["params"]
    assert params["search_query"] == "machine learning"


@pytest.mark.asyncio
async def test_rate_limit_delay(settings):
    """Test that rate limiting adds delay between requests."""
    settings.arxiv_rate_limit_delay = 0.01  # Small delay for test
    client = ArxivClient(settings)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
        <entry>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <updated>2024-01-10T12:00:00Z</updated>
            <published>2024-01-09T12:00:00Z</published>
            <title>Test Paper</title>
            <summary>Test abstract</summary>
            <author><name>Test Author</name></author>
            <arxiv:primary_category term="cs.AI"/>
            <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1"/>
        </entry>
    </feed>"""
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        import time
        start = time.monotonic()
        await client.search(query="test")
        await client.search(query="test")
        elapsed = time.monotonic() - start

    # Should have waited at least the rate limit delay
    assert elapsed >= 0.01


@pytest.mark.asyncio
async def test_context_manager(settings):
    """Test async context manager properly closes client."""
    client = ArxivClient(settings)

    with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
        async with client as c:
            assert c is client

        mock_close.assert_called_once()


def test_paper_model():
    """Test Paper Pydantic model validation."""
    paper = Paper(
        arxiv_id="2401.12345v1",
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="Test abstract",
        category="cs.AI",
        published_date=datetime(2024, 1, 9),
        updated_date=datetime(2024, 1, 10),
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )

    assert paper.arxiv_id == "2401.12345v1"
    assert len(paper.authors) == 2
