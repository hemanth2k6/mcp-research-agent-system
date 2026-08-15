"""Tests for the cache layer."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_research_agent_system.arxiv_client import Paper
from mcp_research_agent_system.cache import cached_search, init_db
from mcp_research_agent_system.config import Settings
from mcp_research_agent_system.db.connection import get_connection


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
            title="Paper One",
            authors=["Author A"],
            abstract="Abstract one",
            category="cs.AI",
            published_date=datetime(2024, 1, 9),
            updated_date=datetime(2024, 1, 10),
            pdf_url="http://arxiv.org/pdf/2401.12345v1",
        ),
        Paper(
            arxiv_id="2401.67890v2",
            title="Paper Two",
            authors=["Author B", "Author C"],
            abstract="Abstract two",
            category="cs.LG",
            published_date=datetime(2024, 1, 11),
            updated_date=datetime(2024, 1, 12),
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


@pytest.mark.asyncio
async def test_cached_search_miss_then_hit(test_settings, sample_papers, mock_arxiv_client):
    """Test cache miss on first call, hit on second call."""
    # First call - cache miss
    papers1 = await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    assert len(papers1) == 2
    assert papers1[0].arxiv_id == "2401.12345v1"
    assert papers1[1].arxiv_id == "2401.67890v2"
    mock_arxiv_client.search.assert_called_once()

    # Reset mock
    mock_arxiv_client.search.reset_mock()

    # Second call - should hit cache
    papers2 = await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    assert len(papers2) == 2
    assert papers2[0].arxiv_id == "2401.12345v1"
    mock_arxiv_client.search.assert_not_called()  # Cache hit!


@pytest.mark.asyncio
async def test_cached_search_expiry(test_settings, sample_papers, mock_arxiv_client, temp_db_path):
    """Test that expired cache entries are refreshed."""
    # First call - populate cache
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )
    mock_arxiv_client.search.reset_mock()

    # Manually expire the cache entry
    with get_connection(test_settings) as conn:
        conn.execute(
            "UPDATE search_cache SET ttl_expires_at = ? WHERE query_hash = ?",
            (
                (datetime.now() - timedelta(hours=1)).isoformat(),
                conn.execute(
                    "SELECT query_hash FROM search_cache WHERE query_text = 'machine learning'"
                ).fetchone()["query_hash"],
            ),
        )

    # Third call - should miss due to expiry
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    mock_arxiv_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_cached_search_different_params_different_cache(test_settings, sample_papers, mock_arxiv_client):
    """Test that different query parameters create separate cache entries."""
    # Search 1
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    mock_arxiv_client.search.reset_mock()

    # Search 2 - different category
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.LG",
        max_results=10,
        client=mock_arxiv_client,
    )

    # Should have called API again because category is different
    mock_arxiv_client.search.assert_called_once()

    mock_arxiv_client.search.reset_mock()

    # Search 3 - different max_results
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=5,  # Different from first call
        client=mock_arxiv_client,
    )

    mock_arxiv_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_cached_search_with_date_from(test_settings, sample_papers, mock_arxiv_client):
    """Test cache with date_from parameter."""
    date_from = datetime(2024, 1, 10)

    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        date_from=date_from,
        client=mock_arxiv_client,
    )

    # Check that date_from was passed to client
    mock_arxiv_client.search.assert_called_once()
    call_kwargs = mock_arxiv_client.search.call_args[1]
    assert call_kwargs["date_from"] == date_from


@pytest.mark.asyncio
async def test_cached_search_preserves_order(test_settings, sample_papers, mock_arxiv_client):
    """Test that cache preserves the order of papers from API."""
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    # Call again to hit cache
    mock_arxiv_client.search.reset_mock()
    papers = await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    assert papers[0].arxiv_id == "2401.12345v1"
    assert papers[1].arxiv_id == "2401.67890v2"


@pytest.mark.asyncio
async def test_cached_search_upserts_papers(test_settings, sample_papers, mock_arxiv_client):
    """Test that papers are upserted into the papers table."""
    await cached_search(
        query="machine learning",
        settings=test_settings,
        category="cs.AI",
        max_results=10,
        client=mock_arxiv_client,
    )

    with get_connection(test_settings) as conn:
        rows = conn.execute("SELECT arxiv_id, title FROM papers ORDER BY arxiv_id").fetchall()

    assert len(rows) == 2
    assert rows[0]["arxiv_id"] == "2401.12345v1"
    assert rows[0]["title"] == "Paper One"
    assert rows[1]["arxiv_id"] == "2401.67890v2"
    assert rows[1]["title"] == "Paper Two"


@pytest.mark.asyncio
async def test_cached_search_updates_existing_paper(test_settings, mock_arxiv_client):
    """Test that existing papers are updated on re-fetch."""
    # First paper with original title
    paper1 = Paper(
        arxiv_id="2401.12345v1",
        title="Original Title",
        authors=["Author A"],
        abstract="Original abstract",
        category="cs.AI",
        published_date=datetime(2024, 1, 9),
        updated_date=datetime(2024, 1, 10),
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )
    mock_arxiv_client.search.return_value = [paper1]

    await cached_search(
        query="test",
        settings=test_settings,
        client=mock_arxiv_client,
    )

    # Second fetch with updated title
    paper1_updated = Paper(
        arxiv_id="2401.12345v1",
        title="Updated Title",
        authors=["Author A"],
        abstract="Updated abstract",
        category="cs.AI",
        published_date=datetime(2024, 1, 9),
        updated_date=datetime(2024, 1, 15),  # Updated date
        pdf_url="http://arxiv.org/pdf/2401.12345v1",
    )
    mock_arxiv_client.search.return_value = [paper1_updated]
    mock_arxiv_client.search.reset_mock()

    # Expire cache to force re-fetch
    with get_connection(test_settings) as conn:
        conn.execute(
            "UPDATE search_cache SET ttl_expires_at = ?",
            ((datetime.now() - timedelta(hours=1)).isoformat(),),
        )

    await cached_search(
        query="test",
        settings=test_settings,
        client=mock_arxiv_client,
    )

    with get_connection(test_settings) as conn:
        row = conn.execute("SELECT title FROM papers WHERE arxiv_id = ?", ("2401.12345v1",)).fetchone()

    assert row["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_init_db_creates_tables(test_settings):
    """Test that init_db creates the required tables."""
    init_db(test_settings)

    with get_connection(test_settings) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    table_names = {row["name"] for row in tables}
    assert "papers" in table_names
    assert "search_cache" in table_names


@pytest.mark.asyncio
async def test_init_db_creates_indexes(test_settings):
    """Test that init_db creates the required indexes."""
    init_db(test_settings)

    with get_connection(test_settings) as conn:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()

    index_names = {row["name"] for row in indexes}
    assert "idx_papers_category" in index_names
    assert "idx_papers_published_date" in index_names
    assert "idx_papers_fetched_at" in index_names
    assert "idx_search_cache_category" in index_names
    assert "idx_search_cache_fetched_at" in index_names
    assert "idx_search_cache_ttl_expires_at" in index_names
