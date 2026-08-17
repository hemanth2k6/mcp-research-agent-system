"""Cache layer tying together the arXiv client and SQLite database."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .arxiv_client import ArxivClient, Paper
from .config import Settings
from .db.connection import get_connection, init_db


def _normalize_query_params(
    query: str,
    category: str | None,
    max_results: int,
    date_from: datetime | None,
) -> str:
    """Create a normalized string of query parameters for hashing."""
    date_str = date_from.isoformat() if date_from else ""
    return json.dumps(
        {
            "query": query.strip().lower(),
            "category": category.strip().lower() if category else "",
            "max_results": max_results,
            "date_from": date_str,
        },
        sort_keys=True,
    )


def _hash_query_params(normalized: str) -> str:
    """Hash normalized query parameters to produce a cache key."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _paper_to_db_row(paper: Paper, fetched_at: str) -> dict[str, Any]:
    """Convert a Paper object to a database row dict."""
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": json.dumps(paper.authors),
        "abstract": paper.abstract,
        "category": paper.category,
        "published_date": paper.published_date.isoformat(),
        "updated_date": paper.updated_date.isoformat(),
        "pdf_url": paper.pdf_url,
        "fetched_at": fetched_at,
    }


def _db_row_to_paper(row: sqlite3.Row) -> Paper:
    """Convert a database row to a Paper object."""
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        abstract=row["abstract"],
        category=row["category"],
        published_date=datetime.fromisoformat(row["published_date"]),
        updated_date=datetime.fromisoformat(row["updated_date"]),
        pdf_url=row["pdf_url"],
    )


def _upsert_paper(conn: sqlite3.Connection, paper: Paper, fetched_at: str) -> None:
    """Insert or update a paper in the database."""
    row = _paper_to_db_row(paper, fetched_at)
    conn.execute(
        """
        INSERT OR REPLACE INTO papers
        (arxiv_id, title, authors, abstract, category, published_date,
         updated_date, pdf_url, fetched_at)
        VALUES
        (:arxiv_id, :title, :authors, :abstract, :category, :published_date,
         :updated_date, :pdf_url, :fetched_at)
        """,
        row,
    )


def _get_cached_papers(conn: sqlite3.Connection, arxiv_ids: list[str]) -> list[Paper]:
    """Retrieve papers from the papers table by their arxiv_ids."""
    placeholders = ",".join("?" for _ in arxiv_ids)
    rows = conn.execute(
        f"SELECT * FROM papers WHERE arxiv_id IN ({placeholders})",
        arxiv_ids,
    ).fetchall()
    # Preserve order from arxiv_ids
    paper_map = {row["arxiv_id"]: _db_row_to_paper(row) for row in rows}
    return [paper_map[pid] for pid in arxiv_ids if pid in paper_map]


async def cached_search(
    query: str,
    settings: Settings | None = None,
    category: str | None = None,
    max_results: int = 10,
    date_from: datetime | None = None,
    client: ArxivClient | None = None,
) -> list[Paper]:
    """Perform a cached arXiv search.

    Checks search_cache for a non-expired hit. If found, returns papers from
    the papers table. Otherwise calls the arXiv API, upserts results into
    papers and search_cache with a TTL.

    Args:
        query: Search query string
        settings: Application settings (uses database and cache TTL)
        category: arXiv category filter
        max_results: Maximum number of results
        date_from: Only return papers after this date
        client: Optional pre-configured ArxivClient

    Returns:
        List of Paper objects
    """
    settings = settings or Settings()
    normalized = _normalize_query_params(query, category, max_results, date_from)
    query_hash = _hash_query_params(normalized)
    now = datetime.now()

    # Ensure database is initialized
    init_db(settings)

    with get_connection(settings) as conn:
        # Check cache for non-expired hit
        cached = conn.execute(
            "SELECT * FROM search_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()

        if cached is not None:
            expires_at = datetime.fromisoformat(cached["ttl_expires_at"])
            if expires_at > now:
                # Cache hit — return papers from papers table
                arxiv_ids = json.loads(cached["arxiv_ids"])
                return _get_cached_papers(conn, arxiv_ids)

        # Cache miss or expired — call arXiv API
        own_client = client is None
        if own_client:
            client = ArxivClient(settings)

        try:
            if client is not None:
                papers = await client.search(
                    query=query,
                    category=category,
                    max_results=max_results,
                    date_from=date_from,
                )
            else:
                papers = []
        finally:
            if own_client and client is not None:
                await client.close()

        # Upsert papers and cache entry
        fetched_at = now.isoformat()
        ttl_expires_at = (now + timedelta(hours=settings.cache_ttl_hours)).isoformat()
        arxiv_ids = [p.arxiv_id for p in papers]

        for paper in papers:
            _upsert_paper(conn, paper, fetched_at)

        conn.execute(
            """
            INSERT OR REPLACE INTO search_cache
            (query_hash, query_text, category, max_results, date_from,
             arxiv_ids, fetched_at, ttl_expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_hash,
                query.strip(),
                category.strip().lower() if category else None,
                max_results,
                date_from.isoformat() if date_from else None,
                json.dumps(arxiv_ids),
                fetched_at,
                ttl_expires_at,
            ),
        )

        return papers
