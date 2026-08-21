"""Async arXiv API client with Atom XML parsing and rate limiting."""

import asyncio
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, Field

from .config import Settings

# arXiv Atom namespace
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class Paper(BaseModel):
    """Pydantic model for parsed arXiv paper data."""

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    category: str = Field(..., description="Primary category")
    published_date: datetime = Field(..., description="Publication date")
    updated_date: datetime = Field(..., description="Last updated date")
    pdf_url: str = Field(..., description="URL to PDF")


class ArxivClient:
    """Async client for the arXiv API with rate limiting."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._client = httpx.AsyncClient(timeout=30.0)
        self._last_request_time: datetime | None = None

    async def __aenter__(self) -> "ArxivClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _rate_limit(self) -> None:
        """Apply rate limiting delay between requests."""
        if self._last_request_time is not None:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.settings.arxiv_rate_limit_delay:
                await asyncio.sleep(self.settings.arxiv_rate_limit_delay - elapsed)
        self._last_request_time = datetime.now()

    async def search(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 10,
        date_from: datetime | None = None,
    ) -> list[Paper]:
        """Search arXiv API and return parsed papers.

        Args:
            query: Search query string
            category: arXiv category to filter (e.g., 'cs.AI')
            max_results: Maximum number of results to return
            date_from: Only return papers published after this date

        Returns:
            List of parsed Paper objects
        """
        """Search arXiv API and return parsed papers.

        Args:
            query: Search query string
            category: arXiv category to filter (e.g., 'cs.AI')
            max_results: Maximum number of results to return
            date_from: Only return papers published after this date

        Returns:
            List of parsed Paper objects
        """
        await self._rate_limit()

        # Build search query
        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})"

        params: dict[str, str | int] = {
            "search_query": search_query,
            "max_results": max_results,
            "start": 0,
        }

        if date_from:
            params["sortBy"] = "submittedDate"
            params["sortOrder"] = "descending"

        response = await self._client.get(self.settings.arxiv_api_base_url, params=params)
        response.raise_for_status()

        return self._parse_atom_response(response.text, date_from)

    def _parse_atom_response(self, xml_text: str, date_from: datetime | None = None) -> list[Paper]:
        """Parse Atom XML response into Paper objects.

        Args:
            xml_text: Raw Atom XML response from arXiv
            date_from: Optional date filter to apply post-parse

        Returns:
            List of Paper objects
        """
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            # Extract arXiv ID from the id URL
            id_elem = entry.find("atom:id", ATOM_NS)
            if id_elem is None or not id_elem.text:
                continue
            arxiv_url = id_elem.text
            # Format: http://arxiv.org/abs/1234.5678v1 -> 1234.5678v1
            arxiv_id = arxiv_url.split("/abs/")[-1]

            title_elem = entry.find("atom:title", ATOM_NS)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            # Extract authors
            authors: list[str] = []
            for author in entry.findall("atom:author", ATOM_NS):
                name_elem = author.find("atom:name", ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            abstract_elem = entry.find("atom:summary", ATOM_NS)
            abstract = (
                abstract_elem.text.strip()
                if abstract_elem is not None and abstract_elem.text
                else ""
            )

            # Extract category (primary category)
            category = ""
            primary_cat = entry.find("arxiv:primary_category", ATOM_NS)
            if primary_cat is not None:
                category = primary_cat.get("term", "")
            if not category:
                # Fall back to first category
                cat_elem = entry.find("atom:category", ATOM_NS)
                if cat_elem is not None:
                    category = cat_elem.get("term", "")

            # Extract dates
            published = entry.find("atom:published", ATOM_NS)
            updated = entry.find("atom:updated", ATOM_NS)
            published_date = (
                datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                if published is not None and published.text
                else datetime.now(UTC)
            )
            updated_date = (
                datetime.fromisoformat(updated.text.replace("Z", "+00:00"))
                if updated is not None and updated.text
                else published_date
            )

            # Extract PDF URL
            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url:
                pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                category=category,
                published_date=published_date,
                updated_date=updated_date,
                pdf_url=pdf_url,
            )

            # Apply date filter if specified
            if date_from is not None:
                # Ensure date_from is timezone-aware for comparison
                if date_from.tzinfo is None:
                    date_from = date_from.replace(tzinfo=UTC)
            if date_from is None or paper.published_date >= date_from:
                papers.append(paper)

        return papers
