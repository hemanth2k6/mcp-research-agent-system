-- SQLite schema for arXiv paper cache

-- Papers table: stores individual paper metadata from arXiv
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,  -- JSON array of author names
    abstract TEXT NOT NULL,
    category TEXT NOT NULL,
    published_date TEXT NOT NULL,  -- ISO 8601 format
    updated_date TEXT NOT NULL,    -- ISO 8601 format
    pdf_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL       -- ISO 8601 timestamp when we cached it
);

-- Indexes for papers table
CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(category);
CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date);
CREATE INDEX IF NOT EXISTS idx_papers_fetched_at ON papers(fetched_at);

-- Search cache table: stores search query results with TTL
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    category TEXT,
    max_results INTEGER NOT NULL,
    date_from TEXT,
    arxiv_ids TEXT NOT NULL,       -- JSON array of paper arxiv_ids
    fetched_at TEXT NOT NULL,      -- ISO 8601 timestamp when we cached it
    ttl_expires_at TEXT NOT NULL   -- ISO 8601 timestamp when cache expires
);

-- Indexes for search_cache table
CREATE INDEX IF NOT EXISTS idx_search_cache_category ON search_cache(category);
CREATE INDEX IF NOT EXISTS idx_search_cache_fetched_at ON search_cache(fetched_at);
CREATE INDEX IF NOT EXISTS idx_search_cache_ttl_expires_at ON search_cache(ttl_expires_at);