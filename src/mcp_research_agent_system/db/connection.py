"""Database connection module for SQLite with WAL mode and schema initialization."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ..config import Settings


def get_db_path(settings: Settings) -> Path:
    """Get the database file path from settings."""
    return Path(settings.database_path)


def init_db(settings: Settings) -> None:
    """Initialize the database: create tables and apply schema."""
    db_path = get_db_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON;")

        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()


@contextmanager
def get_connection(settings: Settings):
    """Context manager for database connections with proper configuration."""
    db_path = get_db_path(settings)
    conn = sqlite3.connect(db_path)
    try:
        # Enable WAL mode and foreign keys on each connection
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        # Return rows as dict-like objects
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()
