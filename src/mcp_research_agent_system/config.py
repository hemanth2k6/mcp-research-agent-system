"""Configuration settings using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_path: str = "data/research_agent.db"

    # arXiv API
    arxiv_api_base_url: str = "http://export.arxiv.org/api/query"
    arxiv_rate_limit_delay: float = 3.0  # seconds between requests

    # Cache
    cache_ttl_hours: int = 24

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
