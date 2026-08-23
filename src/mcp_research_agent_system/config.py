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
    arxiv_api_base_url: str = "https://export.arxiv.org/api/query"
    arxiv_rate_limit_delay: float = 3.0  # seconds between requests

    # Cache
    cache_ttl_hours: int = 24

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # LLM (OpenAI-compatible)
    # Tested default: Gemini (https://generativelanguage.googleapis.com/v1beta/openai/, gemini-3.5-flash)
    # Any OpenAI-compatible endpoint works; configure via LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = "your-gemini-api-key-here"
    llm_model: str = "gemini-3.6-flash"
