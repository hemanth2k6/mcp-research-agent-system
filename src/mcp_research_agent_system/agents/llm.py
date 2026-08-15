"""LLM factory for the research agents."""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import Settings


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured from application settings.

    Reads the OpenAI-compatible endpoint configuration from Settings:
        - LLM_BASE_URL -> base_url
        - LLM_API_KEY -> api_key
        - LLM_MODEL -> model

    Returns:
        A configured ChatOpenAI instance ready for use by agent nodes.
    """
    settings = Settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        model=settings.llm_model,
    )
