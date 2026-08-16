"""Planner agent: decomposes a high-level research goal into arXiv-searchable sub-queries."""

import json
import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..errors import PlannerError
from .llm import get_llm

logger = logging.getLogger(__name__)


class PlannerDecomposition(BaseModel):
    """Structured output for planner goal decomposition."""

    sub_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 focused, non-overlapping arXiv-searchable sub-queries derived from the research goal",
    )


SYSTEM_PROMPT = """You are a research planning assistant. Your task is to break down a high-level research goal into 3-5 specific, focused, non-overlapping sub-queries that can be searched on arXiv.

Guidelines:
- Each sub-query should be a concrete search query suitable for arXiv's search API (keywords, phrases, author names, titles)
- Sub-queries should NOT overlap significantly — each should target a distinct aspect of the research goal
- Use terminology and phrasing that arXiv papers would actually contain
- Aim for 3-5 sub-queries total
- Return ONLY a JSON object with a "sub_queries" key containing a list of strings

Example:
Research goal: "Understand the impact of transformers on NLP benchmarks"
Output: {{"sub_queries": ["transformer architecture NLP benchmarks", "BERT GPT comparison GLUE SuperGLUE", "attention mechanism improvements NLP tasks"]}}"""


def _parse_llm_json_response(response_content: str) -> PlannerDecomposition:
    """Parse LLM response content into PlannerDecomposition, with fallback for malformed output."""
    # Try direct JSON parse first
    try:
        data = json.loads(response_content)
        return PlannerDecomposition(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: try to extract JSON from markdown code blocks or surrounding text
    import re
    json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return PlannerDecomposition(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    raise PlannerError(f"Failed to parse LLM response as valid PlannerDecomposition: {response_content[:200]}")


def decompose_goal(research_goal: str, llm: ChatOpenAI | None = None) -> PlannerDecomposition:
    """Decompose a research goal into 3-5 arXiv-searchable sub-queries.

    Args:
        research_goal: The high-level research question from the user.
        llm: Optional injected LLM instance for testing. If None, creates one via get_llm().

    Returns:
        PlannerDecomposition with sub_queries list.

    Raises:
        PlannerError: If the LLM call fails or returns unparseable output after retries.
    """
    if llm is None:
        llm = get_llm()

    structured_llm = llm.with_structured_output(PlannerDecomposition)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Research goal: {research_goal}"),
    ]

    # First attempt: structured output (most reliable with modern models)
    try:
        result = structured_llm.invoke(messages)
        if isinstance(result, PlannerDecomposition):
            logger.info("Planner decomposition succeeded on first attempt (structured output)")
            return result
    except Exception as e:
        logger.warning(f"Structured output failed, falling back to manual parse: {e}")

    # Fallback: manual JSON parsing with retry
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            # response.content can be str | list[str | dict], ensure we pass str
            content_str = content if isinstance(content, str) else json.dumps(content)
            result = _parse_llm_json_response(content_str)
            logger.info(f"Planner decomposition succeeded on fallback attempt {attempt + 1}")
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Fallback attempt {attempt + 1} failed: {e}")

    raise PlannerError(
        f"Failed to decompose research goal after {max_retries} fallback attempts. "
        f"Last error: {last_error}"
    ) from last_error
