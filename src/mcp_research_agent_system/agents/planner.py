"""Planner agent: decomposes a high-level research goal into arXiv-searchable sub-queries and validates researcher output."""

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..errors import PlannerError
from .llm import get_llm

if TYPE_CHECKING:
    from .researcher import ResearchResult

logger = logging.getLogger(__name__)


class PlannerDecomposition(BaseModel):
    """Structured output for planner goal decomposition."""

    sub_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 focused, non-overlapping arXiv-searchable sub-queries derived from the research goal",
    )


class ValidationOutcome(BaseModel):
    """Structured output for research output validation."""

    is_valid: bool = Field(..., description="Whether the research output is valid for the sub-query")
    reason: str = Field(..., description="Explanation of the validation decision")
    revised_query: str | None = Field(
        default=None,
        description="A revised, more specific or differently-phrased query if invalid; None if valid",
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


def _token_overlap(query: str, text: str) -> float:
    """Calculate token overlap ratio between query and text (0.0 to 1.0).

    Returns the fraction of query tokens that appear in the text.
    """
    if not query or not text:
        return 0.0

    query_tokens = set(re.findall(r'\w+', query.lower()))
    text_tokens = set(re.findall(r'\w+', text.lower()))

    if not query_tokens:
        return 0.0

    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


def _heuristic_validate(sub_query: str, research_result: "ResearchResult") -> ValidationOutcome | None:
    """Heuristic-first validation of research output.

    Returns ValidationOutcome if a clear decision can be made, None if ambiguous.
    Heuristic checks (no LLM call):
    - Empty papers list -> invalid
    - Papers with near-zero keyword overlap with sub_query -> invalid
    - Papers with high keyword overlap -> valid
    """
    papers = research_result.papers

    # Check 1: Empty papers list
    if not papers:
        return ValidationOutcome(
            is_valid=False,
            reason="Research returned zero papers for the sub-query",
            revised_query=_suggest_revised_query(sub_query, "no results"),
        )

    # Check 2: Token overlap with titles/abstracts
    # Calculate average overlap across all papers
    total_overlap = 0.0
    for paper in papers:
        title_overlap = _token_overlap(sub_query, paper.title)
        abstract_overlap = _token_overlap(sub_query, paper.abstract)
        # Weight title higher than abstract
        paper_overlap = (title_overlap * 0.6) + (abstract_overlap * 0.4)
        total_overlap += paper_overlap

    avg_overlap = total_overlap / len(papers)

    # Near-zero overlap threshold -> invalid
    if avg_overlap < 0.1:
        return ValidationOutcome(
            is_valid=False,
            reason=f"Papers have very low keyword overlap with sub-query (avg overlap: {avg_overlap:.2f})",
            revised_query=_suggest_revised_query(sub_query, "off-topic results"),
        )

    # High overlap threshold -> valid
    if avg_overlap >= 0.3:
        return ValidationOutcome(
            is_valid=True,
            reason=f"Papers have high keyword overlap with sub-query (avg overlap: {avg_overlap:.2f})",
            revised_query=None,
        )

    # Ambiguous: some overlap but not strong - fall through to LLM judge
    return None


def _suggest_revised_query(original_query: str, failure_reason: str) -> str:
    """Generate a revised query based on the failure reason."""
    if failure_reason == "no results":
        # Try more specific or different phrasing
        if " " in original_query:
            # Try using just the first few key terms
            words = original_query.split()
            if len(words) > 2:
                return " ".join(words[:3])
        return original_query + " recent"

    elif failure_reason == "off-topic results":
        # Try more specific phrasing
        return original_query + " survey"

    return original_query + " paper"


async def _llm_judge_validate(sub_query: str, research_result: "ResearchResult") -> ValidationOutcome:
    """LLM-judge validation fallback for ambiguous cases.

    Uses structured output with the same fallback-parse pattern as decompose_goal.
    """
    llm = get_llm()

    # Prepare paper summaries for the LLM
    paper_summaries = []
    for i, paper in enumerate(research_result.papers[:5]):  # Limit to first 5
        paper_summaries.append(
            f"Paper {i+1}: {paper.title}\nAbstract: {paper.abstract[:300]}..."
        )

    papers_text = "\n\n".join(paper_summaries) if paper_summaries else "No papers returned."

    system_prompt = """You are a research validation judge. Given a sub-query and a list of paper titles/abstracts
returned by a search, judge whether the results are relevant to the sub-query.

Return ONLY a JSON object with:
- is_valid: boolean (true if papers are relevant to the sub-query)
- reason: string explaining your judgment
- revised_query: string or null (if is_valid is false, provide a genuinely different,
  more specific or differently-phrased query; if true, null)

Be strict but fair. Papers that are completely off-topic should be invalid.
Papers that are tangentially related but not directly relevant should also be invalid.
If suggesting a revised query, make it meaningfully different - not just a rewording."""

    human_prompt = f"""Sub-query: {sub_query}

Papers found:
{papers_text}

Judge the relevance and provide the JSON output."""

    # First attempt: structured output
    structured_llm = llm.with_structured_output(ValidationOutcome)
    try:
        result = await structured_llm.ainvoke([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        if isinstance(result, ValidationOutcome):
            logger.info("LLM judge validation succeeded (structured output)")
            return result
    except Exception as e:
        logger.warning(f"LLM judge structured output failed, falling back to manual parse: {e}")

    # Fallback: manual JSON parsing with retry (same pattern as decompose_goal)
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([
                ("system", system_prompt),
                ("human", human_prompt),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            content_str = content if isinstance(content, str) else json.dumps(content)

            # Try to parse as ValidationOutcome
            try:
                data = json.loads(content_str)
                return ValidationOutcome(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Try extracting from markdown/surrounding text
                json_match = re.search(r"\{.*\}", content_str, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return ValidationOutcome(**data)

        except Exception as e:
            last_error = e
            logger.warning(f"LLM judge fallback attempt {attempt + 1} failed: {e}")

    # If all retries fail, default to invalid with a generic revised query
    logger.error(f"LLM judge validation failed after retries: {last_error}")
    return ValidationOutcome(
        is_valid=False,
        reason="LLM judge validation failed after retries",
        revised_query=_suggest_revised_query(sub_query, "validation error"),
    )


async def validate_research_output(sub_query: str, research_result: "ResearchResult") -> ValidationOutcome:
    """Validate research output for a sub-query.

    Heuristic-first approach:
    1. Check for empty papers -> invalid
    2. Check token overlap -> invalid if near-zero
    3. If ambiguous, fall back to LLM judge

    Args:
        sub_query: The sub-query that was researched
        research_result: ResearchResult from the researcher agent

    Returns:
        ValidationOutcome with is_valid, reason, and revised_query if invalid
    """
    # Heuristic checks
    heuristic_result = _heuristic_validate(sub_query, research_result)
    if heuristic_result is not None:
        logger.info(f"Heuristic validation: {heuristic_result.is_valid} - {heuristic_result.reason}")
        return heuristic_result

    # Ambiguous case: fall back to LLM judge
    logger.info("Heuristic validation inconclusive, falling back to LLM judge")
    return await _llm_judge_validate(sub_query, research_result)


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
