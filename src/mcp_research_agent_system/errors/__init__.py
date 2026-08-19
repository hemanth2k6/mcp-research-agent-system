"""Custom exception hierarchy for the research agent system."""

from typing import Any


class PlannerError(Exception):
    """Raised when the planner agent fails to decompose a research goal."""


class ResearcherError(Exception):
    """Raised when the researcher agent encounters an error (server startup, tool call, timeout, etc.)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class SynthesizerError(Exception):
    """Raised when the synthesizer agent fails to generate the final report."""


__all__ = ["PlannerError", "ResearcherError", "SynthesizerError"]
