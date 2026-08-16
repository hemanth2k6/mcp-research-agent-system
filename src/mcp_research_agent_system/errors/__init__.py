"""Custom exception hierarchy for the research agent system."""


class PlannerError(Exception):
    """Raised when the planner agent fails to decompose a research goal."""


__all__ = ["PlannerError"]
