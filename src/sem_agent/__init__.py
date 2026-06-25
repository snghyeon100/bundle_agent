"""Semantic evidence expansion agent.

Two-stage code-generation pipeline that expands item-level supporting evidence
(Stage 1) and builds bundle context plus candidate-fit narratives (Stage 2),
then feeds them to a reasoning decision model.
"""

from .pipeline import run_sem_agent

__all__ = ["run_sem_agent"]
