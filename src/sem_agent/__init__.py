"""Semantic evidence expansion agent.

Four-agent pipeline: problem analysis, evidence retrieval code generation,
neutral item summary/profile, and final reasoning decision.
"""

from .pipeline import run_sem_agent

__all__ = ["run_sem_agent"]
