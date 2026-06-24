"""Semantic Compatibility Profiling agent.

Two-stage code-generation pipeline that extracts relational ecosystem profiles
(Stage 1) and gap/cross-validation narratives (Stage 2) as pure-text signals,
then feeds them to a reasoning decision model.
"""

from .pipeline import run_sem_agent

__all__ = ["run_sem_agent"]
