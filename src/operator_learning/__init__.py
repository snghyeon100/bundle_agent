"""Hypothesis-conditioned candidate-program learning."""

from .pipeline import (
    admit_verified_programs,
    build_operator_capability_manifest,
    compile_operator_programs,
    deduplicate_raw_operators,
    discover_operator_library,
    induce_raw_operators,
    load_operator_library,
    sample_validation_cases,
    save_operator_artifacts,
    verify_compiled_programs,
)
from .runtime import (
    SourceAPI,
    evaluate_candidate_proposal_set,
    validate_candidate_proposal_set,
)

__all__ = [
    "SourceAPI",
    "admit_verified_programs",
    "build_operator_capability_manifest",
    "compile_operator_programs",
    "deduplicate_raw_operators",
    "discover_operator_library",
    "evaluate_candidate_proposal_set",
    "induce_raw_operators",
    "load_operator_library",
    "sample_validation_cases",
    "save_operator_artifacts",
    "validate_candidate_proposal_set",
    "verify_compiled_programs",
]
