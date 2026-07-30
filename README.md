# Bundle Agent: Candidate-Program Learning MVP

## Research objective

Bundle completion is underdetermined: the same partial bundle can support
multiple plausible completion intents. The MVP therefore discovers several
case-conditioned hypotheses and turns each into a reusable, source-bounded
candidate-retrieval program.

```text
partial bundle
  -> candidate-blind completion hypotheses
  -> reusable candidate-program specifications
  -> deterministic deduplication
  -> offline compilation
  -> held-out execution and admission
  -> verified program library
```

There is no operator compatibility graph, deterministic path, LLM clustering,
or test-sample-specific code generation.

## Program definition

A macro operator is a reusable executable program that operationalizes one
completion hypothesis. Given any partial bundle, it searches permitted sources
for a small set of plausible missing-item candidates and returns source
provenance for every proposed item.

```text
program = one hypothesis
        + one source-bounded retrieval procedure
        + bounded candidate proposals
        + candidate-linked provenance
```

The fixed runtime contract is:

```python
def execute(
    partial_item_ids,
    source_api,
    candidate_budget,
    evidence_budget,
):
    ...
```

Every execution returns `candidate_proposal_set_v1`:

```json
{
  "schema_version": "candidate_proposal_set_v1",
  "program_id": "program_001",
  "hypothesis": "Related historical bundles contain plausible missing items.",
  "candidate_proposals": [
    {
      "item_id": 42,
      "evidence_refs": ["E1"]
    }
  ],
  "evidence_records": [
    {
      "evidence_id": "E1",
      "type": "historical_bundle_context",
      "source": "bundle_item_history",
      "anchor_item_ids": [1, 2],
      "related_item_ids": [42, 51],
      "related_bundle_ids": [9],
      "attributes": {}
    }
  ],
  "execution_trace": {
    "used_sources": ["bundle_item_history"],
    "candidate_budget": 10,
    "evidence_budget": 8
  }
}
```

Programs may use counts, similarities, and other numeric measures internally to
retrieve and compress evidence. Their external task-level output is candidate
items with source provenance, not an opaque answer score.

## First LLM call: candidate-blind induction

Each discovery sample uses one LLM call. The prompt contains:

- partial-item text and metadata;
- source capabilities and GT-independent diagnostics;
- compact candidate-program memory.

It never contains:

- answer options;
- the missing-item ground truth;
- candidate ranks or labels.

The call first emits short sample-conditioned case hypotheses, then maps them
one-to-one to reusable program specifications:

```json
{
  "hypotheses": [
    {
      "id": "H1",
      "observed_cues": [
        "tailored outerwear silhouette",
        "dark coordinated palette"
      ],
      "statement": "The bundle may be assembling a polished coordinated outfit that needs a complementary item role."
    }
  ],
  "operators": [
    {
      "hypothesis_id": "H1",
      "name": "RetrieveRecurringBundleCompanions",
      "hypothesis": "Items recurring across partial-conditioned historical bundles are plausible completions.",
      "required_sources": ["bundle_item_history"],
      "applicability": [
        "partial items have historical bundle coverage"
      ],
      "evidence_types": ["historical_bundle_context"],
      "pseudocode": [
        "retrieve historical bundles containing partial items",
        "collect non-partial items from those bundles",
        "retain a bounded set of recurring candidate items",
        "return representative bundle provenance for each candidate"
      ],
      "output_contract": "candidate_proposals_with_source_provenance"
    }
  ]
}
```

The sample-specific hypotheses and their observed cues are retained in induction
traces. Candidate memory exposes only compact forbidden signatures (`name`,
generalized `hypothesis`, `required_sources`, and `evidence_types`) so previous
programs act as an exclusion list rather than as generation templates.

Run induction:

```powershell
python tests/test_operator_induction.py `
  --config config_operator.yaml `
  --sample_count 3 `
  --output_dir tests/outputs/operators/candidate_program_test
```

## Deterministic deduplication

Deduplication makes no LLM call. It groups exact or high-similarity
specifications only when their required sources and evidence types match, and
records every member operator and discovery-case provenance.

```powershell
python tests/test_operator_clustering.py `
  --config config_operator.yaml `
  --operator_pool tests/outputs/operators/<run>/operator_pool.json
```

Despite the legacy script name, this stage is deterministic program
deduplication, not semantic LLM clustering.

## Second LLM call type: offline compilation

After deduplication, each unique program receives one compilation call. The
compiler sees only:

- the canonical reusable specification;
- its permitted source manifest;
- the shared `SourceAPI` contract;
- the fixed `CandidateProposalSet` output schema.

It does not see the discovery sample, answer options, or ground truth.

```powershell
python tests/test_operator_code_generation.py `
  --config config_operator.yaml `
  --library tests/outputs/dedup/<run>/operator_library.json
```

The generated function is stored with a SHA-256 hash. Any code change
invalidates its validation result. Held-out validation and online inference must
execute the exact same code artifact.

## Verification and admission

`operator_learning.verify_compiled_programs` accepts an injected sandbox runner
and evaluates each immutable program on held-out cases. The runtime validator
checks:

- source scope;
- candidate and evidence budgets;
- candidate-linked provenance;
- output schema;
- execution success.

Initial retrieval metrics include:

- candidate recall under a fixed budget;
- retrieval rank and reciprocal rank;
- candidate-set size;
- execution success rate.

`operator_learning.admit_verified_programs` creates separate verified and
rejected registries. Unverified candidate memory is never treated as an online
library.

## Online hypothesis-conditioned exemplar-retrieval pilot

`online_hypothesis_program` implements a separate two-call path without an
offline operator library:

```text
partial-item text
  -> LLM1: programs containing hypothesis + reference + retrieval strategy + code
  -> hash and fix hypothesis/program/parameters/source scope/budget
  -> runtime executes each retrieve() function once on the partial bundle
  -> retrievers return bounded corpus item IDs with source provenance
  -> runtime resolves IDs into readable hypothesis-specific exemplars
  -> LLM2 receives the answer options and performs a full ranking
```

LLM1 receives only partial-item text and the raw workspace manifest; answer
options remain hidden until retrieval is complete. Each hypothesis constructs a
different reference context and retrieves a small set of plausible completion
exemplars. The runtime verifies item and provenance references, renders related
item/bundle/user text, and only then gives LLM2 the answer options. The pipeline
uses exactly two LLM calls per case and reports Hit@1/3/5, MRR, GT rank,
retrieval counts, answer-option overlap, and GT retrieval.

```powershell
python tests/test_online_hypothesis_program.py `
  --config config_operator.yaml `
  --split test `
  --sample_idx 1
```

To inspect strategy induction separately from Python compilation and execution,
run the code-free diagnostic below. It asks for exactly three distinct
hypotheses and 3--8 stage macro pseudocode plans, then validates their source
scope and structure.

```powershell
python tests/test_online_hypothesis_strategy.py `
  --config config_operator.yaml `
  --split test `
  --sample_idx 19
```

## Direct plausible-set diagnostic

Before attributing errors to retrieval programs or final aggregation, a
one-call diagnostic asks the model to return every answer option that it can
defend under at least one coherent completion hypothesis. It does not impose a
fixed set size. In the same response, the model separately ranks all supplied
options from most to least plausible. Evaluation records plausible-set coverage
and size together with Hit@1/3/5, MRR, and mean GT rank. It also checks whether
the plausible set exactly matches the top-k ranking where k is the model's own
plausible-set size, and reports the resulting self-consistency rate.

The default batch size is 250 samples:

```powershell
python tests/test_direct_plausible_set.py `
  --config config_operator.yaml `
  --split test `
  --sample_count 250
```

Use `--sample_idx 1` for a one-sample spot check. A stopped batch can continue
from the same output directory with `--resume <output-directory>`.
