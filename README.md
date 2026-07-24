# Bundle Agent: Compact Operator Learning MVP

## Research objective

This project learns reusable operations for bundle completion.

The central distinction is:

```text
operator = one reusable input → transformation → output
strategy = multiple operators connected for one prediction case
```

Operator induction must therefore produce atomic transformations, not complete
candidate-ranking pipelines. Workflow composition, code generation, execution,
and rank-based reflection are separate downstream stages.

## Current pipeline

```text
validation cases with hindsight labels
    ↓
compact operator induction
    ↓
raw operator pool
    ↓
semantic clustering
    ↓
reusable operator library
    ↓
test-time workflow composition
    ↓
code generation and execution
    ↓
prediction
```

The current MVP implements induction, clustering, and workflow composition.
Prediction-level operator evaluation and rank-based reflection are planned
after the basic operator abstraction is validated.

## Compact operator representation

Every induced operator has exactly six fields:

```json
{
  "name": "ContrastCandidateIntent",
  "objective": "Expose candidate-specific semantic agreement and conflict.",
  "input": "candidate metadata and a partial-bundle intent hypothesis",
  "operation": "contrast each candidate's attributes against the inferred intent",
  "output": "candidate-indexed semantic agreement and conflict evidence",
  "sources": ["item_metadata"]
}
```

The fields mean:

- `name`: concise reusable operation name;
- `objective`: why the transformation is useful;
- `input`: one logical input artifact in concise natural language;
- `operation`: one central semantic transformation;
- `output`: one reusable intermediate artifact;
- `sources`: capability IDs needed to execute the operation, or `[]`.

The induction output deliberately omits:

- fixed operator kinds;
- nested type contracts;
- anchors;
- preconditions;
- failure-signal lists;
- applicability prose;
- workflow order;
- next-operator references.

This keeps raw operators easy to inspect, compare, embed, and cluster.

## No predefined type catalog

`input` and `output` are concise natural-language artifact descriptions. The
model is not given a fixed type ontology, and lexical equality is not required.

A later workflow composer or code-generation agent must determine whether an
operator output can satisfy another operator input. If descriptions differ but
the underlying artifacts are compatible, the downstream agent may construct an
explicit adapter.

This is an experimental choice: the MVP tests whether semantic interfaces are
sufficient before introducing a human-designed type system.

## Source capabilities are not operators

The source-capability manifest describes executable data access primitives,
including:

- item metadata;
- bundle-item history;
- user-item history;
- content and description embeddings;
- collaborative embeddings;
- dataset statistics.

An induced operator may list these capability IDs in `sources`, but a lookup
alone is not considered a learned operator.

```text
source capability:
    retrieve item embeddings

induced operator:
    transform bundle and candidate embeddings into candidate-specific
    supporting and conflicting evidence
```

The distinction prevents the raw operator pool from becoming a renamed list of
available files or APIs.

## Offline stage 1: operator induction

Induction samples labeled examples only from the validation split:

- `bi_valid_input.txt`
- `bi_valid_gt.txt`

The test split is not used for operator discovery.

For each sampled validation case, the model receives:

- partial-bundle item descriptions;
- a finite candidate set;
- the ground-truth candidate as hindsight evidence;
- the source-capability manifest.

Ground truth is used only to discover what evidence would have been
discriminative. It must never appear in an operator's deployable fields.

The prompt requires each operator to:

- describe one atomic transformation;
- remain reusable across cases;
- avoid product names, candidate labels, and item IDs;
- avoid references to other generated operators or execution order;
- produce an intermediate artifact rather than a final choice;
- avoid rank, prediction, and score-only outputs;
- use only available capability IDs in `sources`.

The deterministic validator enforces:

- the exact six-field schema;
- non-empty textual fields;
- valid and unique source IDs;
- no ground-truth or correct-answer dependency.

Rank, prediction, final-choice, and score-only outputs remain prohibited by the
induction prompt, but are not rejected by a brittle keyword-based validator.

Run induction:

```powershell
.venv\Scripts\python.exe tests\test_operator_induction.py `
  --config config_operator.yaml `
  --sample_count 3
```

The run writes:

```text
tests/outputs/operators/<dataset>_<timestamp>/
├── run.json
├── source_capabilities.json
├── validation_samples.json
├── operator_pool.json
├── operators_by_sample.json
├── summary.json
├── cases/
│   └── <case_id>/
│       ├── input.txt
│       ├── output.txt
│       ├── parsed_response.json
│       ├── validation_issues.json
│       ├── connection_diagnostics.json
│       └── operators.json
└── samples/
    └── <case_id>.json
```

## Offline stage 2: semantic clustering

Clustering operates on the saved raw operator pool. Operators are merged only
when the following describe the same reusable transition:

1. objective;
2. input artifact;
3. central operation;
4. output artifact.

Shared sources or similar names are not sufficient grounds for merging.

Each refined operator keeps the same compact six-field representation and adds:

```json
{
  "derived_from": ["bundle_20517__op3", "bundle_28141__op2"]
}
```

Clusters additionally record their member IDs, representative operator, and
merge rationale. Every raw operator must belong to exactly one cluster.

Run clustering:

```powershell
.venv\Scripts\python.exe tests\test_operator_clustering.py `
  --config config_operator.yaml `
  --operator_pool tests\outputs\operators\<run>\operator_pool.json
```

The clustered library uses:

```json
{
  "schema_version": "compact_operator_library_v1"
}
```

## Test-time workflow composition

The composer receives:

- one held-out test case without a label;
- the compact operator library;
- available sources.

It creates multiple workflows by connecting operator outputs to semantically
compatible inputs. It must state any required adaptation explicitly and may not
pretend that incompatible interfaces connect.

The composer chooses a recommended workflow from case and source applicability
only. It has no ground truth, historical candidate rank, or operator reward.

## Future execution and reflection

The next research step is to determine whether a selected operator or workflow
actually improves prediction.

A minimal evaluation loop is:

```text
compose workflow
    ↓
generate executable implementation
    ↓
run on validation cases
    ↓
measure ranking change
    ↓
attribute gains or failures to operators
    ↓
retain, revise, or remove operators
```

Useful measurements include:

- prediction accuracy or ranking gain over a baseline;
- operator selection frequency;
- execution success rate;
- marginal gain when an operator is added or removed;
- redundant operator pairs;
- source cost and latency;
- semantic-interface connection failures.

Rank-based reflection should be introduced only after code generation can
reliably execute the compact operator interfaces. Otherwise execution failures
and operator quality become confounded.

## MVP interpretation

The induction result should be judged on three separate levels:

1. **Atomicity**: is each object one transformation rather than a strategy?
2. **Reusability**: is it independent of the current answer and item identity?
3. **Composability**: can a later agent infer meaningful input/output links?

Passing JSON validation establishes only structural validity. It does not prove
that an operator is useful, executable, or improves prediction. Those claims
require downstream execution and ranking experiments.
