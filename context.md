# Bundle Agent Research Context

## Current Research Direction

The current MVP studies train-free, LLM-designed strategies for multiple-choice
bundle completion. The method is implemented under `src/operator_learning/`.

For a partial bundle \(P\) and candidate items \(c_1,\ldots,c_n\), exactly one
candidate is the actual missing item. The method treats
\(B_i=P\cup\{c_i\}\) as competing hypothetical completions and asks an LLM to
design source-grounded strategies that distinguish these alternatives.

The current end-to-end flow is:

```text
partial bundle + candidate items + source diagnostics
  -> LLM1: three completion intents
           + three immutable strategy specifications
           + three executable Python programs
  -> guarded runtime: execute each program over every candidate
  -> candidate-specific textual evidence contexts
  -> LLM2: baseline-shaped multiple-choice prompt + strategy evidence
  -> complete candidate ranking
```

This is currently an online, per-sample experimental pipeline. An offline
strategy pool and online selection/composition stage remain possible future
extensions, but they are not part of the current end-to-end evaluation.

## Method Identity

The central object is not a graph path or a fixed base operator. It is a
reusable relational strategy:

```text
strategy =
    one plausible completion intent
  + one shared reference constructed from the partial bundle and sources
  + one candidate relation applied identically to every candidate
  + one source-grounded textual evidence route
```

A strategy should answer:

1. Under what coherent interpretation could this partial bundle be completed?
2. What shared evaluation basis should be constructed from the partial bundle?
3. What relation between an arbitrary candidate and that basis would
   distinguish plausible from implausible completions?
4. Which related item texts or historical bundle compositions make that
   relation visible to the final prediction model?

The reference is not a candidate and is not the answer. It is a common basis
constructed once within a strategy and used to evaluate every candidate.

## Why the Current Form Was Chosen

Earlier experiments exposed several failure modes:

- Candidate-blind retrieval often produced plausible corpus items that did not
  overlap the benchmark answer options.
- Direct partial-to-candidate checks often collapsed into shallow one-hop or
  text-only comparisons.
- Forcing numeric scores encouraged the prediction model to treat the largest
  number as the answer without interpreting the underlying relation.
- Restricting outputs to generic related texts could produce the same context
  for every candidate.
- Asking for executable code before fixing the strategy caused the intended
  reasoning and the implementation to diverge.
- A rigid helper or skeleton improved execution stability but narrowed strategy
  diversity and encouraged template-like implementations.
- Automatic fallbacks often replaced the declared strategy with a different,
  weaker procedure and obscured whether the original strategy worked.

The current design therefore:

- exposes both the partial bundle and all answer candidates to LLM1;
- hides the ground-truth identity;
- fixes the strategy specification before presenting its code in the same LLM
  response;
- applies one declared candidate relation consistently to all candidates;
- returns candidate-specific source context instead of an opaque final score;
- permits free internal Python structure under a small external I/O contract;
- returns an empty context list when the declared relation finds no evidence;
- does not invent a fallback.

## LLM1: Spec-First Strategy and Program Generation

LLM1 receives:

1. partial item IDs, text, and metadata;
2. candidate labels, IDs, text, and metadata;
3. available source components and exact formats;
4. source diagnostics for the current partial bundle.

The ground-truth label is never included.

LLM1 must first complete exactly three strategy specifications and then provide
exactly three Python programs in the same JSON response. Strategy IDs are
`S1`, `S2`, and `S3`.

Each specification contains:

```json
{
  "strategy_id": "S1",
  "intent": "one plausible interpretation of the completed bundle",
  "name": "ConcisePascalCaseName",
  "description": "candidate ambiguity resolved by this strategy",
  "reference_construction": "how one shared evaluation basis is built",
  "candidate_relation": "the same relation applied to every candidate",
  "evidence_route": ["ordered source-grounded stages"],
  "required_sources": ["exact source IDs"],
  "pseudocode": ["ordered reusable computation steps"]
}
```

The `strategy_specs` array must appear before `programs`. This is a spec-first
constraint within one LLM call: the model is told to finish all three strategy
designs before writing code and then implement those specifications without
replacing or simplifying them.

The three strategies must differ meaningfully in at least two of:

- reference construction;
- candidate relation;
- evidence route.

Changing only wording, a metric, threshold, embedding modality, or source file
does not count as a distinct strategy.

## Source Diagnostics

Each source component has an `availability` field and a
`partial_coverage` field.

```text
availability = available
  The source exists and can be read by the generated program.

partial_coverage = full
  Every current partial item has at least one direct record or relation.

partial_coverage = partial
  Only some current partial items have a direct record or relation.

partial_coverage = none
  No current partial item has a direct record or relation.
```

Diagnostics describe feasibility, not relevance or correctness. A strategy
should not confuse an available source with a semantically appropriate source.

Typical configured components are:

- `dataset_statistics`;
- `item_metadata`;
- `bundle_item_history`;
- `user_item_history`;
- `item_content_embedding`;
- `item_description_embedding`;
- `user_collaborative_embedding`;
- `bundle_collaborative_embedding`.

The manifest provides the exact parsing and entity-alignment contract for each
component. Generated programs may access only sources listed in their own
`required_sources`.

## Generated Program Contract

Every generated program defines:

```python
def run(
    partial_items,
    candidate_items,
    source_paths,
    max_contexts_per_candidate=5,
):
    ...
```

This is only an external contract. No internal retrieval helper, computation
skeleton, vector helper, or fixed graph traversal is imposed. The program may
choose its own helper functions, indexing, multi-hop computation, aggregation,
and joint or individual candidate comparison.

The result contains exactly one row per candidate in the original order:

```json
[
  {
    "label": "A",
    "item_id": 123,
    "contexts": [
      {
        "text": "related item text or historical bundle item-text composition",
        "sources": ["bundle_item_history", "item_metadata"],
        "supporting_item_ids": [10, 20],
        "supporting_bundle_ids": [30]
      }
    ]
  }
]
```

Final context values must be:

- related item text; or
- a historical bundle's item-text composition.

Programs may use scores, similarities, counts, and embeddings internally to
retrieve and select contexts. These numeric values are not the final evidence
shown to LLM2.

Every returned context must result from the declared candidate relation and
must concretely connect that candidate to the shared reference. A shared
reference context selected independently of the candidate is invalid. When no
such context exists, the candidate receives `contexts: []`.

## Runtime

Programs are executed in guarded, timeout-bounded subprocesses by:

- `src/operator_learning/spec_first_runtime.py`;
- `src/operator_learning/spec_first_worker.py`.

The runtime:

- checks generated code with the existing code guard;
- scopes `source_paths` to the strategy's declared sources;
- executes each strategy separately;
- validates one result row per candidate;
- validates candidate order and identity;
- requires non-empty textual contexts and declared source IDs;
- records failures without a repair LLM call.

A sample can proceed when at least one of the three generated programs executes
successfully. Failed programs are omitted from the evidence passed to LLM2.

## LLM2: Baseline-Shaped Evidence-Grounded Ranking

The current prediction prompt deliberately follows the text-only baseline
shape. This makes the experimental difference easier to interpret:

```text
Text-only baseline:
  partial-item text + candidate-option text
  -> prediction

Current method:
  the same partial-item text + candidate-option text
  + generated strategy evidence
  -> full ranking
```

LLM2 receives:

- partial item text;
- candidate labels and text;
- each successful strategy's `intent`;
- `reference_construction`;
- `candidate_relation`;
- candidate-specific contexts containing only `sources` and `text`.

LLM2 does not receive:

- generated Python code;
- pseudocode;
- strategy name or general description;
- required-source declarations;
- raw partial/candidate item IDs;
- raw supporting item, bundle, or user IDs.

The prompt states that:

- a context is evidence, not a vote;
- more contexts do not automatically make a candidate better;
- context repeated unchanged for all candidates is non-discriminative;
- missing context is not automatic contradiction.

The required output is:

```json
{
  "prediction": "top-ranked label",
  "ranking": ["every candidate label exactly once"],
  "rationale": "at most two evidence-grounded sentences"
}
```

`prediction` must equal `ranking[0]`.

## Current Code Map

```text
src/
  main_baseline.py
    Text-only, one-call, top-1 baseline.

  main.py
    Older src/code code-generation/evidence baseline.
    It is not the current spec-first method entry point.

  operator_learning/
    prompts.py
      LLM1 induction and LLM2 prediction prompts.
    pipeline.py
      Case construction, source manifest, and source diagnostics.
    schemas.py
      Strategy/program schemas and validation.
    spec_first_runtime.py
      Guarded program execution and context validation.
    spec_first_worker.py
      Subprocess worker.
    spec_first_prediction.py
      Joins strategy specs with runtime contexts and computes ranking metrics.

tests/
  test_operator_induction.py
    LLM1-only generation inspection.

  test_spec_first_operator_prediction.py
    Prediction wiring over previously generated/executed strategies.

  test_spec_first_operator_batch.py
    Current end-to-end online MVP and primary evaluation entry point.
```

`src/online_hypothesis_program/` and
`src/counterfactual_reinterpretation/` contain separate earlier experiments.
They are not the active spec-first batch path.

## Running the Current MVP

Inspect five LLM1 generations without batch prediction:

```powershell
python tests/test_operator_induction.py `
  --config config_operator.yaml `
  --sample_count 5 `
  --output_dir tests/outputs/operators/strategy_inspection
```

Run one end-to-end sample:

```powershell
python tests/test_spec_first_operator_batch.py `
  --config config_operator.yaml `
  --split test `
  --sample_idx 0
```

Run the comparable 250-sample evaluation:

```powershell
python tests/test_spec_first_operator_batch.py `
  --config config_operator.yaml `
  --split test `
  --start_idx 0 `
  --sample_count 250
```

Each valid sample normally uses exactly two LLM calls:

1. intent/spec/code generation;
2. evidence-grounded ranking.

With 250 samples, the normal total is approximately 500 LLM calls, excluding
provider-level retries.

Resume an interrupted batch:

```powershell
python tests/test_spec_first_operator_batch.py `
  --config config_operator.yaml `
  --resume tests/outputs/spec_first_operator_batch/<run-directory> `
  --split test `
  --start_idx 0 `
  --sample_count 250
```

Default outputs are written to:

```text
tests/outputs/spec_first_operator_batch/{dataset}_{timestamp}/
```

The batch reports:

- valid sample count;
- Hit@1, Hit@3, and Hit@5;
- mean reciprocal rank;
- mean ground-truth rank;
- mean successful program count;
- total LLM calls.

## Deliberately Removed or Deferred Designs

The current MVP does not use:

- operator compatibility graphs;
- deterministic graph paths;
- graph-query tools;
- manually defined atomic/base operators;
- LLM clustering or deduplication in the online path;
- candidate-blind exemplar retrieval as the final method;
- a forced `prepare()`/`evaluate()` skeleton;
- a fixed `SourceAPI` helper surface;
- generated-code repair calls;
- automatic fallback strategies;
- numeric evidence as the final LLM2 context;
- a planner module;
- an evidence-summary LLM.

These choices are not permanent claims. They reflect the present attempt to
isolate whether an LLM can design diverse, executable, candidate-discriminative
strategies without overly constraining the implementation.

## Open Questions and Planned Ablations

The next evaluation should separate four issues:

1. **Strategy diversity**
   - Do the three strategies actually differ in reference, candidate relation,
     and evidence route?

2. **Specification-to-code fidelity**
   - Does each program implement the declared process, or does it collapse into
     a simpler partial-only retrieval?

3. **Candidate specificity**
   - How often do all candidates receive identical contexts?
   - How often does a strategy return no context for every candidate?

4. **Code stability**
   - What fraction of programs execute and satisfy the result contract?

Useful ablations include:

- text-only baseline versus strategy evidence;
- one strategy versus three strategies;
- raw contexts versus a lightweight evidence consolidator;
- numeric-valued versus context-valued evidence;
- source diagnostics present versus absent;
- online strategy generation versus an offline validated pool followed by
  online selection or composition.

If an evidence consolidator is added, it should run once per sample, perform
extractive deduplication and organization, preserve source text and identifiers,
and never rank candidates or introduce new evidence.

## Current Research Claim Boundary

At this stage, the strongest defensible claim is not that generated programs
already outperform trained bundle recommenders. The current contribution being
tested is:

> An LLM can design multiple completion-intent-conditioned, source-grounded
> relational strategies, compile them into executable programs, and use their
> candidate-specific contexts to reinterpret ambiguous bundle-completion
> options without task-specific training.

Whether this becomes a publishable method depends on demonstrating:

- measurable gains over the matched text-only baseline;
- real strategy diversity beyond prompt-level paraphrases;
- high spec-to-code fidelity and execution success;
- robustness on sparse as well as dense bundle data;
- ablations showing which parts of the design produce the gains.
