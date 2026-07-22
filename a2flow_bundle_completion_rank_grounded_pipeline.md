# A²Flow 파이프라인 요약 및 Rank-Grounded Bundle Completion 적용안

## 0. 문서 목적

이 문서는 다음 두 내용을 정리한다.

1. **A²Flow의 원래 파이프라인**
2. A²Flow의 아이디어를 단순화하여, 현재의 **Zero-shot LLM Bundle Completion / Bundle Agentic RAG**에 적용하는 초기 파이프라인

현재 연구의 핵심 문제는 LLM이 자유롭게 retrieval strategy를 생성하도록 했을 때, 대부분의 전략이 **bundle-item affiliation(BI) 기반 co-occurrence retrieval**로 수렴한다는 것이다.  
따라서 단일 free-form strategy generation 대신, validation 사례들로부터 다양한 retrieval operator를 유도하고 정제한 뒤, 새로운 bundle sample에서 이 operator들을 선택·조합하도록 한다.

---

# Part I. A²Flow 원래 파이프라인

## 1. A²Flow의 문제의식

기존 agentic workflow optimization 방법은 보통 다음과 같은 operator를 사람이 미리 정의한다.

- Planner
- Executor
- Validator
- Review
- Revise
- Ensemble

이후 search algorithm이 operator의 순서, 반복, 분기 등을 탐색한다.

문제는 operator 자체가 사람이 정의한 search space이므로, 새로운 task에서 필요한 operation이 operator set에 없으면 탐색할 수 없다는 것이다.

A²Flow의 핵심 아이디어는 다음과 같다.

> Expert demonstrations에서 task-specific operation을 먼저 추출하고,  
> 이를 clustering과 abstraction을 통해 재사용 가능한 operator로 정제한 뒤,  
> 정제된 operator들을 이용해 workflow를 자동 탐색한다.

---

## 2. 전체 파이프라인

```text
Expert demonstrations
        ↓
Case-based initial operator generation
        ↓
Operator clustering and preliminary abstraction
        ↓
Deep extraction for abstract execution operators
        ↓
Final operator library
        ↓
MCTS-based workflow generation
        ↓
Optimized workflow
```

A²Flow는 크게 두 부분으로 나뉜다.

### Part A. Operator discovery

```text
Expert cases
→ Case-specific operators
→ Functional clustering
→ Abstract reusable operators
```

### Part B. Workflow search

```text
Operator library
+ Nodes
+ Edges
+ Prompts
+ Control flow
→ MCTS
→ Best workflow
```

---

## 3. Case-based Initial Operator Generation

A²Flow는 validation set의 problem과 resolution을 expert demonstration으로 사용한다.

각 사례를 LLM에게 제공하고, 해당 사례의 해결 과정에 필요한 구체적인 operation을 Python operator 형태로 추출한다.

예를 들어 embodied task에서는 초기에 다음과 같이 매우 구체적인 operator가 나올 수 있다.

```text
ObserveEnvironment
CreatePlan
ExecuteAction
ValidateAction

PotatoLocationFinder
PotatoRetriever
PotCleaner
MicrowavePreparer
MicrowavePlacer
```

이 단계의 목적은 처음부터 완벽하게 일반화된 operator를 만드는 것이 아니다.

> 각 사례를 해결하는 데 필요했던 세부적인 실행 기능을 최대한 풍부하게 수집하는 단계이다.

---

## 4. Operator Clustering and Preliminary Abstraction

초기 operator는 사례별로 만들어졌기 때문에 중복되거나 지나치게 구체적이다.

예:

```text
PotatoLocationFinder
TomatoLocationFinder
AlarmClockFinder
VaseFinder
```

이들은 모두 목표 물체의 위치를 찾는 동일한 기능을 수행하므로 다음과 같이 통합할 수 있다.

```text
Locator
```

A²Flow는 일반적인 k-means나 embedding clustering을 사용하는 것이 아니라, LLM에게 다음을 수행하도록 한다.

- 기능적으로 유사한 operator grouping
- 중복 operator 제거
- 지나치게 세부적인 operator abstraction
- 간결한 operator name 생성
- Python code 형식 유지

예시 결과:

```text
TaskPlanner
Locator
Navigator
ObjectInteractor
Validator
```

중요한 점은 **같은 source를 사용한다는 이유로 묶는 것이 아니라, 실제 functional role이 같은지를 기준으로 묶는 것**이다.

---

## 5. Deep Extraction for Abstract Execution Operators

Preliminary operator도 여전히 task-specific하거나 중복될 수 있다.

A²Flow는 동일한 preliminary operator set에 대해 여러 개의 독립적인 reasoning path를 생성하고, 각 path 안에서 operator를 반복적으로 추상화한다.

```text
Preliminary operators
        ↓
Reasoning path 1: abstraction → refinement → final abstraction
Reasoning path 2: abstraction → refinement → final abstraction
...
Reasoning path 6: abstraction → refinement → final abstraction
        ↓
Aggregation
        ↓
Final abstract operator library
```

여기서 병렬 reasoning path는 **서로 다른 workflow를 생성하는 단계가 아니다.**

> 동일한 preliminary operator 집합을 여러 방식으로 추상화하는 독립적인 operator-abstraction 시도이다.

그 후 여러 path의 결과를 다시 aggregate하여 최종 operator set을 만든다.

ALFWorld 예시에서는 최종적으로 다음과 같이 압축된다.

```text
Planner
Executor
Validator
```

---

## 6. Operator Memory Mechanism

기존 sequential workflow에서는 현재 operator가 직전 operator의 출력만 참고할 수 있다.

A²Flow는 이전 모든 operator output을 memory에 누적한다.

```text
Task
→ Operator 1 output
→ Operator 2 output
→ Operator 3 output
→ ...
```

각 operator는 다음을 입력으로 받는다.

- 원래 task
- 현재 operator의 instruction
- 이전까지 누적된 모든 intermediate output

이를 통해 operator 간 정보 손실을 줄이고, 이후 operator가 전체 reasoning history를 참고하도록 한다.

---

## 7. MCTS-based Workflow Search

최종 operator library가 만들어진 뒤, A²Flow는 MCTS를 이용하여 workflow 구조를 탐색한다.

탐색 대상은 다음과 같다.

- operator 추가 및 삭제
- operator 순서 변경
- 반복 구조 추가
- 조건문 추가
- prompt 수정
- parallel execution
- validation 및 revision loop

예:

```text
Initial:
Executor
```

```text
Candidate 1:
Planner → Executor
```

```text
Candidate 2:
Executor → Validator
```

```text
Candidate 3:
Planner → Executor → Validator
```

```text
Candidate 4:
Planner → Executor → Planner refinement → Executor
```

각 workflow를 validation set에서 실행하고 성능을 평가한 뒤, 점수가 좋은 branch를 더 확장한다.

---

## 8. 용어 구분

### Operator

재사용 가능한 단일 실행 기능이다.

```text
Planner
CandidateContrast
SemanticFilter
```

### Node

특정 workflow 안에서 operator가 실제로 한 번 호출되는 위치이다.

같은 operator가 workflow 안에서 여러 번 사용되면 서로 다른 node가 된다.

### Edge

node 사이의 실행 순서와 정보 흐름이다.

```text
Planner → Executor
```

### Workflow

여러 operator node와 edge를 연결한 전체 실행 구조이다.

```text
Planner
→ Executor
→ Validator
→ if failed: Executor
```

### Parallel reasoning path

A²Flow의 operator abstraction 단계에서 동일한 preliminary operator set을 서로 다르게 추상화하는 독립적인 reasoning trajectory이다.

### MCTS search path

workflow 후보가 iterative modification을 통해 진화하는 탐색 경로이다.

두 path는 서로 다른 단계에 속한다.

---

# Part II. 현재 Bundle Agentic RAG의 문제

## 1. 현재 구조

현재 방법은 다음과 같다.

```text
Partial bundle
→ Bundle intent inference
→ Free-form bundle-specific strategy generation
→ Python code generation
→ Evidence retrieval
→ Evidence-augmented prediction
```

## 2. 관찰된 한계

LLM에게 자유롭게 strategy를 생성하도록 했지만, 실제 생성 결과는 대부분 다음으로 수렴한다.

```text
Partial item을 포함하는 historical bundle 검색
→ Co-occurring item 집계
→ Candidate와 연결
```

즉, 표현은 달라도 실제로는 BI 기반 co-occurrence retrieval인 경우가 많다.

주요 원인은 다음과 같다.

- BI는 bundle completion task와 직접 연결되어 있다.
- 코드로 구현하기 쉽다.
- sparse하지 않은 경우 안정적으로 evidence를 생성한다.
- LLM 입장에서 가장 안전한 retrieval choice이다.
- “창의적인 전략을 생성하라”는 자연어 instruction만으로는 search space가 실제로 넓어지지 않는다.

---

# Part III. 우리 Task에 적용하는 A²Flow-lite

## 1. 적용 원칙

초기 버전에서는 A²Flow 전체를 구현하지 않는다.

제외하는 요소:

- 병렬 reasoning path
- MCTS
- operator memory
- iterative workflow evolution
- multiple-strategy fusion

우선 다음 핵심만 적용한다.

> Validation 사례에서 retrieval operator를 유도하고,  
> 비슷한 operator를 한 번 clustering하여 operator library를 만든 뒤,  
> 새로운 bundle sample에서 해당 operator를 선택·조합하도록 한다.

---

## 2. 전체 파이프라인

### Offline: Retrieval Operator Discovery

```text
Validation set
→ Random n samples
→ Label-guided operator induction
→ Raw operator pool
→ Functional clustering and refinement
→ Retrieval operator environment
```

### Online: Bundle-specific Strategy Composition

```text
Partial bundle
→ Bundle intent inference
→ Operator environment 제공
→ Operator selection and composition
→ Retrieval strategy
→ Code generation and execution
→ Evidence-augmented prediction
```

---

# Part IV. Offline Operator Discovery

## Step 1. Validation set에서 random n개 추출

Validation set에서 seed를 고정하고 random sample을 뽑는다.

초기 권장값:

```text
n = 20~30 per dataset
seed = 42
```

이 단계에서는 기존 Agentic RAG의 성공 여부나 rank를 기준으로 sample을 고르지 않는다.

목적은 사람이 사례를 하나씩 검토하지 않고도 다양한 partial bundle과 candidate 구성을 확보하는 것이다.

---

## Step 2. Label-Guided Operator Induction

각 validation sample에 다음 정보를 제공한다.

- Partial bundle
- Candidate items
- Ground-truth candidate
- 사용 가능한 source 목록
- 각 source의 schema와 의미

중요한 점은 이 단계에서 **현재 Agentic RAG 전체를 다시 실행하지 않는다는 것**이다.

기존 Agentic RAG를 실행하면 이미 확인된 BI 중심 전략이 다시 수집되고, 그 편향이 operator library에 그대로 고정될 수 있다.

대신 LLM에게 다음과 같이 요청한다.

> Ground-truth candidate가 다른 candidate보다 적합하다는 것을 source-grounded evidence로 드러내기 위해 어떤 retrieval operation이 필요할 수 있는지 분석하라.  
> 하나의 end-to-end strategy가 아니라 재사용 가능한 atomic operator들로 분해하라.

이 방식은 다음과 같이 부를 수 있다.

- Hindsight Operator Induction
- Label-Guided Retrieval Operator Discovery
- Case-Based Retrieval Operator Elicitation

---

## Step 3. 자연어 또는 Structured Specification 수준에서 operator 생성

초기 버전에서는 executable code를 생성하지 않는다.

이유:

- 코드 생성부터 요구하면 구현하기 쉬운 BI lookup, count, top-k로 다시 수렴할 수 있다.
- 현재 목표는 실행 프로그램이 아니라 retrieval strategy의 기본 단위를 발견하는 것이다.
- 실행 가능성보다 먼저 strategy space를 넓히는 것이 목적이다.

Operator는 structured JSON으로 생성한다.

### 권장 schema

```json
{
  "name": "CandidateExclusiveSupport",
  "purpose": "Find evidence that supports one candidate but not its competitors.",
  "anchor": "candidate differences",
  "inputs": [
    "partial bundle",
    "candidate items"
  ],
  "sources": [
    "bundle-item affiliation",
    "item metadata"
  ],
  "relation_path": "candidate → historical bundles → co-items",
  "operation": "Remove evidence shared by many candidates and retain candidate-exclusive supporting patterns.",
  "output": "candidate-specific discriminative evidence",
  "when_useful": "When generic evidence is shared across many candidates."
}
```

각 sample에서 3~5개의 operator를 생성한다.

---

## Step 4. Operator Diversity Constraint

각 sample에서 BI 변형만 여러 개 나오지 않도록 최소한의 다양성 조건을 준다.

예시 instruction:

```text
Generate four atomic retrieval operators.

Constraints:
1. No two operators may have the same combination of anchor, source, and retrieval principle.
2. At most one operator may rely only on bundle-item affiliation.
3. At least one operator must use one of:
   - candidate comparison,
   - negative evidence,
   - missing-role inference,
   - cross-source reasoning,
   - popularity debiasing,
   - multimodal evidence.
4. Do not generate a complete end-to-end strategy.
5. Each operator must be reusable across multiple bundle samples.
```

---

## Step 5. Raw Operator Pool 구성

예를 들어 30개 sample에서 sample당 4개 operator를 생성하면 약 120개의 raw operator가 생성된다.

```text
Sample 1:
- InferAccessorySlot
- CandidateExclusiveSupport
- SemanticConflictCheck
- CrossSourceAgreement

Sample 2:
- DetectMoodTransition
- AudioPrototypeMatch
- UserGroupConsensus
- PopularityDebias

Sample 3:
- InferMissingShoeRole
- BridgeItemDiscovery
- NegativeCompatibility
- SourceFallback
```

이 결과들을 하나의 pool로 합친다.

---

## Step 6. Functional Clustering and Refinement

전체 raw operator를 LLM에게 제공하고 기능적으로 비슷한 operator를 묶는다.

### Clustering 기준

다음이 유사한 경우에만 merge한다.

1. Retrieval objective
2. Anchor
3. Relation path
4. Computation or filtering principle
5. Output evidence type

같은 source를 사용한다는 이유만으로 merge하지 않는다.

### 예시 1

```text
InferAccessorySlot
InferMissingShoeRole
InferMissingGenreRole
        ↓
InferMissingRole
```

### 예시 2

```text
NormalizeByPopularity
ComputeConditionalLift
DownweightFrequentItems
        ↓
DebiasAssociation
```

### Merge하면 안 되는 예시

다음 operator들은 모두 BI를 사용하더라도 목적이 다르므로 별개로 유지한다.

```text
RetrieveRecurringPattern
CandidateExclusiveSupport
BridgeItemDiscovery
DebiasAssociation
```

---

## Step 7. Middle-level Operator Library 생성

최종 operator는 지나치게 구체적이지도, 지나치게 일반적이지도 않아야 한다.

### 너무 추상적인 결과

```text
Analyze
Retrieve
Process
Validate
```

이 수준은 내부 동작이 다시 free-form이 되기 때문에 BI collapse를 막지 못한다.

### 권장 abstraction level

```text
IntentDecompose
InferMissingRole
SemanticFilter
RetrieveRecurringPattern
BridgeDiscovery
CandidateContrast
NegativeEvidence
DebiasAssociation
CrossSourceAgreement
SourceFallback
MultimodalMatch
```

초기 operator library 크기는 약 8~12개가 적절하다.

---

# Part V. Online Bundle-specific Strategy Composition

## Step 1. Bundle Intent Inference

새로운 sample의 partial bundle을 보고 semantic intent를 추론한다.

예:

```json
{
  "style": "romantic vintage",
  "missing_role": "simple complementary accessory",
  "coherence": "feminine and classic"
}
```

---

## Step 2. Operator Environment 제공

LLM에게 validation 사례에서 정제한 operator library를 제공한다.

예:

```text
Available retrieval operators:

1. InferMissingRole
2. SemanticFilter
3. RetrieveRecurringPattern
4. BridgeDiscovery
5. CandidateContrast
6. NegativeEvidence
7. DebiasAssociation
8. CrossSourceAgreement
9. SourceFallback
10. MultimodalMatch
```

각 operator는 이름뿐 아니라 다음 정보를 포함한다.

- 목적
- 입력
- source
- relation path
- operation
- output
- when useful

---

## Step 3. Operator Selection and Composition

LLM은 현재 partial bundle에 적합한 operator 2~4개를 선택하고, 실행 순서와 정보 흐름을 구성한다.

### 예시 A: Missing-role 기반 전략

```text
InferMissingRole
→ SemanticFilter
→ CandidateContrast
```

### 예시 B: Historical pattern 기반 전략

```text
RetrieveRecurringPattern
→ BridgeDiscovery
→ DebiasAssociation
```

### 예시 C: Cross-source 전략

```text
SemanticFilter
→ CrossSourceAgreement
→ NegativeEvidence
```

이때 LLM은 operator library 밖의 완전히 자유로운 전략을 만드는 것이 아니라, 정제된 retrieval primitives를 조합한다.

---

## Step 4. Strategy를 Executable Code로 변환

선택된 operator sequence와 operator specification을 기존 code generator에 전달한다.

Code generator는 다음을 구현한다.

- 필요한 source loading
- relation traversal
- filtering
- aggregation
- candidate-specific evidence generation
- JSON output

예:

```text
Selected workflow:
InferMissingRole
→ SemanticFilter
→ CandidateContrast
```

```text
Code implementation:
1. Infer missing semantic/category role from partial metadata.
2. Filter candidate metadata using the inferred role.
3. Compare candidate-specific evidence and retain exclusive support.
```

---

## Step 5. Evidence-Augmented Prediction

생성된 코드를 실행하여 다음을 얻는다.

- Partial-bundle evidence
- Candidate-specific evidence

최종 prediction LLM은 다음을 입력으로 받는다.

```text
Partial bundle
+ Candidate items
+ Bundle intent
+ Selected operator workflow
+ Retrieved evidence
```

그리고 ground-truth candidate의 rank를 포함한 full ranking 또는 최종 선택을 출력한다.

---

# Part VI. Rank의 역할

## 1. Operator discovery 단계

초기 operator discovery는 label-guided hindsight 방식이므로, 기존 Agentic RAG 실행 rank를 사용하지 않는다.

즉:

```text
Random validation sample
+ Ground-truth label
+ Source schema
→ Atomic operator induction
```

이다.

---

## 2. Operator library 평가 단계

Operator library가 완성된 후, 별도의 validation-evaluation subset에서 실제 code를 실행하고 rank를 측정한다.

Validation set을 다음처럼 나눌 수 있다.

```text
Validation-Discovery
- Random sample에서 operator 유도
- Clustering
- Operator library 생성

Validation-Evaluation
- Operator library를 이용해 strategy 생성
- Code 실행
- Rank 측정
```

초기 권장 예시:

```text
Discovery: 20~30 samples
Evaluation: 30~50 samples
```

---

## 3. 평가 지표

기존 free-form Agentic RAG와 비교한다.

### HitRate@1

```text
Ground-truth rank = 1인 비율
```

### Average Ground-truth Rank

```text
정답 candidate의 평균 rank
```

### Rank Improvement

```text
Δrank = free-form rank - operator-based rank
```

- Δrank > 0: operator 기반 전략이 개선
- Δrank = 0: 동일
- Δrank < 0: 악화

### BI-only Strategy Ratio

생성된 전략 중 BI만 사용하는 전략의 비율

### Operator Usage Distribution

어떤 operator에 지나치게 쏠리는지 분석

### Unique Workflow Ratio

서로 다른 operator sequence가 얼마나 생성되는지 분석

---

# Part VII. Reflection의 후속 적용

초기 MVP에서는 reflection을 사용하지 않는다.

Operator library의 기본 효과를 확인한 후 다음 단계로 추가할 수 있다.

```text
Initial operator workflow
→ Code execution
→ Prediction rank
→ Rank-based feedback
→ Operator replacement or reordering
→ Re-execution
```

Reflection trigger 예시:

```text
operator-based rank ≥ free-form rank
```

또는:

```text
operator-based rank > 3
```

Reflection instruction 예시:

```text
The current operator workflow failed to improve the ground-truth rank.
Replace at least one operator with another operator based on a structurally different evidence principle.
Do not repeat the same BI-centered retrieval path.
```

수정 전략은 다음 조건에서만 채택한다.

```text
revised rank < original rank
```

---

# Part VIII. 초기 MVP 실험

## Baseline

```text
Partial bundle
→ Free-form strategy generation
→ Code execution
→ Evidence
→ Prediction
```

## A²Flow-lite Variant

```text
Offline:
Random validation samples
→ Label-guided operator induction
→ One-time clustering
→ Operator library

Online:
Partial bundle
→ Operator selection and composition
→ Code execution
→ Evidence
→ Prediction
```

## 핵심 연구 질문

> Can a retrieval operator environment induced from labeled validation cases reduce the BI-centered strategy collapse of free-form LLM generation?

## 예상 분석

1. Free-form 대비 HitRate@1 변화
2. 평균 rank 변화
3. BI-only 전략 비율 변화
4. 생성된 operator sequence 다양성
5. source 조합 다양성
6. evidence empty 비율
7. candidate-specific evidence 구분력

---

# Part IX. 구현 순서

## Phase 1. Operator Discovery

1. Validation-discovery sample random 추출
2. Ground truth와 source schema를 포함한 operator induction prompt 작성
3. Sample당 3~5개 structured operator 생성
4. Raw operator pool 저장
5. Functional clustering prompt 작성
6. 8~12개의 refined operator 생성
7. Operator library JSON 저장

## Phase 2. Operator-based Strategy Generation

1. 기존 Stage 1 prompt에 operator library 추가
2. 2~4개 operator 선택
3. Operator 순서와 data flow 출력
4. 기존 code generator에 전달
5. Code 실행
6. Evidence 저장
7. 기존 Stage 2 prediction 수행

## Phase 3. Evaluation

1. Free-form Agentic RAG 실행 결과와 비교
2. HitRate@1 및 average rank 계산
3. Operator usage 분석
4. BI-only 비율 계산
5. Unique workflow와 source combination 분석

## Phase 4. 후속 확장

초기 결과가 유의미할 때만 다음을 추가한다.

- Rank-based reflection
- Multiple strategy generation
- Best-strategy selection
- Operator-level mutation
- Strategy fusion
- Parallel abstraction paths
- MCTS workflow search
- Operator memory

---

# Part X. 핵심 차이 요약

## A²Flow 원형

```text
Expert problem-resolution cases
→ Case-specific executable operators
→ Clustering
→ Multi-path deep abstraction
→ Operator library
→ MCTS workflow search
→ Global optimized workflow
```

## 우리 Task의 초기 적용

```text
Random labeled validation cases
→ Hindsight natural-language retrieval operators
→ One-time functional clustering
→ Retrieval operator environment
→ Sample-conditioned operator composition
→ Code generation and execution
→ Evidence-augmented prediction
```

## 주요 단순화

| 구성 요소 | A²Flow | 우리 초기 적용 |
|---|---|---|
| Expert data | Problem + resolution | Partial bundle + candidates + GT |
| Initial operator | Executable Python operator | Structured natural-language operator |
| Clustering | LLM functional clustering | LLM functional clustering |
| Parallel reasoning | 사용 | 미사용 |
| Operator memory | 사용 | 미사용 |
| Workflow search | MCTS | LLM selection/composition 1회 |
| Workflow scope | Task-level global workflow | Sample-specific retrieval strategy |
| Evaluation | Validation performance | Ground-truth rank, HitRate@1 |

---

# 최종 요약

초기 적용의 핵심은 기존 Agentic RAG의 BI 중심 trajectory를 다시 수집하는 것이 아니다.

대신:

> Labeled validation 사례에서 다양한 atomic retrieval operator를 hindsight 방식으로 유도하고,  
> 이를 기능적으로 정제한 operator environment를 만든 뒤,  
> 새로운 bundle sample에서 적합한 operator를 선택·조합해 실제 retrieval code를 생성한다.

이를 통해 기존의 완전한 free-form strategy generation을 다음과 같이 바꾼다.

```text
Free-form strategy generation
        ↓
Validation-induced operator composition
```

첫 단계에서는 operator environment 자체가 전략 다양성과 성능을 개선하는지 확인한다.  
Reflection, multiple strategies, fusion, MCTS는 그 효과를 확인한 이후에 추가한다.

---

# Part XI. Dataset-specific Operator Discovery

본 연구에서는 operator extraction부터 evaluation까지 **데이터셋별로 독립적으로 수행**한다.

```text
POG validation
→ POG raw operator induction
→ POG clustering
→ POG rank grounding
→ POG operator environment
```

```text
POG-Dense validation
→ POG-Dense raw operator induction
→ POG-Dense clustering
→ POG-Dense rank grounding
→ POG-Dense operator environment
```

```text
Spotify validation
→ Spotify raw operator induction
→ Spotify clustering
→ Spotify rank grounding
→ Spotify operator environment
```

```text
Spotify-Sparse validation
→ Spotify-Sparse raw operator induction
→ Spotify-Sparse clustering
→ Spotify-Sparse rank grounding
→ Spotify-Sparse operator environment
```

데이터셋별로 분리하는 이유는 다음과 같다.

- POG와 Spotify는 bundle semantics가 다르다.
- Dense/sparse 환경에서 신뢰할 수 있는 source가 다르다.
- Metadata, interaction, bundle affiliation의 sparsity와 utility가 다르다.
- 하나의 공통 library로 합치면 operator가 지나치게 일반화될 수 있다.

따라서 초기 실험에서는 각 데이터셋에서 별도의 operator environment를 유도한다.

---

# Part XII. Rank-Grounded Operator Refinement

초기 operator induction은 label-guided hindsight 방식으로 수행하지만, 자연어 operator만으로는 실제 utility를 보장할 수 없다.

따라서 clustering 후 만들어진 provisional operator library를 별도의 validation subset에서 실제 실행하고, **ground-truth rank를 이용해 operator를 정제**한다.

## Revised Pipeline

```text
Random labeled validation samples
        ↓
Raw operator induction
        ↓
Functional clustering
        ↓
Provisional operator library
        ↓
Operator-based strategies 생성
        ↓
Code generation and execution
        ↓
Ground-truth rank 측정
        ↓
Rank-based utility analysis
        ↓
Rank-contrastive reflection
        ↓
Final rank-grounded operator environment
```

Rank는 다음 역할에 사용할 수 있다.

1. Operator utility 측정
2. 유사 operator 중 대표 구현 선택
3. 좋은 전략과 나쁜 전략의 pairwise preference 생성
4. Rank-based reflection을 통한 operator 사용 조건 정제
5. 이후 strategy selector의 supervision 생성

---

# Part XIII. Rank Feedback의 기본 표현

각 sample \(i\)와 strategy \(s\)에 대해 ground-truth candidate의 rank를 다음처럼 둔다.

\[
r_i^s \in \{1,\dots,10\}
\]

작을수록 좋은 전략이다.

## Baseline 대비 Rank Gain

\[
\Delta r_i^s
=
r_i^{base}-r_i^s
\]

- \(\Delta r_i^s > 0\): baseline보다 개선
- \(\Delta r_i^s = 0\): 동일
- \(\Delta r_i^s < 0\): baseline보다 악화

예:

```text
Base rank: 8
Operator strategy rank: 3
Rank gain: +5
```

## Reciprocal Rank Gain

Top-rank 개선을 더 크게 반영하기 위해 reciprocal rank를 사용할 수 있다.

\[
RR_i^s=rac{1}{r_i^s}
\]

\[
\Delta RR_i^s
=
rac{1}{r_i^s}
-
rac{1}{r_i^{base}}
\]

초기 실험에서는 `rank_gain`과 `reciprocal_rank_gain`을 함께 기록한다.

---

# Part XIV. Rank를 활용한 Operator Utility

각 operator가 포함된 workflow들의 rank 결과를 모아 operator별 통계를 계산한다.

## Mean Rank Gain

\[
	ext{MeanRankGain}(o)
=
\mathbb{E}
[
r^{base}-r^{workflow}
\mid o \in workflow
]
\]

## Mean Reciprocal Rank Gain

\[
	ext{MeanRRGain}(o)
=
\mathbb{E}
\left[
rac{1}{r^{workflow}}
-
rac{1}{r^{base}}
\mid o \in workflow
ight]
\]

## Win Rate

\[
	ext{WinRate}(o)
=
P(
r^{workflow}<r^{base}
\mid o\in workflow
)
\]

## Top-1 Rate

\[
	ext{Top1Rate}(o)
=
P(
r^{workflow}=1
\mid o\in workflow
)
\]

## Unique Win Rate

\[
	ext{UniqueWin}(o)
=
P(
r^s=1
\land
r^{s'}>1,\ orall s'
eq s
\mid o\in s
)
\]

Unique win은 평균적으로 자주 쓰이는 BI operator뿐 아니라 특정 sample에서 결정적으로 유용한 niche operator를 보존하는 데 도움을 준다.

---

# Part XV. Rank를 활용한 Clustering Representative Selection

Functional clustering 이후 하나의 cluster 안에 여러 구현 후보가 남을 수 있다.

예:

```text
NormalizeByPopularity
ConditionalLift
RarePatternWeighting
GlobalFrequencyPenalty
        ↓
DebiasAssociation
```

이때 단순히 LLM이 대표 연산을 정하도록 하지 않고, validation rank를 이용해 가장 유용한 구현을 선택한다.

| Candidate implementation | Mean rank gain |
|---|---:|
| NormalizeByPopularity | +0.4 |
| ConditionalLift | +1.8 |
| RarePatternWeighting | +1.2 |
| GlobalFrequencyPenalty | +0.2 |

최종 operator 예:

```json
{
  "name": "DebiasAssociation",
  "core_principle": "Discount globally frequent relations.",
  "preferred_implementation": "conditional lift",
  "alternative_implementations": [
    "rarity weighting",
    "global frequency penalty"
  ]
}
```

즉 clustering은 다음처럼 수행한다.

```text
Functional similarity로 cluster 형성
→ Rank utility로 대표 implementation 선택
```

---

# Part XVI. Pairwise Rank Preference

같은 sample에서 여러 operator workflow를 실행하면 strategy 간 pairwise preference를 만들 수 있다.

예:

```text
Workflow A rank: 2
Workflow B rank: 7
Workflow C rank: 4
```

따라서:

```text
A > C > B
```

라는 preference가 생성된다.

저장 형식 예:

```json
{
  "sample_id": "...",
  "preferred_workflow": [
    "InferMissingRole",
    "SemanticFilter",
    "CandidateContrast"
  ],
  "rejected_workflow": [
    "RetrieveRecurringPattern",
    "BridgeDiscovery"
  ],
  "preferred_rank": 2,
  "rejected_rank": 7
}
```

Pairwise preference는 이후 다음에 활용할 수 있다.

- LLM strategy selector의 few-shot example
- 별도 selector model 학습
- Preference optimization
- Sample condition과 operator 효용의 연결 분석

초기 MVP에서는 preference 사례를 operator environment의 `when_useful` 정보로 변환하는 정도로 사용한다.

---

# Part XVII. Rank-Based Reflection

Rank-based reflection은 같은 sample에서 rank가 좋았던 workflow와 나빴던 workflow를 함께 제공하고, **두 전략의 구조적 차이로부터 재사용 가능한 operator-level lesson을 추출**한다.

## Reflection Input

```text
Same validation sample

Workflow A:
RetrieveRecurringPattern
→ BridgeDiscovery

Ground-truth rank: 7

Workflow B:
InferMissingRole
→ SemanticFilter
→ CandidateContrast

Ground-truth rank: 2
```

## Reflection Prompt Concept

```text
Compare the two workflows for the same sample.

Identify which structural difference is most likely responsible for the
ground-truth rank improvement.

Do not merely state that Workflow B is better.

Extract a reusable lesson about:
- when an operator should be used,
- which failure mode it addresses,
- which operator should be replaced or added,
- and which evidence principle should be avoided.
```

## Reflection Output

```json
{
  "failure_pattern": "Shared co-occurrence evidence supported multiple candidates.",
  "successful_change": "The better workflow inferred a missing semantic role and compared candidates contrastively.",
  "operator_lesson": "When relational support is non-discriminative, apply CandidateContrast after semantic-role filtering.",
  "operator_to_add": "CandidateContrast",
  "operator_to_replace": "BridgeDiscovery",
  "when_useful": [
    "multiple candidates receive similar relational evidence"
  ]
}
```

Reflection 결과는 operator specification을 보강하는 데 사용한다.

```json
{
  "name": "CandidateContrast",
  "when_useful": [
    "multiple candidates receive similar evidence",
    "relational retrieval is dense but non-discriminative"
  ],
  "failure_modes_addressed": [
    "shared BI co-occurrence",
    "candidate-insensitive evidence"
  ],
  "avoid_when": [
    "candidate-specific evidence is already sparse and unique"
  ]
}
```

즉 rank-based reflection은 다음을 학습하는 단계이다.

> 어떤 상황에서 어떤 operator가 유용한가.

---

# Part XVIII. Rank-Based Reflection Trigger

초기 MVP에서는 모든 sample에 reflection을 수행하지 않는다.

다음과 같은 경우에만 reflection을 수행한다.

## Trigger A. Baseline보다 악화

\[
r^{operator} \geq r^{base}
\]

## Trigger B. Strategy 간 큰 Rank Gap

\[
r^{worst}-r^{best}\geq 3
\]

## Trigger C. Oracle과 Selected Strategy의 Gap

\[
r^{selected}-r^{oracle}\geq 2
\]

여기서:

\[
r^{oracle}=\min_s r^s
\]

이다.

Trigger B는 같은 sample에서 구조적으로 다른 workflow들이 큰 rank 차이를 보였기 때문에 operator-level lesson을 얻는 데 특히 유용하다.

---

# Part XIX. Rank-Based Reflection의 두 종류

## Library Refinement Reflection

목적:

- operator description 보강
- `when_useful` 추가
- failure mode 추가
- 유사 operator merge/split
- preferred implementation 선택

```text
Validation workflows
→ Rank contrast
→ Operator-level lesson
→ Operator environment update
```

## Strategy Revision Reflection

목적:

- 현재 workflow에서 operator 교체
- operator 순서 변경
- source 변경
- candidate comparison 추가

```text
Initial workflow
→ Code execution
→ Poor rank
→ Rank-based reflection
→ Revised workflow
→ Re-execution
```

초기 MVP에서는 **Library Refinement Reflection**을 먼저 사용한다.

Test-time에는 ground-truth rank를 알 수 없기 때문에 Strategy Revision Reflection을 직접 사용할 수 없다. 따라서 test에서는 validation에서 학습된 `when_useful`, `failure_mode`, rank prior를 활용한다.

---

# Part XX. Final Rank-Grounded Operator Environment

최종 operator environment는 단순한 operator name과 description만 포함하지 않는다.

권장 schema:

```json
{
  "name": "CandidateContrast",
  "purpose": "Retain evidence that distinguishes one candidate from competing candidates.",
  "required_inputs": [
    "candidate-specific evidence for all candidates"
  ],
  "allowed_sources": [
    "bundle-item affiliation",
    "user-item interaction",
    "metadata"
  ],
  "required_computation": [
    "cross-candidate comparison",
    "shared-evidence removal"
  ],
  "forbidden_shortcuts": [
    "independent candidate co-occurrence count only"
  ],
  "when_useful": [
    "multiple candidates share similar relational evidence"
  ],
  "failure_modes_addressed": [
    "non-discriminative BI retrieval"
  ],
  "rank_statistics": {
    "mean_rank_gain": 2.1,
    "mean_reciprocal_rank_gain": 0.18,
    "win_rate": 0.64,
    "top1_rate": 0.31,
    "unique_win_rate": 0.12
  }
}
```

새로운 sample에서 LLM은 다음을 함께 참고한다.

```text
Current bundle characteristics
+ Available source conditions
+ Operator functional specification
+ Historical rank statistics
+ Rank-reflection-based when-useful rules
```

---

# Part XXI. Revised Initial Experiment

## Stage 1. Dataset-specific Discovery

```text
Dataset-specific validation-discovery samples
→ GT-guided raw operator induction
→ One-time functional clustering
→ Provisional operator library
```

## Stage 2. Rank Grounding

```text
Validation-grounding samples
→ Strategy 3개 생성
→ 각 strategy code 실행
→ Ground-truth rank 측정
→ Operator utility 계산
→ Pairwise preference 생성
→ Rank-based reflection
→ Final operator environment
```

## Stage 3. Held-out Evaluation

```text
Held-out validation/test sample
→ Final operator environment 제공
→ Operator selection and composition
→ Code execution
→ Evidence-augmented prediction
```

---

# Part XXII. 초기 MVP에서 사용할 Rank 요소

처음부터 모든 rank 기반 분석을 구현할 필요는 없다.

초기에는 다음 다섯 가지만 사용한다.

1. **Rank Gain**
   ```text
   base rank - operator workflow rank
   ```

2. **Reciprocal Rank Gain**
   ```text
   1/operator rank - 1/base rank
   ```

3. **Win Rate**
   ```text
   baseline보다 rank를 개선한 비율
   ```

4. **Same-sample Pairwise Preference**
   ```text
   workflow A rank < workflow B rank
   ```

5. **Rank-Based Reflection**
   ```text
   best workflow와 poor workflow의 차이에서
   reusable operator-level lesson 추출
   ```

Top-1 rate와 unique-win rate는 함께 기록하되, sample 수가 충분히 확보된 이후 해석한다.

---

# Part XXIII. Updated Core Research Question

기존 질문:

> Can a retrieval operator environment induced from labeled validation cases reduce the BI-centered strategy collapse of free-form LLM generation?

Rank grounding을 추가한 질문:

> Can a dataset-specific, rank-grounded retrieval operator environment discover not only diverse but empirically useful bundle-specific retrieval strategies?

Rank-based reflection을 강조하면 다음처럼 표현할 수 있다.

> Can pairwise rank feedback reveal when specific retrieval operators are useful and refine the strategy environment beyond free-form generation?

---

# Part XXIV. 핵심 요약

최종 파이프라인은 다음과 같다.

```text
Dataset-specific labeled validation cases
        ↓
Hindsight raw operator induction
        ↓
Functional clustering
        ↓
Provisional operator environment
        ↓
Multiple operator workflows
        ↓
Code execution
        ↓
Ground-truth ranks
        ↓
Operator utility + Pairwise preference
        ↓
Rank-based reflection
        ↓
Final rank-grounded operator environment
        ↓
Sample-conditioned operator composition
        ↓
Bundle completion prediction
```

Rank의 가장 중요한 역할은 단순히 좋은 operator를 점수화하는 것이 아니다.

> 같은 sample에서 잘한 전략과 못한 전략의 차이를 통해, 특정 operator가 언제 필요하고 어떤 실패를 해결하는지를 학습하는 것.

따라서 rank-based reflection은 operator environment를 단순한 전략 목록이 아니라, **사용 조건과 실패 모드를 포함한 경험 기반 retrieval policy space**로 정제하는 역할을 한다.
