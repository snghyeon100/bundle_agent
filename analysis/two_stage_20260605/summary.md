# 2-stage 결과 분석 (2026-06-05)

> 이 리포트는 후보군 수정 전 결과를 분석한 문서다. 후보군을 baseline과 250/250 동일하게 맞춘 최신 분석은 `analysis/two_stage_20260605_0828_fixed_candidates/summary.md`를 참고한다.

## 결론

- `pog_dense`의 상승은 실제 retrieval 효과가 크다. 전체 기준 `0.336 -> 0.548`(+21.2%p), 후보가 완전히 같은 103개만 비교해도 `0.282 -> 0.485`(+20.4%p)다.
- `pog`는 `0.328 -> 0.372`(+4.4%p)이지만, 정답 복구 42개와 기존 정답 손상 31개가 거의 상쇄된다. 후보가 같은 샘플은 16개뿐이고 그 안에서는 오히려 -12.5%p라 순수 효과를 확정할 수 없다.
- Dense 성능의 핵심은 stage 1의 직접 `bi_signal`이다. Dense에서는 164개 샘플에서 이 신호가 비동률이고, 그중 정답이 top인 122개의 최종 정확도는 86.1%다.
- POG에서는 `bi_signal`이 비동률인 샘플이 13개뿐이다. 최종 성능은 retrieval보다 stage 2의 텍스트 기반 스타일 추론에 더 의존한다.
- `pog_dense`와 `pog`는 별도 데이터셋 평가다. Dense에서 얻은 evidence나 성능이 POG로 전달되는 구조가 아니며, POG graph 자체가 훨씬 sparse해서 dense 상승이 POG로 이어지지 않는다.

## Baseline -> 2-stage 전이

| dataset | baseline | 2-stage | base 실패 -> 성공 | base 성공 -> 실패 | 둘 다 실패 | 둘 다 성공 |
|---|---:|---:|---:|---:|---:|---:|
| POG | 82/250 (32.8%) | 93/250 (37.2%) | 42 (16.8%) | 31 (12.4%) | 126 (50.4%) | 51 (20.4%) |
| POG-dense | 84/250 (33.6%) | 137/250 (54.8%) | 74 (29.6%) | 21 (8.4%) | 92 (36.8%) | 63 (25.2%) |

- POG는 baseline 실패의 25.0%를 복구하지만 baseline 정답의 37.8%를 망친다. 순증가는 11개이며 McNemar exact `p=0.242`로 유의하지 않다.
- POG-dense는 baseline 실패의 44.6%를 복구하고 baseline 정답의 25.0%를 망친다. 순증가는 53개이며 `p=4.25e-8`이다.
- “2-stage가 불필요하게 정답을 망치는 비율”은 전체 샘플 기준 POG 12.4%, POG-dense 8.4%다. baseline 정답만 분모로 두면 각각 37.8%, 25.0%다.

## 후보 고정 비교

Baseline과 2-stage의 후보 10개가 완전히 같은 샘플만 비교한 결과다.

| dataset | 동일 후보 샘플 | baseline | 2-stage | 실패 -> 성공 | 성공 -> 실패 |
|---|---:|---:|---:|---:|---:|
| POG | 16/250 (6.4%) | 50.0% | 37.5% | 1 | 3 |
| POG-dense | 103/250 (41.2%) | 28.2% | 48.5% | 27 | 6 |

- POG-dense는 후보를 고정해도 +20.4%p이며 `p=0.000324`다. Dense에서 2-stage가 실제로 도움이 된다는 가장 강한 근거다.
- POG는 동일 후보 표본이 16개뿐이라 결론을 내리기 어렵다. 현재 수치만 보면 -12.5%p다.
- 전체 비교에서 평균 후보 교집합은 POG 6.95/10, POG-dense 7.86/10이다. 후보 변경 효과가 전체 전이 수치에 섞여 있다.

## Stage별 기여

Stage 1은 설계상 정답 선택이 금지되어 있다. 따라서 “stage 1 결과”는 각 비동률 numeric evidence를 min-max 정규화하고 동일 가중 평균한 evidence-only proxy다. 이는 실제 stage 출력이 아니라 사후 진단용 근사치다.

| dataset | evidence 생성 | proxy 유효 | stage 1 proxy | 같은 샘플의 stage 2 | stage 2 순개선 |
|---|---:|---:|---:|---:|---:|
| POG | 248/250 | 209/250 | 41/209 (19.6%) | 79/209 (37.8%) | +18.2%p |
| POG-dense | 246/250 | 227/250 | 107/227 (47.1%) | 126/227 (55.5%) | +8.4%p |

### POG

- Stage 1 proxy 실패 168개 중 stage 2가 48개(28.6%)를 교정했다.
- Stage 1 proxy 정답 41개 중 stage 2가 10개(24.4%)를 망쳤다.
- Stage 2가 proxy 선택과 일치한 비율은 41.6%다.
- 해석: retrieval은 약하고 noisy하다. 최종 성능 대부분은 stage 2가 evidence를 무시하거나 텍스트로 다시 푼 결과다.

### POG-dense

- Stage 1 proxy 실패 120개 중 stage 2가 28개(23.3%)를 교정했다.
- Stage 1 proxy 정답 107개 중 stage 2가 9개(8.4%)를 망쳤다.
- Stage 2가 proxy 선택과 일치한 비율은 65.2%다.
- 해석: stage 1이 이미 강하고, stage 2는 신호를 선택적으로 가중하면서 추가 개선한다.

## 신호별 기여

| signal | POG: 비동률 / 정답 top | Dense: 비동률 / 정답 top | 핵심 해석 |
|---|---:|---:|---|
| `bi_signal` | 13 / 12 | 164 / 122 | Dense 상승의 핵심 직접 신호 |
| `ui_signal` | 7 / 5 | 117 / 44 | Dense에서만 의미 있는 coverage |
| `bi_lightgcn_similarity` | 192 / 34 | 184 / 82 | Dense에서는 강하지만 틀리면 손상 위험 |
| `ui_lightgcn_similarity` | 186 / 31 | 182 / 37 | 양쪽 모두 top-1 식별력은 제한적 |
| `embedding_similarity` | 51 / 6 | 112 / 8 | 단순 content 유사도는 정답 top 비율이 매우 낮음 |
| `metadata_fit` | 70 / 1 | 66 / 0 | 같은 category 선호는 outfit completion에 부적합 |

- Dense에서 `bi_signal` 정답 top일 때 최종 정확도는 86.1%, 오답 top일 때는 4.8%다. 최종 정답 137개 중 105개(76.6%)가 이 조건에서 나왔다.
- Dense에서 `bi_lightgcn_similarity` 정답 top일 때 최종 정확도는 91.5%, 오답 top일 때는 23.5%다.
- POG 최종 정답 93개 중 `bi_signal` 정답 top과 겹치는 것은 12개(12.9%)뿐이다.
- 단순 category match와 content similarity는 “함께 입을 보완 아이템”보다 “비슷한 아이템”을 선호해 오답을 만들기 쉽다.

## Dense와 POG 차이 원인

Agent가 실제로 읽은 raw graph 기준:

| 항목 | POG | POG-dense |
|---|---:|---:|
| 평균 input item 수 | 1.54 | 2.02 |
| BI train edges / item | 1.04 | 2.37 |
| UI edges / item | 1.27 | 203.26 |
| UI에 등장하는 item coverage | 12,258/48,676 (25.2%) | 24,137/31,217 (77.3%) |
| UI 등장 item의 평균 degree | 5.06 | 262.88 |

- Dense UI graph는 총 edge가 POG보다 약 102배, item당 약 160배 많다. Stage 1이 candidate 간 차이를 만들 수 있는 확률이 본질적으로 높다.
- POG `ui_full.txt`는 61,987 unique edge만 포함한다. `count.json`의 237,519와 일치하는 파일은 `ui_full_with_duplicates.txt`지만 현재 agent 허용 파일이 아니다.
- POG의 `bi_signal`과 `ui_signal`은 각각 235/248, 241/248 evidence row에서 전부 0으로 동률이다. Dense는 각각 82/246, 129/246이다.
- POG input도 더 짧아 candidate와 연결할 단서가 적다.
- 결과적으로 Dense에서는 direct graph evidence가 판단을 주도하지만, POG에서는 약한 LightGCN/content 신호와 텍스트 추론만 남는다.

## 대표 샘플

- Dense `bundle_id=273`, `true_indice=13625`: 동일 후보에서 baseline `G` 오답 -> 2-stage `A` 정답. `bi_signal`과 BI LightGCN이 정답을 지지했고 최종 판단이 이를 올바르게 우선했다.
- Dense `bundle_id=1265`, `true_indice=23371`: 동일 후보에서 baseline `D` 정답 -> 2-stage `F` 오답. embedding/UI LightGCN/BI LightGCN의 오답 신호를 high confidence로 따라 정답을 망쳤다.
- Dense `bundle_id=1790`, `true_indice=4451`: 둘 다 실패. 잘못된 `bi_signal`과 BI LightGCN을 stage 2가 그대로 따라갔다.
- POG `bundle_id=1292`, `true_indice=4517`: baseline 오답 -> 2-stage 정답이지만 stage 1 proxy는 만들 수 없었다. stage 2의 텍스트 기반 스타일 추론으로 복구한 사례다.
- POG `bundle_id=49`, `true_indice=40631`: 동일 후보에서 baseline 정답 -> 2-stage 오답. 잘못된 LightGCN 신호와 텍스트 추론이 기존 정답을 망쳤다.

## 실험 유효성 이슈

1. Baseline과 2-stage 후보가 대부분 다르다. 특히 Dense baseline은 `cfg_use_hard_negative=True`이고 현재 2-stage 후보 생성은 random negative다.
2. POG baseline에는 model/seed/config가 기록되어 있지 않아 predictor 자체가 동일 조건인지 확인할 수 없다.
3. `agent_allow_interaction_embeddings=False`이고 allowed files에도 LightGCN 파일이 없지만, persistent workspace에 `ui_item_embeddings.pt`와 `bi_item_embeddings.pt`가 남아 있다. Prompt도 이 파일 사용을 권장해 실제 evidence에 포함됐다.
4. Stage 1은 샘플마다 LLM이 다른 retrieval code를 생성한다. 같은 signal 이름도 계산 방식이 일정하지 않아 정량 비교에 noise가 있다.
5. Evidence-only proxy는 신호를 동일 가중한 진단 도구다. Stage 2가 신뢰도 높은 BI 신호를 선택적으로 우선하는 능력을 과소평가할 수 있다.

## 다음 실험 우선순위

1. 2-stage CSV의 `input_indices`와 `candidate_indices`를 그대로 재사용해 같은 모델로 text-only baseline을 다시 실행한다.
2. 동일 후보에서 `text-only`, `BI direct only`, `BI+UI direct`, `LightGCN only`, `all evidence` ablation을 실행한다.
3. POG에서는 direct BI/UI가 비동률일 때만 evidence를 사용하고, category/content/LightGCN 단독 신호는 gate하거나 낮은 가중치를 준다.
4. Dense에서는 BI direct와 BI LightGCN이 충돌할 때 fallback해 기존 정답 손상률을 낮춘다.
5. Workspace를 실행 전 정리하고 LightGCN 허용 여부를 config와 실제 파일 목록에 일치시킨다.
6. POG UI frequency가 필요하다면 `ui_full_with_duplicates.txt`를 명시적으로 허용하고 frequency-aware 신호를 별도 ablation한다.

## 산출물

- `metrics.json`: 전체 수치와 신호별 지표
- `sample_transitions_pog.csv`: POG 샘플별 전이, stage 1 proxy, 신호 진단
- `sample_transitions_pog_dense.csv`: POG-dense 샘플별 전이, stage 1 proxy, 신호 진단
- `utils/analyze_two_stage_results.py`: 재현용 분석 스크립트
