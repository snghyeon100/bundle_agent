# 후보군 수정 후 2-stage 분석

분석 대상:

- POG 2-stage: `results_pog_two_stage_agent_C10_T5_20260605_082843.csv`
- POG-dense 2-stage: `results_pog_dense_two_stage_agent_C10_T5_20260605_082934.csv`
- Baseline: 기존 2026-04 POG / POG-dense 결과

## 핵심 결론

- 이번 결과는 baseline과 2-stage의 input, 후보 10개, 정답 위치, 문제 문자열이 두 데이터셋 모두 **250/250 완전히 동일**하다.
- POG는 `32.8% -> 34.8%`, 순개선은 5개(+2.0%p)에 불과하며 유의하지 않다.
- POG-dense는 `33.6% -> 54.4%`, 순개선은 52개(+20.8%p)이며 강하게 유의하다.
- Dense 상승의 핵심은 stage 1의 직접 `bi_signal`이다. POG에서는 이 신호가 거의 항상 전부 0이라 dense의 상승이 이어지지 않는다.
- Stage 2는 stage 1 evidence-only proxy보다 양쪽 모두 개선하지만, POG에서는 약한 retrieval을 텍스트 추론으로 보완하는 역할이 크다.
- 기존 정답 손상의 주요 위험은 잘못된 BI LightGCN 신호와 이를 신뢰하는 최종 판단이다.

## 샘플별 변화

| dataset | baseline | 2-stage | base 실패 -> 성공 | base 성공 -> 실패 | 둘 다 실패 | 둘 다 성공 |
|---|---:|---:|---:|---:|---:|---:|
| POG | 82/250 (32.8%) | 87/250 (34.8%) | 33 (13.2%) | 28 (11.2%) | 135 (54.0%) | 54 (21.6%) |
| POG-dense | 84/250 (33.6%) | 136/250 (54.4%) | 74 (29.6%) | 22 (8.8%) | 92 (36.8%) | 62 (24.8%) |

### POG

- Baseline 실패 168개 중 33개를 복구했다: **19.6% 복구율**
- Baseline 정답 82개 중 28개를 망쳤다: **34.1% 손상률**
- 전체 샘플 기준 불필요한 손상률: **11.2%**
- 순증가: 5개, +2.0%p
- McNemar exact `p=0.609`: 유의한 개선이라고 보기 어렵다.

### POG-dense

- Baseline 실패 166개 중 74개를 복구했다: **44.6% 복구율**
- Baseline 정답 84개 중 22개를 망쳤다: **26.2% 손상률**
- 전체 샘플 기준 불필요한 손상률: **8.8%**
- 순증가: 52개, +20.8%p
- McNemar exact `p=9.44e-8`: 유의한 개선이다.

샘플별 상세 전이는 아래 CSV의 `baseline_to_two_stage` 열로 필터링할 수 있다.

- `sample_transitions_pog.csv`
- `sample_transitions_pog_dense.csv`

## 각 Stage의 기여도

Stage 1은 설계상 정답을 직접 선택하지 않는다. 아래 stage 1 성능은 비동률 numeric evidence를 정규화하고 동일 가중 평균한 **evidence-only proxy**다. 실제 stage 1 출력 정확도가 아니라 진단용 근사치다.

| dataset | evidence 생성 성공 | proxy 유효 | stage 1 proxy | 같은 샘플의 stage 2 | stage 2 순개선 |
|---|---:|---:|---:|---:|---:|
| POG | 241/250 | 211/250 | 43/211 (20.4%) | 71/211 (33.6%) | +13.3%p |
| POG-dense | 247/250 | 222/250 | 101/222 (45.5%) | 124/222 (55.9%) | +10.4%p |

### Stage 2가 stage 1을 얼마나 개선했는가

| dataset | stage 1 실패 -> stage 2 성공 | stage 1 성공 -> stage 2 실패 | proxy 선택과 최종 선택 일치 |
|---|---:|---:|---:|
| POG | 40/168 (23.8%) | 12/43 (27.9%) | 43.6% |
| POG-dense | 28/121 (23.1%) | 5/101 (5.0%) | 69.4% |

- POG stage 1 evidence는 약하고 충돌이 많다. Stage 2가 proxy 실패를 40개 교정하지만 proxy 정답도 12개 망친다.
- Dense stage 1은 이미 강하다. Stage 2는 proxy 정답을 거의 보존하면서 28개를 추가 교정한다.
- POG에서 reasoning이 evidence를 명시적으로 언급한 153개 정확도는 30.7%, 언급하지 않은 97개는 41.2%다.
- Dense에서 evidence를 언급한 225개 정확도는 56.0%, 언급하지 않은 25개는 40.0%다.
- 해석: Dense에서는 evidence 사용이 유용하지만, POG에서는 evidence가 오히려 텍스트 판단을 방해하는 경우가 많다.

## 신호별 기여도

| signal | POG: 비동률 / 정답 top | Dense: 비동률 / 정답 top | 해석 |
|---|---:|---:|---|
| `bi_signal` | 12 / 12 | 160 / 120 | Dense 상승의 핵심 |
| `ui_signal` | 6 / 5 | 110 / 45 | Dense에서만 의미 있는 coverage |
| `bi_lightgcn_similarity` | 179 / 38 | 169 / 71 | 유용하지만 오답 top일 때 손상 위험 |
| `ui_lightgcn_similarity` | 177 / 36 | 166 / 36 | 단독 top-1 식별력은 낮음 |
| `embedding_similarity` | 60 / 5 | 105 / 9 | 단순 content 유사도는 약함 |
| `metadata_fit` | 68 / 0 | 75 / 1 | 같은 category 선호는 completion에 부적합 |

### Direct BI 신호

- Dense에서 `bi_signal`이 비동률인 샘플은 160개이며, 정답 top 비율은 120/160(75.0%)이다.
- Dense에서 `bi_signal` 정답 top일 때 최종 정확도는 86.7%, 오답 top일 때는 0%다.
- Dense 최종 정답 136개 중 104개(76.5%)가 `bi_signal` 정답 top 사례다.
- Dense 복구 74개 중 58개(78.4%)가 정답 `bi_signal`을 가진다.
- POG에서 `bi_signal`이 비동률인 샘플은 12개뿐이다. 12개 모두 정답 top이고 최종도 모두 정답이지만 coverage가 너무 작다.
- POG 복구 33개 중 정답 `bi_signal`을 가진 사례는 9개(27.3%)뿐이다.

### LightGCN과 손상

- POG 손상 28개 중 19개(67.9%)에 잘못된 BI LightGCN unique-top 신호가 있다.
- Dense 손상 22개 중 16개(72.7%)에 잘못된 BI LightGCN unique-top 신호가 있다.
- Dense에서 BI LightGCN 정답 top이면 최종 정확도 88.7%, 오답 top이면 25.5%다.
- 잘못된 LightGCN 신호를 high/medium confidence로 따라가는 것이 기존 정답을 망치는 주요 패턴이다.

## POG-dense와 POG 차이 원인

Agent가 실제 읽는 raw graph 기준:

| 항목 | POG | POG-dense |
|---|---:|---:|
| 평균 input item 수 | 1.54 | 2.02 |
| BI train edges / item | 1.04 | 2.37 |
| UI edges / item | 1.27 | 203.26 |
| UI item coverage | 25.2% | 77.3% |
| UI 등장 item의 평균 degree | 5.06 | 262.88 |

- Dense UI graph는 총 edge가 POG보다 약 102배, item당 약 160배 많다.
- POG evidence row 241개 중 `bi_signal`은 229개(95.0%), `ui_signal`은 235개(97.5%)에서 전부 0으로 동률이다.
- Dense는 `bi_signal` 87/247(35.2%), `ui_signal` 137/247(55.5%)만 전부 0이다.
- POG는 input item 수도 더 적어 candidate와 연결할 graph 단서가 적다.
- Dense에서는 direct BI/UI가 후보를 구분하지만, POG에서는 대부분의 후보가 direct signal상 동일해진다.

## Dense 상승이 POG로 이어지지 않는 이유

POG-dense와 POG는 하나의 파이프라인 앞뒤 단계가 아니라 **서로 다른 데이터셋에서 별도로 실행한 평가**다. Dense에서 얻은 graph evidence가 POG 실행으로 전달되지 않는다.

Dense:

```text
풍부한 BI/UI graph
-> stage 1이 정답 후보를 직접 구분
-> stage 2가 강한 BI 신호를 따름
-> 큰 성능 상승
```

POG:

```text
희소한 BI/UI graph
-> 거의 모든 후보의 direct signal이 0
-> 약한 LightGCN/content 신호 또는 텍스트 추론에 의존
-> 복구와 손상이 비슷하게 발생
-> 최종 순개선이 작음
```

## 대표 샘플

- POG `bundle_id=341`, `true_indice=38297`: baseline `A` 오답 -> 2-stage `G` 정답. `bi_signal`과 BI LightGCN이 모두 정답을 지지했다.
- POG `bundle_id=3204`, `true_indice=36038`: stage 1 proxy 없이 stage 2의 텍스트 스타일 추론으로 복구했다.
- POG `bundle_id=49`, `true_indice=40631`: baseline `B` 정답 -> 2-stage `F` 오답. 잘못된 UI/BI LightGCN 신호와 텍스트 추론이 기존 정답을 망쳤다.
- Dense `bundle_id=273`, `true_indice=13625`: baseline `G` 오답 -> 2-stage `A` 정답. 높은 `bi_signal=10`과 BI LightGCN을 올바르게 우선했다.
- Dense `bundle_id=1265`, `true_indice=23371`: baseline `D` 정답 -> 2-stage `F` 오답. 잘못된 embedding/UI/BI LightGCN을 high confidence로 따랐다.
- Dense `bundle_id=1790`, `true_indice=4451`: 둘 다 실패. 잘못된 direct BI와 BI LightGCN을 최종 판단이 그대로 따랐다.

## 실행 실패

- POG: `ERR_EX` 4개, 이 중 baseline 정답 2개가 손상으로 집계됐다.
- POG-dense: `ERR_EX` 1개, baseline도 오답이어서 추가 손상은 없었다.
- POG의 유효 응답만 계산한 정확도는 87/246 = 35.4%다.

## 개선 우선순위

1. POG에서는 `bi_signal`/`ui_signal`이 비동률일 때만 retrieval evidence를 강하게 사용한다.
2. Direct BI가 없을 때 category/content/LightGCN만으로 high confidence 결정을 내리지 않도록 gate한다.
3. Dense에서는 direct BI와 BI LightGCN이 충돌하면 text-only 판단 또는 fallback을 사용한다.
4. Stage 1의 샘플별 생성 코드를 deterministic retriever로 바꿔 같은 signal 계산 방식을 보장한다.
5. POG의 UI graph coverage를 높이거나 frequency-aware UI 데이터를 별도 ablation한다.
6. 실행 실패를 최종 오답으로 남기지 말고 text-only fallback으로 처리한다.

## 비교 시 주의사항

- 후보군과 입력은 이번에 완전히 통제됐다.
- 다만 POG 2026-04 baseline CSV에는 model/config가 기록되어 있지 않아 baseline과 2-stage predictor 모델이 동일한지는 확인할 수 없다.
- Stage 1 proxy는 실제 stage 출력이 아니라 evidence 품질을 보기 위한 사후 진단 지표다.
