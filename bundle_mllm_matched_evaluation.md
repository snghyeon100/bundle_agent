# Bundle-MLLM을 현재 Spec-First 평가와 동일한 문제로 비교하는 방법

## 1. 목표

현재 실험 명령:

```powershell
python tests/test_spec_first_operator_batch.py `
  --config config_operator.yaml `
  --split test `
  --start_idx 0 `
  --sample_count 250
```

위 명령이 사용하는 **동일한 250개 test sample**에 Bundle-MLLM을 적용한다.

여기서 “동일한 문제”란 다음 값이 sample별로 완전히 같다는 뜻이다.

- `sample_idx`
- `bundle_id`
- partial bundle에 포함되는 item ID와 순서
- 10개 candidate item ID와 A–J label 순서
- ground-truth item ID와 ground-truth label

Bundle-MLLM이 자체적으로 test sample이나 negative candidate를 다시 만들게 두면 동일한 문제가 되지 않는다. 현재 evaluator가 만든 sample을 고정 manifest로 내보내고, Bundle-MLLM은 그 manifest만 소비해야 한다.

## 2. 현재 비교 대상의 정확한 평가 계약

현재 `config_operator.yaml`의 주요 설정은 다음과 같다.

```yaml
dataset: pog_dense
num_cans: 10
num_token: 5
toy_eval: -1
seed: 45
shuffle_seed: 41
```

`tests/test_spec_first_operator_batch.py`는 다음 순서로 sample을 구성한다.

1. `set_seed(seed=45)`를 호출한다.
2. `BundleZeroShotDataset(..., split="test")`를 생성한다.
3. evaluator 내부에서 `toy_eval=-1`로 전체 test sample을 먼저 생성한다.
4. 생성된 순서에서 `[start_idx:start_idx + sample_count]`, 즉 `[0:250]`을 선택한다.
5. bundle별 partial item은 `shuffle_seed=41`을 사용해 결정하고 최대 5개만 남긴다.
6. ground-truth 한 개와 negative 9개로 candidate 10개를 만든다.
7. candidate 순서는 sample 생성 시 정해진 A–J 순서를 그대로 사용한다.

현재 실행이 저장한 기준 manifest는 다음 파일이다.

```text
tests/outputs/spec_first_operator_batch/
  pog_dense_20260730_114851/
    sample_manifest.json
```

이 manifest에는 현재 다음 필드가 있다.

```json
{
  "sample_idx": 0,
  "bundle_id": 9388,
  "candidate_item_ids": [
    9877,
    21880,
    26773,
    14120,
    1368,
    2179,
    9204,
    30353,
    2839,
    27985
  ]
}
```

이 파일은 candidate 동일성을 검증하는 기준으로 사용한다. 다만 Bundle-MLLM 평가에는 partial item과 GT도 필요하므로 아래의 확장 manifest를 별도로 생성하는 것이 안전하다.

## 3. 공통 evaluation manifest

새 exporter가 현재 `BundleZeroShotDataset`을 그대로 사용해 다음 형식의 파일을 생성해야 한다.

권장 파일명:

```text
tests/fixtures/matched_eval/
  pog_dense_test_start0_count250.json
```

권장 형식:

```json
{
  "metadata": {
    "dataset": "pog_dense",
    "split": "test",
    "start_idx": 0,
    "sample_count": 250,
    "num_cans": 10,
    "num_token": 5,
    "seed": 45,
    "shuffle_seed": 41,
    "source_config": "config_operator.yaml",
    "reference_manifest": "tests/outputs/spec_first_operator_batch/pog_dense_20260730_114851/sample_manifest.json"
  },
  "samples": [
    {
      "sample_idx": 0,
      "bundle_id": 9388,
      "partial_item_ids": [27786, 4787],
      "candidate_item_ids": [
        9877,
        21880,
        26773,
        14120,
        1368,
        2179,
        9204,
        30353,
        2839,
        27985
      ],
      "candidate_labels": [
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J"
      ],
      "true_item_id": 27985,
      "true_option_idx": 9,
      "true_label": "J"
    }
  ]
}
```

export 시 다음 조건을 assertion으로 검사한다.

- sample 수가 정확히 250개다.
- 모든 sample의 candidate 수가 정확히 10개다.
- candidate item ID에 중복이 없다.
- `true_item_id`가 candidate 안에 정확히 한 번 존재한다.
- `candidate_item_ids[true_option_idx] == true_item_id`다.
- `true_label == candidate_labels[true_option_idx]`다.
- 기존 `sample_manifest.json`과 `sample_idx`, `bundle_id`, `candidate_item_ids`가 순서까지 완전히 같다.
- 완성된 manifest의 SHA-256을 함께 기록한다.

이후 세 모델의 결과에 동일한 manifest SHA-256을 저장하면 sample 불일치를 탐지할 수 있다.

## 4. Bundle-MLLM 원본 evaluator를 그대로 쓰면 안 되는 이유

Bundle-MLLM의 `utils.BundleTestDataset`은 원본 상태에서:

- test GT pair를 다시 random permutation하고,
- negative candidate를 다시 무작위 추출하고,
- negative를 replacement 방식으로 뽑을 수 있으며,
- candidate 순서를 다시 shuffle하고,
- test DataLoader도 `shuffle=True`로 구성한다.

따라서 원본 `python main.py -o test ...`를 바로 실행하면 현재 Spec-First의 250개 문제와 candidate가 달라질 수 있다.

또한 원본 `main.py::test()`는 생성된 첫 문자를 읽어 top-1 `hitrate`만 계산한다. 현재 결과와 비교하려면 A–J 전체 점수가 필요하고, 이 점수로 full ranking을 만들어야 한다.

결론적으로 원본 dataloader와 원본 metric loop는 사용하지 않고 다음 두 부분만 재사용한다.

- `BundleMLLM` 모델 및 checkpoint loading
- graph, multimodal feature, item metadata loading

sample selection과 metric 계산은 현재 `bundle_agent` 쪽 평가 계약을 따른다.

## 5. 권장 구현 구조

현재 repository에 다음 파일을 추가하는 방식을 권장한다.

```text
src/comparisons/
  __init__.py
  matched_manifest.py
  bundle_mllm_adapter.py

tests/
  export_matched_evaluation_manifest.py
  test_bundle_mllm_batch.py

tests/outputs/
  bundle_mllm_matched/
    pog_dense_<timestamp>/
      run.json
      sample_manifest.json
      results.json
      results.csv
      summary.json
      samples/
```

역할은 다음과 같다.

### `matched_manifest.py`

- 현재 `BundleZeroShotDataset`으로 exact sample을 생성한다.
- 기존 Spec-First manifest와 동일성을 검증한다.
- partial, candidates, GT가 모두 들어간 확장 manifest를 읽고 쓴다.

### `bundle_mllm_adapter.py`

- 외부 `Bundle-MLLM` repository를 import한다.
- base LLM과 공개 checkpoint를 로드한다.
- manifest sample을 Bundle-MLLM 입력 tensor로 변환한다.
- A–J candidate score를 추출한다.
- score 내림차순으로 full ranking을 반환한다.

### `test_bundle_mllm_batch.py`

- `test_spec_first_operator_batch.py`와 비슷한 CLI를 제공한다.
- `--split`, `--start_idx`, `--sample_count` 계약을 동일하게 유지한다.
- resume, partial result, summary 저장을 담당한다.
- 모든 결과에 manifest SHA-256을 기록한다.

## 6. Bundle-MLLM 입력 변환

manifest의 한 sample은 다음 두 tensor로 변환할 수 있다.

```python
input_indices = sample["partial_item_ids"]
candidate_indices = sample["candidate_item_ids"]
```

Bundle-MLLM의 `evaluate(indices, candidates, generation_config)`가 기대하는 형태:

```text
indices:
  shape = [batch_size, padded_partial_length]
  padding value = num_items

candidates:
  shape = [batch_size, 10]
  order = manifest의 A–J 순서 그대로
```

중요한 조건:

- Bundle-MLLM의 자체 `dataset.test_loader`에서 sample을 꺼내지 않는다.
- partial item을 다시 shuffle하거나 truncate하지 않는다.
- candidate를 다시 생성하거나 shuffle하지 않는다.
- manifest에 저장된 item ID 순서를 그대로 tensor로 만든다.
- candidate index `0..9`를 label `A..J`로 고정한다.

Bundle-MLLM은 item ID를 이용해 자체 `item_info`, content feature, description/text feature, CF feature, bundle feature를 조회할 수 있다. 따라서 동일한 item ID 문제를 사용하되, Bundle-MLLM은 자신의 native multimodal representation으로 판단하게 된다.

이는 “동일 sample에서 각 방법을 원래 설계대로 비교”하는 primary protocol이다. Bundle-MLLM에 현재 Spec-First가 본 문자열만 강제로 제공하는 text-controlled ablation은 별도 실험으로 분리해야 한다.

## 7. 공개 checkpoint 로딩

공개 저장소:

```text
https://huggingface.co/xhLiu/Bundle-MLLM/tree/main
```

POG-dense checkpoint:

```text
pog_dense/checkpoint/
```

checkpoint에는 대략 다음 추가 가중치가 포함된다.

```text
adapter_config.json
adapter_model.bin
fusion.bin
projector_model_fuse.bin
prompt_token.bin
```

이는 full base LLM이 아니라 LoRA 및 Bundle-MLLM 추가 모듈이다. 따라서 다음이 모두 필요하다.

1. `adapter_config.json`과 일치하는 base LLM
2. LoRA adapter
3. fusion module
4. projector
5. prompt token

base model은 먼저 `adapter_config.json`의 `base_model_name_or_path`를 확인해 결정한다. repository 코드의 기본값은 `meta-llama/Llama-2-7b-hf`지만, checkpoint metadata가 다르면 metadata를 우선한다.

공개 README의 checkpoint 사용 예시는 `soft_prompt=True`, `del_sp=True`를 사용한다. checkpoint tensor shape가 맞도록 checkpoint가 학습될 때 사용한 flags를 그대로 복구해야 한다.

주의할 점:

- repository의 `model.py`는 내부에서 `self.mode = "text+mm"`로 다시 설정하므로 실제 checkpoint 평가가 multimodal 경로를 사용한다.
- 공개 README의 progressive stage는 20 candidates를 예시로 들지만, 현재 비교에서는 동일 문제를 위해 반드시 10 candidates만 사용한다.
- 20-candidate native 결과가 필요하면 별도로 실행할 수 있지만, 현재 10-candidate 결과와 같은 표에 직접 비교하면 안 된다.

## 8. A–J score에서 full ranking 만들기

Bundle-MLLM 원본 test는 문자를 생성해 top-1만 얻는다. 현재 비교에는 candidate별 score가 필요하다.

권장 방법은 첫 generation step의 vocabulary logits에서 checkpoint가 학습에 사용한 A–J token 점수를 추출하는 것이다.

원본 코드는 `MAP_LETTER`에 A–T token ID를 하드코딩한다. 10-candidate 평가에서는 첫 10개 A–J token만 사용한다.

개념적인 처리:

```python
outputs = model.evaluate(indices, candidates, generation_config)
first_step_logits = outputs.scores[0]

label_token_ids = token_ids_for_A_through_J
candidate_scores = first_step_logits[:, label_token_ids]

ranked_indices = candidate_scores.argsort(dim=-1, descending=True)
ranking = [candidate_labels[index] for index in ranked_indices]
prediction = ranking[0]
```

실행 시작 시 반드시 검증한다.

```text
tokenizer.decode(label_token_ids[0]) -> A
...
tokenizer.decode(label_token_ids[9]) -> J
```

base tokenizer에 따라 label이 한 token이 아니거나 공백 prefix가 다르면 하드코딩 token ID를 그대로 사용하면 안 된다. 이 경우 각 답안 문자열 `"A"`부터 `"J"`까지의 conditional log-likelihood를 teacher forcing으로 계산해 candidate score로 사용한다.

공통 ranking metric에는 생성 샘플 결과가 아니라 candidate score의 내림차순을 사용한다. 디버깅 목적으로 다음 두 값은 함께 저장할 수 있다.

- `generated_prediction`: 원본 generation으로 나온 문자
- `score_prediction`: candidate score가 가장 큰 label

공식 비교의 `prediction`은 `ranking[0]`, 즉 `score_prediction`으로 정의한다.

ranking만 필요하므로 generation 설정은 다음처럼 deterministic하게 두는 것이 적절하다.

```text
do_sample = false
max_new_tokens = 1
```

## 9. 공통 result 형식

sample별 권장 결과:

```json
{
  "sample_idx": 0,
  "bundle_id": 9388,
  "candidate_item_ids": [
    9877,
    21880,
    26773,
    14120,
    1368,
    2179,
    9204,
    30353,
    2839,
    27985
  ],
  "candidate_labels": [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J"
  ],
  "candidate_scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "prediction": "H",
  "ranking": ["H", "B", "J", "D", "A", "C", "I", "G", "F", "E"],
  "true_item_id": 27985,
  "true_label": "J",
  "gt_rank": 3,
  "reciprocal_rank": 0.3333333333,
  "hit_at_1": false,
  "hit_at_3": true,
  "hit_at_5": true,
  "valid": true,
  "error": null,
  "manifest_sha256": "..."
}
```

구조 검증 조건:

- `ranking`은 A–J를 정확히 한 번씩 포함한다.
- `prediction == ranking[0]`이다.
- `gt_rank == ranking.index(true_label) + 1`이다.
- score와 ranking 순서가 일치한다.
- 모든 row의 manifest SHA-256이 동일하다.

## 10. 공통 summary

현재 Spec-First와 같은 metric을 계산한다.

- Hit@1
- Hit@3
- Hit@5
- Mean Reciprocal Rank
- Mean GT Rank
- valid sample count
- invalid/error sample count

다만 현재 `aggregate_prediction_rows()`는 valid row만 분모로 사용하는 valid-only metric이다. 모델 간 비교 표에는 다음 두 종류를 함께 저장하는 것이 안전하다.

### Valid-only

구조적으로 유효한 row만 분모로 사용한다. 기존 Spec-First summary와 직접 대응한다.

### Strict all-requested

요청된 250개 전체를 분모로 사용하고 invalid/error sample은 Hit=0, RR=0으로 처리한다. 모델별 invalid 비율 차이로 특정 모델이 유리해지는 것을 막기 위해 primary comparison에는 strict metric을 권장한다.

예:

```json
{
  "requested_sample_count": 250,
  "completed_sample_count": 250,
  "valid_sample_count": 250,
  "invalid_or_error_sample_count": 0,
  "valid_rate": 1.0,
  "valid_only": {
    "hit_rate_at_1": 0.0,
    "hit_rate_at_3": 0.0,
    "hit_rate_at_5": 0.0,
    "mean_reciprocal_rank": 0.0,
    "mean_gt_rank": 0.0
  },
  "strict_all_requested": {
    "hit_rate_at_1": 0.0,
    "hit_rate_at_3": 0.0,
    "hit_rate_at_5": 0.0,
    "mean_reciprocal_rank": 0.0
  }
}
```

## 11. 권장 CLI

아래 명령은 관련 script를 구현한 뒤 사용할 목표 인터페이스다.

### 11.1 공통 manifest export

```powershell
python tests/export_matched_evaluation_manifest.py `
  --config config_operator.yaml `
  --split test `
  --start_idx 0 `
  --sample_count 250 `
  --reference_manifest "tests/outputs/spec_first_operator_batch/pog_dense_20260730_114851/sample_manifest.json" `
  --output "tests/fixtures/matched_eval/pog_dense_test_start0_count250.json"
```

### 11.2 Bundle-MLLM matched evaluation

```powershell
python tests/test_bundle_mllm_batch.py `
  --config config_operator.yaml `
  --manifest "tests/fixtures/matched_eval/pog_dense_test_start0_count250.json" `
  --bundle_mllm_root "C:\Users\wotjs\Desktop\bundle\Bundle-MLLM" `
  --checkpoint_path "C:\Users\wotjs\Desktop\bundle\Bundle-MLLM\checkpoints_hf\pog_dense\checkpoint" `
  --base_model "<adapter_config.json과 일치하는 base model>" `
  --split test `
  --start_idx 0 `
  --sample_count 250
```

`--config`, `--split`, `--start_idx`, `--sample_count`는 현재 Spec-First 명령과 같은 의미를 유지한다. `--manifest`가 전달되면 재생성한 sample보다 manifest가 우선하며, 재생성 sample은 검증에만 사용한다.

## 12. 환경 준비

Bundle-MLLM은 별도 CUDA 환경에서 실행하는 것이 좋다. 현재 `bundle_agent/.venv`는 CPU PyTorch 환경이고 `transformers`, `peft`, `fastchat` 등이 설치되어 있지 않다.

필요 구성:

- CUDA가 연결된 PyTorch
- Transformers
- PEFT
- bitsandbytes
- scipy, numpy, pyyaml
- base LLM 접근 권한과 로컬 저장 공간
- Hugging Face의 POG-dense Bundle-MLLM checkpoint

repository가 구버전 PEFT API인 `prepare_model_for_int8_training`을 사용하므로 최신 PEFT에서는 `prepare_model_for_kbit_training`으로 호환 패치가 필요할 수 있다. 비교 코드 수정과 모델 의미 변경을 분리하기 위해 Bundle-MLLM 전용 환경을 권장한다.

## 13. 비교 시 보고해야 할 차이

동일 sample을 사용해도 모델이 사용하는 정보는 다르다.

| 방법 | 입력 및 학습 조건 |
|---|---|
| Spec-First operator | partial/candidate text, 실행된 retrieval evidence, GPT 기반 판단 |
| Bundle-MLLM | partial/candidate item text, multimodal/CF/bundle feature, task-specific fine-tuned LLM |

따라서 결과표에는 다음을 명시한다.

- 동일 item-level sample, candidate set, candidate order를 사용했다.
- Bundle-MLLM은 공개 task-specific checkpoint를 사용했다.
- Bundle-MLLM은 native multimodal feature를 사용했다.
- Spec-First는 공개 checkpoint 학습 없이 현재 retrieval/operator pipeline을 사용했다.
- candidate 수는 두 방법 모두 10개다.
- 두 방법 모두 동일 manifest SHA-256을 사용했다.

## 14. 완료 판정 체크리스트

- [ ] 확장 manifest가 250개 sample을 포함한다.
- [ ] 기존 Spec-First `sample_manifest.json`과 candidate 순서가 완전히 같다.
- [ ] partial item ID와 GT item ID까지 manifest에 고정되어 있다.
- [ ] POG-dense Bundle-MLLM checkpoint와 정확한 base model을 로드한다.
- [ ] `adapter_model.bin`, `fusion.bin`, projector, prompt token이 모두 로드되었다는 로그가 나온다.
- [ ] Bundle-MLLM 자체 random candidate 생성 경로를 사용하지 않는다.
- [ ] 모든 sample에서 A–J score 10개를 얻는다.
- [ ] 모든 ranking이 A–J 전체를 정확히 한 번 포함한다.
- [ ] `prediction == ranking[0]`이다.
- [ ] 결과 row가 정확히 250개다.
- [ ] Hit@1/3/5, MRR, mean GT rank를 동일 코드로 계산한다.
- [ ] valid-only와 strict-all-requested metric을 함께 저장한다.
- [ ] Spec-First와 Bundle-MLLM 결과에 동일 manifest SHA-256이 기록된다.

## 15. 최종 권장안

가장 안전한 비교 순서는 다음과 같다.

1. 현재 evaluator에서 250개 exact sample을 확장 manifest로 한 번만 export한다.
2. 현재 Spec-First 결과가 그 manifest와 동일함을 검증한다.
3. 공개 POG-dense Bundle-MLLM checkpoint와 대응 base LLM을 준비한다.
4. Bundle-MLLM의 dataloader는 우회하고 manifest의 partial/candidate tensor를 직접 입력한다.
5. 첫 decode step의 A–J score로 full ranking을 만든다.
6. 공통 evaluator에서 두 결과의 Hit@1/3/5, MRR, mean GT rank를 계산한다.
7. primary table에는 동일 250개 전체를 분모로 한 strict metric을 사용하고 valid-only metric도 함께 보고한다.

이 방식이면 두 모델이 판단에 사용하는 내부 표현은 각자 다르더라도, 평가받는 bundle, partial items, candidates, GT는 완전히 동일하게 유지된다.
