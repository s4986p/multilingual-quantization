# 결과

## 공통 설정

#### 범위
- 따로 적지 않으면 모든 실험에 적용
- 언어별 손실 이후로는 mgsm만, mmlu-prox-lite에서는 en보다 눈에 띄게 더 잃는 언어가 없기 때문
- 언어별 손실 이후로는 gemma만, gemma는 거의 반복하지 않아 손실만 보기 좋기 때문

#### 용어
- 프롬프트: 지시문, 질문
- 지시문: 질문 앞 텍스트
- 질문: 문제의 텍스트
- 추론: thinking mode에서 쓴 텍스트
- 응답: 추론 뒤 텍스트

#### 모델
- qwen3.5-2b, gemma-4-e2b-it, thinking mode 사용
- bf16 기준
- 4 bit, linear layer weight만 양자화
- gptq, awq group size 128, nf4 group size 64, mxfp4 group size 32
- gptq, awq는 en wikipedia chunk 128개로 calibration, chunk마다 2048 token

#### 생성
- vllm 0.30.0, temperature 1.0, top-p 0.95
- qwen top-k 20, presence penalty 1.5, 최대 생성 길이 32768
- gemma top-k 64, presence penalty 0, 최대 생성 길이 16384
- 따로 적지 않으면 seed 0, 1, 2 평균

#### 평가
- en, zh, sw, ja, te, 해당 언어 프롬프트, zero shot
- 언어마다 mgsm 250개, mmlu-prox-lite 588개 문제
- mgsm 지시문은 [simple-evals](https://github.com/openai/simple-evals/blob/main/mgsm_eval.py)에서 가져옴
- mmlu-prox-lite 지시문은 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_prox/lang_libs.py)에서 가져옴
- 추론이 아니라 응답만 채점

## 언어별 손실
[the multilingual quantization tax](https://arxiv.org/abs/2608.09941)에서 말한 대로 언어마다 손실이 다른지 확인

#### 설정
- mmlu-prox-lite는 bf16 seed 0, 1, 양자화 모델 seed 0, 언어마다 손실이 비슷해 보이기 때문
- 잘림: 최대 생성 길이에 닿고 틀린 문제
- 반복: 추론이나 응답 끝에서 200~150번째 글자 50개가 그 앞에 5번 이상 나온 문제
- 모든 문제 사용, qwen bf16이 sw와 te에서 푼 문제가 너무 적기 때문

#### 요약
![](../figures/Qwen3.5-2B/mgsm_loss.png)
![](../figures/gemma-4-E2B-it/mgsm_loss.png)
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
- mgsm에서, awq가 가장 적게 잃고 mxfp4가 가장 많이 잃음
- mgsm에서, gemma가 qwen보다 적게 잃음
- qwen mgsm에서, 양자화 모델이 많이 잘리고 그 대부분이 반복
- qwen nf4와 mxfp4 mgsm에서, zh가 크게 잃고 그 대부분이 잘림
- qwen bf16 mgsm에서, sw와 te가 약함
- gemma mgsm에서, 거의 잘리지 않음, 따라서 손실은 대부분 오답에서 발생
- **gemma mgsm에서, sw와 te가 가장 많이 잃음**
- gemma mgsm에서, ja도 en보다 더 잃음, 하지만 sw와 te보다 적음
- mmlu-prox-lite에서, en보다 눈에 띄게 더 잃는 언어 없음

## calibration 언어 민감도
calibration 언어가 중요한지 확인

#### 설정
- awq만, 직관적으로 awq가 calibration에 더 의존하기 때문
- all: en, zh, sw, ja, te, 언어마다 chunk 약 26개
- en, sw, te, all로 calibration

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
- **sw와 te에서, 해당 언어로 calibration해도 눈에 띄게 나아지지 않음**
- all로 calibration해도 눈에 띄게 나아지지 않음
