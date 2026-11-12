# 결과

## 공통 설정

#### 용어
- 프롬프트: 지시문, 질문
- 지시문: 질문 앞 텍스트
- 질문: 문제의 텍스트
- 추론: thinking mode에서 쓴 텍스트
- 응답: 추론 뒤 텍스트
- 해당 언어: 문제와 같은 언어
- 손실: bf16보다 떨어진 정확도
- 영어와의 손실 차이: 해당 언어에서 더 생긴 손실, 해당 언어 손실에서 영어 손실을 뺀 값
- seed 노이즈: seed 사이의 표준편차, plot의 ± 값
- 잘림: 최대 생성 길이에 닿고 틀린 문제
- 반복: 같은 글을 되풀이한 문제, 추론 또는 응답 끝에서 200~150번째 글자 50개가 그 앞에 5번 이상 나오는지로 판단
- 문제 이해: 추론 시작에서 목표와 주어진 정보를 영어로 쓰는 부분, 제목에 calculate, determine, compute, solve, perform, set up, formulate가 처음 나오는 단계 앞까지, 추론의 약 30%
- 영어 문제 이해: bf16이 영어 프롬프트에 쓴 문제 이해
- 풀이: 문제 이해 뒤 추론
- 추론 위치: 추론에서 어디쯤인지, 문제마다 추론 token을 5등분한 구간

#### 범위
- 따로 적지 않으면 모든 실험에 적용
- 언어별 손실 이후로는 mgsm만, mmlu-prox-lite에서는 영어보다 눈에 띄게 더 잃는 언어가 없기 때문
- 언어별 손실 이후로는 gemma만, gemma는 거의 반복하지 않아 반복과 섞이지 않은 손실을 보기 좋기 때문

#### 모델
- qwen3.5-2b, gemma-4-e2b-it, thinking mode 사용
- bf16 기준
- 4 bit, language model layer의 linear weight만 양자화
- gptq, awq group size 128, nf4 group size 64, mxfp4 group size 32
- gptq, awq는 영어 wikipedia chunk 128개로 calibration, chunk마다 2048 token

#### 생성
- vllm 0.30.0, temperature 1.0, top-p 0.95
- qwen top-k 20, presence penalty 1.5, 최대 생성 길이 32768
- gemma top-k 64, presence penalty 0, 최대 생성 길이 16384
- 따로 적지 않으면 seed 0, 1, 2 평균

#### 평가
- 영어, 중국어, 스와힐리어, 일본어, 텔루구어, 해당 언어 프롬프트, zero shot
- 언어마다 mgsm 250개, mmlu-prox-lite 588개 문제
- mgsm 지시문은 [simple-evals](https://github.com/openai/simple-evals/blob/main/mgsm_eval.py)에서 가져옴
- mmlu-prox-lite 지시문은 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_prox/lang_libs.py)에서 가져옴
- 추론이 아니라 응답만 채점

## 언어별 손실
[the multilingual quantization tax](https://arxiv.org/abs/2608.09941)에서 말한 대로 언어마다 손실이 다른지 확인

#### 설정
- mmlu-prox-lite는 bf16 seed 0, 1, 양자화 모델 seed 0, 언어마다 손실이 비슷해 보이기 때문
- 모든 문제 사용, qwen bf16이 스와힐리어와 텔루구어에서 푼 문제가 너무 적기 때문

#### 요약
![](../figures/Qwen3.5-2B/mgsm_loss.png)
![](../figures/gemma-4-E2B-it/mgsm_loss.png)
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
- mgsm에서, mxfp4가 가장 많이 잃음
- qwen mgsm에서, awq가 가장 적게 잃음
- mgsm에서, gemma가 qwen보다 적게 잃음
- qwen gptq, nf4, mxfp4 mgsm에서, 많이 잘리고 그 대부분이 반복
- qwen nf4와 mxfp4 mgsm에서, 중국어가 크게 잃고 그 대부분이 잘림
- qwen bf16 mgsm에서, 스와힐리어와 텔루구어가 약함
- gemma mgsm에서, 거의 잘리지 않음, 따라서 손실은 대부분 틀린 응답에서 발생
- **gemma mgsm에서, 스와힐리어와 텔루구어가 가장 많이 잃음**
- gemma mxfp4 mgsm에서, 일본어도 영어보다 더 잃음, 그러나 스와힐리어와 텔루구어보다 적음
- mmlu-prox-lite에서, 영어보다 눈에 띄게 더 잃는 언어 없음

## calibration 언어 민감도
calibration 언어가 중요한지 확인

#### 설정
- all: 영어, 중국어, 스와힐리어, 일본어, 텔루구어, 언어마다 chunk 약 26개
- awq만, 직관적으로 awq가 calibration에 더 의존하기 때문
- 영어, 스와힐리어, 텔루구어, all로 calibration

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
- **스와힐리어와 텔루구어에서, 해당 언어로 calibration해도 눈에 띄게 나아지지 않음**
- all로 calibration해도 눈에 띄게 나아지지 않음

## 추론 위치별 예측 차이
언어마다 양자화로 예측이 얼마나 바뀌는지 확인

#### 설정
- 추론 위치별 kl은 mxfp4만
- 모든 bf16 seed가 푼 문제에서 bf16 seed 0 추론 추출
- 그 추론을 bf16과 양자화 모델에 입력
- 추론 token마다 두 모델의 다음 token 예측을 kl로 비교

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_teacher_forcing.png)
- 언어 사이에서는 kl 순서가 손실 순서와 맞음, 그러나 방법 사이에서는 맞지 않음
- **스와힐리어와 텔루구어에서, 영어와의 kl 차이가 추론 위치 0~20%에서 가장 큼**

## 추론 교체
[추론 위치별 예측 차이](#추론-위치별-예측-차이)에서 스와힐리어와 텔루구어의 영어와의 kl 차이가 추론 위치 0~20%에서 가장 큼, 따라서 먼저 추론 전체를 바꿔서 추론이 정확도를 결정하는지 확인

#### 설정
- bf16 또는 mxfp4가 저장한 추론 추출
- 그 추론에 이어서 bf16 또는 mxfp4로 응답 생성, 조합 4개

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_reasoning_swap.png)
- **응답이 아니라 추론이 정확도를 결정**

## 오류 유형
[추론 교체](#추론-교체)에서 응답이 아니라 추론이 정확도를 결정, 따라서 추론이 어떻게, 어디서 틀리는지 확인

#### 설정
- 사실 오류: 질문의 숫자, 단위, 단어, 이름을 잘못 옮김
- 관계 오류: 양 사이의 관계를 잘못 이해함
- 목표 오류: 묻는 것과 다른 양을 구함
- 조건 무시: 주어진 조건을 없다고 하고 버림
- 계산 오류: 식은 맞지만 계산이 틀림
- 기타: 위에 없음
- 모든 bf16 seed가 푼 문제에서 mxfp4가 저장한 틀린 추론 추출
- 틀린 추론과 bf16 추론을 qwen3.5-27b에 입력
- qwen3.5-27b가 처음 틀린 줄과 오류 유형 선택

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_error_types.png)
- **스와힐리어와 텔루구어에서, 오류는 계산 오류가 아니라 대부분 사실 오류와 관계 오류**, 따라서 질문을 잘못 이해한 것일 수 있음
- 첫 오류는 대부분 추론 위치 0~60%에서 발생
- 사실 오류는 대부분 추론 위치 0~20%에서 발생

## 영어 문제 이해 교체
[오류 유형](#오류-유형)에서 스와힐리어와 텔루구어 오류는 대부분 사실 오류와 관계 오류이고 첫 오류는 대부분 추론 위치 0~60%에서 발생, 따라서 문제 이해만 영어 문제 이해로 바꿔도 영어와의 차이를 메우는지 확인

#### 설정
- 해당 언어 프롬프트 뒤, 추론 시작에 영어 문제 이해 입력
- 이어서 bf16 또는 양자화 모델로 풀이와 응답 생성
- 해당 언어 프롬프트만 넣었을 때와 비교
- 영어 문제 이해가 있는 문제만 사용

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_en_understanding_swap.png)
- bf16과 각 양자화 방법에서, 영어 문제 이해 교체가 영어와의 차이를 절반 넘게 메움, 따라서 영어와의 차이는 풀이보다 문제 이해에서 더 많이 발생
- **mxfp4에서, 영어 문제 이해 교체가 영어와의 손실 차이는 대부분 메움**, 따라서 영어와의 손실 차이는 대부분 문제 이해에서 발생
- mxfp4 스와힐리어에서, 영어와의 손실 차이가 일부 남음
- gptq, awq, nf4에서, 영어와의 손실 차이가 작아 seed 노이즈와 구분하기 어려움

## 영어 질문 삽입
[영어 문제 이해 교체](#영어-문제-이해-교체)에서 영어와의 차이는 풀이보다 문제 이해에서 더 많이 발생, 따라서 bf16 문제 이해 없이 영어 질문만 넣어도 영어와의 차이를 메우는지 확인

#### 설정
- 해당 언어 프롬프트 뒤, 추론 시작에 같은 문제의 영어 질문 입력
- 이어서 bf16 또는 양자화 모델로 추론과 응답 생성
- 해당 언어 프롬프트만 넣었을 때와 비교

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_en_question_insert.png)
- bf16과 각 양자화 방법에서, 영어 질문 삽입이 영어와의 차이를 대부분 메움, 따라서 영어와의 차이는 대부분 해당 언어 질문을 이해하는 데서 발생
- **mxfp4에서, 영어 질문 삽입이 영어와의 손실 차이도 대부분 메움**, 따라서 영어와의 손실 차이도 대부분 해당 언어 질문을 이해하는 데서 발생
- mxfp4 스와힐리어에서, 영어와의 손실 차이가 절반쯤 남음
- gptq, awq, nf4 텔루구어에서도, 영어 질문 삽입이 영어와의 손실 차이를 대부분 메움
- gptq, awq, nf4의 중국어, 스와힐리어, 일본어에서, 영어와의 손실 차이가 작아 seed 노이즈와 구분하기 어려움

## 중간 결론
- gemma mgsm에서, 손실은 언어마다 다르고 스와힐리어와 텔루구어가 가장 많이 잃음
- 응답이 아니라 추론이 정확도를 결정
- 영어와의 차이는 풀이보다 문제 이해에서 더 많이 발생
- mxfp4에서, 영어 문제 이해 교체와 영어 질문 삽입이 영어와의 손실 차이를 대부분 메움, 따라서 영어와의 손실 차이는 대부분 해당 언어 질문을 이해하는 데서 발생
- mxfp4 스와힐리어에서, 영어와의 손실 차이가 일부 남음

## 영어 문제 이해 학습
[영어 질문 삽입](#영어-질문-삽입)에서 영어와의 차이는 대부분 해당 언어 질문을 이해하는 데서 발생, 따라서 학습으로 문제 이해를 영어처럼 바꿀 수 있는지 확인

#### 설정
- ce: cross entropy
- kl: 영어 프롬프트를 넣은 bf16의 다음 token 예측과의 kl
- on policy: 학습 중인 모델이 해당 언어 프롬프트에 직접 쓴 텍스트
- on policy는 kl만, ce는 모델이 쓴 텍스트를 그대로 다시 학습하기 때문
- 양자화 없이 bf16 weight에서 학습, 양자화 모델을 학습하기 전에 먼저 확인하기 때문
- gsm8k train 질문을 [mcot](https://huggingface.co/datasets/laihuiyuan/mCoT-MATH)에서 중국어, 스와힐리어, 일본어, 텔루구어로 가져옴
- adamw lr 1e-5, weight decay 없음, cosine schedule, warmup 5%, batch size 8, 1 epoch
- on policy는 학습 중인 모델로 해당 언어 프롬프트에 추론 생성, 문제 이해는 512 token, 추론은 2048 token까지
- 해당 언어 프롬프트 뒤, 추론 시작에 학습할 텍스트 입력
- kl은 같은 텍스트를 영어 프롬프트 뒤에도 넣어 bf16에 입력
- 그 텍스트 위치에서만 다음 token 예측을 ce 또는 kl로 학습
- 해당 언어 프롬프트에서 bf16과 비교
- bf16 seed 0이 영어와 해당 언어에서 모두 풀고 두 추론에 모두 문제 이해가 있는 문제 중 앞 4000개, 언어마다 약 1000개
- on policy는 문제 이해가 없는 추론 제외

#### 요약
![](../figures/gemma-4-E2B-it/mgsm_finetune.png)
- **어느 조합도 bf16을 눈에 띄게 넘지 못함**
