# 전체 실험
- 공통 설정은 [README](README.md#공통-설정)

### 언어별 손실
- [the multilingual quantization tax](https://arxiv.org/abs/2608.09941)의 언어별 손실이 thinking mode를 켜도 나타나는지 확인
- bf16과 양자화 모델로 mgsm과 mmlu-prox-lite를 풀고, 언어마다 손실 비교
#### 설정
- 잘림: 최대 생성 길이에 닿고 틀린 문제
- 반복: reasoning 또는 response 끝에서 200~150번째 글자 50개가 그 앞에 5번 이상 나오는 문제
- mmlu-prox-lite는 bf16 seed 0, 1, 양자화 모델 seed 0
#### 결과
![](figures/gemma-4-E2B-it/mgsm_loss.png)
![](figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
![](figures/Qwen3.5-2B/mgsm_loss.png)
![](figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
- gemma mgsm에서, 스와힐리어와 텔루구어가 가장 많이 잃음
- 양자화 방법 중에서는 mxfp4가 가장 많이 잃음
- mmlu-prox-lite에서는, 영어보다 눈에 띄게 더 잃는 언어가 없음
- qwen은 양자화하면 많이 잘리고, 잘린 문제는 대부분 반복
- gemma는 거의 잘리지 않음
- 따라서 이후 실험은 반복이 적은 gemma와 언어별 손실이 나타난 mgsm만 사용

### calibration 언어별 손실
- gptq와 awq는 calibration 언어에 따라 손실이 달라질 수 있음
- 해당 언어로도 calibration해 손실 비교
#### 설정
- 양자화 방법은 gptq, awq
- calibration 언어는 영어, 스와힐리어, 텔루구어, 다섯 언어 혼합
- 다섯 언어 혼합은 언어마다 chunk 25~26개
#### 결과
![](figures/gemma-4-E2B-it/mgsm_calibration.png)
- gptq와 awq 모두, 해당 언어나 다섯 언어 혼합으로 calibration해도 스와힐리어와 텔루구어 손실이 눈에 띄게 줄지 않음
- [언어별 손실](#언어별-손실)에서도 calibration이 없는 mxfp4가 가장 많이 잃음
- 따라서 calibration 언어는 손실의 원인이 아님

### reasoning 위치별 kl
- 양자화 모델의 예측이 reasoning의 어디서 bf16과 가장 달라지는지 확인
- bf16 reasoning을 bf16과 양자화 모델에 똑같이 넣고, reasoning token마다 다음 token 예측의 kl 비교
#### 설정
- reasoning 위치: 문제마다 reasoning token을 5등분한 구간
- kl은 문제마다 token 평균을 낸 뒤 문제 평균
- reasoning은 bf16 seed 0
- 문제는 모든 bf16 seed가 맞힌 문제만
- reasoning 위치별 kl은 mxfp4만
#### 결과
![](figures/gemma-4-E2B-it/mgsm_position_kl.png)
- 스와힐리어와 텔루구어에서, 영어와의 kl 차이가 reasoning 위치 0~20%에서 가장 큼
- kl이 큰 언어가 손실도 큼
- 그러나 양자화 방법끼리는 그렇지 않고, gptq는 kl이 크지만 손실은 작음
- 따라서 스와힐리어와 텔루구어의 손실은 reasoning 앞부분에서 생길 수 있음

### 오류 유형
- [reasoning 위치별 kl](#reasoning-위치별-kl)에서, 스와힐리어와 텔루구어는 영어와의 kl 차이가 reasoning 앞부분에서 가장 큼
- 그렇다면 양자화 모델이 무엇을 틀리는지, 판정 모델로 대략적인 경향 확인
- mxfp4가 틀린 reasoning을 영어 question, 정답, bf16 reasoning과 함께 판정 모델에 넣고, 처음 틀린 줄과 오류 유형을 고름
#### 설정
- 사실 오류: question의 숫자, 단위, 단어, 이름을 잘못 옮김
- 관계 오류: 양 사이의 관계를 잘못 이해함
- 목표 오류: 묻는 것과 다른 양을 구함
- 조건 무시: 주어진 조건을 없거나, 애매하거나, 상관없다고 보고 버림
- 계산 오류: 식은 맞지만 계산이 틀림
- 기타: 위에 없거나 판정을 읽지 못함
- 양자화 방법은 mxfp4
- 판정 모델은 qwen3.5-27b
- 문제는 모든 bf16 seed가 맞히고 mxfp4가 틀린 문제만
#### 결과
![](figures/gemma-4-E2B-it/mgsm_error_types.png)
- 스와힐리어와 텔루구어에서, 오류는 대부분 사실 오류와 관계 오류이고 계산 오류는 적음
- 첫 오류는 대부분 reasoning 위치 0~60%
- 사실 오류는 대부분 reasoning 위치 0~20%
- 따라서 question을 잘못 이해해서 틀린 것일 수 있음

### understanding 삽입
- [reasoning 위치별 kl](#reasoning-위치별-kl)과 [오류 유형](#오류-유형)에서, 양자화 모델은 reasoning 앞부분의 question 이해에서 bf16과 어긋남
- 생성된 reasoning을 보면, gemma는 앞부분에 question을 정리하는 understanding 단계가 있음
- 그렇다면 understanding만 bf16 것으로 삽입해 손실이 회복되는지 확인
#### 설정
- understanding은 같은 seed의 bf16이 해당 언어 prompt에 쓴 것
- understanding 구분: 제목에 calculate, determine, compute, solve, perform, set up, formulate가 처음 나오는 단계 앞까지
- bf16은 자기 understanding을 삽입해도 결과가 거의 같아서, 삽입 없는 bf16을 기준으로 사용
- 문제는 해당 언어와 영어 bf16 reasoning 모두에 understanding이 있는 문제만
- 문제 수는 seed마다 달라 그림에는 seed 평균
#### 결과
![](figures/gemma-4-E2B-it/mgsm_understanding_insert.png)
- mxfp4에서, understanding만 삽입해도 모든 언어에서 손실이 절반 넘게 줄어듦
- 스와힐리어와 텔루구어가 영어보다 더 잃는 정도도 크게 줄어듦
- gptq, awq, nf4는 손실이 작아 차이를 보기 어려움
- [reasoning 앞부분 길이별 삽입](#reasoning-앞부분-길이별-삽입)에서도, mxfp4 손실은 understanding 길이까지 크게 줄고 그 뒤로는 거의 그대로
- 따라서 손실은 solving보다 주로 understanding에서 생김

### reasoning 앞부분 길이별 삽입
- [understanding 삽입](#understanding-삽입)에서, mxfp4는 understanding만 삽입해도 손실이 절반 넘게 줄어듦
- 그러나 understanding 내용 덕분인지, 단지 bf16 reasoning 앞부분을 삽입한 덕분인지 모름
- bf16 reasoning을 understanding 길이의 여러 배에서 잘라 삽입하고 손실 비교
#### 설정
- reasoning은 같은 seed의 bf16이 해당 언어 prompt에 쓴 것
- 자르는 길이는 understanding token 수의 0.5, 0.75, 1, 1.5, 2배
- 문제는 reasoning이 understanding token 수의 2배 이상인 문제만
- 문제 수는 seed마다 달라 그림에는 seed 평균
#### 결과
![](figures/gemma-4-E2B-it/mgsm_reasoning_prefix_insert.png)
- mxfp4에서, 손실이 1배까지 크게 줄고 그 뒤로는 거의 그대로
- gptq, awq, nf4는 손실이 작아 차이를 보기 어려움
- 따라서 손실이 줄어든 것은 길이보다 understanding 내용 덕분

### 영어 understanding 삽입
- [understanding 삽입](#understanding-삽입)에서, bf16 understanding만 삽입해도 손실이 크게 회복됨
- 그렇다면 해당 언어 question을 영어처럼 곧이곧대로 이해한 understanding을 삽입하면, 영어만큼 회복되는지 확인
- bf16이 영어 prompt에 쓴 understanding을 bf16과 양자화 모델에 삽입하고 손실 비교
#### 설정
- understanding은 같은 seed의 bf16이 영어 prompt에 쓴 것
- bf16도 자기와 다른 understanding을 받으므로 bf16에도 삽입
- 문제는 understanding 삽입과 같음
- 문제 수는 seed마다 달라 그림에는 seed 평균
#### 결과
![](figures/gemma-4-E2B-it/mgsm_en_understanding_insert.png)
- 모든 양자화 방법에서, 영어 외 모든 언어가 bf16을 넘음
- bf16도 영어 외 모든 언어에서 크게 오름
- 그러나 영어 외 언어는 여전히 영어 정확도에 못 미침
- 따라서 영어처럼 이해하면 bf16을 넘을 만큼 회복되지만, understanding만으로는 영어만큼 회복되지 않음

### 영어 question 삽입
- [영어 understanding 삽입](#영어-understanding-삽입)에서, understanding만으로는 영어만큼 회복되지 않음
- 그렇다면 [why do multilingual reasoning gaps emerge in reasoning language models](https://arxiv.org/abs/2510.27269)처럼 영어 question 자체를 삽입하면, 양자화 모델도 더 회복되는지 확인
- 같은 문제의 영어 question을 bf16과 양자화 모델에 삽입하고 손실 비교
#### 설정
- 해당 논문과 달리 문장으로 감싸지 않고 영어 question만 삽입
- 문제는 모든 문제
#### 결과
![](figures/gemma-4-E2B-it/mgsm_en_question_insert.png)
- 모든 양자화 방법에서, 영어 외 모든 언어가 bf16을 넘음
- 영어 understanding 삽입보다 대부분 더 오름
- bf16은 해당 논문처럼 영어 외 언어가 영어 정확도에 가까워짐
- 그러나 양자화 모델은 아직 영어 정확도에 못 미침
- 따라서 양자화 모델도 영어 question을 받으면 해당 언어에서 bf16을 넘음

### bf16 번역 삽입
- [영어 question 삽입](#영어-question-삽입)에서, 양자화 모델도 영어 question을 받으면 bf16을 넘음
- 그러나 실제로는 영어 question이 없으므로, finetune 없이 bf16이 먼저 번역해 삽입하는 것으로 충분한지 확인
- bf16이 해당 언어 question을 영어로 번역하고, 그 번역을 bf16과 mxfp4에 삽입해 손실 비교
#### 설정
- 번역은 bf16이 thinking mode 없이 "Translate the following question into English. Output only the translation."으로 생성
- 번역은 seed 0, 최대 생성 길이 512
- 영어는 번역할 것이 없어 영어 question 그대로 삽입
- 양자화 방법은 mxfp4
#### 결과
![](figures/gemma-4-E2B-it/mgsm_bf16_translation_insert.png)
- bf16은 영어 question 삽입보다 훨씬 적게 오름
- 텔루구어는 bf16에서 거의 오르지 않음
- mxfp4는 손실이 줄지만 bf16을 넘지 못함
- 따라서 번역해 삽입하는 것으로는 부족하고, 모델이 스스로 영어처럼 이해하도록 finetune이 필요함

### 영어 understanding finetune

### 영어 question finetune

### 영어 question과 reasoning finetune

### qat
