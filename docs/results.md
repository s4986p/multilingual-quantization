# results

## common setup
![](diagrams/terms.png)
all experiments use this setup unless noted
- qwen3.5-2b, gemma-4-e2b-it, thinking mode on
- bf16 baseline
- 4 bit, weights of linear layers only
- gptq, awq group size 128, nf4 group size 64, mxfp4 group size 32
- gptq, awq calibrating on 128 en wikipedia chunks, 2048 tokens each
- 5 languages (en, zh, sw, ja, te), prompts in each language, zero shot
- mgsm 250, mmlu-prox-lite 588 problems per language
- mgsm instructions from [simple-evals](https://github.com/openai/simple-evals/blob/main/mgsm_eval.py)
- mmlu-prox-lite instructions from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_prox/lang_libs.py)
- vllm 0.30.0, temperature 1.0, top-p 0.95
- qwen top-k 20, presence penalty 1.5, max generation len 32768
- gemma top-k 64, presence penalty 0, max generation len 16384
- mean of seed 0, 1, 2 unless noted
- mgsm only after loss section, because on mmlu-prox-lite, no language clearly loses more than en
- gemma only after loss section, because it almost never loops
- mxfp4 only after teacher forcing section, because it loses most
- prompt means instruction, question
- instruction means text before question
- question means text of problem
- reasoning means text in thinking mode
- understanding means part at start of reasoning that writes goal, given information in en
- solving means reasoning after understanding
- response means text after reasoning
- score only response

## loss
check if language matters, as paper says

#### setup
- on mmlu-prox-lite, seed 0, 1 for bf16, seed 0 for quantized, because losses look similar across languages
- truncated means it hit max generation len and got wrong answer
- looped means the 50 characters from 200 to 150 before end of reasoning or response appear 5 or more times earlier
- use all problems, because qwen bf16 solved too few on sw and te

#### summary
![](../figures/Qwen3.5-2B/mgsm_loss.png)
![](../figures/gemma-4-E2B-it/mgsm_loss.png)
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
- on mgsm, awq loses least, and mxfp4 loses most
- on mgsm, gemma loses less than qwen
- on qwen mgsm, quantized truncate a lot, mostly from loops
- on qwen bf16 mgsm, sw and te are weak
- on gemma mgsm, models almost never truncate, so they lose through wrong answers
- **on gemma mgsm, sw and te lose most**
- on mmlu-prox-lite, no language clearly loses more than en

## calibration
check if calibration language matters

#### setup
- awq only, because intuitively awq relies more on calibration
- seed 0 only
- all means en, zh, sw, ja, te with about 26 chunks each
- calibrating on en, sw, te, all

#### summary
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
- **on sw and te, calibrating on their own language does not clearly help**
- calibrating on all does not clearly help

## teacher forcing
check if prediction changes differ by language

#### setup
![](diagrams/teacher_forcing.png)
- kl by reasoning progress is awq only
- take bf16 seed 0 reasoning on problems solved by bf16 seed 0 and 1
- give that reasoning to bf16 and quantized
- compare their next token predictions with kl at each reasoning position

#### summary
![](../figures/gemma-4-E2B-it/mgsm_teacher_forcing.png)
- kl matches loss between languages, but not clearly between methods
- **on sw and te, kl gap with en is largest at start of reasoning**

## reasoning swap
[teacher forcing](#teacher-forcing) shows kl gap with en is largest at start of reasoning, so check if reasoning or response decides accuracy

#### setup
![](diagrams/reasoning_swap.png)
- take saved reasoning of bf16 or quantized on problems that reached response, correct or not
- write response after it with bf16 or quantized, so four cases

#### summary
![](../figures/gemma-4-E2B-it/mgsm_reasoning_swap.png)
- **reasoning decides accuracy, not response**

## error types
[reasoning swap](#reasoning-swap) shows reasoning decides accuracy, not response, so check how and where reasoning goes wrong

#### setup
- wrong reasoning means quantized reasoning on problems it got wrong
- fact means a number, unit, word, name is copied wrong
- relation means a relation between quantities is read wrong
- target means it solves for a different quantity than asked
- doubt means a given condition is called missing, then dropped
- arithmetic means setup is right, but computation is wrong
- other means none of these
- take wrong reasoning on problems solved by all bf16 seeds
- give wrong reasoning and bf16 reasoning to qwen3.5-27b
- qwen3.5-27b picks first wrong line and error type

#### summary
![](../figures/gemma-4-E2B-it/mgsm_error_types.png)
- **on sw and te, errors are mostly fact and relation, not arithmetic**, so they may read question wrong
- first errors mostly come in first 60% of reasoning
- fact errors mostly come in first 20% of reasoning
- on claude opus 5.5 as judge, types differ in places, but overall pattern looks similar

## en question insert
[error types](#error-types) shows sw and te may read question wrong, so check if en question closes gap with en

#### setup
![](diagrams/en_question_insert.png)
- native instruction means instruction in each language
- en question means mgsm question of same problem in en
- en question insert means en question put at start of reasoning, after prompt with native instruction
- continue solving and response after en question with bf16 or quantized
- compare en question insert with native instruction

#### summary
![](../figures/gemma-4-E2B-it/mgsm_en_question_insert.png)
- **en question insert closes most of gap with en**, so gap with en mostly comes from reading question
- after en question, model mostly skips its own understanding

## en understanding swap
[en question insert](#en-question-insert) shows gap with en mostly comes from reading question, so check if it comes from understanding or from solving

#### setup
![](diagrams/en_understanding_swap.png)
- en understanding means understanding of bf16 reasoning on en prompt, up to first step whose title says calculate, determine, compute, solve, perform, set up, formulate
- en understanding swap means en understanding put at start of reasoning, after prompt with native instruction
- continue solving and response after en understanding with bf16 or quantized
- compare en understanding swap with native instruction
- use problems that have en understanding

#### summary
![](../figures/gemma-4-E2B-it/mgsm_en_understanding_swap.png)
- **en understanding swap also closes most of gap with en**, so gap with en mostly comes from understanding, not from solving

## qat
[en understanding swap](#en-understanding-swap) shows gap with en mostly comes from understanding, not from solving, so check if qat on en understanding closes gap with en

#### setup
- native understanding means understanding of bf16 reasoning on prompt in each language
- qat en means qat on en understanding
- qat native means qat on native understanding
- use gsm8k train questions from [mcot](https://huggingface.co/datasets/laihuiyuan/mCoT-MATH)
- put en or native understanding at start of reasoning, after prompt with native instruction in zh, sw, ja and te
- train next token predictions on that understanding only, with cross entropy
- train from bf16 weights with mxfp4 rounding
- adamw lr 1e-5, no weight decay, cosine schedule, 5% warmup, batch size 8, 1 epoch
- compare qat en and qat native with bf16 and mxfp4 on native instruction
- use gsm8k problems solved by bf16 seed 0 in en and in that language, with both understandings

#### summary
![](../figures/gemma-4-E2B-it/mgsm_qat.png)
- qat en and qat native both recover most of mxfp4 loss
- **qat en does not beat qat native**, so training on en understanding may not close gap with en
