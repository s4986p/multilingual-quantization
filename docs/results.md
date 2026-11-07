# results

## summary
- **on mgsm, some languages lose more than others, but not on mmlu-prox-lite**
- awq loses least, but rtn, mxfp4 lose most
- qwen mostly loses through loops, but gemma loses through wrong answers, so later experiments use mostly gemma
- calibrating on a language does not clearly help, so gap may not come from calibrating only on en
- kl partly matches loss on gemma, but not on qwen
- gemma sw, te kl gap is largest at start of reasoning with awq
- gemma sw loses more through solving, but te loses through reading, solving with mxfp4
- gemma sw, te lose most through reasoning, not response with mxfp4
- **gemma sw, te lose mostly through fact, relation, so they mostly copy, read question wrong with mxfp4**
- gemma sw, te lose about equally through translation, solving, so translation is not the only cause with mxfp4
- gemma sw, te get higher accuracy when translation, solving are split than at once with mxfp4
- **gemma sw, te get higher accuracy with en instruction, but less than split, so quantized model loses more with more non-en text with mxfp4**
- gemma sw, te question have higher hidden error than en, but zh, ja do not, so quantized model may lose with sw, te text, not with all non-en text with mxfp4
- **gemma sw, te question hidden error gap with en starts at layer 3, so they may already copy, read question wrong before reasoning with mxfp4**
- **gemma sw, te recover with bf16 front 15, but not with bf16 front 3, so hidden of layers 4-15 matters most for sw, te with mxfp4**
- we do not know yet why hidden of layers 4-15 matters most

## common setting
all experiments use these settings unless noted

#### terms
- prompt means instruction, question
- instruction means text before question (e.g. "solve this math problem ...")
- question means text of each problem
- reasoning means text in thinking mode
- response means text after reasoning

#### model
- qwen3.5-2b, gemma-4-e2b-it
- thinking mode on
- bf16 baseline

#### quantization
- 4 bit, weights of linear layers only
- rtn (group size 128)
- gptq (group size 128, block size 128, damp ratio 0.01)
- awq (group size 128, 20 grid steps, max shrink 0.5)
- nf4 (group size 64)
- mxfp4 (group size 32)

#### calibration
- gptq, awq only
- 128 en wikipedia chunks, 2048 tokens each

#### benchmark
- 5 languages (en, zh, sw, ja, te)
- mgsm 250 problems per language
- mmlu-prox-lite 588 problems per language
- zero shot
- prompts in each language
- mgsm prompts from [simple-evals](https://github.com/openai/simple-evals/blob/main/mgsm_eval.py)
- mmlu-prox-lite prompts from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_prox/lang_libs.py)

#### generation
- vllm 0.30.0, temperature 1.0, top-p 0.95
- mgsm seed 0, 1, 2
- mmlu-prox-lite seed 0, 1 for bf16, seed 0 for quantized models
- qwen3.5-2b top-k 20, presence penalty 1.5, max generation len 32768
- gemma-4-e2b-it top-k 64, presence penalty 0, max generation len 16384

#### scoring
- use only text after reasoning
- take last number (mgsm) or letter (mmlu-prox-lite) after answer prefix
- if there is no prefix, take last one in response
- loop means a 50 character piece near end of reasoning or response appears 5 or more times earlier

## loss
check if other languages lose more than en, as paper says

#### setting
- compare rtn, gptq, awq, nf4, mxfp4 with bf16
- use all problems, because qwen bf16 solved too few on sw, te
- panels are truncated, wrong, looped
- truncated means it hit max generation len and got wrong answer
- wrong means it finished and got wrong answer
- first bar is bf16, and ± shows seed noise
- quantized bars are mean of seeds

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_loss.png)
- most truncation comes from loops
- zh loses most (-14.4%p, -35.7%p vs en -4.7%p, -18.4%p) with nf4, mxfp4
- sw, ja lose more than en, zh (-10.3%p, -8.9%p vs -4.3%p, -4.0%p) with awq
- te has little left to lose, because bf16 is weak on te (16.1%)

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_loss.png)
- gemma almost never truncates, so it loses through wrong answers
- sw, te lose most (-4.8%p to -14.7%p vs en -0.7%p to -4.4%p)

#### mmlu-prox-lite
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
- no language clearly loses more than en

#### common summary
- gemma loses less than qwen
- awq loses least (-3.2%p to -6.5%p on mgsm)
- rtn, mxfp4 lose most (-8.1%p to -27.8%p on mgsm)

## calibration
check if calibration language matters, because calibration uses only en

#### setting
- awq only, gptq not tested
- mgsm only, because only mgsm shows a gap with en
- calibrate on en, sw, te, all
- all means en, zh, sw, ja, te with about 26 chunks each
- panels are same as loss
- seed 0 only, so small differences may be noise

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_calibration.png)
- calibrating on sw, te does not clearly help sw, te (-11.9%p, -4.1%p vs en -10.3%p, -4.9%p)
- calibrating on all may help mean loss (-4.7%p vs en -7.2%p)

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
- calibrating on sw, te does not clearly help sw, te (-6.8%p, -3.2%p vs en -3.6%p, -4.4%p)
- calibrating on all does not help mean loss (-3.0%p vs en -2.8%p)

## teacher forcing
check if prediction changes differ by language

#### setting
- give bf16 reasoning to bf16, quantized models
- compare their next token predictions with kl at each position
- use bf16 seed 0 reasoning on problems solved by bf16 seed 0, 1
- compare reasoning only, not response
- progress panel is awq only

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_teacher_forcing.png)
- kl does not match loss (zh 0.084 vs en 0.083, -35.7%p vs -18.4%p with mxfp4)

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_teacher_forcing.png)
- kl matches which languages lose more, but not size of loss (te 0.143 vs 0.142, -5.2%p vs -14.7%p with gptq, mxfp4)
- sw, te have highest kl (0.094, 0.090 vs en 0.065) with awq
- sw, te kl gap is largest in first 20% of reasoning (0.144, 0.132 vs en 0.077) with awq

## prefix swap
check if quantization hurts reading or solving, because kl gap is largest at start of reasoning

#### setting
- gemma only, because its loss is not from loops
- awq, mxfp4 only, because awq loses least, but mxfp4 has largest gap with en
- cut saved reasoning after line with last question number
- reading means text before cut
- solving means rest of reasoning, response
- continue from cut with bf16 or quantized
- bars show accuracy on problems cut in all four cases

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_prefix_swap.png)
- sw loses more through solving (-8.4%p vs reading -3.1%p) with mxfp4
- te loses through reading, solving (-4.9%p, -5.7%p) with mxfp4
- reading, solving do not clearly differ (sw -1.8%p vs -3.8%p, te -0.3%p vs -2.2%p) with awq

## reasoning swap
check if quantization hurts reasoning or response, because solving mixes reasoning, response

#### setting
- gemma, mxfp4 only, because mxfp4 has largest gap with en
- use saved reasoning of bf16 or quantized
- write response with bf16 or quantized

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_reasoning_swap.png)
- sw, te lose most through reasoning (-13.1%p, -13.6%p vs en -4.0%p) with mxfp4
- response does not clearly lose (-0.7%p to +0.4%p) with mxfp4

## error types
check how reasoning goes wrong, because sw, te lose most through reasoning with mxfp4

#### setting
- gemma, mxfp4 only
- use wrong reasoning on problems solved by bf16 seed 0, 1, 2
- give bf16 reasoning, wrong reasoning to qwen3.5-27b, and get one error type
- fact means a number, word or name is copied wrong
- relation means words like more than, per, half of are read wrong
- bars show ratio of these problems that mxfp4 gets wrong
- progress panel uses first wrong line picked by qwen3.5-27b

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_error_types.png)
- sw, te lose mostly through fact, relation, but rarely through arithmetic with mxfp4
- te errors start most often in first 20% of reasoning, but sw errors spread over first 60% with mxfp4
- qwen3.5-27b, claude opus 5.5 give different types in places, so use types, positions only as a rough guide

## translation
check if quantization hurts translating sw, te question, because sw, te lose mostly through fact, relation, which copy, read question wrong early in reasoning

#### setting
- gemma, mxfp4 only
- translate each question into en with bf16 or quantized
- give translation instruction in en, so only question is in each language
- use response as translation
- solve translation with bf16 or quantized, with en prompt
- solving means reasoning, response on translation
- en is also translated, so it shows loss from copying en
- bars show accuracy on problems translated by both bf16, quantized
- at once means quantized reasoning, quantized response of reasoning swap

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_translation.png)
- sw, te lose about equally through translation, solving (sw -5.4%p, -5.5%p, te -4.2%p, -4.7%p vs en -1.4%p, -2.8%p) with mxfp4
- sw, te get higher accuracy when translation, solving are split than at once (67.4%, 69.3% vs 59.9%, 61.2%) with mxfp4

## instruction language
check if en instruction alone raises sw, te accuracy, because sw, te get higher accuracy when split, which also uses en instruction

#### setting
- gemma, bf16, mxfp4 only
- keep question in each language, but give instruction in en
- native instruction means instruction in each language, as in mgsm
- score with en answer prefix

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_instruction_language.png)
- en instruction raises sw, te accuracy (62.9%, 66.5% vs 59.7%, 60.9%) with mxfp4
- en instruction raises sw, te less than split (62.9%, 66.5% vs 67.4%, 69.3%) with mxfp4

## hidden error
check if quantization changes non-en tokens more, because quantized model loses more with more non-en text with mxfp4

#### setting
- gemma, mxfp4 only
- use en instruction, as in instruction language
- give same prompt to bf16, quantized, without generation
- hidden error means distance between bf16, quantized hidden of each layer, divided by bf16 hidden size
- use all problems, because there is no generation
- question panel uses question tokens, and instruction panel uses instruction tokens

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_hidden_error.png)
- sw, te question have higher hidden error than en (0.233, 0.211 vs 0.195), but zh, ja do not with mxfp4
- gap with en starts at layer 3, and stays to last layers with mxfp4

## layer swap
check if front layers cause sw, te loss, because sw, te question hidden error gap with en starts at layer 3 with mxfp4

#### setting
- gemma, mxfp4 only
- use en instruction, as in instruction language
- front n means first n layers, and back means the rest
- layer swap means front from bf16 or quantized, and back from the other
- swap is done by copying front weights
- use front 15, because layers 16-35 reuse key, value of layers 1-15
- bf16, quantized bars come from instruction language

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_layer_swap.png)
- bf16 front 3 does not clearly help sw, te, but bf16 front 15 mostly does (-1.3%p, +0.8%p vs quantized -12.5%p, -10.5%p) with mxfp4
