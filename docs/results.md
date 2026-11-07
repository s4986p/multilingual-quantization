# results

## common setting
all experiments use these settings unless noted

#### model
- use qwen3.5-2b and gemma-4-e2b-it
- turn on thinking mode
- use bf16 as baseline

#### quantization
- quantize only weights of linear layers to 4 bit
- use rtn (group size 128)
- use gptq (group size 128, block size 128, damp ratio 0.01)
- use awq (group size 128, 20 grid steps, max shrink 0.5)
- use nf4 (group size 64)
- use mxfp4 (group size 32)

#### calibration
- only gptq and awq use calibration
- use 128 english wikipedia chunks
- each chunk has 2048 tokens

#### benchmark
- use 5 languages (en, zh, sw, ja, te)
- mgsm has 250 problems per language
- mmlu-prox-lite has 588 problems per language
- use zero shot
- prompts are written in each language
- mgsm prompts come from [simple-evals](https://github.com/openai/simple-evals/blob/main/mgsm_eval.py)
- mmlu-prox-lite prompts come from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu_prox/lang_libs.py)

#### generation
- use vllm 0.30.0
- use temperature 1.0 and top-p 0.95
- use seed 0, 1 and 2 on mgsm
- use seed 0 and 1 for bf16 and seed 0 for quantized models on mmlu-prox-lite
- qwen3.5-2b uses top-k 20, presence penalty 1.5 and max generation len 32768
- gemma-4-e2b-it uses top-k 64, presence penalty 0 and max generation len 16384

#### scoring
- use only text after reasoning
- take last number (mgsm) or letter (mmlu-prox-lite) after answer prefix
- if there is no prefix, take last one in whole response
- loop means a 50 character piece, 150 characters before end of reasoning or response, appears 5 or more times earlier in same text

## does quantization hurt non english more than english?

#### setting
- compare rtn, gptq, awq, nf4 and mxfp4 with bf16
- use all problems, because qwen bf16 got too few right on sw and te
- show bf16 as first bar, with mean and std of its seeds
- first panel is truncated, second panel is wrong, third panel is looped
- truncated means generation hit max generation len, even after reasoning, and answer is wrong
- wrong means generation finished but answer is wrong
- all panels show ratio of all problems
- bar labels show ratio and change from bf16 mean
- on mgsm, quantized bars show mean of seed 0, 1 and 2
- on mmlu-prox-lite, quantized bars show seed 0 only

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_loss.png)

- method decides how much truncation happens
- almost all rtn, gptq and mxfp4 truncation is loops
- bf16 is weak on sw and te (43.1%, 16.1%)
- te has little left to lose, so its drop in points looks small
- with awq, sw and ja lose more than en and zh (-10.3%p, -8.9%p vs -4.3%p, -4.0%p)
- zh clearly loses most with nf4 and mxfp4 (-14.4%p, -35.7%p)
- zh truncates most with every method

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_loss.png)

- gemma almost never truncates
- sw and te lose most with every method (-4.8%p to -14.7%p, en -0.7%p to -4.4%p)

#### mmlu-prox-lite
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)

- non english does not clearly lose more than en on both models
- qwen bf16 is weak on sw and te (37.8%, 34.7%)
- gemma never truncates

#### common summary
- gemma loses less than qwen
- awq loses least on average (-3.2%p to -6.5%p on mgsm)
- rtn and mxfp4 lose most (-8.1%p to -27.8%p on mgsm)

## does calibration language matter?

#### setting
- use only awq and mgsm
- calibrate on en, sw, te and all (en, zh, sw, ja, te)
- all has 128 chunks in total, about 26 per language
- panels and bar labels are same as above
- use seed 0 only, so small differences could be noise

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_calibration.png)

- all calibration may help (-4.7%p vs -7.2%p with en)
- te calibration hurts (-10.8%p vs -7.2%p with en)
- sw and te calibration add truncation and loops on en and zh

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)

- all calibration does not help (-3.0%p vs -2.8%p with en)

#### common summary
- calibrating on a language does not clearly help that language
