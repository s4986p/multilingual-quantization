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
- use seed 0, 1 and 2 for bf16 on mgsm
- use seed 0 and 1 for bf16 on mmlu-prox-lite
- use seed 0 for quantized models
- qwen3.5-2b uses top-k 20, presence penalty 1.5 and max generation len 32768
- gemma-4-e2b-it uses top-k 64, presence penalty 0 and max generation len 16384

#### scoring
- use only text after reasoning
- take last number (mgsm) or letter (mmlu-prox-lite) after answer prefix
- if there is no prefix, take last one in whole response
- loop means a 50 character piece, 150 characters before end, appears 5 or more times earlier

## does quantization hurt non english more than english?

![](../figures/gemma-4-E2B-it/mgsm_loss.png)
![](../figures/gemma-4-E2B-it/mmlu_prox_lite_loss.png)
![](../figures/Qwen3.5-2B/mgsm_loss.png)
![](../figures/Qwen3.5-2B/mmlu_prox_lite_loss.png)

#### setting
- compare rtn, gptq, awq, nf4 and mxfp4 with bf16
- use all problems, because qwen bf16 got too few right on sw and te
- show bf16 as first bar, with mean and std of its seeds
- first panel is truncated, second panel is wrong, third panel is looped
- wrong means reasoning finished but answer is wrong
- all panels show ratio of all problems
- bar labels show ratio and change from bf16 mean

#### gemma summary
- gemma loses less than qwen
- gemma almost never truncates
- with awq on mgsm, sw and en differ by only 2.4p, which could be noise

#### qwen summary
- method decides how much truncation happens
- most rtn and mxfp4 truncation comes from loops
- bf16 is weak on sw and te on mgsm (43.1%, 16.1%)
- with awq on mgsm, en and zh drop less than sw and ja
- with awq on mgsm, ja loses as much as sw (-10.9p, -10.3p), even though bf16 does much better on ja (77.7%, 43.1%)

#### common summary
- awq loses least on average (-2.8p to -7.2p)
- rtn and mxfp4 lose most
- on mmlu-prox-lite, non english does not clearly lose more than en

## does calibration language matter?

![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
![](../figures/Qwen3.5-2B/mgsm_calibration.png)

#### setting
- use only awq and mgsm
- calibrate on en, sw, te and all (en, zh, sw, ja, te)
- all has 128 chunks in total, about 26 per language
- panels and bar labels are same as above

#### gemma summary
- all calibration does not help (-3.0p vs -2.8p with en)

#### qwen summary
- all calibration helps (-4.7p vs -7.2p with en)
- sw and te calibration add truncation and loops on en and zh

#### common summary
- calibrating on a language does not clearly help that language
