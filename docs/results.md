# results

## summary
- on mgsm, some languages lose more than others
- on mmlu-prox-lite, no language clearly loses more than en
- qwen mostly loses through loops with rtn, mxfp4
- gemma loses through wrong answers
- calibrating on a language does not clearly help that language
- we do not know yet why some languages lose more

## common setting
all experiments use these settings unless noted

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
- 128 english wikipedia chunks, 2048 tokens each

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

## does quantization hurt non english more than english?

#### setting
- compare rtn, gptq, awq, nf4, mxfp4 with bf16
- use all problems, because qwen bf16 solved too few on sw, te
- panels are truncated, wrong, looped
- truncated means it hit max generation len and got wrong answer
- wrong means it finished and got wrong answer
- ratio is out of all problems
- first bar is bf16, and ± shows seed noise
- mgsm quantized bars are mean of seed 0, 1, 2
- mmlu-prox-lite quantized bars are seed 0

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_loss.png)
- most truncation comes from loops
- zh loses most with nf4, mxfp4 (-14.4%p, -35.7%p vs en -4.7%p, -18.4%p)
- sw, ja lose more than en, zh with awq (-10.3%p, -8.9%p vs -4.3%p, -4.0%p)
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

## does calibration language matter?

#### setting
- awq, mgsm only
- calibrate on en, sw, te, all
- all means en, zh, sw, ja, te with about 26 chunks each
- panels are same as above
- seed 0 only, so small differences could be noise

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_calibration.png)
- calibrating on all languages may help (-4.7%p vs en -7.2%p)
- calibrating on te hurts (-10.8%p vs en -7.2%p)

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_calibration.png)
- calibrating on all languages does not help (-3.0%p vs en -2.8%p)

#### common summary
- calibrating on a language does not clearly help that language

## where does quantization change predictions?

#### setting
- give bf16 reasoning to bf16, quantized models
- compare their next token predictions with kl at each position
- use bf16 seed 0 reasoning
- use mgsm problems solved by bf16 seed 0, 1
- skip response
- progress panel is awq only

#### qwen mgsm
![](../figures/Qwen3.5-2B/mgsm_teacher_forcing.png)
- kl does not match loss
- zh loses most with nf4, mxfp4 (-14.4%p, -35.7%p vs en -4.7%p, -18.4%p), but zh kl stays close to en (0.049, 0.084 vs en 0.048, 0.083)

#### gemma mgsm
![](../figures/gemma-4-E2B-it/mgsm_teacher_forcing.png)
- kl matches loss
- sw, te have highest kl with awq (0.094, 0.090 vs en 0.065)
