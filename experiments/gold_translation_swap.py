from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, get_mgsm_question, load_mgsm
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import THINK_HEADER, continue_vllm, get_prompt_ids, get_think_ids, load_vllm
from utils.paths import make_target_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to evaluate
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = "gold_translation_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)

    # 1) put gold translation at start of reasoning, after prompt in each language
    # e.g. sw prompt + "<|channel>thought\n" + "Janet's ducks lay 16 eggs per day. ..." + blank line
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    gold_translations = [get_mgsm_question(prompt) for prompt, _ in problems["en"]]
    input_ids = [torch.cat([get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, THINK_HEADER + gold_translation + "\n\n")]) for language in languages for (prompt, _), gold_translation in zip(problems[language], gold_translations)]
    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)

    # 2) continue after gold translation and score it
    log_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, config.max_generation_len)
    result = {}
    start = 0
    for language in languages:
        end = start + len(problems[language])
        result[language] = evaluate_mgsm(outputs[start:end], [answer for _, answer in problems[language]], language, tokenizer, config.model_name)
        start = end
        print(f"{target} {language} accuracy {result[language]['accuracy']:.2%} think len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result(entry_name, target, file_name, result)


if __name__ == "__main__":
    main()
