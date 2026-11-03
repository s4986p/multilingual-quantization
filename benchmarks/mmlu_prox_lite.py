from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_mmlu_prox_lite
from utils.evaluation import evaluate_mmlu_prox_lite
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import generate_vllm, load_vllm
from utils.paths import make_target_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to evaluate
    model_name: str = "Qwen/Qwen3.5-2B"
    method: Literal["bf16", "gptq", "awq"] = "bf16"
    languages: tuple[Literal["en", "zh", "sw"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists("mmlu_prox_lite", target, file_name):
        print(f"{target} already evaluated")
        return
    problems = load_mmlu_prox_lite()

    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    log_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, [prompt for language_problems in problems.values() for prompt, _ in language_problems], config.max_generation_len)
    result = {}
    start = 0
    for language, language_problems in problems.items():
        end = start + len(language_problems)
        result[language] = evaluate_mmlu_prox_lite(outputs[start:end], [answer for _, answer in language_problems], language, tokenizer, config.model_name)
        start = end
        print(f"{target} {language} accuracy {result[language]['accuracy']:.2%} think len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result("mmlu_prox_lite", target, file_name, result)


if __name__ == "__main__":
    main()
