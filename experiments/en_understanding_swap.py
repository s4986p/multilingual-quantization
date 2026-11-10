from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import continue_vllm, get_prompt_ids, get_think_ids, load_vllm
from utils.paths import make_bf16_path, make_target_path
from utils.understanding import get_understanding


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to evaluate
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4", "understanding_qat"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = "en_understanding_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)
    bf16_result = load_result("mgsm", make_bf16_path(config.model_name), file_name)

    # 1) use problems that have en understanding
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    understandings = [get_understanding(think) for think in bf16_result["en"]["thinks"]]
    problem_indices = [problem_index for problem_index, understanding in enumerate(understandings) if understanding is not None]

    # 2) put en understanding at start of reasoning, after prompt with native instruction
    input_ids = [torch.cat([get_prompt_ids(tokenizer, problems[language][problem_index][0]), get_think_ids(tokenizer, config.model_name, understandings[problem_index])]) for language in languages for problem_index in problem_indices]
    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)

    # 3) continue after en understanding and score it
    log_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, config.max_generation_len)
    result = {}
    start = 0
    for language in languages:
        end = start + len(problem_indices)
        answers = [problems[language][problem_index][1] for problem_index in problem_indices]
        result[language] = {**evaluate_mgsm(outputs[start:end], answers, language, tokenizer, config.model_name), "problem_indices": problem_indices}
        start = end
        print(f"{target} {language} accuracy {result[language]['accuracy']:.2%} think len {mean(result[language]['think_lens']):.0f} used {len(problem_indices)}/{len(problems[language])}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result(entry_name, target, file_name, result)


if __name__ == "__main__":
    main()
