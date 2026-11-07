from dataclasses import dataclass
import re
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

READ_METHODS = ["bf16", "quantized"]
NUMBER_PATTERN = r"\d[\d,]*(?:\.\d+)?"


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to compare with bf16
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "awq"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # model that continues reasoning and writes response
    solve_method: Literal["bf16", "quantized"] = "bf16"


def get_read_end(
    prompt: str,
    think: str,
) -> int | None:
    numbers = {number.replace(",", "") for number in re.findall(NUMBER_PATTERN, prompt)}
    if not numbers:
        return None
    for match in re.finditer(NUMBER_PATTERN, think):
        numbers.discard(match.group().replace(",", ""))
        if not numbers:
            line_end = think.find("\n", match.end())
            return len(think) if line_end == -1 else line_end
    return None


def main() -> None:
    config = parse(Config)
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    entry_name = "prefix_swap/mgsm"
    file_names = {read_method: f"read_method={read_method},solve_method={config.solve_method},max_generation_len={config.max_generation_len},seed={config.seed}" for read_method in READ_METHODS}
    missing_read_methods = [read_method for read_method in READ_METHODS if not result_exists(entry_name, target, file_names[read_method])]
    if not missing_read_methods:
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)
    source_file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    source_results = {"bf16": load_result("mgsm", make_bf16_path(config.model_name), source_file_name), "quantized": load_result("mgsm", target, source_file_name)}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_path = config.model_name if config.solve_method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)

    for read_method in missing_read_methods:
        # 1) take text up to line of last question number as reading
        problem_indices = {}
        input_ids = []
        for language in languages:
            problem_indices[language] = []
            for problem_index, ((prompt, _), think) in enumerate(zip(problems[language], source_results[read_method][language]["thinks"])):
                read_end = get_read_end(prompt, think)
                if read_end is not None:
                    problem_indices[language].append(problem_index)
                    input_ids.append(torch.cat([get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, think[:read_end])]))

        # 2) continue from reading as solving and score it
        log_time = time.perf_counter()
        outputs = continue_vllm(model, config.model_name, input_ids, config.max_generation_len)
        result = {}
        start = 0
        for language in languages:
            end = start + len(problem_indices[language])
            answers = [problems[language][problem_index][1] for problem_index in problem_indices[language]]
            result[language] = {**evaluate_mgsm(outputs[start:end], answers, language, tokenizer, config.model_name), "problem_indices": problem_indices[language]}
            start = end
            print(f"{target} {read_method} {language} accuracy {result[language]['accuracy']:.2%} used {len(answers)}/{len(problems[language])}")
        print(f"{target} {read_method} time {format_duration(time.perf_counter() - log_time)}")
        save_result(entry_name, target, file_names[read_method], result)


if __name__ == "__main__":
    main()
