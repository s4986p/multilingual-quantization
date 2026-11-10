from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import GENERATION_CONFIGS, continue_vllm, get_prompt_ids, get_think_ids, load_vllm
from utils.paths import make_bf16_path, make_target_path

REASONING_METHODS = ["bf16", "quantized"]


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to compare with bf16
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # model that writes response
    response_method: Literal["bf16", "quantized"] = "bf16"


def main() -> None:
    config = parse(Config)
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    entry_name = "reasoning_swap/mgsm"
    file_names = {reasoning_method: f"reasoning_method={reasoning_method},response_method={config.response_method},max_generation_len={config.max_generation_len},seed={config.seed}" for reasoning_method in REASONING_METHODS}
    missing_reasoning_methods = [reasoning_method for reasoning_method in REASONING_METHODS if not result_exists(entry_name, target, file_names[reasoning_method])]
    if not missing_reasoning_methods:
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)
    source_file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    source_results = {"bf16": load_result("mgsm", make_bf16_path(config.model_name), source_file_name), "quantized": load_result("mgsm", target, source_file_name)}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model_path = config.model_name if config.response_method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)
    think_end = GENERATION_CONFIGS[config.model_name]["think_end"]
    think_end_id = tokenizer.convert_tokens_to_ids(think_end)

    for reasoning_method in missing_reasoning_methods:
        # 1) take whole reasoning of problems that reached response
        problem_indices = {}
        input_ids = []
        for language in languages:
            problem_indices[language] = []
            source_result = source_results[reasoning_method][language]
            for problem_index, ((prompt, _), think, response) in enumerate(zip(problems[language], source_result["thinks"], source_result["responses"])):
                if response is not None:
                    problem_indices[language].append(problem_index)
                    input_ids.append(torch.cat([get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, think + think_end)]))

        # 2) write response and score it
        log_time = time.perf_counter()
        outputs = continue_vllm(model, config.model_name, input_ids, config.max_generation_len)
        # think_end is in input, but split_outputs looks for it in output
        outputs = [SimpleNamespace(token_ids=[think_end_id, *output.token_ids], finish_reason=output.finish_reason) for output in outputs]
        result = {}
        start = 0
        for language in languages:
            end = start + len(problem_indices[language])
            answers = [problems[language][problem_index][1] for problem_index in problem_indices[language]]
            result[language] = {**evaluate_mgsm(outputs[start:end], answers, language, tokenizer, config.model_name), "problem_indices": problem_indices[language]}
            start = end
            print(f"{target} {reasoning_method} {language} accuracy {result[language]['accuracy']:.2%} used {len(answers)}/{len(problems[language])}")
        print(f"{target} {reasoning_method} time {format_duration(time.perf_counter() - log_time)}")
        save_result(entry_name, target, file_names[reasoning_method], result)


if __name__ == "__main__":
    main()
