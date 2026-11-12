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
from utils.generation import continue_vllm, load_vllm
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import GENERATION_CONFIGS, get_prompt_ids, get_think_ids
from utils.paths import make_target_path, make_uncalibrated_path

REASONING_METHODS = ["bf16", "quantized"]


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # bf16과 비교할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # 응답을 쓸 모델
    response_method: Literal["bf16", "quantized"] = "bf16"


def main() -> None:
    config = parse(Config)
    script_name = "reasoning_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    source_targets = {"bf16": make_uncalibrated_path("bf16", config.model_name), "quantized": target}
    file_names = {(seed, reasoning_method): f"reasoning_method={reasoning_method},response_method={config.response_method},max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds for reasoning_method in REASONING_METHODS}

    # 1. seed와 추론 방법마다 hub 결과에서 빠진 것 확인
    missing_cases = [case for case, file_name in file_names.items() if not result_exists(script_name, target, file_name)]
    if not missing_cases:
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    think_end = GENERATION_CONFIGS[config.model_name]["think_end"]

    # 2. 응답에 닿은 문제에서 저장된 추론 추출
    problem_indices = {}
    input_ids = []
    prompt_seeds = []
    for seed, reasoning_method in missing_cases:
        source_result = load_result("mgsm", source_targets[reasoning_method], f"max_generation_len={config.max_generation_len},seed={seed}")
        for language in MGSM_PROMPTS:
            problem_indices[seed, reasoning_method, language] = []
            for problem_index, ((prompt, _), think, response) in enumerate(zip(problems[language], source_result[language]["thinks"], source_result[language]["responses"])):
                if response is not None:
                    problem_indices[seed, reasoning_method, language].append(problem_index)
                    input_ids.append(torch.cat([get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, think + think_end)]))
                    prompt_seeds.append(seed)

    # 3. 그 추론에 이어서 응답을 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
    model_path = config.model_name if config.response_method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name)
    start_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, prompt_seeds, config.max_generation_len)
    print(f"{target} time {format_duration(time.perf_counter() - start_time)}")
    # 출력 앞에 추론 끝 token 추가, split_outputs가 그 token으로 추론과 응답을 나누기 때문
    # 예시) 응답 → "<channel|>" + 응답
    think_end_id = tokenizer.convert_tokens_to_ids(think_end)
    outputs = [SimpleNamespace(token_ids=[think_end_id, *output.token_ids], finish_reason=output.finish_reason) for output in outputs]

    # 4. seed와 추론 방법마다 채점하고 저장
    start = 0
    for seed, reasoning_method in missing_cases:
        result = {}
        for language in MGSM_PROMPTS:
            language_problem_indices = problem_indices[seed, reasoning_method, language]
            end = start + len(language_problem_indices)
            answers = [problems[language][problem_index][1] for problem_index in language_problem_indices]
            result[language] = {**evaluate_mgsm(outputs[start:end], answers, language, tokenizer, config.model_name), "problem_indices": language_problem_indices}
            start = end
            print(f"{target} seed {seed} {reasoning_method} {language} accuracy {result[language]['accuracy']:.2%} used {len(answers)} of {len(problems[language])}")
        save_result(script_name, target, file_names[seed, reasoning_method], result)


if __name__ == "__main__":
    main()
