from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, load_mgsm
from utils.evaluation import continue_vllm, evaluate_outputs, load_vllm, parse_mgsm_prediction
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import get_prompt_ids, get_reasoning_ids
from utils.paths import make_hub_path, make_result_name
from utils.understanding import get_understanding


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    experiment_name = "bf16_native_understanding_insert/mgsm"
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    file_names = {seed: make_result_name(config.max_generation_len, seed) for seed in config.seeds}

    # 1. seed마다 hub result에서 빠진 것 확인, 빠진 것이 없으면 종료
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(experiment_name, hub_path, file_name)]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return
    problems = load_mgsm(LANGUAGES)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. 같은 seed의 bf16이 해당 언어 prompt에 쓴 reasoning에서 understanding을 추출하고 해당 언어 prompt 뒤, reasoning 시작에 삽입
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + bf16 스와힐리어 understanding
    problem_indices = {}
    input_ids = []
    prompt_seeds = []
    for seed in missing_seeds:
        bf16_result = load_result("mgsm", make_hub_path("bf16", config.model_name), file_names[seed])
        for language in LANGUAGES:
            understandings = [get_understanding(reasoning) for reasoning in bf16_result[language]["reasonings"]]
            problem_indices[seed, language] = [problem_index for problem_index, understanding in enumerate(understandings) if understanding is not None]
            for problem_index in problem_indices[seed, language]:
                input_ids.append(torch.cat([get_prompt_ids(tokenizer, problems[language][problem_index][0]), get_reasoning_ids(tokenizer, config.model_name, understandings[problem_index])]))
                prompt_seeds.append(seed)

    # 3. vllm은 띄우는 데 오래 걸리기 때문에 understanding에 이어서 solving과 response를 모아 한 번에 생성
    model_path = config.model_name if config.method == "bf16" else download_model(hub_path)
    model = load_vllm(model_path, config.model_name)
    start_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, prompt_seeds, config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 4. outputs를 seed와 언어별로 나눠 채점하고 저장
    start = 0
    for seed in missing_seeds:
        result = {}
        for language in LANGUAGES:
            end = start + len(problem_indices[seed, language])
            answers = [problems[language][problem_index][1] for problem_index in problem_indices[seed, language]]
            result[language] = {**evaluate_outputs(outputs[start:end], answers, language, tokenizer, config.model_name, parse_mgsm_prediction), "problem_indices": problem_indices[seed, language]}
            start = end
            print(f"{hub_path} seed {seed} {language} accuracy {result[language]['accuracy']:.2%} reasoning len {mean(result[language]['reasoning_lens']):.0f} used {len(answers)} of {len(problems[language])}")
        save_result(experiment_name, hub_path, file_names[seed], result)


if __name__ == "__main__":
    main()
