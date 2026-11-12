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
from utils.generation import continue_vllm, load_vllm
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import get_prompt_ids, get_think_ids
from utils.paths import make_target_path, make_uncalibrated_path
from utils.understanding import get_understanding


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    script_name = "en_understanding_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    file_names = {seed: f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds}

    # 1. seed마다 hub 결과에서 빠진 것 확인
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(script_name, target, file_name)]
    if not missing_seeds:
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. bf16 영어 추론에서 영어 문제 이해를 추출하고 해당 언어 프롬프트 뒤, 추론 시작에 입력
    problem_indices = {}
    input_ids = []
    prompt_seeds = []
    for seed in missing_seeds:
        bf16_result = load_result("mgsm", make_uncalibrated_path("bf16", config.model_name), file_names[seed])
        understandings = [get_understanding(think) for think in bf16_result["en"]["thinks"]]
        problem_indices[seed] = [problem_index for problem_index, understanding in enumerate(understandings) if understanding is not None]
        for language in MGSM_PROMPTS:
            for problem_index in problem_indices[seed]:
                input_ids.append(torch.cat([get_prompt_ids(tokenizer, problems[language][problem_index][0]), get_think_ids(tokenizer, config.model_name, understandings[problem_index])]))
                prompt_seeds.append(seed)

    # 3. 영어 문제 이해에 이어서 풀이와 응답을 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name)
    start_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, prompt_seeds, config.max_generation_len)
    print(f"{target} time {format_duration(time.perf_counter() - start_time)}")

    # 4. seed와 언어마다 채점하고 저장
    start = 0
    for seed in missing_seeds:
        result = {}
        for language in MGSM_PROMPTS:
            end = start + len(problem_indices[seed])
            answers = [problems[language][problem_index][1] for problem_index in problem_indices[seed]]
            result[language] = {**evaluate_mgsm(outputs[start:end], answers, language, tokenizer, config.model_name), "problem_indices": problem_indices[seed]}
            start = end
            print(f"{target} seed {seed} {language} accuracy {result[language]['accuracy']:.2%} reasoning len {mean(result[language]['think_lens']):.0f} used {len(answers)} of {len(problems[language])}")
        save_result(script_name, target, file_names[seed], result)


if __name__ == "__main__":
    main()
