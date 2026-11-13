from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_outputs, parse_mgsm_prediction
from utils.format import format_duration
from utils.generation import continue_vllm, load_vllm
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import get_prompt_ids, get_think_ids
from utils.paths import make_target_path, make_uncalibrated_path
from utils.understanding import get_understanding

CUT_RATIOS = [0.5, 0.75, 1, 1.5, 2]


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    script_name = "bf16_understanding_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    file_names = {(seed, cut_ratio): f"cut_ratio={cut_ratio},max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds for cut_ratio in CUT_RATIOS}

    # 1. seed와 자르는 비율마다 hub 결과에서 빠진 것 확인
    missing_cases = [case for case, file_name in file_names.items() if not result_exists(script_name, target, file_name)]
    if not missing_cases:
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. bf16 해당 언어 추론을 문제 이해 token 수의 0.5배, 0.75배, 1배, 1.5배, 2배에서 자르고 해당 언어 프롬프트 뒤에 입력
    # 추론이 문제 이해 token 수의 2배보다 긴 문제만 사용, 비율끼리 같은 문제로 비교하기 때문
    # 예시) 스와힐리어 프롬프트 + "<|channel>thought\n" + bf16 스와힐리어 추론 앞부분
    problem_indices = {}
    input_ids = []
    prompt_seeds = []
    for seed, cut_ratio in missing_cases:
        bf16_result = load_result("mgsm", make_uncalibrated_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}")
        for language in MGSM_PROMPTS:
            think_ids = [get_think_ids(tokenizer, config.model_name, think) for think in bf16_result[language]["thinks"]]
            understandings = [get_understanding(think) for think in bf16_result[language]["thinks"]]
            understanding_lens = {problem_index: len(get_think_ids(tokenizer, config.model_name, understanding)) for problem_index, understanding in enumerate(understandings) if understanding is not None}
            problem_indices[seed, cut_ratio, language] = [problem_index for problem_index, understanding_len in understanding_lens.items() if len(think_ids[problem_index]) >= max(CUT_RATIOS) * understanding_len]
            for problem_index in problem_indices[seed, cut_ratio, language]:
                input_ids.append(torch.cat([get_prompt_ids(tokenizer, problems[language][problem_index][0]), think_ids[problem_index][: round(cut_ratio * understanding_lens[problem_index])]]))
                prompt_seeds.append(seed)

    # 3. bf16 추론 앞부분에 이어서 추론과 응답을 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
    model = load_vllm(download_model(target), config.model_name)
    start_time = time.perf_counter()
    outputs = continue_vllm(model, config.model_name, input_ids, prompt_seeds, config.max_generation_len)
    print(f"{target} time {format_duration(time.perf_counter() - start_time)}")

    # 4. seed, 자르는 비율, 언어마다 채점하고 저장
    start = 0
    for seed, cut_ratio in missing_cases:
        result = {}
        for language in MGSM_PROMPTS:
            end = start + len(problem_indices[seed, cut_ratio, language])
            answers = [problems[language][problem_index][1] for problem_index in problem_indices[seed, cut_ratio, language]]
            result[language] = {**evaluate_outputs(outputs[start:end], answers, language, tokenizer, config.model_name, parse_mgsm_prediction), "problem_indices": problem_indices[seed, cut_ratio, language]}
            start = end
            print(f"{target} seed {seed} cut ratio {cut_ratio} {language} accuracy {result[language]['accuracy']:.2%} reasoning len {mean(result[language]['think_lens']):.0f} used {len(answers)} of {len(problems[language])}")
        save_result(script_name, target, file_names[seed, cut_ratio], result)


if __name__ == "__main__":
    main()
