from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, get_mgsm_question, load_mgsm
from utils.evaluation import evaluate_outputs, parse_mgsm_prediction
from utils.format import format_duration
from utils.generation import continue_vllm, load_vllm
from utils.hub import download_model, result_exists, save_result
from utils.model import get_prompt_ids, get_think_ids
from utils.paths import Language, make_target_path

# gemma가 저장한 추론은 추론 시작 token 뒤에 이 텍스트로 시작
THINK_HEADER = "thought\n"


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    script_name = "en_question_insert/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    file_names = {seed: f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds}

    # 1. seed마다 hub 결과에서 빠진 것 확인
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(script_name, target, file_name)]
    if not missing_seeds:
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. 해당 언어 프롬프트 뒤, 추론 시작에 같은 문제의 영어 질문 입력
    # 예시) 스와힐리어 프롬프트 + "<|channel>thought\n" + 영어 질문 + "\n\n"
    en_questions = [get_mgsm_question(prompt) for prompt, _ in problems["en"]]
    language_input_ids = [torch.cat([get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, THINK_HEADER + en_question + "\n\n")]) for language in MGSM_PROMPTS for (prompt, _), en_question in zip(problems[language], en_questions)]
    input_ids = language_input_ids * len(missing_seeds)
    prompt_seeds = [seed for seed in missing_seeds for _ in language_input_ids]

    # 3. 영어 질문에 이어서 추론과 응답을 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
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
            end = start + len(problems[language])
            result[language] = evaluate_outputs(outputs[start:end], [answer for _, answer in problems[language]], language, tokenizer, config.model_name, parse_mgsm_prediction)
            start = end
            print(f"{target} seed {seed} {language} accuracy {result[language]['accuracy']:.2%} reasoning len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
        save_result(script_name, target, file_names[seed], result)


if __name__ == "__main__":
    main()
