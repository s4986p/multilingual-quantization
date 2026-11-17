from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, get_mgsm_question, load_mgsm
from utils.evaluation import continue_vllm, evaluate_outputs, load_vllm, parse_mgsm_prediction
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import GENERATION_CONFIGS, get_prompt_ids, get_reasoning_ids
from utils.paths import make_hub_path, make_result_name, make_translation_name

# 삽입할 영어 question마다 hub 결과 경로, bf16_translation은 bf16이 thinking mode 없이 번역한 question
EXPERIMENT_NAMES = {"gold": "en_question_insert/mgsm", "bf16_translation": "bf16_translation_insert/mgsm"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    # 삽입할 영어 question
    en_question: Literal["gold", "bf16_translation"] = "gold"
    translation_seed: int = 0
    max_translation_len: int = 512
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    experiment_name = EXPERIMENT_NAMES[config.en_question]
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    file_names = {seed: make_result_name(config.max_generation_len, seed) for seed in config.seeds}

    # 1. seed마다 hub result에서 빠진 것 확인, 빠진 것이 없으면 종료
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(experiment_name, hub_path, file_name)]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return
    problems = load_mgsm(LANGUAGES)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 2. 해당 언어 prompt 뒤, reasoning 시작에 같은 문제의 영어 question 삽입
    # bf16_translation에서 영어는 번역할 것이 없어 영어 question 그대로 삽입
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + 영어 question + "\n\n"
    gold_questions = [get_mgsm_question(prompt) for prompt, _ in problems["en"]]
    if config.en_question == "gold":
        en_questions = {language: gold_questions for language in LANGUAGES}
    else:
        translation_result = load_result("translation/mgsm", make_hub_path("bf16", config.model_name), make_translation_name(config.max_translation_len, config.translation_seed))
        en_questions = {"en": gold_questions} | {language: translation_result[language]["translations"] for language in LANGUAGES[1:]}
    reasoning_header = GENERATION_CONFIGS[config.model_name]["reasoning_header"]
    language_input_ids = [torch.cat([get_prompt_ids(tokenizer, prompt), get_reasoning_ids(tokenizer, config.model_name, reasoning_header + en_question + "\n\n")]) for language in LANGUAGES for (prompt, _), en_question in zip(problems[language], en_questions[language])]
    input_ids = language_input_ids * len(missing_seeds)
    prompt_seeds = [seed for seed in missing_seeds for _ in language_input_ids]

    # 3. vllm은 띄우는 데 오래 걸리기 때문에 영어 question에 이어서 reasoning과 response를 모아 한 번에 생성
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
            end = start + len(problems[language])
            result[language] = evaluate_outputs(outputs[start:end], [answer for _, answer in problems[language]], language, tokenizer, config.model_name, parse_mgsm_prediction)
            start = end
            print(f"{hub_path} seed {seed} {language} accuracy {result[language]['accuracy']:.2%} reasoning len {mean(result[language]['reasoning_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
        save_result(experiment_name, hub_path, file_names[seed], result)


if __name__ == "__main__":
    main()
