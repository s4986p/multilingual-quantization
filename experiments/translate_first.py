from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_mgsm
from utils.evaluation import evaluate_benchmark, parse_mgsm_prediction
from utils.paths import make_hub_path

# 해당 언어 prompt 뒤에 붙이는 instruction, finetune이나 삽입 없이 reasoning 시작에 모델이 직접 영어로 번역하게 함
TRANSLATE_FIRST_INSTRUCTION = "Start your thinking by writing the problem above in English, then solve it."


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


def load_translate_first_problems(
    languages: list[str],
) -> dict[str, list[tuple[str, int]]]:
    # mgsm 해당 언어 prompt 뒤에 instruction을 붙임
    # 예시) 스와힐리어 instruction + "\n\n" + 스와힐리어 question → ... + "\n\n" + TRANSLATE_FIRST_INSTRUCTION
    return {language: [(f"{prompt}\n\n{TRANSLATE_FIRST_INSTRUCTION}", answer) for prompt, answer in problems] for language, problems in load_mgsm(languages).items()}


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    evaluate_benchmark("translate_first/mgsm", load_translate_first_problems, parse_mgsm_prediction, hub_path, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
