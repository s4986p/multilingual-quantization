from dataclasses import dataclass
import re
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_mgsm
from utils.evaluation import evaluate_benchmark
from utils.paths import make_hub_path

MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}


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


def parse_mgsm_prediction(
    response: str | None,
    language: str,
) -> float | None:
    if response is None:
        return None
    # 마지막 answer prefix 뒤 text 추출, prefix가 없으면 response 전체 사용
    # 예시) "5+3=8，所以答案：8" → "8"
    # 예시) "Answer :7" → "7"
    # 예시) "Answer - 7" → "Answer - 7"
    # 예시) "Answer: 5. Answer: 8" → " 8"
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    # text의 마지막 숫자 추출, 숫자가 없으면 response 전체에서 추출
    # 예시) "so 366.\n答え:" → 366.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    evaluate_benchmark("mgsm", load_mgsm, parse_mgsm_prediction, hub_path, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
