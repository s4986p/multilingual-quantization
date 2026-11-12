from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_benchmark, evaluate_mgsm
from utils.paths import make_target_path


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
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    evaluate_benchmark("mgsm", list(MGSM_PROMPTS), load_mgsm, evaluate_mgsm, target, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
