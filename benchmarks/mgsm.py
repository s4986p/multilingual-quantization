from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_mgsm
from utils.evaluation import evaluate_benchmark, parse_mgsm_prediction
from utils.paths import make_finetune_path, make_hub_path


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4", "finetune"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    loss: Literal["ce", "kl", "hidden"] = "ce"
    reasoning_source: Literal["bf16_en", "on_policy"] = "bf16_en"
    train_part: Literal["understanding", "reasoning"] = "understanding"


def main() -> None:
    config = parse(Config)
    if config.method == "finetune":
        hub_path = make_finetune_path(config.model_name, config.loss, config.reasoning_source, config.train_part)
    else:
        hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    evaluate_benchmark("mgsm", load_mgsm, parse_mgsm_prediction, hub_path, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
