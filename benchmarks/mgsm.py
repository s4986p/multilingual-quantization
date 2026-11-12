from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse

from utils.config import Language
from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_benchmark, parse_mgsm_prediction
from utils.paths import make_finetune_path, make_target_path


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4", "finetune"] = "bf16"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    loss: Literal["ce", "kl"] = "ce"
    think_source: Literal["en", "on_policy"] = "en"
    think_part: Literal["understanding", "reasoning"] = "understanding"


def main() -> None:
    config = parse(Config)
    if config.method == "finetune":
        target = make_finetune_path(config.model_name, config.loss, config.think_source, config.think_part)
    else:
        target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    evaluate_benchmark("mgsm", list(MGSM_PROMPTS), load_mgsm, parse_mgsm_prediction, target, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
