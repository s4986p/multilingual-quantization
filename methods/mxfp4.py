from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn import Linear

from utils.hub import download_model, save_model
from utils.model import load_model
from utils.paths import make_hub_path, make_training_path
from utils.quantization import quantize_mxfp4


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    # bf16 대신 finetune한 모델을 양자화, 설정은 methods/finetune.py와 같음
    finetune: bool = False
    loss: Literal["ce", "kl", "hidden", "ce_kl"] = "ce"
    reasoning_source: Literal["bf16_en", "on_policy", "bf16_en_question_insert"] = "bf16_en"
    train_part: Literal["understanding", "reasoning", "en_question"] = "understanding"
    num_samples: int | None = None
    num_trained_samples: int | None = None


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    if config.finetune:
        model = load_model(download_model(make_training_path("finetune", config.model_name, config.loss, config.reasoning_source, config.train_part, config.num_samples, config.num_trained_samples)))
        hub_path = make_training_path("finetune_mxfp4", config.model_name, config.loss, config.reasoning_source, config.train_part, config.num_samples, config.num_trained_samples)
    else:
        model = load_model(config.model_name)
        hub_path = make_hub_path("mxfp4", config.model_name)
    for linear in [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]:
        linear.weight.copy_(quantize_mxfp4(linear.weight))
    save_model(hub_path, model)


if __name__ == "__main__":
    main()
