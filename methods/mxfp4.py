from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear

from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_uncalibrated_path

MXFP4_GROUP_SIZE = 32


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"


def find_fp4_scale(
    weight: Tensor,
) -> Tensor:
    weight_max = weight.abs().amax(dim=-1).clamp(min=2**-125)
    return 2 ** (torch.floor(torch.log2(weight_max)) - 2)


def round_to_fp4(
    weight: Tensor,
    scale: Tensor,
) -> Tensor:
    magnitude = (weight.abs() / scale).clamp(max=6)
    # fp4 값 [0, 0.5, 1, 1.5, 2, 3, 4, 6] 중 가까운 값으로 반올림
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.sign(weight) * torch.round(magnitude / step) * step * scale


def quantize_weight(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, MXFP4_GROUP_SIZE)
    return round_to_fp4(groups, find_fp4_scale(groups)[..., None]).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    for linear in [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]:
        linear.weight.copy_(quantize_weight(linear.weight))
    save_model(make_uncalibrated_path("mxfp4", config.model_name), model)


if __name__ == "__main__":
    main()
