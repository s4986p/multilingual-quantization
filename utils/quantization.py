import torch
from torch import Tensor

MXFP4_GROUP_SIZE = 32


def find_grid(
    weight: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    weight_min = weight.amin(dim=-1).clamp(max=0)
    weight_max = weight.amax(dim=-1).clamp(min=0)
    scale = ((weight_max - weight_min) / (2**num_bits - 1)).clamp(min=1e-8)
    zero = torch.round(-weight_min / scale)
    return scale, zero


def round_to_grid(
    weight: Tensor,
    scale: Tensor,
    zero: Tensor,
    num_bits: int,
) -> Tensor:
    level = (torch.round(weight / scale) + zero).clamp(0, 2**num_bits - 1)
    return (level - zero) * scale


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
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.sign(weight) * torch.round(magnitude / step) * step * scale
