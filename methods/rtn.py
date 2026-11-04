from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear

from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_rtn_path
from utils.quantization import find_grid, round_to_grid


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    num_bits: int = 4
    group_size: int = 128


def quantize_weight(
    weight: Tensor,
    num_bits: int,
    group_size: int,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, group_size)
    scale, zero = find_grid(groups, num_bits)
    return round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    layers = model.model.language_model.layers

    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        for linear in layer.modules():
            if not isinstance(linear, Linear):
                continue
            linear.weight.copy_(quantize_weight(linear.weight, config.num_bits, config.group_size))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_rtn_path(config.model_name, config.num_bits, config.group_size), model)


if __name__ == "__main__":
    main()
