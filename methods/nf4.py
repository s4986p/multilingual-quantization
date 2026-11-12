from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear

from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_uncalibrated_path

NF4_GROUP_SIZE = 64
# bitsandbytes와 같은 값
NF4_VALUES = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"


def quantize_weight(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, NF4_GROUP_SIZE)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    values = torch.tensor(NF4_VALUES, device=weight.device)
    value_indices = torch.bucketize(groups / scale, (values[1:] + values[:-1]) / 2)
    return (values[value_indices] * scale).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    layers = model.model.language_model.layers

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        for linear in [module for module in layer.modules() if isinstance(module, Linear)]:
            linear.weight.copy_(quantize_weight(linear.weight))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_uncalibrated_path("nf4", config.model_name), model)


if __name__ == "__main__":
    main()
