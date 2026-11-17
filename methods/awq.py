from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from utils.capture import capture_layer_inputs, compute_abs_mean_and_hessian
from utils.dataset import load_wikipedia
from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import make_hub_path
from utils.quantization import find_grid, round_to_grid


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    num_bits: int = 4
    group_size: int = 128
    num_search_steps: int = 20
    max_shrink: float = 0.5
    # calibration 데이터
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_chunks: int = 128
    chunk_len: int = 2048
    shuffle_seed: int = 0


def find_linear_groups(
    layer: Module,
    layer_args: tuple,
    layer_kwargs: dict,
) -> list[list[Linear]]:
    # 같은 activations를 받는 linear끼리 묶음, smooth는 묶음마다 하나
    linear_groups = []

    def hook(
        module: Linear,
        args: tuple[Tensor, ...],
    ) -> None:
        for activations, linears in linear_groups:
            if activations is args[0]:
                linears.append(module)
                return
        linear_groups.append((args[0], [module]))

    handles = [module.register_forward_pre_hook(hook) for module in layer.modules() if isinstance(module, Linear)]
    layer(*layer_args, **layer_kwargs)
    for handle in handles:
        handle.remove()
    return [linears for _, linears in linear_groups]


def forward_layer(
    layer: Module,
    layer_inputs: list[tuple[tuple, dict]],
) -> list[Tensor]:
    return [layer(*layer_args, **layer_kwargs) for layer_args, layer_kwargs in layer_inputs]


def quantize_weight(
    weight: Tensor,
    smooth: Tensor,
    clip: Tensor | None,
    num_bits: int,
    group_size: int,
) -> Tensor:
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    clipped_groups = groups if clip is None else groups.clamp(-clip[..., None], clip[..., None])
    scale, zero = find_grid(clipped_groups, num_bits)
    return round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(weight.shape) / smooth


def find_smooth(
    layer: Module,
    linears: list[Linear],
    layer_inputs: list[tuple[tuple, dict]],
    abs_mean: Tensor,
    num_bits: int,
    group_size: int,
    num_search_steps: int,
) -> Tensor:
    # 원래 awq는 하위 module output을 비교하지만, 단순하게 layer 전체 output으로 비교
    bf16_outputs = forward_layer(layer, layer_inputs)
    bf16_weights = [linear.weight.clone() for linear in linears]

    best_error = math.inf
    for ratio in np.linspace(0.0, 1.0, num_search_steps, endpoint=False):
        smooth = abs_mean.pow(ratio).clamp(min=1e-4)
        smooth = smooth / (smooth.max() * smooth.min()).sqrt()
        for linear, bf16_weight in zip(linears, bf16_weights):
            linear.weight.copy_(quantize_weight(bf16_weight, smooth, None, num_bits, group_size))
        error = sum((bf16_output - quantized_output).float().pow(2).mean().item() for bf16_output, quantized_output in zip(bf16_outputs, forward_layer(layer, layer_inputs)))
        if error < best_error:
            best_error, best_smooth = error, smooth

    for linear, bf16_weight in zip(linears, bf16_weights):
        linear.weight.copy_(bf16_weight)
    return best_smooth


def find_clip(
    weight: Tensor,
    smooth: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    num_search_steps: int,
    max_shrink: float,
) -> Tensor:
    # outlier가 grid 간격을 넓히기 때문에 group마다 clip 탐색
    # 1. weight와 hessian에 smooth 적용 (w * s, h = x.T @ x에서 x / s)
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    num_groups = groups.shape[1]
    smoothed_hessian = hessian / smooth[:, None] / smooth[None, :]
    # 각 group의 대각 block만 유지
    hessian_blocks = smoothed_hessian.reshape(num_groups, group_size, num_groups, group_size).diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    # 2. shrink를 줄여 가며 group마다 error가 가장 작은 clip 탐색
    weight_max = groups.abs().amax(dim=-1)
    best_clip = weight_max.clone()
    best_error = torch.full_like(weight_max, math.inf)
    for shrink in np.linspace(1.0, 1.0 - max_shrink, int(max_shrink * num_search_steps), endpoint=False):
        clip = weight_max * shrink
        scale, zero = find_grid(groups.clamp(-clip[..., None], clip[..., None]), num_bits)
        quantized_groups = round_to_grid(groups, scale[..., None], zero[..., None], num_bits)
        delta = quantized_groups - groups
        # group 사이 hessian 항은 무시하고 group마다 error 계산
        # 그래야 group마다 clip을 따로 고를 수 있음
        error = torch.einsum("ogi,gij,ogj->og", delta, hessian_blocks, delta)
        is_better = error < best_error
        best_error = torch.where(is_better, error, best_error)
        best_clip = torch.where(is_better, clip, best_clip)
    return best_clip


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    chunks = load_wikipedia(config.model_name, config.calibration_languages, config.num_chunks, config.chunk_len, config.shuffle_seed)
    layers = model.model.language_model.layers

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        layer_inputs = capture_layer_inputs(model, layer, chunks)
        for linears in find_linear_groups(layer, *layer_inputs[0]):
            abs_mean, hessian = compute_abs_mean_and_hessian(layer, linears[0], layer_inputs)
            smooth = find_smooth(layer, linears, layer_inputs, abs_mean, config.num_bits, config.group_size, config.num_search_steps)
            for linear in linears:
                clip = find_clip(linear.weight, smooth, hessian, config.num_bits, config.group_size, config.num_search_steps, config.max_shrink)
                linear.weight.copy_(quantize_weight(linear.weight, smooth, clip, config.num_bits, config.group_size))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_hub_path("awq", config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks), model)


if __name__ == "__main__":
    main()
