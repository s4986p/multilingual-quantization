from dataclasses import dataclass, field
import math
import time

import numpy as np
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from primitives.dataset import load_wikipedia
from primitives.hub import make_awq_path, save_quantized
from primitives.model import load_model
from primitives.modules import capture_layer_inputs, find_linear_groups, get_layers
from primitives.quantization import find_grid, round_to_grid
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    languages: list[str] = field(default_factory=lambda: ["en"])
    num_bits: int = 4
    group_size: int = 128
    num_grid: int = 20
    max_shrink: float = 0.5
    num_samples: int = 128
    max_sequence_len: int = 2048
    seed: int = 0


def compute_statistics(
    layer: Module,
    linear: Linear,
    inputs: list[tuple[tuple, dict]],
) -> tuple[Tensor, Tensor]:
    abs_sum = torch.zeros(linear.in_features, device=linear.weight.device)
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)
    num_tokens = 0

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        nonlocal num_tokens
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        abs_sum.add_(x.abs().sum(dim=0))
        hessian.add_(x.T @ x)
        num_tokens += x.shape[0]

    handle = linear.register_forward_pre_hook(hook)
    for args, kwargs in inputs:
        layer(*args, **kwargs)
    handle.remove()
    return abs_sum / num_tokens, hessian


def forward_layer(
    layer: Module,
    inputs: list[tuple[tuple, dict]],
) -> list[Tensor]:
    outputs = []
    for args, kwargs in inputs:
        output = layer(*args, **kwargs)
        outputs.append(output[0] if isinstance(output, tuple) else output)
    return outputs


def quantize_weight(
    weight: Tensor,
    smooth: Tensor,
    clip_value: Tensor | None,
    num_bits: int,
    group_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    clipped = groups if clip_value is None else groups.clamp(-clip_value[..., None], clip_value[..., None])
    scale, zero = find_grid(clipped, num_bits)
    quantized = round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(weight.shape) / smooth
    return quantized, scale, zero


def find_smooth(
    layer: Module,
    linears: list[Linear],
    inputs: list[tuple[tuple, dict]],
    abs_mean: Tensor,
    num_bits: int,
    group_size: int,
    num_grid: int,
) -> tuple[Tensor, float]:
    outputs = forward_layer(layer, inputs)
    weights = [linear.weight.clone() for linear in linears]

    best_error = math.inf
    for ratio in np.linspace(0.0, 1.0, num_grid, endpoint=False):
        smooth = abs_mean.pow(ratio).clamp(min=1e-4)
        smooth = smooth / (smooth.max() * smooth.min()).sqrt()
        for linear, weight in zip(linears, weights):
            linear.weight.copy_(quantize_weight(weight, smooth, None, num_bits, group_size)[0])
        error = sum((output - quantized_output).float().pow(2).mean().item() for output, quantized_output in zip(outputs, forward_layer(layer, inputs)))
        if error < best_error:
            best_error, best_smooth, best_ratio = error, smooth, float(ratio)

    for linear, weight in zip(linears, weights):
        linear.weight.copy_(weight)
    return best_smooth, best_ratio


def find_clip(
    weight: Tensor,
    smooth: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    num_grid: int,
    max_shrink: float,
) -> Tensor:
    # min-max 양자화는 group의 범위를 outlier가 정해, 나머지 weight의 step이 거칠어짐.
    # 범위를 줄여 outlier를 자르면 나머지는 정밀해지지만, 잘린 outlier의 오차는 커짐.
    # 어느 쪽이 나은지는 결국 그 weight에 곱해지는 activation의 크기에 달려 있음.
    # 그리고 weight 오차가 layer 출력에 미치는 영향은 activation으로 만든 hessian으로 표현 가능.
    # 따라서 clip 후보마다 다시 forward하지 않고, hessian으로 출력 오차를 바로 계산.

    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    num_groups = groups.shape[1]
    # smooth를 곱한 weight에는 smooth로 나눈 activation이 곱해지므로, hessian에도 같은 smooth를 반영.
    smoothed_hessian = hessian / smooth[:, None] / smooth[None, :]
    # clip 값은 group마다 따로 정하므로, hessian에서 같은 group 안의 채널끼리의 블록만 남김.
    hessian_blocks = smoothed_hessian.reshape(num_groups, group_size, num_groups, group_size).diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    weight_max = groups.abs().amax(dim=-1)
    best_clip_value = weight_max.clone()
    best_error = torch.full_like(weight_max, math.inf)
    for shrink in np.linspace(1.0, 1.0 - max_shrink, int(max_shrink * num_grid), endpoint=False):
        # smooth처럼 clip도 grid search로 찾음.
        clip_value = weight_max * shrink
        scale, zero = find_grid(groups.clamp(-clip_value[..., None], clip_value[..., None]), num_bits)
        quantized = round_to_grid(groups, scale[..., None], zero[..., None], num_bits)
        delta = quantized - groups
        # 행과 group마다 weight의 양자화 오차가 만드는 출력 오차를 smooth를 반영한 hessian으로 구함.
        error = torch.einsum("ogi,gij,ogj->og", delta, hessian_blocks, delta)
        # 행과 group마다 최적의 clip 값만 남김.
        is_better = error < best_error
        best_error = torch.where(is_better, error, best_error)
        best_clip_value = torch.where(is_better, clip_value, best_clip_value)
    return best_clip_value


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model, tokenizer = load_model(config.model_name, config.model_name)
    samples = load_wikipedia(tokenizer, config.languages, config.num_samples, config.max_sequence_len, config.seed)
    layers = get_layers(model)

    grid = {}
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        inputs = capture_layer_inputs(model, layer, samples)
        linear_groups = find_linear_groups(layer, *inputs[0])
        ratios = []
        for linear_group in linear_groups:
            # 입력을 공유하는 linear들 끼리는 같은 수치? 사용. (결국 같이 들어가니까 연산 효율적)
            abs_mean, hessian = compute_statistics(layer, next(iter(linear_group.values())), inputs)
            smooth, ratio = find_smooth(layer, list(linear_group.values()), inputs, abs_mean, config.num_bits, config.group_size, config.num_grid)
            ratios.append(ratio)
            for name, linear in linear_group.items():
                clip_value = find_clip(linear.weight, smooth, hessian, config.num_bits, config.group_size, config.num_grid, config.max_shrink) if not name.endswith(("q_proj", "k_proj")) else None
                quantized, scale, zero = quantize_weight(linear.weight, smooth, clip_value, config.num_bits, config.group_size)
                linear.weight.copy_(quantized)
                grid[f"{layer_index}.{name}.scale"] = scale
                grid[f"{layer_index}.{name}.zero"] = zero
                grid[f"{layer_index}.{name}.smooth"] = smooth

        print(f"layer {layer_index + 1}/{len(layers)} groups {[list(linear_group) for linear_group in linear_groups]} ratios {ratios} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_quantized(make_awq_path(config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples), model, grid)


if __name__ == "__main__":
    main()
