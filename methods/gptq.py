from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear

from utils.capture import capture_layer_inputs, compute_abs_mean_and_hessian
from utils.dataset import load_wikipedia
from utils.format import format_duration, format_memory
from utils.hub import save_model
from utils.model import load_model
from utils.paths import Language, make_calibrated_path
from utils.quantization import find_grid, round_to_grid


@dataclass
class Config:
    num_bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damp_ratio: float = 0.01
    model_name: str = "google/gemma-4-E2B-it"
    # calibration할 데이터
    calibration_languages: tuple[Language, ...] = ("en",)
    num_samples: int = 128
    max_sequence_len: int = 2048
    shuffle_seed: int = 0


def quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    block_size: int,
    damp_ratio: float,
) -> Tensor:
    weight = weight.float().clone()
    # 1. hessian 대각이 0인 열은 weight 0, 대각 1로 설정, 대각이 0이면 cholesky 분해가 실패하기 때문
    is_zero_diagonal = hessian.diagonal() == 0
    weight[:, is_zero_diagonal] = 0
    hessian.diagonal()[is_zero_diagonal] = 1

    # 2. 대각에 damp 추가, 역행렬이 안정적이어야 하기 때문
    hessian.diagonal().add_(damp_ratio * hessian.diagonal().mean())
    hessian_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)

    # 3. block 단위로 열을 하나씩 양자화
    quantized = torch.zeros_like(weight)
    scale = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    zero = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    error = torch.zeros(weight.shape[0], block_size, device=weight.device)
    for block_start in range(0, weight.shape[1], block_size):
        block_end = min(block_start + block_size, weight.shape[1])
        for column_index in range(block_start, block_end):
            group_index = column_index // group_size
            # group 시작 시 grid 설정, 앞 열이 갱신한 weight를 써야 하기 때문
            if column_index % group_size == 0:
                scale[:, group_index], zero[:, group_index] = find_grid(weight[:, column_index : column_index + group_size], num_bits)

            # 4. 열을 양자화하고 그 오차를 같은 block 안 뒤 열로 전파
            quantized[:, column_index] = round_to_grid(weight[:, column_index], scale[:, group_index], zero[:, group_index], num_bits)
            error[:, column_index - block_start] = (weight[:, column_index] - quantized[:, column_index]) / hessian_inv[column_index, column_index]
            weight[:, column_index:block_end] -= torch.outer(error[:, column_index - block_start], hessian_inv[column_index, column_index:block_end])

        # 5. block 오차를 block 뒤 모든 열로 전파
        weight[:, block_end:] -= error[:, : block_end - block_start] @ hessian_inv[block_start:block_end, block_end:]
    return quantized


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    samples = load_wikipedia(config.model_name, config.calibration_languages, config.num_samples, config.max_sequence_len, config.shuffle_seed)
    layers = model.model.language_model.layers

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(layers):
        layer_inputs = capture_layer_inputs(model, layer, samples)
        for linear in [module for module in layer.modules() if isinstance(module, Linear)]:
            _, hessian = compute_abs_mean_and_hessian(layer, linear, layer_inputs)
            linear.weight.copy_(quantize_weight(linear.weight, hessian, config.num_bits, config.group_size, config.block_size, config.damp_ratio))

        print(f"layer {layer_index + 1}/{len(layers)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_model(make_calibrated_path("gptq", config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples), model)


if __name__ == "__main__":
    main()
