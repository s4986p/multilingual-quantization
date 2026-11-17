from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse
import torch
from torch import Tensor

from utils.dataset import LANGUAGES
from utils.hub import load_results, load_tensors
from utils.metrics import get_bf16_solved_mask
from utils.paths import make_hub_path, make_result_name
from utils.plot import LANGUAGE_COLORS, NUM_POSITION_BINS, POSITION_LABELS, QUANTIZATION_METHODS, add_language_legend, get_figsize, get_title, plot_language_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def split_problems(
    values: Tensor,
    problem_indices: Tensor,
    bf16_solved_mask: list[bool],
) -> list[Tensor]:
    # token마다 저장된 값을 문제마다 나누고, 모든 bf16 seed가 맞힌 문제만 남김
    unique_indices, counts = problem_indices.unique_consecutive(return_counts=True)
    return [problem_values for problem_index, problem_values in zip(unique_indices.tolist(), values.split(counts.tolist())) if bf16_solved_mask[problem_index]]


def main() -> None:
    config = parse(Config)
    results = {}
    for method in QUANTIZATION_METHODS:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        results[method] = load_tensors("position_kl/mgsm", hub_path, make_result_name(config.max_generation_len, config.seed))
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.bf16_seeds)

    # 1. 문제마다 kl 평균을 내고, 그 문제 평균
    # 2. mxfp4는 문제마다 reasoning을 NUM_POSITION_BINS 구간으로 나눠 구간별로도 평균
    kls = {language: [] for language in LANGUAGES}
    position_kls = {}
    num_problems = {}
    for language in LANGUAGES:
        bf16_solved_mask = get_bf16_solved_mask(bf16_results, language)
        num_problems[language] = sum(bf16_solved_mask)
        for method, result in results.items():
            problem_kls = split_problems(result[f"{language}/kl"], result[f"{language}/problem_indices"], bf16_solved_mask)
            kls[language].append(torch.stack([kl.mean() for kl in problem_kls]).mean().item())
            if method == "mxfp4":
                position_kls[language] = torch.stack([torch.stack([bin_kl.mean() for bin_kl in kl.tensor_split(NUM_POSITION_BINS)]) for kl in problem_kls]).mean(dim=0).tolist()

    fig, axes = plt.subplots(2, 1, figsize=get_figsize(len(QUANTIZATION_METHODS), 2), layout="constrained")
    plot_language_bars(axes[0], QUANTIZATION_METHODS, kls, "kl")
    for language in LANGUAGES:
        axes[1].plot(range(NUM_POSITION_BINS), position_kls[language], color=LANGUAGE_COLORS[language], marker="o", linewidth=2)
    axes[1].set_xticks(range(NUM_POSITION_BINS), POSITION_LABELS)
    axes[1].set_title("mxfp4")
    for ax in axes:
        ax.set_ylabel("kl")
        ax.margins(y=0.15)

    fig.suptitle(get_title(config.model_name, "mgsm", "reasoning 위치별 kl"))
    add_language_legend(fig, bf16_results, num_problems)
    save_figure(fig, config.model_name, "mgsm_position_kl")


if __name__ == "__main__":
    main()
