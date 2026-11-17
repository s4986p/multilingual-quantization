from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import INSERT_LOSS_YMAX, INSERT_LOSS_YMIN, INSERT_LOSS_YTICK, QUANTIZATION_METHODS, add_language_legend, get_figsize, get_insert_handles, get_subset_accuracy, get_title, plot_insert_bars, save_figure
from utils.understanding import LENGTH_RATIOS

# 예시) ["understanding 0.5배", ..., "understanding 2배"]
CASES = [f"understanding {length_ratio}배" for length_ratio in LENGTH_RATIOS]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)

    fig, axes = plt.subplots(len(QUANTIZATION_METHODS), 1, figsize=get_figsize(len(CASES), len(QUANTIZATION_METHODS)), layout="constrained")
    for ax, method in zip(axes, QUANTIZATION_METHODS):
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        no_insert_results = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        insert_results = [load_results("bf16_native_reasoning_prefix_insert/mgsm", hub_path, config.max_generation_len, config.seeds, length_ratio) for length_ratio in LENGTH_RATIOS]
        # 모든 길이에서 같은 문제를 쓰기 때문에 첫 길이의 문제로 bf16, 삽입 없음, 삽입 정확도를 계산
        problem_indices = {language: [set(result[language]["problem_indices"]) for result in insert_results[0]] for language in LANGUAGES}
        losses = {}
        for language in LANGUAGES:
            bf16_accuracy, no_insert_accuracy = [mean(get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices[language])) for results in [bf16_results, no_insert_results]]
            losses[language] = [(bf16_accuracy - no_insert_accuracy, bf16_accuracy - mean(get_subset_accuracy(result[language], indices) for result, indices in zip(length_results, problem_indices[language]))) for length_results in insert_results]
        plot_insert_bars(ax, CASES, losses, [])
        ax.set_title(method)
        ax.set_ylabel("손실")
        ax.set_ylim(INSERT_LOSS_YMIN, INSERT_LOSS_YMAX)
        ax.yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))

    fig.suptitle(get_title(config.model_name, "mgsm", "reasoning 앞부분 길이별 삽입"))
    num_problems = {language: round(mean(len(indices) for indices in problem_indices[language])) for language in LANGUAGES}
    add_language_legend(fig, bf16_results, num_problems, get_insert_handles())
    save_figure(fig, config.model_name, "mgsm_reasoning_prefix_insert")


if __name__ == "__main__":
    main()
