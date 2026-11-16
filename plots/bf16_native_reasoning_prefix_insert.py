from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import LANGUAGE_COLORS, QUANTIZATION_METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, get_subset_accuracy, plot_bars, save_figure

LENGTH_RATIOS = [0.5, 0.75, 1, 1.5, 2]
CASES = [("mgsm", ""), *(("bf16_native_reasoning_prefix_insert/mgsm", f"length_ratio={length_ratio},") for length_ratio in LENGTH_RATIOS)]
CASE_LABELS = ["삽입 없음", *(f"understanding 길이 {length_ratio}배" for length_ratio in LENGTH_RATIOS)]


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
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]

    fig, axes = plt.subplots(len(QUANTIZATION_METHODS), 1, figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], len(QUANTIZATION_METHODS)), layout="constrained")
    for method, ax in zip(QUANTIZATION_METHODS, axes):
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        case_results = [[load_result(experiment_name, hub_path, f"{file_prefix}max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for experiment_name, file_prefix in CASES]
        # 모든 길이에서 같은 문제를 쓰기 때문에 첫 길이의 문제로 계산
        problem_indices = {language: [set(result[language]["problem_indices"]) for result in case_results[1]] for language in LANGUAGES}

        series = []
        for language in LANGUAGES:
            bf16_accuracy = mean(get_subset_accuracy(bf16_result[language], indices) for bf16_result, indices in zip(bf16_results, problem_indices[language]))
            losses = [bf16_accuracy - mean(get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices[language])) for results in case_results]
            series.append((losses, LANGUAGE_COLORS[language], [f"{loss:.1%}p" for loss in losses]))
        plot_bars(ax, CASE_LABELS, series)
        # 방법끼리 비교하기 쉽게 같은 범위로 그림
        ax.set_ylim(-0.05, 0.2)
        ax.set_ylabel("손실")
        ax.set_title(method)
        ax.axhline(0, color="#808080", linewidth=0.8)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))
    fig.suptitle(f"{config.model_name} mgsm bf16이 해당 언어 prompt에 쓴 reasoning 앞부분 길이별 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig)
    fit_bar_width(fig, list(axes))
    save_figure(fig, config.model_name, "mgsm_bf16_native_reasoning_prefix_insert")


if __name__ == "__main__":
    main()
