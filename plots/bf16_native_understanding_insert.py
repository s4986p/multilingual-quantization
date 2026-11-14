from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_NAMES, QUANTIZATION_METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, get_subset_accuracy, plot_insert_bars, plot_loss_gap_bars, save_figure
from utils.understanding import get_understanding_problem_indices

CASES = [("bf16", "mgsm"), ("quantized", "mgsm"), ("quantized", "bf16_native_understanding_insert/mgsm")]
CASE_LABELS = {"mgsm": "삽입 없음", "bf16_native_understanding_insert/mgsm": "해당 언어 understanding 삽입"}


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
    panel_widths = [get_panel_width(2, len(LANGUAGES)), get_panel_width(2, len(LANGUAGES) - 1)]
    fig, axes = plt.subplots(len(QUANTIZATION_METHODS), 2, figsize=get_figsize(panel_widths, len(QUANTIZATION_METHODS)), width_ratios=panel_widths, layout="constrained")
    for (accuracy_ax, loss_gap_ax), method in zip(axes, QUANTIZATION_METHODS):
        hub_paths = {"bf16": make_hub_path("bf16", config.model_name), "quantized": make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)}
        case_results = [[load_result(experiment_name, hub_paths[model_method], f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for model_method, experiment_name in CASES]
        case_accuracies = {}
        num_problems = {}
        for language in LANGUAGES:
            problem_indices = [get_understanding_problem_indices(bf16_result, language) for bf16_result in case_results[0]]
            num_problems[language] = mean(len(indices) for indices in problem_indices)
            case_accuracies[language] = [[get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices)] for results in case_results]

        plot_insert_bars(accuracy_ax, ["bf16", method], {language: [(accuracies[0], None), (accuracies[1], accuracies[2])] for language, accuracies in case_accuracies.items()})
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(f"{method} (사용한 문제 수 {', '.join(f'{LANGUAGE_NAMES[language]} {num_problems[language]:.0f}' for language in LANGUAGES)})")
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_gap_bars(loss_gap_ax, [CASE_LABELS[experiment_name] for _, experiment_name in CASES[1:]], {language: [accuracies[0]] * 2 for language, accuracies in case_accuracies.items()}, {language: accuracies[1:] for language, accuracies in case_accuracies.items()})
        loss_gap_ax.set_ylabel("영어와의 손실 격차")
        loss_gap_ax.set_title(f"{method} 영어와의 손실 격차")

    fig.suptitle(f"{config.model_name} mgsm bf16이 해당 언어 prompt에 쓴 understanding 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, insert_label="해당 언어 understanding 삽입으로 오른 정확도")
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "mgsm_bf16_native_understanding_insert")


if __name__ == "__main__":
    main()
