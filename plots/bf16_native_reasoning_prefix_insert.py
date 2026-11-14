from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGE_NAMES, QUANTIZATION_METHODS, add_language_legend, fit_bar_width, format_accuracy, get_figsize, get_panel_width, get_subset_accuracy, plot_bars, plot_loss_gap_bars, save_figure

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

    fig, axes = plt.subplots(len(QUANTIZATION_METHODS) * 2, 1, figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], len(QUANTIZATION_METHODS) * 2), layout="constrained")
    for method, accuracy_ax, loss_gap_ax in zip(QUANTIZATION_METHODS, axes[::2], axes[1::2]):
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        case_results = [[load_result(experiment_name, hub_path, f"{file_prefix}max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for experiment_name, file_prefix in CASES]
        problem_indices = {language: [set(result[language]["problem_indices"]) for result in case_results[1]] for language in LANGUAGES}
        bf16_accuracies = {language: [get_subset_accuracy(bf16_result[language], indices) for bf16_result, indices in zip(bf16_results, problem_indices[language])] for language in LANGUAGES}
        case_accuracies = {language: [[get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices[language])] for results in case_results] for language in LANGUAGES}

        series = []
        for language in LANGUAGES:
            accuracies = [mean(seed_accuracies) for seed_accuracies in case_accuracies[language]]
            series.append((accuracies, LANGUAGE_COLORS[language], [format_accuracy(accuracy, mean(bf16_accuracies[language])) for accuracy in accuracies]))
        plot_bars(accuracy_ax, CASE_LABELS, series, [mean(bf16_accuracies[language]) for language in LANGUAGES])
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(f"{method} (사용한 문제 수 {', '.join(f'{LANGUAGE_NAMES[language]} {mean(len(indices) for indices in problem_indices[language]):.0f}' for language in LANGUAGES)})")
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_gap_bars(loss_gap_ax, CASE_LABELS, {language: [bf16_accuracies[language]] * len(CASES) for language in LANGUAGES}, case_accuracies)
        loss_gap_ax.set_ylabel("영어와의 손실 격차")
        loss_gap_ax.set_title(f"{method} 영어와의 손실 격차")

    fig.suptitle(f"{config.model_name} mgsm bf16이 해당 언어 prompt에 쓴 reasoning 앞부분 길이별 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, [{language: {"accuracy": bf16_accuracies[language][seed_index]} for language in LANGUAGES} for seed_index in range(len(config.seeds))])
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_bf16_native_reasoning_prefix_insert")


if __name__ == "__main__":
    main()
