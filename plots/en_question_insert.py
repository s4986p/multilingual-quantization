from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, QUANTIZATION_METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, plot_insert_bars, plot_loss_gap_bars, save_figure

CASES = [("bf16", "mgsm"), ("bf16", "en_question_insert/mgsm"), ("quantized", "mgsm"), ("quantized", "en_question_insert/mgsm")]
CASE_LABELS = {"mgsm": "삽입 없음", "en_question_insert/mgsm": "영어 question 삽입"}


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
        case_accuracies = {language: [[result[language]["accuracy"] for result in results] for results in case_results] for language in LANGUAGES}

        plot_insert_bars(accuracy_ax, ["bf16", method], {language: [(accuracies[0], accuracies[1]), (accuracies[2], accuracies[3])] for language, accuracies in case_accuracies.items()})
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(method)
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_gap_bars(loss_gap_ax, [CASE_LABELS[experiment_name] for _, experiment_name in CASES[:2]], {language: accuracies[:2] for language, accuracies in case_accuracies.items()}, {language: accuracies[2:] for language, accuracies in case_accuracies.items()})
        loss_gap_ax.set_ylabel("영어와의 손실 격차")
        loss_gap_ax.set_title(f"{method} 영어와의 손실 격차")

    fig.suptitle(f"{config.model_name} mgsm 영어 question 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, insert_label="영어 question 삽입으로 오른 정확도")
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "mgsm_en_question_insert")


if __name__ == "__main__":
    main()
