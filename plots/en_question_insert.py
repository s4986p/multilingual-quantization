from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_target_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGES, METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, plot_case_bars, plot_loss_difference_bars, save_figure

CASES = [("bf16", "mgsm"), ("bf16", "en_question_insert/mgsm"), ("quantized", "mgsm"), ("quantized", "en_question_insert/mgsm")]
CASE_LABELS = {"mgsm": "해당 언어 프롬프트", "en_question_insert/mgsm": "영어 질문 삽입"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    panel_widths = [get_panel_width(len(CASES), len(LANGUAGES)), get_panel_width(2, len(LANGUAGES) - 1)]
    fig, axes = plt.subplots(len(METHODS), 2, figsize=get_figsize(panel_widths, len(METHODS)), width_ratios=panel_widths, layout="constrained")
    for (accuracy_ax, loss_difference_ax), method in zip(axes, METHODS):
        targets = {"bf16": make_uncalibrated_path("bf16", config.model_name), "quantized": make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)}
        case_results = [[load_result(script_name, targets[model_method], f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for model_method, script_name in CASES]
        case_accuracies = {language: [[result[language]["accuracy"] for result in results] for results in case_results] for language in LANGUAGES}

        method_names = {"bf16": "bf16", "quantized": method}
        plot_case_bars(accuracy_ax, [f"{method_names[model_method]}, {CASE_LABELS[script_name]}" for model_method, script_name in CASES], case_accuracies)
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(method)
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_difference_bars(loss_difference_ax, [CASE_LABELS[script_name] for _, script_name in CASES[:2]], {language: accuracies[:2] for language, accuracies in case_accuracies.items()}, {language: accuracies[2:] for language, accuracies in case_accuracies.items()})
        loss_difference_ax.set_ylabel("영어와의 손실 차이")
        loss_difference_ax.set_title(f"{method} 영어와의 손실 차이")

    fig.suptitle(f"{config.model_name} mgsm en question insert\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig)
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "mgsm_en_question_insert")


if __name__ == "__main__":
    main()
