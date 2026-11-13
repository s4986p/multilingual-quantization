from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_target_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGE_NAMES, LANGUAGES, METHODS, add_language_legend, fit_bar_width, format_ratio, get_figsize, get_panel_width, get_used_accuracy, plot_bars, plot_loss_difference_bars, save_figure

CUT_RATIOS = [0.5, 0.75, 1, 1.5, 2]
CASES = [("mgsm", ""), *(("bf16_understanding_swap/mgsm", f"cut_ratio={cut_ratio},") for cut_ratio in CUT_RATIOS), ("en_understanding_swap/mgsm", "")]
CASE_LABELS = ["해당 언어 프롬프트", *(f"bf16 문제 이해 {cut_ratio}배" for cut_ratio in CUT_RATIOS), "영어 문제 이해 교체"]


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
    bf16_results = [load_result("mgsm", make_uncalibrated_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]
    fig, axes = plt.subplots(len(METHODS) * 2, 1, figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], len(METHODS) * 2), layout="constrained")
    for method, accuracy_ax, loss_difference_ax in zip(METHODS, axes[::2], axes[1::2]):
        target = make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
        case_results = [[load_result(script_name, target, f"{file_prefix}max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for script_name, file_prefix in CASES]
        seed_used = {language: [set(cut_result[language]["problem_indices"]) & set(en_result[language]["problem_indices"]) for cut_result, en_result in zip(case_results[1], case_results[-1])] for language in LANGUAGES}
        bf16_accuracies = {language: [get_used_accuracy(bf16_result[language], used) for bf16_result, used in zip(bf16_results, seed_used[language])] for language in LANGUAGES}
        case_accuracies = {language: [[get_used_accuracy(result[language], used) for result, used in zip(results, seed_used[language])] for results in case_results] for language in LANGUAGES}

        series = []
        for language in LANGUAGES:
            accuracies = [mean(seed_accuracies) for seed_accuracies in case_accuracies[language]]
            series.append((accuracies, LANGUAGE_COLORS[language], [format_ratio(accuracy, mean(bf16_accuracies[language])) for accuracy in accuracies]))
        plot_bars(accuracy_ax, CASE_LABELS, series, [mean(bf16_accuracies[language]) for language in LANGUAGES])
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(f"{method} (사용한 문제 {', '.join(f'{LANGUAGE_NAMES[language]} {mean(len(used) for used in seed_used[language]):.0f}' for language in LANGUAGES)})")
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_difference_bars(loss_difference_ax, CASE_LABELS, {language: [bf16_accuracies[language]] * len(CASES) for language in LANGUAGES}, case_accuracies)
        loss_difference_ax.set_ylabel("영어와의 손실 차이")
        loss_difference_ax.set_title(f"{method} 영어와의 손실 차이")

    fig.suptitle(f"{config.model_name} mgsm bf16 understanding swap\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, [{language: {"accuracy": bf16_accuracies[language][seed_index]} for language in LANGUAGES} for seed_index in range(len(config.seeds))])
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_bf16_understanding_swap")


if __name__ == "__main__":
    main()
