from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.config import Language
from utils.hub import load_result
from utils.paths import make_target_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_NAMES, LANGUAGES, METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, plot_case_bars, plot_loss_difference_bars, save_figure

CASES = [("bf16", "mgsm"), ("bf16", "en_understanding_swap/mgsm"), ("quantized", "mgsm"), ("quantized", "en_understanding_swap/mgsm")]
CASE_LABELS = {"mgsm": "해당 언어 프롬프트", "en_understanding_swap/mgsm": "영어 문제 이해 교체"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def get_used_accuracy(
    result: dict,
    used: set[int],
) -> float:
    problem_indices = result.get("problem_indices", range(len(result["corrects"])))
    return mean(correct for problem_index, correct in zip(problem_indices, result["corrects"]) if problem_index in used)


def main() -> None:
    config = parse(Config)
    panel_widths = [get_panel_width(len(CASES), len(LANGUAGES)), get_panel_width(2, len(LANGUAGES) - 1)]
    fig, axes = plt.subplots(len(METHODS), 2, figsize=get_figsize(panel_widths, len(METHODS)), width_ratios=panel_widths, layout="constrained")
    for (accuracy_ax, loss_difference_ax), method in zip(axes, METHODS):
        targets = {"bf16": make_uncalibrated_path("bf16", config.model_name), "quantized": make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)}
        case_results = [[load_result(script_name, targets[model_method], f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for model_method, script_name in CASES]
        swap_results = case_results[CASES.index(("bf16", "en_understanding_swap/mgsm"))]
        case_accuracies = {}
        used_counts = {}
        for language in LANGUAGES:
            seed_used = [set(swap_result[language]["problem_indices"]) for swap_result in swap_results]
            used_counts[language] = mean(len(used) for used in seed_used)
            case_accuracies[language] = [[get_used_accuracy(result[language], used) for result, used in zip(results, seed_used)] for results in case_results]

        method_names = {"bf16": "bf16", "quantized": method}
        plot_case_bars(accuracy_ax, [f"{method_names[model_method]}, {CASE_LABELS[script_name]}" for model_method, script_name in CASES], case_accuracies)
        accuracy_ax.set_ylim(0, ACCURACY_YMAX)
        accuracy_ax.set_ylabel("정확도")
        accuracy_ax.set_title(f"{method} (사용한 문제 {', '.join(f'{LANGUAGE_NAMES[language]} {used_counts[language]:.0f}' for language in LANGUAGES)})")
        accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        plot_loss_difference_bars(loss_difference_ax, [CASE_LABELS[script_name] for _, script_name in CASES[:2]], {language: accuracies[:2] for language, accuracies in case_accuracies.items()}, {language: accuracies[2:] for language, accuracies in case_accuracies.items()})
        loss_difference_ax.set_ylabel("영어와의 손실 차이")
        loss_difference_ax.set_title(f"{method} 영어와의 손실 차이")

    fig.suptitle(f"{config.model_name} mgsm en understanding swap\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig)
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "mgsm_en_understanding_swap")


if __name__ == "__main__":
    main()
