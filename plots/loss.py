from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import Language, make_target_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGES, METHODS, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, plot_language_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(think) or is_loop(response) for think, response in zip(result["thinks"], result["responses"]))


def main() -> None:
    config = parse(Config)
    bf16_target = make_uncalibrated_path("bf16", config.model_name)
    bf16_results = [load_result(config.benchmark, bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    results = {}
    for method in METHODS:
        target = make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
        results[method] = [load_result(config.benchmark, target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]

    panels = [
        (get_accuracy, "정확도", "정확도", ACCURACY_YMAX),
        (get_truncated_ratio, "잘림", "전체 문제 중 비율", 0.6),
        (get_loop_ratio, "반복", "전체 문제 중 비율", 0.6),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=get_figsize([get_panel_width(len(results), len(LANGUAGES))], len(panels)), layout="constrained")
    for ax, (get_value, title, ylabel, ymax) in zip(axes, panels):
        plot_language_bars(ax, results, bf16_results, get_value)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} {config.benchmark} loss\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))},bf16_seeds={','.join(map(str, config.bf16_seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, f"{config.benchmark}_loss")


if __name__ == "__main__":
    main()
