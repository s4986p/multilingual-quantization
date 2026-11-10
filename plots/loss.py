from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_target_path
from utils.plot import ACCURACY_YMAX, BF16_COLOR, LANGUAGE_COLORS, LANGUAGES, METHODS, fit_bar_width, format_bf16_accuracy, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def get_accuracy(
    result: dict,
) -> float:
    return result["accuracy"]


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
    bf16_target = make_bf16_path(config.model_name)
    bf16_file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.bf16_seeds]
    bf16_results = [load_result(config.benchmark, bf16_target, bf16_file_name) for bf16_file_name in bf16_file_names]

    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    results = {}
    for method in METHODS:
        target = make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
        results[method] = [load_result(config.benchmark, target, file_name) for file_name in file_names]

    panels = [
        (get_accuracy, "accuracy", "accuracy", ACCURACY_YMAX),
        (get_truncated_ratio, "truncated", "ratio of all problems", 0.6),
        (get_loop_ratio, "looped", "ratio of all problems", 0.6),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=get_figsize([get_panel_width(len(results), len(LANGUAGES))], len(panels)), layout="constrained")
    for ax, (get_value, title, ylabel, ymax) in zip(axes, panels):
        groups = []
        bf16_values = []
        for language in LANGUAGES:
            bf16_values.append(mean(get_value(bf16_result[language]) for bf16_result in bf16_results))
            values = [mean(get_value(result[language]) for result in method_results) for method_results in results.values()]
            groups.append((values, LANGUAGE_COLORS[language], [format_ratio(value, bf16_values[-1]) for value in values]))

        plot_bars(ax, list(results), groups, bf16_values)
        # same scale for every model, benchmark, so figures can be compared
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} {config.benchmark} loss\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} ({format_bf16_accuracy([bf16_result[language]['accuracy'] for bf16_result in bf16_results])})") for language in LANGUAGES]
    fig.legend(handles=[*handles, Patch(color=BF16_COLOR, label="bf16")], loc="outside lower center", ncol=len(handles) + 1)
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, f"{config.benchmark}_loss")


if __name__ == "__main__":
    main()
