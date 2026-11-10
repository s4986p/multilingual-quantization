from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_target_path
from utils.plot import ACCURACY_YMAX, BF16_COLOR, LANGUAGE_COLORS, LANGUAGES, fit_bar_width, format_bf16_accuracy, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

CALIBRATIONS = {"en": ("en",), "sw": ("sw",), "te": ("te",), "all": ("en", "zh", "sw", "ja", "te")}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    bf16_target = make_bf16_path(config.model_name)
    bf16_file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.bf16_seeds]
    bf16_results = [load_result(config.benchmark, bf16_target, bf16_file_name) for bf16_file_name in bf16_file_names]

    results = {calibration: load_result(config.benchmark, make_target_path("awq", config.model_name, calibration_languages, config.num_bits, config.group_size, config.num_samples), file_name) for calibration, calibration_languages in CALIBRATIONS.items()}

    groups = []
    bf16_accuracies = []
    for language in LANGUAGES:
        bf16_accuracies.append(mean(bf16_result[language]["accuracy"] for bf16_result in bf16_results))
        accuracies = [result[language]["accuracy"] for result in results.values()]
        groups.append((accuracies, LANGUAGE_COLORS[language], [format_ratio(accuracy, bf16_accuracies[-1]) for accuracy in accuracies]))

    labels = [f"calibrating on {calibration}" for calibration in results]
    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(labels), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, labels, groups, bf16_accuracies)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("accuracy")
    ax.set_title("awq")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} {config.benchmark} calibration\n{file_name}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} ({format_bf16_accuracy([bf16_result[language]['accuracy'] for bf16_result in bf16_results])})") for language in LANGUAGES]
    fig.legend(handles=[*handles, Patch(color=BF16_COLOR, label="bf16")], loc="outside lower center", ncol=len(handles) + 1)
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, f"{config.benchmark}_calibration")


if __name__ == "__main__":
    main()
