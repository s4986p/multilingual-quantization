from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_target_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGES, METHODS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

CELLS = [("bf16", "mgsm"), ("bf16", "en_question_insert/mgsm"), ("quantized", "mgsm"), ("quantized", "en_question_insert/mgsm")]
CELL_LABELS = {"mgsm": "해당 언어 프롬프트", "en_question_insert/mgsm": "en 질문 삽입"}


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
    fig, axes = plt.subplots(len(METHODS), 1, figsize=get_figsize([get_panel_width(len(CELLS), len(LANGUAGES))], len(METHODS)), layout="constrained")
    for ax, method in zip(axes, METHODS):
        targets = {"bf16": make_uncalibrated_path("bf16", config.model_name), "quantized": make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)}
        seed_results = [[load_result(script_name, targets[model_method], f"max_generation_len={config.max_generation_len},seed={seed}") for model_method, script_name in CELLS] for seed in config.seeds]

        series = []
        for language in LANGUAGES:
            seed_accuracies = [[cell_result[language]["accuracy"] for cell_result in cell_results] for cell_results in seed_results]
            accuracies = [mean(cell_accuracies) for cell_accuracies in zip(*seed_accuracies)]
            texts = [format_bf16_ratio([cell_accuracies[0] for cell_accuracies in seed_accuracies]), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]
            series.append((accuracies, LANGUAGE_COLORS[language], texts))

        method_names = {"bf16": "bf16", "quantized": method}
        plot_bars(ax, [f"{method_names[model_method]}, {CELL_LABELS[script_name]}" for model_method, script_name in CELLS], series)
        ax.set_ylim(0, ACCURACY_YMAX)
        ax.set_ylabel("정확도")
        ax.set_title(method)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm en question insert\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_en_question_insert")


if __name__ == "__main__":
    main()
