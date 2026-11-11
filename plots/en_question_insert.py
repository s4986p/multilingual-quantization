from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_target_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGES, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

CELLS = [("bf16", "mgsm"), ("bf16", "en_question_insert/mgsm"), ("quantized", "mgsm"), ("quantized", "en_question_insert/mgsm")]
LABELS = ["bf16, native prompt", "bf16, en question insert", "quantized, native prompt", "quantized, en question insert"]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    targets = {"bf16": make_bf16_path(config.model_name), "quantized": make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)}
    seed_results = [[load_result(entry_name, targets[model_method], f"max_generation_len={config.max_generation_len},seed={seed}") for model_method, entry_name in CELLS] for seed in config.seeds]

    groups = []
    for language in LANGUAGES:
        seed_accuracies = [[cell_result[language]["accuracy"] for cell_result in cell_results] for cell_results in seed_results]
        accuracies = [mean(cell_accuracies) for cell_accuracies in zip(*seed_accuracies)]
        texts = [format_bf16_ratio([cell_accuracies[0] for cell_accuracies in seed_accuracies]), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]
        groups.append((accuracies, LANGUAGE_COLORS[language], texts))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(CELLS), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, LABELS, groups)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("accuracy")
    ax.set_title(config.method)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm en question insert\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_en_question_insert")


if __name__ == "__main__":
    main()
