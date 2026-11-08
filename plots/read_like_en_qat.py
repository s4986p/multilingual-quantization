from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_mxfp4_path, make_qat_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

LANGUAGES = list(LANGUAGE_COLORS)


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en", "zh", "sw", "ja", "te")
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    targets = {
        "bf16": make_bf16_path(config.model_name),
        "mxfp4": make_mxfp4_path(config.model_name),
        "qat same": make_qat_path(config.model_name, config.languages, "same"),
        "qat en": make_qat_path(config.model_name, config.languages, "en"),
    }
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    results = {label: [load_result("mgsm", target, file_name) for file_name in file_names] for label, target in targets.items()}

    groups = []
    for language in LANGUAGES:
        bf16_accuracies = [result[language]["accuracy"] for result in results["bf16"]]
        accuracies = [mean(result[language]["accuracy"] for result in label_results) for label_results in results.values()]
        groups.append((accuracies, LANGUAGE_COLORS[language], [format_bf16_ratio(bf16_accuracies), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(targets), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, list(targets), groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm read like en qat\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_read_like_en_qat")


if __name__ == "__main__":
    main()
