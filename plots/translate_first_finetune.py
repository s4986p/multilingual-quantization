from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_translate_first_finetune_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

LANGUAGES = list(LANGUAGE_COLORS)


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    languages: tuple[Literal["zh", "sw", "ja", "te"], ...] = ("zh", "sw", "ja", "te")
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_target = make_bf16_path(config.model_name)
    cells = {
        "bf16, native instruction": ("mgsm", bf16_target),
        "bf16, translate first instruction": ("translate_first/mgsm", bf16_target),
        "translate first finetune": ("mgsm", make_translate_first_finetune_path(config.model_name, config.languages)),
    }
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    results = {label: [load_result(entry_name, target, file_name) for file_name in file_names] for label, (entry_name, target) in cells.items()}

    groups = []
    for language in LANGUAGES:
        bf16_accuracies = [result[language]["accuracy"] for result in results["bf16, native instruction"]]
        accuracies = [mean(result[language]["accuracy"] for result in label_results) for label_results in results.values()]
        groups.append((accuracies, LANGUAGE_COLORS[language], [format_bf16_ratio(bf16_accuracies), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(cells), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, list(cells), groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("accuracy")
    ax.set_title("finetune")
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm translate first finetune\nlanguages={','.join(config.languages)},max_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_translate_first_finetune")


if __name__ == "__main__":
    main()
