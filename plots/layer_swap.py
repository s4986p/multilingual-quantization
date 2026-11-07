from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.paths import make_bf16_path, make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

LANGUAGES = list(LANGUAGE_COLORS)
# (model_method, entry_name, front_method), and bf16, quantized bars come from instruction language
CELLS = [("bf16", "instruction_language/mgsm", None), ("quantized", "layer_swap/mgsm", "bf16"), ("quantized", "layer_swap/mgsm", "quantized"), ("quantized", "instruction_language/mgsm", None)]
LABELS = ["bf16", "bf16 front, quantized back", "quantized front, bf16 back", "quantized"]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    num_front_layers: tuple[int, ...] = (3, 15)
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    targets = {"bf16": make_bf16_path(config.model_name), "quantized": make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)}

    # 1) keep num_front_layers that have all cells of all seeds
    results = {}
    for num_front_layers in config.num_front_layers:
        seed_file_names = []
        for seed in config.seeds:
            file_name = f"max_generation_len={config.max_generation_len},seed={seed}"
            seed_file_names.append([file_name if front_method is None else f"front_method={front_method},num_front_layers={num_front_layers},{file_name}" for _, _, front_method in CELLS])
        if all(result_exists(entry_name, targets[model_method], file_name) for file_names in seed_file_names for (model_method, entry_name, _), file_name in zip(CELLS, file_names)):
            results[num_front_layers] = [[load_result(entry_name, targets[model_method], file_name) for (model_method, entry_name, _), file_name in zip(CELLS, file_names)] for file_names in seed_file_names]

    panel_widths = [get_panel_width(len(CELLS), len(LANGUAGES))]
    fig, axes = plt.subplots(len(results), 1, figsize=get_figsize(panel_widths, len(results)), layout="constrained", squeeze=False)
    for ax, (num_front_layers, seed_results) in zip(axes[:, 0], results.items()):
        groups = []
        for language in LANGUAGES:
            seed_accuracies = [[cell_result[language]["accuracy"] for cell_result in cell_results] for cell_results in seed_results]
            accuracies = [mean(cell_accuracies) for cell_accuracies in zip(*seed_accuracies)]
            texts = [format_bf16_ratio([cell_accuracies[0] for cell_accuracies in seed_accuracies]), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]
            groups.append((accuracies, LANGUAGE_COLORS[language], texts))

        plot_bars(ax, LABELS, groups)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{config.method} (front {num_front_layers})")
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm layer swap\nnum_front_layers={','.join(map(str, results))},max_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0, 0]])
    save_figure(fig, config.model_name, "mgsm_layer_swap")


if __name__ == "__main__":
    main()
