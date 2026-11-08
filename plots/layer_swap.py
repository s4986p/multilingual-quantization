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
LAYER_RANGES = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 35)]
PANELS = {"recovery": "bf16", "damage": "quantized"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    entry_name = "layer_swap/mgsm"
    targets = {"bf16": make_bf16_path(config.model_name), "quantized": make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)}
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]

    # 1) keep layer ranges that have both panels of all seeds
    layer_ranges = [(start_layer, end_layer) for start_layer, end_layer in LAYER_RANGES if all(result_exists(entry_name, targets["quantized"], f"swap_method={swap_method},start_layer={start_layer},end_layer={end_layer},{file_name}") for swap_method in PANELS.values() for file_name in file_names)]

    panel_widths = [get_panel_width(len(layer_ranges) + 1, len(LANGUAGES))]
    fig, axes = plt.subplots(len(PANELS), 1, figsize=get_figsize(panel_widths, len(PANELS)), layout="constrained")
    for ax, (panel, swap_method) in zip(axes, PANELS.items()):
        # 2) first bar is model without swap, from instruction language
        base_method = "quantized" if swap_method == "bf16" else "bf16"
        base_results = [load_result("instruction_language/mgsm", targets[base_method], file_name) for file_name in file_names]
        swap_results = [[load_result(entry_name, targets["quantized"], f"swap_method={swap_method},start_layer={start_layer},end_layer={end_layer},{file_name}") for file_name in file_names] for start_layer, end_layer in layer_ranges]
        groups = []
        for language in LANGUAGES:
            base_accuracies = [base_result[language]["accuracy"] for base_result in base_results]
            accuracies = [mean(seed_result[language]["accuracy"] for seed_result in seed_results) for seed_results in swap_results]
            texts = [format_bf16_ratio(base_accuracies), *[format_ratio(accuracy, mean(base_accuracies)) for accuracy in accuracies]]
            groups.append(([mean(base_accuracies), *accuracies], LANGUAGE_COLORS[language], texts))

        plot_bars(ax, [base_method, *[f"{start_layer}~{end_layer}" for start_layer, end_layer in layer_ranges]], groups)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{config.method} {panel}")
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm layer swap\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_layer_swap")


if __name__ == "__main__":
    main()
