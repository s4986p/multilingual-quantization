from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_wrong_ratio, get_loop_ratio, get_truncated_ratio
from utils.paths import make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 32768
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    bf16_target = make_target_path("bf16", config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    bf16_file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in range(config.seed, config.seed + 3)]
    bf16_results = [load_result(config.benchmark, bf16_target, bf16_file_name) for bf16_file_name in bf16_file_names if result_exists(config.benchmark, bf16_target, bf16_file_name)]
    languages = list(bf16_results[0])

    results = {}
    for method in ["rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        if result_exists(config.benchmark, target, file_name):
            results[method] = load_result(config.benchmark, target, file_name)

    panels = [
        (get_truncated_ratio, [], "truncated", 0.7),
        (get_wrong_ratio, [], "wrong", 1),
        (get_loop_ratio, [], "looped", 0.8),
    ]
    for get_ratio, groups, _, _ in panels:
        for language in languages:
            bf16_ratios = [get_ratio(bf16_result[language]) for bf16_result in bf16_results]
            ratios = [get_ratio(result[language]) for result in results.values()]
            groups.append(([mean(bf16_ratios), *ratios], LANGUAGE_COLORS[language], [format_bf16_ratio(bf16_ratios), *[format_ratio(ratio, mean(bf16_ratios)) for ratio in ratios]]))

    labels = ["bf16", *results]
    panel_widths = [get_panel_width(len(labels), len(languages))]
    fig, axes = plt.subplots(3, 1, figsize=get_figsize(panel_widths, 3), layout="constrained")
    for ax, (_, groups, title, ymax) in zip(axes, panels):
        plot_bars(ax, labels, groups)
        # same scale for every model and benchmark, so figures can be compared
        ax.set_ylim(0, ymax)
        ax.set_ylabel("ratio of all problems")
        ax.set_title(title)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} {config.benchmark} loss\n{file_name}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} (bf16 {mean(bf16_result[language]['accuracy'] for bf16_result in bf16_results):.1%})") for language in languages]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, f"{config.benchmark}_loss")


if __name__ == "__main__":
    main()
