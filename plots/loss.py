from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_loop_ratio, get_truncated_ratio, get_wrong_ratio
from utils.paths import make_bf16_path, make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_target = make_bf16_path(config.model_name)
    bf16_file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.bf16_seeds]
    bf16_results = [load_result(config.benchmark, bf16_target, bf16_file_name) for bf16_file_name in bf16_file_names if result_exists(config.benchmark, bf16_target, bf16_file_name)]
    languages = list(bf16_results[0])

    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    results = {}
    for method in ["rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        method_results = [load_result(config.benchmark, target, file_name) for file_name in file_names if result_exists(config.benchmark, target, file_name)]
        if method_results:
            results[method] = method_results

    panels = [
        (get_truncated_ratio, [], "truncated", 0.7),
        (get_wrong_ratio, [], "wrong", 1),
        (get_loop_ratio, [], "looped", 0.8),
    ]
    for get_ratio, groups, _, _ in panels:
        for language in languages:
            bf16_ratios = [get_ratio(bf16_result[language]) for bf16_result in bf16_results]
            ratios = [mean(get_ratio(result[language]) for result in method_results) for method_results in results.values()]
            groups.append(([mean(bf16_ratios), *ratios], LANGUAGE_COLORS[language], [format_bf16_ratio(bf16_ratios), *[format_ratio(ratio, mean(bf16_ratios)) for ratio in ratios]]))

    labels = ["bf16", *results]
    panel_widths = [get_panel_width(len(labels), len(languages))]
    fig, axes = plt.subplots(3, 1, figsize=get_figsize(panel_widths, 3), layout="constrained")
    for ax, (_, groups, title, ymax) in zip(axes, panels):
        plot_bars(ax, labels, groups)
        # same scale for every model, benchmark, so figures can be compared
        ax.set_ylim(0, ymax)
        ax.set_ylabel("ratio of all problems")
        ax.set_title(title)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} {config.benchmark} loss\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} (bf16 {mean(bf16_result[language]['accuracy'] for bf16_result in bf16_results):.1%})") for language in languages]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, f"{config.benchmark}_loss")


if __name__ == "__main__":
    main()
