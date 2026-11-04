from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_length_ratio, get_loop_ratio
from utils.paths import make_target_path
from utils.plot import METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    languages: tuple[Literal["en", "zh", "sw"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 32768
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    results = {}
    for method in ["bf16", "rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        if result_exists("mmlu_prox_lite", target, file_name):
            results[method] = load_result("mmlu_prox_lite", target, file_name)
    methods = list(results)
    bf16_result = results["bf16"]
    languages = list(bf16_result)

    panel_widths = [get_panel_width(len(methods), 1)] * 3
    fig = plt.figure(figsize=get_figsize(panel_widths, len(languages)), layout="constrained")
    for subfig, language in zip(fig.subfigures(len(languages), 1, squeeze=False)[:, 0], languages):
        accuracies = []
        truncateds = []
        length_ratios = []
        loop_ratios = []
        for method in methods:
            result = results[method][language]
            accuracies.append(result["accuracy"])
            truncateds.append(mean(result["truncateds"]))
            length_ratios.append(get_length_ratio(result, bf16_result[language]))
            loop_ratios.append(get_loop_ratio(result))
        panels = [
            ("accuracy", accuracies, [f"{accuracy:.1%} ({accuracy / accuracies[0] - 1:+.1%})\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
            ("median reasoning tokens / bf16, both correct", length_ratios, [f"{length_ratio:.2f}x" for length_ratio in length_ratios]),
            ("looped reasoning ratio", loop_ratios, [f"{loop_ratio:.1%}" for loop_ratio in loop_ratios]),
        ]

        axes = subfig.subplots(1, len(panels), width_ratios=panel_widths)
        for ax, (ylabel, values, texts) in zip(axes, panels):
            plot_bars(ax, methods, [(values, [METHOD_COLORS[method] for method in methods], texts)])
            ax.set_ylabel(ylabel)
            ax.margins(y=0.2)
            ax.set_ylim(bottom=0)
        for ax in [axes[0], axes[2]]:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        subfig.suptitle(language)

    fig.suptitle(f"{config.model_name} mmlu_prox_lite {file_name}")
    fit_bar_width(fig, axes)
    save_figure(fig, config.model_name, "mmlu_prox_lite")


if __name__ == "__main__":
    main()
