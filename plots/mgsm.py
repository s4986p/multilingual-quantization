from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_length_ratio, get_loop_ratio
from utils.paths import make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
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
    bf16_result = load_result("mgsm", bf16_target, file_name)
    languages = list(bf16_result)

    results = {}
    # bf16 with another seed shows sampling noise
    noise_file_name = f"max_generation_len={config.max_generation_len},seed={config.seed + 1}"
    if result_exists("mgsm", bf16_target, noise_file_name):
        results["bf16"] = load_result("mgsm", bf16_target, noise_file_name)
    for method in ["rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        if result_exists("mgsm", target, file_name):
            results[method] = load_result("mgsm", target, file_name)
    method_labels = [f"bf16 seed={config.seed + 1}" if method == "bf16" else method for method in results]

    accuracy_groups = []
    length_groups = []
    loop_groups = []
    for language in languages:
        accuracy_ratios = [result[language]["accuracy"] / bf16_result[language]["accuracy"] for result in results.values()]
        length_ratios = [get_length_ratio(result[language], bf16_result[language]) for result in results.values()]
        loop_ratios = [get_loop_ratio(result[language]) for result in results.values()]
        accuracy_groups.append((accuracy_ratios, LANGUAGE_COLORS[language], [f"{accuracy_ratio:.0%}" for accuracy_ratio in accuracy_ratios]))
        length_groups.append((length_ratios, LANGUAGE_COLORS[language], [f"{length_ratio:.2f}x" for length_ratio in length_ratios]))
        loop_groups.append((loop_ratios, LANGUAGE_COLORS[language], [f"{loop_ratio:.1%}" for loop_ratio in loop_ratios]))
    panels = [
        ("accuracy / bf16", accuracy_groups),
        ("median reasoning tokens / bf16, both correct", length_groups),
        ("looped reasoning ratio", loop_groups),
    ]
    panel_widths = [get_panel_width(len(results), len(languages))]
    fig, axes = plt.subplots(len(panels), 1, figsize=get_figsize(panel_widths, len(panels)), layout="constrained")
    for ax, (ylabel, groups) in zip(axes, panels):
        plot_bars(ax, method_labels, groups)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.set_ylim(bottom=0)
    for ax in [axes[0], axes[2]]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f"{config.model_name} mgsm {file_name}\nvs bf16 seed={config.seed}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} (bf16 {bf16_result[language]['accuracy']:.1%})") for language in languages]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm")


if __name__ == "__main__":
    main()
