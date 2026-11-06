from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_loss_ratios
from utils.paths import make_target_path
from utils.plot import fit_bar_width, get_figsize, get_panel_width, plot_stacked_bars, save_figure


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
    bf16_result = load_result("mmlu_prox_lite", bf16_target, file_name)
    languages = list(bf16_result)

    results = {}
    # bf16 with another seed shows sampling noise
    noise_file_name = f"max_generation_len={config.max_generation_len},seed={config.seed + 1}"
    if result_exists("mmlu_prox_lite", bf16_target, noise_file_name):
        results["bf16"] = load_result("mmlu_prox_lite", bf16_target, noise_file_name)
    for method in ["rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        if result_exists("mmlu_prox_lite", target, file_name):
            results[method] = load_result("mmlu_prox_lite", target, file_name)
    method_labels = [f"bf16 seed={config.seed + 1}" if method == "bf16" else method for method in results]

    ratios = [get_loss_ratios(result[language], bf16_result[language]) for result in results.values() for language in languages]
    truncated_ratios, finished_ratios = zip(*ratios)
    colors = {"truncated": "tab:orange", "finished but wrong": "tab:blue"}

    panel_widths = [get_panel_width(len(results), len(languages))]
    fig, ax = plt.subplots(figsize=get_figsize(panel_widths, 1), layout="constrained")
    plot_stacked_bars(ax, method_labels, languages, [(truncated_ratios, colors["truncated"]), (finished_ratios, colors["finished but wrong"])])
    # same scale for every model and benchmark, so figures can be compared
    ax.set_ylim(0, 0.8)
    ax.set_ylabel(f"lost among bf16 seed={config.seed} correct")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f"{config.model_name} mmlu_prox_lite loss\n{file_name}")
    handles = [Patch(color=color, label=label) for label, color in colors.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mmlu_prox_lite_loss")


if __name__ == "__main__":
    main()
