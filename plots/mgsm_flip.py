from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_flip_ratios
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
    labels = {method: f"bf16 seed={config.seed + 1}" if method == "bf16" else method for method in results}

    gained_groups = []
    lost_groups = []
    for method, result in results.items():
        lost_ratios, gained_ratios = zip(*[get_flip_ratios(result[language], bf16_result[language]) for language in languages])
        gained_groups.append((gained_ratios, METHOD_COLORS[method], [f"+{gained_ratio:.1%}" for gained_ratio in gained_ratios]))
        lost_groups.append(([-lost_ratio for lost_ratio in lost_ratios], METHOD_COLORS[method], [f"-{lost_ratio:.1%}" for lost_ratio in lost_ratios]))

    panel_widths = [get_panel_width(len(languages), len(results))]
    fig, ax = plt.subplots(figsize=get_figsize(panel_widths, 1), layout="constrained")
    plot_bars(ax, languages, gained_groups)
    plot_bars(ax, languages, lost_groups)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"gained (up) and lost (down) vs bf16 seed={config.seed}")
    ax.margins(y=0.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f"{config.model_name} mgsm flip\n{file_name}\nratio of {len(bf16_result[languages[0]]['corrects'])} problems")
    handles = [Patch(color=METHOD_COLORS[method], label=label) for method, label in labels.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_flip")


if __name__ == "__main__":
    main()
