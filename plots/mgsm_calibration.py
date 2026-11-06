from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_bf16_path, make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure

CALIBRATIONS = {"en": ("en",), "sw": ("sw",), "te": ("te",), "all": ("en", "zh", "sw", "ja", "te")}


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 32768
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    bf16_result = load_result("mgsm", make_bf16_path(config.model_name), file_name)
    languages = list(bf16_result)

    results = {}
    for calibration, calibration_languages in CALIBRATIONS.items():
        target = make_target_path("awq", config.model_name, calibration_languages, config.num_bits, config.group_size, config.num_samples)
        results[calibration] = load_result("mgsm", target, file_name)

    accuracy_groups = []
    truncated_groups = []
    for language in languages:
        accuracies = [result[language]["accuracy"] for result in results.values()]
        truncated_ratios = [mean(result[language]["truncateds"]) for result in results.values()]
        accuracy_groups.append((accuracies, LANGUAGE_COLORS[language], [f"{accuracy:.1%}" for accuracy in accuracies]))
        truncated_groups.append((truncated_ratios, LANGUAGE_COLORS[language], [f"{truncated_ratio:.1%}" for truncated_ratio in truncated_ratios]))

    panel_widths = [get_panel_width(len(results), len(languages))]
    fig, axes = plt.subplots(2, 1, figsize=get_figsize(panel_widths, 2), layout="constrained")
    for ax, groups, ylabel, top in zip(axes, [accuracy_groups, truncated_groups], ["accuracy", "truncated"], [1, 0.25]):
        plot_bars(ax, list(results), groups)
        # same scale for every model and benchmark, so figures can be compared
        ax.set_ylim(0, top)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("awq calibration language")
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f"{config.model_name} mgsm calibration\n{file_name}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} (bf16 {bf16_result[language]['accuracy']:.1%})") for language in languages]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    # panels share one column, so fit width by one panel
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_calibration")


if __name__ == "__main__":
    main()
