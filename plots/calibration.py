from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, plot_language_bars, save_figure

METHODS = ["gptq", "awq"]
CASES = {"영어": ("en",), "스와힐리어": ("sw",), "텔루구어": ("te",), "다섯 언어 혼합": tuple(LANGUAGES)}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    fig, axes = plt.subplots(len(METHODS), 1, figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], len(METHODS)), layout="constrained")
    for ax, method in zip(axes, METHODS):
        results = {}
        for case, calibration_languages in CASES.items():
            hub_path = make_hub_path(method, config.model_name, calibration_languages, config.num_bits, config.group_size, config.num_chunks)
            results[case] = [load_result("mgsm", hub_path, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]
        plot_language_bars(ax, results, bf16_results, get_accuracy)
        ax.set_ylim(0, ACCURACY_YMAX)
        ax.set_ylabel("정확도")
        ax.set_title(method)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm calibration 언어 영향\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))},bf16_seeds={','.join(map(str, config.bf16_seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_calibration")


if __name__ == "__main__":
    main()
