from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import VALUE_MARGIN, add_language_legend, get_figsize, get_losses, get_title, plot_language_bars, save_figure

CALIBRATION_METHODS = ["gptq", "awq"]
# x축 이름과 calibration 언어
CALIBRATION_CASES = {"영어": ("en",), "스와힐리어": ("sw",), "텔루구어": ("te",), "다섯 언어 혼합": tuple(LANGUAGES)}


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
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.bf16_seeds)

    fig, axes = plt.subplots(len(CALIBRATION_METHODS), 1, figsize=get_figsize(len(CALIBRATION_CASES), len(CALIBRATION_METHODS)), sharey=True, layout="constrained")
    for ax, method in zip(axes, CALIBRATION_METHODS):
        results = {}
        for case, calibration_languages in CALIBRATION_CASES.items():
            hub_path = make_hub_path(method, config.model_name, calibration_languages, config.num_bits, config.group_size, config.num_chunks)
            results[case] = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        plot_language_bars(ax, list(results), get_losses(bf16_results, results, list(results)), "%p")
        ax.set_title(method)
        ax.set_ylabel("손실")
    axes[0].margins(y=VALUE_MARGIN)

    fig.suptitle(get_title(config.model_name, "mgsm", "calibration 언어별 손실"))
    add_language_legend(fig, bf16_results)
    save_figure(fig, config.model_name, "mgsm_calibration")


if __name__ == "__main__":
    main()
