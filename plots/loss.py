from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import CASE_WIDTH, QUANTIZATION_METHODS, add_language_legend, get_figsize, get_losses, get_mean_values, get_title, plot_language_bars, save_figure

# 잘림과 반복은 모델끼리 비교하기 쉽게 y축 고정
RATIO_YMAX = 0.6


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))


def main() -> None:
    config = parse(Config)
    results = {}
    for method, seeds in [("bf16", config.bf16_seeds)] + [(method, config.seeds) for method in QUANTIZATION_METHODS]:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        results[method] = load_results(config.benchmark, hub_path, config.max_generation_len, seeds)

    # 잘림과 반복은 bf16도 함께 표시
    fig, axes = plt.subplots(3, 1, figsize=get_figsize(len(results), 3), layout="constrained")
    plot_language_bars(axes[0], QUANTIZATION_METHODS, get_losses(results["bf16"], results, QUANTIZATION_METHODS), "%p")
    plot_language_bars(axes[1], list(results), get_mean_values(results, list(results), get_truncated_ratio), "%")
    plot_language_bars(axes[2], list(results), get_mean_values(results, list(results), get_loop_ratio), "%")
    for ax, ylabel in zip(axes, ["손실", "잘림", "반복"]):
        ax.set_ylabel(ylabel)
    # 손실에는 bf16 case가 없어서, 잘림, 반복과 양자화 방법 위치를 맞추려고 bf16 자리를 비워 둠
    axes[0].set_xlim(-1.5 * CASE_WIDTH, (len(QUANTIZATION_METHODS) - 0.5) * CASE_WIDTH)
    axes[0].margins(y=0.15)
    for ax in axes[1:]:
        ax.set_ylim(0, RATIO_YMAX)

    fig.suptitle(get_title(config.model_name, config.benchmark, "언어별 손실"))
    add_language_legend(fig, results["bf16"])
    save_figure(fig, config.model_name, f"{config.benchmark}_loss")


if __name__ == "__main__":
    main()
