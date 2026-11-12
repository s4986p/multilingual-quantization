from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_target_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGES, fit_bar_width, format_bf16_ratio, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure

CELLS = [("bf16", "bf16"), ("quantized", "bf16"), ("bf16", "quantized"), ("quantized", "quantized")]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    cell_results = [[load_result("reasoning_swap/mgsm", target, f"reasoning_method={reasoning_method},response_method={response_method},max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds] for reasoning_method, response_method in CELLS]

    series = []
    for language in LANGUAGES:
        cell_accuracies = [[result[language]["accuracy"] for result in results] for results in cell_results]
        accuracies = [mean(seed_accuracies) for seed_accuracies in cell_accuracies]
        texts = [format_bf16_ratio(cell_accuracies[0]), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]
        series.append((accuracies, LANGUAGE_COLORS[language], texts))

    method_names = {"bf16": "bf16", "quantized": config.method}
    slot_labels = [f"{method_names[reasoning_method]} 추론, {method_names[response_method]} 응답" for reasoning_method, response_method in CELLS]
    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(CELLS), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, slot_labels, series)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("정확도")
    ax.set_title(config.method)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm reasoning swap\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_reasoning_swap")


if __name__ == "__main__":
    main()
