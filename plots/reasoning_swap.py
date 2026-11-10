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
LABELS = ["bf16 reasoning, bf16 response", "quantized reasoning, bf16 response", "bf16 reasoning, quantized response", "quantized reasoning, quantized response"]


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
    entry_name = "reasoning_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    seed_results = [[load_result(entry_name, target, f"reasoning_method={reasoning_method},response_method={response_method},max_generation_len={config.max_generation_len},seed={seed}") for reasoning_method, response_method in CELLS] for seed in config.seeds]

    groups = []
    used_counts = {}
    for language in LANGUAGES:
        # compare cells on problems all four used
        seed_accuracies = []
        seed_used_counts = []
        for cell_results in seed_results:
            used = set.intersection(*[set(cell_result[language]["problem_indices"]) for cell_result in cell_results])
            seed_used_counts.append(len(used))
            seed_accuracies.append([mean(correct for problem_index, correct in zip(cell_result[language]["problem_indices"], cell_result[language]["corrects"]) if problem_index in used) for cell_result in cell_results])
        accuracies = [mean(cell_accuracies) for cell_accuracies in zip(*seed_accuracies)]
        used_counts[language] = mean(seed_used_counts)
        texts = [format_bf16_ratio([cell_accuracies[0] for cell_accuracies in seed_accuracies]), *[format_ratio(accuracy, accuracies[0]) for accuracy in accuracies[1:]]]
        groups.append((accuracies, LANGUAGE_COLORS[language], texts))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(CELLS), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, LABELS, groups)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{config.method} (problems used {', '.join(f'{language} {used_counts[language]:.0f}' for language in LANGUAGES)})")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm reasoning swap\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=LANGUAGE_COLORS[language], label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_reasoning_swap")


if __name__ == "__main__":
    main()
