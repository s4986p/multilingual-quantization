from dataclasses import dataclass
from typing import Literal

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse
from transformers import AutoTokenizer

from utils.hub import load_result, result_exists
from utils.metrics import get_both_correct_mask
from utils.paths import make_bf16_path, make_target_path
from utils.plot import LANGUAGE_COLORS, get_figsize, get_panel_width, save_figure

LANGUAGES = list(LANGUAGE_COLORS)
NUM_PROGRESS_BINS = 5
ERROR_TYPE_COLORS = {"fact": "#2a78d6", "relation": "#eb6834", "question": "#1baf7a", "doubt": "#eda100", "arithmetic": "#e87ba4", "other": "#a3a29b"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    entry_name = "error_types/mgsm"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    file_names = [file_name for file_name in file_names if result_exists(entry_name, target, file_name)]
    results = [load_result(entry_name, target, file_name) for file_name in file_names]
    quantized_results = [load_result("mgsm", target, file_name) for file_name in file_names]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_results = [load_result("mgsm", make_bf16_path(config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in range(3)]

    ratios = {}
    used_counts = {}
    for language in LANGUAGES:
        used_counts[language] = sum(get_both_correct_mask([bf16_result[language] for bf16_result in bf16_results]))
        error_types = [error_type or "other" for result in results for error_type in result[language]["error_types"]]
        ratios[language] = {error_type: error_types.count(error_type) / (used_counts[language] * len(results)) for error_type in ERROR_TYPE_COLORS}

    progress_ratios = {}
    for language in LANGUAGES:
        bin_counts = [0] * NUM_PROGRESS_BINS
        for result, quantized_result in zip(results, quantized_results):
            for problem_index, error_line in zip(result[language]["problem_indices"], result[language]["error_lines"]):
                if error_line is None:
                    continue
                lines = quantized_result[language]["thinks"][problem_index].split("\n")
                error_start = len(tokenizer("\n".join(lines[: error_line - 1]), add_special_tokens=False)["input_ids"])
                bin_counts[min(error_start * NUM_PROGRESS_BINS // quantized_result[language]["think_lens"][problem_index], NUM_PROGRESS_BINS - 1)] += 1
        progress_ratios[language] = [bin_count / (used_counts[language] * len(results)) for bin_count in bin_counts]

    fig, axes = plt.subplot_mosaic([["types"], ["progress"]], figsize=get_figsize([get_panel_width(len(LANGUAGES), 1)], 2), layout="constrained")
    ax = axes["types"]
    for x, language in enumerate(LANGUAGES):
        bottom = 0
        for error_type, color in ERROR_TYPE_COLORS.items():
            ratio = ratios[language][error_type]
            ax.bar(x, ratio, bottom=bottom, width=0.7, color=color, edgecolor="white", linewidth=2)
            if ratio >= 0.015:
                ax.text(x, bottom + ratio / 2, f"{ratio:.1%}", ha="center", va="center", fontsize=8)
            bottom += ratio
        ax.annotate(f"{bottom:.1%}", (x, bottom), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(LANGUAGES)), LANGUAGES)
    ax.set_ylim(0, max(sum(ratios[language].values()) for language in LANGUAGES) * 1.15)
    ax.set_ylabel("ratio of problems bf16 solved")
    ax.set_title(f"{config.method} (problems used {', '.join(f'{language} {used_counts[language]}' for language in LANGUAGES)})")
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    ax = axes["progress"]
    for language in LANGUAGES:
        ax.plot(range(1, NUM_PROGRESS_BINS + 1), progress_ratios[language], color=LANGUAGE_COLORS[language], marker="o", linewidth=2)
    ax.set_xticks(range(1, NUM_PROGRESS_BINS + 1), [f"{(index - 1) * 20}-{index * 20}%" for index in range(1, NUM_PROGRESS_BINS + 1)])
    ax.set_xlabel("reasoning progress")
    ax.set_ylabel("ratio of problems bf16 solved")
    ax.set_title(f"{config.method} first error by reasoning progress")
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(handles=[Line2D([], [], color=LANGUAGE_COLORS[language], marker="o", linewidth=2, label=language) for language in LANGUAGES], ncol=len(LANGUAGES))

    fig.suptitle(f"{config.model_name} mgsm error types\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(color=color, label=error_type) for error_type, color in ERROR_TYPE_COLORS.items()], loc="outside lower center", ncol=len(ERROR_TYPE_COLORS))
    save_figure(fig, config.model_name, "mgsm_error_types")


if __name__ == "__main__":
    main()
