from dataclasses import dataclass
from typing import Literal

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from simple_parsing import parse
import torch

from utils.hub import load_tensors, result_exists
from utils.metrics import split_problems
from utils.paths import make_target_path
from utils.plot import LANGUAGE_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure

LANGUAGES = list(LANGUAGE_COLORS)
NUM_PROGRESS_BINS = 5


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "awq"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 32768
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = f"teacher_forcing/{config.benchmark}"

    results = {}
    for method in ["rtn", "gptq", "awq", "nf4", "mxfp4"]:
        target = make_target_path(method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
        if result_exists(entry_name, target, file_name, "safetensors"):
            results[method] = load_tensors(entry_name, target, file_name)

    kl_groups = []
    for language in LANGUAGES:
        kl_means = []
        for result in results.values():
            problem_kls = torch.stack([kl.mean() for kl in split_problems(result[f"{language}/kl"], result[f"{language}/problem_index"])])
            kl_means.append(problem_kls.mean().item())
        kl_groups.append((kl_means, LANGUAGE_COLORS[language], [f"{kl_mean:.3f}" for kl_mean in kl_means]))

    panel_widths = [get_panel_width(len(results), len(LANGUAGES))]
    fig, axes = plt.subplot_mosaic([["kl"], ["progress"]], figsize=get_figsize(panel_widths, 2), layout="constrained")
    plot_bars(axes["kl"], list(results), kl_groups)
    axes["kl"].set_ylim(0, max(max(values) for values, _, _ in kl_groups) * 1.15)
    axes["kl"].set_ylabel("mean kl per problem")
    result = results[config.method]
    used_counts = {language: len(result[f"{language}/problem_index"].unique()) for language in LANGUAGES}
    axes["kl"].set_title(f"kl (problems used {', '.join(f'{language} {used_counts[language]}' for language in LANGUAGES)})")

    for language in LANGUAGES:
        progress_kls = torch.stack([torch.stack([bin_kl.mean() for bin_kl in kl.tensor_split(NUM_PROGRESS_BINS)]) for kl in split_problems(result[f"{language}/kl"], result[f"{language}/problem_index"])])
        axes["progress"].plot(range(1, NUM_PROGRESS_BINS + 1), progress_kls.mean(dim=0), color=LANGUAGE_COLORS[language], marker="o", linewidth=2)
    axes["progress"].set_xticks(range(1, NUM_PROGRESS_BINS + 1), [f"{(index - 1) * 20}-{index * 20}%" for index in range(1, NUM_PROGRESS_BINS + 1)])
    axes["progress"].set_xlabel("reasoning progress")
    axes["progress"].set_ylabel("mean kl per problem")
    axes["progress"].set_title(f"{config.method} kl by reasoning progress")

    for ax in axes.values():
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(f"{config.model_name} {config.benchmark} teacher forcing\n{file_name}")
    fig.legend(handles=[Line2D([], [], color=LANGUAGE_COLORS[language], marker="o", linewidth=2, label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [axes["kl"]])
    save_figure(fig, config.model_name, f"{config.benchmark}_teacher_forcing")


if __name__ == "__main__":
    main()
