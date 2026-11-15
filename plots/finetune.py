from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.metrics import get_bf16_solved_mask, get_bf16_unsolved_mask
from utils.paths import make_finetune_path, make_hub_path
from utils.plot import ACCURACY_YMAX, LANGUAGE_COLORS, LANGUAGE_NAMES, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, get_subset_accuracy, plot_bars, plot_language_bars, save_figure

CASES = {
    "ce, bf16 영어 understanding": ("ce", "bf16_en", "understanding"),
    "kl, bf16 영어 understanding": ("kl", "bf16_en", "understanding"),
    "kl, on policy understanding": ("kl", "on_policy", "understanding"),
    "kl, on policy reasoning": ("kl", "on_policy", "reasoning"),
    "hidden, on policy understanding": ("hidden", "on_policy", "understanding"),
}
GROUPS = {"bf16이 맞힌 문제": get_bf16_solved_mask, "bf16이 틀린 문제": get_bf16_unsolved_mask}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), file_name) for file_name in file_names]
    results = {}
    for case, (loss, reasoning_source, train_part) in CASES.items():
        hub_path = make_finetune_path(config.model_name, loss, reasoning_source, train_part)
        results[case] = [load_result("mgsm", hub_path, file_name) for file_name in file_names]

    fig, axes = plt.subplots(1 + len(GROUPS), 1, figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], 1 + len(GROUPS)), layout="constrained")
    plot_language_bars(axes[0], results, bf16_results, get_accuracy)
    axes[0].set_title("모든 문제")

    for ax, (group, get_mask) in zip(axes[1:], GROUPS.items()):
        series = []
        bf16_values = []
        num_problems = {}
        for language in LANGUAGES:
            seed_problem_indices = []
            for seed_index in range(len(config.seeds)):
                mask = get_mask(bf16_results[:seed_index] + bf16_results[seed_index + 1 :], language)
                seed_problem_indices.append({problem_index for problem_index, selected in enumerate(mask) if selected})
            num_problems[language] = mean(map(len, seed_problem_indices))
            bf16_accuracies = [get_subset_accuracy(bf16_result[language], problem_indices) for bf16_result, problem_indices in zip(bf16_results, seed_problem_indices)]
            bf16_values.append(mean(bf16_accuracies))
            values = []
            texts = []
            for case_results in results.values():
                accuracies = [get_subset_accuracy(result[language], problem_indices) for result, problem_indices in zip(case_results, seed_problem_indices)]
                num_changed = mean((accuracy - bf16_accuracy) * len(problem_indices) for accuracy, bf16_accuracy, problem_indices in zip(accuracies, bf16_accuracies, seed_problem_indices))
                values.append(mean(accuracies))
                texts.append(f"{values[-1]:.1%}\n{round(num_changed):+d}문제")
            series.append((values, LANGUAGE_COLORS[language], texts))
        plot_bars(ax, list(CASES), series, bf16_values)
        ax.set_title(f"{group}\n(평균 문제 수 {', '.join(f'{LANGUAGE_NAMES[language]} {num_problems[language]:.0f}' for language in LANGUAGES)})")

    for ax in axes:
        ax.set_ylim(0, ACCURACY_YMAX)
        ax.set_ylabel("정확도")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm finetune\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_finetune")


if __name__ == "__main__":
    main()
