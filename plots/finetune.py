from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_finetune_path, make_uncalibrated_path
from utils.plot import ACCURACY_YMAX, LANGUAGES, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, plot_language_bars, save_figure

CASES = {
    "ce, 영어 문제 이해": ("ce", "en", "understanding"),
    "kl, 영어 문제 이해": ("kl", "en", "understanding"),
    "kl, on policy 문제 이해": ("kl", "on_policy", "understanding"),
    "kl, on policy 추론": ("kl", "on_policy", "reasoning"),
}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_target = make_uncalibrated_path("bf16", config.model_name)
    bf16_results = [load_result("mgsm", bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    results = {}
    for label, (loss, think_source, think_part) in CASES.items():
        target = make_finetune_path(config.model_name, loss, think_source, think_part)
        results[label] = [load_result("mgsm", target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(results), len(LANGUAGES))], 1), layout="constrained")
    plot_language_bars(ax, results, bf16_results, get_accuracy)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("정확도")
    ax.set_title("정확도")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm finetune\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))},bf16_seeds={','.join(map(str, config.bf16_seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_finetune")


if __name__ == "__main__":
    main()
