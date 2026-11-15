from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_finetune_path, make_hub_path
from utils.plot import ACCURACY_YMAX, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, plot_language_bars, save_figure

CASES = {
    "ce, bf16 영어 understanding": ("ce", "bf16_en", "understanding"),
    "kl, bf16 영어 understanding": ("kl", "bf16_en", "understanding"),
    "kl, on policy understanding": ("kl", "on_policy", "understanding"),
    "kl, on policy reasoning": ("kl", "on_policy", "reasoning"),
    "hidden, on policy understanding": ("hidden", "on_policy", "understanding"),
}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    results = {}
    for case, (loss, reasoning_source, train_part) in CASES.items():
        hub_path = make_finetune_path(config.model_name, loss, reasoning_source, train_part)
        results[case] = [load_result("mgsm", hub_path, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.seeds]

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(CASES), len(LANGUAGES))], 1), layout="constrained")
    plot_language_bars(ax, results, bf16_results, get_accuracy)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("정확도")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm finetune\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))},bf16_seeds={','.join(map(str, config.bf16_seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_finetune")


if __name__ == "__main__":
    main()
