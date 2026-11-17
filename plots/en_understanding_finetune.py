from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.paths import make_hub_path, make_training_path
from utils.plot import VALUE_MARGIN, add_language_legend, get_figsize, get_losses, get_title, plot_language_bars, save_figure

# x축 이름과 (loss, reasoning_source, train_part)
CASES = {
    "ce\nbf16 영어 understanding": ("ce", "bf16_en", "understanding"),
    "kl\nbf16 영어 understanding": ("kl", "bf16_en", "understanding"),
    "kl\non policy understanding": ("kl", "on_policy", "understanding"),
    "kl\non policy reasoning": ("kl", "on_policy", "reasoning"),
    "hidden\non policy understanding": ("hidden", "on_policy", "understanding"),
}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    results = {case: load_results("mgsm", make_training_path("finetune", config.model_name, loss, reasoning_source, train_part), config.max_generation_len, config.seeds) for case, (loss, reasoning_source, train_part) in CASES.items()}

    fig, ax = plt.subplots(figsize=get_figsize(len(CASES), 1), layout="constrained")
    plot_language_bars(ax, list(CASES), get_losses(bf16_results, results, list(CASES)), "%p")
    ax.set_ylabel("손실")
    ax.margins(y=VALUE_MARGIN)

    fig.suptitle(get_title(config.model_name, "mgsm", "영어 understanding finetune"))
    add_language_legend(fig, bf16_results)
    save_figure(fig, config.model_name, "mgsm_en_understanding_finetune")


if __name__ == "__main__":
    main()
