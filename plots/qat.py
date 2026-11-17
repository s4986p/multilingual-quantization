from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.paths import make_hub_path, make_training_path
from utils.plot import add_language_legend, get_figsize, get_losses, get_title, plot_language_bars, save_figure

NUM_SAMPLES = 20000
CE_KL_SETTINGS = ("ce_kl", "bf16_en_question_insert", "en_question", NUM_SAMPLES, NUM_SAMPLES)


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    # x축 이름과 hub 경로
    cases = {
        "mxfp4": make_hub_path("mxfp4", config.model_name),
        "ce_kl finetune": make_training_path("finetune", config.model_name, *CE_KL_SETTINGS),
        "ce_kl finetune\nmxfp4": make_training_path("finetune_mxfp4", config.model_name, *CE_KL_SETTINGS),
        "kl qat": make_training_path("qat", config.model_name, "kl", "bf16_native", "reasoning", NUM_SAMPLES, NUM_SAMPLES),
        "ce_kl qat": make_training_path("qat", config.model_name, *CE_KL_SETTINGS),
    }
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    results = {case: load_results("mgsm", hub_path, config.max_generation_len, config.seeds) for case, hub_path in cases.items()}

    fig, ax = plt.subplots(figsize=get_figsize(len(cases), 1), layout="constrained")
    plot_language_bars(ax, list(cases), get_losses(bf16_results, results, list(cases)), "%p")
    ax.set_ylabel("손실")
    ax.margins(y=0.15)

    fig.suptitle(get_title(config.model_name, "mgsm", "qat"))
    add_language_legend(fig, bf16_results)
    save_figure(fig, config.model_name, "mgsm_qat")


if __name__ == "__main__":
    main()
