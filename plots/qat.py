from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.metrics import get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path, make_training_path
from utils.plot import ACCURACY_YMAX, add_language_legend, fit_bar_width, get_accuracy, get_figsize, get_panel_width, plot_language_bars, save_figure

NUM_SAMPLES = 20000


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    ce_kl_settings = ("ce_kl", "bf16_en_question_insert", "en_question", NUM_SAMPLES, NUM_SAMPLES)
    cases = {
        "mxfp4": make_hub_path("mxfp4", config.model_name),
        "ce_kl finetune": make_training_path("finetune", config.model_name, *ce_kl_settings),
        "ce_kl finetune mxfp4": make_training_path("finetune_mxfp4", config.model_name, *ce_kl_settings),
        "kl qat": make_training_path("qat", config.model_name, "kl", "bf16_native", "reasoning", NUM_SAMPLES, NUM_SAMPLES),
        "ce_kl qat": make_training_path("qat", config.model_name, *ce_kl_settings),
    }
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), file_name) for file_name in file_names]
    results = {case: [load_result("mgsm", hub_path, file_name) for file_name in file_names] for case, hub_path in cases.items()}

    panels = [
        (get_accuracy, "정확도", "정확도", ACCURACY_YMAX),
        (get_truncated_ratio, "잘림", "전체 문제 중 비율", 0.6),
        (get_loop_ratio, "반복", "전체 문제 중 비율", 0.6),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=get_figsize([get_panel_width(len(cases), len(LANGUAGES))], len(panels)), layout="constrained")
    for ax, (get_value, title, ylabel, ymax) in zip(axes, panels):
        plot_language_bars(ax, results, bf16_results, get_value)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm 영어 question과 reasoning qat\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, bf16_results)
    fit_bar_width(fig, [axes[0]])
    save_figure(fig, config.model_name, "mgsm_qat")


if __name__ == "__main__":
    main()
