from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import QUANTIZATION_METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, get_subset_accuracy, plot_insert_loss_bars, save_figure
from utils.understanding import get_understanding_problem_indices


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), file_name) for file_name in file_names]
    problem_indices = {language: [get_understanding_problem_indices(bf16_result, language) for bf16_result in bf16_results] for language in LANGUAGES}
    case_losses = {language: [] for language in LANGUAGES}
    for method in QUANTIZATION_METHODS:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        results = [[load_result(experiment_name, hub_path, file_name) for file_name in file_names] for experiment_name in ("mgsm", "bf16_native_understanding_insert/mgsm")]
        for language in LANGUAGES:
            # 순서대로 bf16, 삽입 없음, 삽입 정확도
            accuracies = [mean(get_subset_accuracy(result[language], indices) for result, indices in zip(case_results, problem_indices[language])) for case_results in [bf16_results, *results]]
            case_losses[language].append((accuracies[0] - accuracies[1], accuracies[0] - accuracies[2]))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(QUANTIZATION_METHODS), len(LANGUAGES))], 1), layout="constrained")
    plot_insert_loss_bars(ax, QUANTIZATION_METHODS, case_losses, ("mxfp4",))
    ax.set_ylim(0, 0.22)
    ax.set_ylabel("손실")
    fig.suptitle(f"{config.model_name} mgsm bf16이 해당 언어 prompt에 쓴 understanding 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, insert_label="삽입 없음 손실")
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_bf16_native_understanding_insert")


if __name__ == "__main__":
    main()
