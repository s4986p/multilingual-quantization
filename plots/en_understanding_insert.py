from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX, INSERT_LOSS_YTICK, METHODS, add_language_legend, get_figsize, get_no_insert_handle, get_subset_accuracy, get_title, plot_insert_bars, save_figure
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
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    # understanding 삽입과 같은 문제로 비교
    problem_indices = {language: [get_understanding_problem_indices(bf16_result, language) for bf16_result in bf16_results] for language in LANGUAGES}

    # 같은 seed의 삽입 없는 bf16 정확도를 기준으로 삽입 없음, 삽입 손실, bf16을 넘으면 음수
    losses = {language: [] for language in LANGUAGES}
    for method in METHODS:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        no_insert_results = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        insert_results = load_results("bf16_en_understanding_insert/mgsm", hub_path, config.max_generation_len, config.seeds)
        for language in LANGUAGES:
            bf16_accuracy, no_insert_accuracy, insert_accuracy = [mean(get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices[language])) for results in [bf16_results, no_insert_results, insert_results]]
            losses[language].append((bf16_accuracy - no_insert_accuracy, bf16_accuracy - insert_accuracy))

    fig, ax = plt.subplots(figsize=get_figsize(len(METHODS), 1), layout="constrained")
    plot_insert_bars(ax, METHODS, losses, [])
    ax.set_ylabel("손실")
    ax.set_ylim(EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))

    fig.suptitle(get_title(config.model_name, "mgsm", "영어 understanding 삽입"))
    num_problems = {language: round(mean(len(indices) for indices in problem_indices[language])) for language in LANGUAGES}
    add_language_legend(fig, bf16_results, num_problems, (get_no_insert_handle(),))
    save_figure(fig, config.model_name, "mgsm_en_understanding_insert")


if __name__ == "__main__":
    main()
