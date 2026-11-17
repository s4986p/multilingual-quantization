from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX, INSERT_LOSS_YTICK, METHODS, add_language_legend, get_accuracy, get_figsize, get_mean_values, get_no_insert_handle, get_title, plot_insert_bars, save_figure

# 삽입할 영어 question마다 hub 결과 경로, 섹션 이름, 그림 이름, 결과가 있는 방법
EN_QUESTIONS = {
    "mgsm": ("en_question_insert/mgsm", "영어 question 삽입", "mgsm_en_question_insert", METHODS),
    "bf16_translation": ("bf16_translation_insert/mgsm", "bf16 번역 삽입", "mgsm_bf16_translation_insert", ["bf16", "mxfp4"]),
}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    en_question: Literal["mgsm", "bf16_translation"] = "mgsm"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    experiment_name, section_name, file_name, methods = EN_QUESTIONS[config.en_question]
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    no_insert_results = {}
    insert_results = {}
    for method in methods:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        no_insert_results[method] = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        insert_results[method] = load_results(experiment_name, hub_path, config.max_generation_len, config.seeds)

    # 삽입 없는 bf16 정확도를 기준으로 삽입 없음, 삽입 손실, bf16을 넘으면 음수
    bf16_accuracies = get_mean_values({"bf16": bf16_results}, ["bf16"], get_accuracy)
    no_insert_accuracies = get_mean_values(no_insert_results, methods, get_accuracy)
    insert_accuracies = get_mean_values(insert_results, methods, get_accuracy)
    losses = {language: [(bf16_accuracies[language][0] - no_insert_accuracy, bf16_accuracies[language][0] - insert_accuracy) for no_insert_accuracy, insert_accuracy in zip(no_insert_accuracies[language], insert_accuracies[language])] for language in LANGUAGES}

    fig, ax = plt.subplots(figsize=get_figsize(len(methods), 1), layout="constrained")
    plot_insert_bars(ax, methods, losses, [])
    ax.set_ylabel("손실")
    ax.set_ylim(EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))

    fig.suptitle(get_title(config.model_name, "mgsm", section_name))
    add_language_legend(fig, bf16_results, extra_handles=(get_no_insert_handle(),))
    save_figure(fig, config.model_name, file_name)


if __name__ == "__main__":
    main()
