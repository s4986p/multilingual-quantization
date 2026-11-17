from dataclasses import dataclass
from functools import partial
import re
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from simple_parsing import parse

from utils.dataset import LANGUAGES, get_mgsm_question, load_mgsm
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import CASE_WIDTH, EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX, INSERT_LOSS_YTICK, METHODS, VALUE_MARGIN, add_language_legend, get_accuracy, get_figsize, get_mean_values, get_insert_handles, get_title, plot_insert_bars, plot_language_bars, save_figure

# reasoning 앞에서 영어 question 글자 수의 이 배까지만 봄, gemma는 번역 전에 할 일을 정리하는 단락을 먼저 씀
EN_QUESTION_WINDOW_RATIO = 4
# 영어 question 단어 중 이 비율 이상이 reasoning 앞부분에 나오면 번역을 쓴 것으로 봄
MIN_EN_QUESTION_WORD_RATIO = 0.7
# 번역을 쓴 비율의 기준이 되는 instruction 없는 bf16
NO_INSTRUCTION_CASE = "instruction 없음\nbf16"


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def get_words(
    text: str,
) -> set[str]:
    # 예시) "Janet's ducks lay 16 eggs" → {"janet", "s", "ducks", "lay", "16", "eggs"}
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def wrote_translation(
    reasoning: str,
    en_question: str,
) -> bool:
    en_words = get_words(en_question)
    window = reasoning[: EN_QUESTION_WINDOW_RATIO * len(en_question)]
    return len(en_words & get_words(window)) / len(en_words) >= MIN_EN_QUESTION_WORD_RATIO


def get_translation_ratio(
    en_questions: list[str],
    result: dict,
) -> float:
    # reasoning 앞부분에 같은 문제의 영어 번역을 쓴 비율
    return mean(wrote_translation(reasoning, en_question) for reasoning, en_question in zip(result["reasonings"], en_questions))


def main() -> None:
    config = parse(Config)
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    no_instruction_results = {}
    instruction_results = {}
    for method in METHODS:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        no_instruction_results[method] = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        instruction_results[method] = load_results("translate_first/mgsm", hub_path, config.max_generation_len, config.seeds)

    # 1. instruction 없는 bf16 정확도를 기준으로 instruction 없음, instruction 손실, bf16을 넘으면 음수
    bf16_accuracies = get_mean_values({"bf16": bf16_results}, ["bf16"], get_accuracy)
    no_instruction_accuracies = get_mean_values(no_instruction_results, METHODS, get_accuracy)
    instruction_accuracies = get_mean_values(instruction_results, METHODS, get_accuracy)
    losses = {language: [(bf16_accuracies[language][0] - no_instruction_accuracy, bf16_accuracies[language][0] - instruction_accuracy) for no_instruction_accuracy, instruction_accuracy in zip(no_instruction_accuracies[language], instruction_accuracies[language])] for language in LANGUAGES}

    # 2. instruction을 따라 reasoning 앞부분에 번역을 썼는지, instruction 없는 bf16과 비교
    en_questions = [get_mgsm_question(prompt) for prompt, _ in load_mgsm(["en"])["en"]]
    ratio_results = {NO_INSTRUCTION_CASE: bf16_results} | instruction_results
    translation_ratios = get_mean_values(ratio_results, list(ratio_results), partial(get_translation_ratio, en_questions))

    fig, axes = plt.subplots(2, 1, figsize=get_figsize(len(ratio_results), 2), layout="constrained")
    plot_insert_bars(axes[0], METHODS, losses, [])
    axes[0].set_ylabel("손실")
    axes[0].set_ylim(EN_INSERT_LOSS_YMIN, INSERT_LOSS_YMAX)
    axes[0].yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))
    # 손실에는 instruction 없는 bf16 case가 없어서, 번역 비율과 위치를 맞추려고 그 자리를 비워 둠
    axes[0].set_xlim(-1.5 * CASE_WIDTH, (len(METHODS) - 0.5) * CASE_WIDTH)
    plot_language_bars(axes[1], list(ratio_results), translation_ratios, "%")
    axes[1].set_ylabel("번역 씀")
    axes[1].margins(y=VALUE_MARGIN)

    fig.suptitle(get_title(config.model_name, "mgsm", "번역 먼저 쓰기"))
    add_language_legend(fig, bf16_results, extra_handles=get_insert_handles("instruction 없음", "instruction 있음"))
    save_figure(fig, config.model_name, "mgsm_translate_first")


if __name__ == "__main__":
    main()
