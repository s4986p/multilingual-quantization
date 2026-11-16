from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import QUANTIZATION_METHODS, add_language_legend, fit_bar_width, get_figsize, get_panel_width, plot_insert_loss_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    # qwen은 mxfp4만 돌림
    methods: tuple[Literal["gptq", "awq", "nf4", "mxfp4"], ...] = tuple(QUANTIZATION_METHODS)
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    methods = ["bf16", *config.methods]
    bf16_results = [load_result("mgsm", make_hub_path("bf16", config.model_name), file_name) for file_name in file_names]

    # 손실은 아무것도 넣지 않은 bf16 기준이라 bf16도 삽입하면 손실이 생김
    case_losses = {language: [] for language in LANGUAGES}
    for method in methods:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        results = [[load_result(experiment_name, hub_path, file_name) for file_name in file_names] for experiment_name in ("mgsm", "en_question_insert/mgsm")]
        for language in LANGUAGES:
            # 순서대로 bf16, 삽입 없음, 삽입 정확도
            accuracies = [mean(result[language]["accuracy"] for result in case_results) for case_results in [bf16_results, *results]]
            case_losses[language].append((accuracies[0] - accuracies[1], accuracies[0] - accuracies[2]))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(methods), len(LANGUAGES))], 1), layout="constrained")
    plot_insert_loss_bars(ax, methods, case_losses)
    # 삽입으로 오른 정확도가 모델마다 달라 범위를 데이터에 맞춤
    ax.margins(y=0.2)
    ax.set_ylabel("손실")
    fig.suptitle(f"{config.model_name} mgsm 영어 question 삽입\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    add_language_legend(fig, insert_label="삽입 없음 손실")
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_en_question_insert")


if __name__ == "__main__":
    main()
