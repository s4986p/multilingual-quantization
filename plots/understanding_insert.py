from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import CASE_WIDTH, INSERT_LOSS_YMAX, INSERT_LOSS_YMIN, INSERT_LOSS_YTICK, LANGUAGE_COLORS, QUANTIZATION_METHODS, add_language_legend, get_figsize, get_subset_accuracy, get_title, save_figure
from utils.understanding import get_understanding_problem_indices

# 삽입 없음 손실을 같은 언어 색의 연한 막대로 표시
NO_INSERT_ALPHA = 0.35
# 회복 비율을 표시할 양자화 방법, 나머지는 손실이 작아 비율이 의미 없음
RECOVERY_METHODS = ["mxfp4"]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def plot_insert_loss_bars(
    ax: Axes,
    cases: list[str],
    losses: dict[str, list[tuple[float, float]]],
) -> None:
    # case마다 (삽입 없음, 삽입) 손실을 받아 삽입 없음 막대 위에 삽입 손실 막대를 겹쳐 그림
    # 예시) 막대 위 "13.6%p\n→5.8%p\n57% 회복"
    for language_index, language in enumerate(LANGUAGES):
        for case_index, (no_insert_loss, insert_loss) in enumerate(losses[language]):
            x = case_index * CASE_WIDTH + language_index - (len(LANGUAGES) - 1) / 2
            ax.bar(x, no_insert_loss, width=1, color=LANGUAGE_COLORS[language], alpha=NO_INSERT_ALPHA)
            ax.bar(x, insert_loss, width=1, color=LANGUAGE_COLORS[language])
            texts = [f"{round(no_insert_loss * 100, 1) + 0.0:.1f}%p", f"→{round(insert_loss * 100, 1) + 0.0:.1f}%p"]
            if cases[case_index] in RECOVERY_METHODS:
                texts.append(f"{(no_insert_loss - insert_loss) / no_insert_loss:.0%} 회복")
            ax.annotate("\n".join(texts), (x, max(no_insert_loss, insert_loss, 0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#808080", linewidth=0.8)
    ax.set_xticks([case_index * CASE_WIDTH for case_index in range(len(cases))], cases)
    ax.set_xlim(-CASE_WIDTH / 2, (len(cases) - 0.5) * CASE_WIDTH)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))


def main() -> None:
    config = parse(Config)
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    # seed마다 해당 언어와 영어 bf16 reasoning 모두에 understanding이 있는 문제
    problem_indices = {language: [get_understanding_problem_indices(bf16_result, language) for bf16_result in bf16_results] for language in LANGUAGES}

    # 같은 seed의 bf16, 삽입 없음, 삽입 정확도를 같은 문제로 계산해 손실 비교
    losses = {language: [] for language in LANGUAGES}
    for method in QUANTIZATION_METHODS:
        hub_path = make_hub_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
        no_insert_results = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
        insert_results = load_results("bf16_native_understanding_insert/mgsm", hub_path, config.max_generation_len, config.seeds)
        for language in LANGUAGES:
            bf16_accuracy, no_insert_accuracy, insert_accuracy = [mean(get_subset_accuracy(result[language], indices) for result, indices in zip(results, problem_indices[language])) for results in [bf16_results, no_insert_results, insert_results]]
            losses[language].append((bf16_accuracy - no_insert_accuracy, bf16_accuracy - insert_accuracy))

    fig, ax = plt.subplots(figsize=get_figsize(len(QUANTIZATION_METHODS), 1), layout="constrained")
    plot_insert_loss_bars(ax, QUANTIZATION_METHODS, losses)
    ax.set_ylabel("손실")
    ax.set_ylim(INSERT_LOSS_YMIN, INSERT_LOSS_YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))

    fig.suptitle(get_title(config.model_name, "mgsm", "understanding 삽입"))
    num_problems = {language: round(mean(len(indices) for indices in problem_indices[language])) for language in LANGUAGES}
    add_language_legend(fig, bf16_results, num_problems, (Patch(color="#808080", alpha=NO_INSERT_ALPHA, label="삽입 없음 손실"),))
    save_figure(fig, config.model_name, "mgsm_understanding_insert")


if __name__ == "__main__":
    main()
