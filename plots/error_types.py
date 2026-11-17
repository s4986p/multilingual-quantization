from dataclasses import dataclass
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES
from utils.hub import load_results
from utils.metrics import get_bf16_solved_mask
from utils.paths import make_hub_path
from utils.plot import LANGUAGE_NAMES, NUM_POSITION_BINS, POSITION_LABELS, get_figsize, get_title, save_figure

ERROR_TYPE_NAMES = {"fact": "사실 오류", "relation": "관계 오류", "target": "목표 오류", "doubt": "조건 무시", "arithmetic": "계산 오류", "other": "기타"}
ERROR_TYPE_COLORS = {"fact": "#2a78d6", "relation": "#eb6834", "target": "#1baf7a", "doubt": "#eda100", "arithmetic": "#e87ba4", "other": "#a3a29b"}
# 가장 높은 막대의 이 비율보다 작은 조각은 값 생략
MIN_LABEL_RATIO = 0.05
# 막대 위 합이 제목과 겹치지 않게 가장 높은 막대보다 y축을 늘리는 비율
YMAX_RATIO = 1.15


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def plot_error_type_bars(
    ax: Axes,
    cases: list[str],
    ratios: list[dict[str, float]],
) -> None:
    # case마다 오류 유형을 쌓은 막대를 그리고, 조각 안에 비율, 막대 위에 합 표시
    max_total = max(sum(case_ratios.values()) for case_ratios in ratios)
    for x, case_ratios in enumerate(ratios):
        bottom = 0
        for error_type, color in ERROR_TYPE_COLORS.items():
            ratio = case_ratios[error_type]
            ax.bar(x, ratio, bottom=bottom, width=0.7, color=color, edgecolor="white", linewidth=2)
            if ratio >= max_total * MIN_LABEL_RATIO:
                ax.text(x, bottom + ratio / 2, f"{ratio:.1%}", ha="center", va="center", fontsize=8)
            bottom += ratio
        ax.annotate(f"{bottom:.1%}", (x, bottom), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(cases)), cases)
    ax.set_ylim(0, max_total * YMAX_RATIO)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    results = load_results("error_types/mgsm", hub_path, config.max_generation_len, config.seeds)
    quantized_results = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)
    bf16_results = load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.bf16_seeds)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 1. 언어마다 모든 bf16 seed가 맞힌 문제 중 오류 유형별 비율, 판정을 못 읽으면 기타
    language_ratios = []
    num_problems = {}
    for language in LANGUAGES:
        num_problems[language] = sum(get_bf16_solved_mask(bf16_results, language))
        error_types = [error_type or "other" for result in results for error_type in result[language]["error_types"]]
        language_ratios.append({error_type: error_types.count(error_type) / (num_problems[language] * len(results)) for error_type in ERROR_TYPE_COLORS})

    # 2. 모든 언어의 첫 오류를 그 줄의 reasoning 위치 구간으로 나눠 유형별 비율
    position_counts = [dict.fromkeys(ERROR_TYPE_COLORS, 0) for _ in range(NUM_POSITION_BINS)]
    for language in LANGUAGES:
        for result, quantized_result in zip(results, quantized_results):
            for problem_index, error_line, error_type in zip(result[language]["problem_indices"], result[language]["error_lines"], result[language]["error_types"]):
                if error_line is None:
                    continue
                lines = quantized_result[language]["reasonings"][problem_index].split("\n")
                error_token_index = len(tokenizer("\n".join(lines[: error_line - 1]), add_special_tokens=False)["input_ids"])
                position_index = min(error_token_index * NUM_POSITION_BINS // quantized_result[language]["reasoning_lens"][problem_index], NUM_POSITION_BINS - 1)
                position_counts[position_index][error_type or "other"] += 1
    num_errors = sum(sum(counts.values()) for counts in position_counts)
    position_ratios = [{error_type: count / num_errors for error_type, count in counts.items()} for counts in position_counts]

    # reasoning 위치 눈금 이름이 길어서 언어 막대 묶음 3개 너비
    fig, axes = plt.subplots(2, 1, figsize=get_figsize(3, 2), layout="constrained")
    # 예시) "스와힐리어\n166문제"
    plot_error_type_bars(axes[0], [f"{LANGUAGE_NAMES[language]}\n{num_problems[language]}문제" for language in LANGUAGES], language_ratios)
    axes[0].set_title(config.method)
    axes[0].set_ylabel("오류 비율")
    plot_error_type_bars(axes[1], POSITION_LABELS, position_ratios)
    axes[1].set_title(f"{config.method} 모든 언어")
    axes[1].set_ylabel("첫 오류 중 비율")

    fig.suptitle(get_title(config.model_name, "mgsm", "오류 유형"))
    fig.legend(handles=[Patch(color=color, label=ERROR_TYPE_NAMES[error_type]) for error_type, color in ERROR_TYPE_COLORS.items()], loc="outside lower center", ncol=3)
    save_figure(fig, config.model_name, "mgsm_error_types")


if __name__ == "__main__":
    main()
