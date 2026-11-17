from pathlib import Path
from statistics import mean, stdev

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils.dataset import LANGUAGES

QUANTIZATION_METHODS = ["gptq", "awq", "nf4", "mxfp4"]
MODEL_NAMES = {"Qwen/Qwen3.5-2B": "qwen3.5-2b", "google/gemma-4-E2B-it": "gemma-4-e2b-it"}
BENCHMARK_NAMES = {"mgsm": "mgsm", "mmlu_prox_lite": "mmlu-prox-lite"}
LANGUAGE_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "sw": "스와힐리어", "te": "텔루구어"}
LANGUAGE_COLORS = {"en": "#3c5a9a", "zh": "#de2910", "ja": "#f4a6b7", "sw": "#1eb53a", "te": "#ff9933"}
BAR_INCHES = 0.45
BAR_GAP = 1
CASE_WIDTH = len(LANGUAGES) + BAR_GAP
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.titlesize": 13,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.wspace": 0.08,
    "figure.constrained_layout.hspace": 0.08,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def get_title(
    model_name: str,
    benchmark: str,
    experiment_name: str,
) -> str:
    # 예시) "gemma-4-e2b-it mgsm 언어별 손실"
    return f"{MODEL_NAMES[model_name]} {BENCHMARK_NAMES[benchmark]} {experiment_name}"


def get_figsize(
    num_cases: int,
    num_panels: int,
) -> tuple[float, float]:
    return BAR_INCHES * CASE_WIDTH * num_cases + PANEL_MARGIN, PANEL_HEIGHT * num_panels + 1


def plot_language_bars(
    ax: Axes,
    cases: list[str],
    values: dict[str, list[float]],
    unit: str,
) -> None:
    # case마다 언어 막대를 나란히 그리고 막대 위에 값 표시
    # 예시) unit "%p" → 막대 위 "5.2%p", y축 "5%p"
    for language_index, language in enumerate(LANGUAGES):
        xs = [case_index * CASE_WIDTH + language_index - (len(LANGUAGES) - 1) / 2 for case_index in range(len(cases))]
        ax.bar(xs, values[language], width=1, color=LANGUAGE_COLORS[language])
        for x, value in zip(xs, values[language]):
            ax.annotate(f"{value * 100 + 0.0:.1f}{unit}", (x, max(value, 0)), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#808080", linewidth=0.8)
    ax.set_xticks([case_index * CASE_WIDTH for case_index in range(len(cases))], cases)
    ax.set_xlim(-CASE_WIDTH / 2, (len(cases) - 0.5) * CASE_WIDTH)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=unit))


def add_language_legend(
    fig: Figure,
    bf16_results: list[dict],
) -> None:
    # 언어마다 bf16 정확도의 seed 평균과 표준편차 표시
    handles = []
    for language in LANGUAGES:
        bf16_accuracies = [bf16_result[language]["accuracy"] for bf16_result in bf16_results]
        handles.append(Patch(color=LANGUAGE_COLORS[language], label=f"{LANGUAGE_NAMES[language]} (bf16 {mean(bf16_accuracies):.1%} ±{stdev(bf16_accuracies):.1%}p)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3)


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
