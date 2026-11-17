from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

from utils.dataset import LANGUAGES

QUANTIZATION_METHODS = ["gptq", "awq", "nf4", "mxfp4"]
# bf16도 삽입을 받는 plot은 bf16을 함께 표시
METHODS = ["bf16", *QUANTIZATION_METHODS]
MODEL_NAMES = {"Qwen/Qwen3.5-2B": "qwen3.5-2b", "google/gemma-4-E2B-it": "gemma-4-e2b-it"}
BENCHMARK_NAMES = {"mgsm": "mgsm", "mmlu_prox_lite": "mmlu-prox-lite"}
LANGUAGE_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "sw": "스와힐리어", "te": "텔루구어"}
LANGUAGE_COLORS = {"en": "#3c5a9a", "zh": "#de2910", "ja": "#f4a6b7", "sw": "#1eb53a", "te": "#ff9933"}
BAR_INCHES = 0.45
BAR_GAP = 1
CASE_WIDTH = len(LANGUAGES) + BAR_GAP
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9
# 손실은 %p, 비율은 %, 나머지는 format_value에서 kl과 배로 표시
PERCENT_UNITS = ["%p", "%"]
# 잘림과 반복은 plot끼리 비교하기 쉽게 y축 고정
RATIO_YMAX = 1.0
# bf16 대비 배는 0.5배 간격 눈금
RATIO_YTICK = 0.5
# case가 적어도 3열 범례가 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 삽입 plot끼리 비교하기 쉽게 y축 고정, 삽입하면 bf16보다 높아져 손실이 음수일 수 있음
INSERT_LOSS_YMIN = -0.05
INSERT_LOSS_YMAX = 0.2
INSERT_LOSS_YTICK = 0.05
# 영어 삽입은 bf16을 넘어 손실이 크게 음수
EN_INSERT_LOSS_YMIN = -0.2
# 막대 위 값이 그림 밖으로 잘리지 않게 y축 위아래 여백 비율
VALUE_MARGIN = 0.15
# 삽입 없음을 같은 언어 색의 연한 막대로 표시
NO_INSERT_ALPHA = 0.35
# 막대 위 값 글자 크기와 막대에서 띄우는 point 수
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
# 기준선과 삽입 범례의 회색
NEUTRAL_COLOR = "#808080"
BASELINE_WIDTH = 0.8
NUM_POSITION_BINS = 5
# 예시) ["reasoning 0~20%", ..., "reasoning 80~100%"]
POSITION_LABELS = [f"reasoning {index * 100 // NUM_POSITION_BINS}~{(index + 1) * 100 // NUM_POSITION_BINS}%" for index in range(NUM_POSITION_BINS)]

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


def get_accuracy(
    result: dict,
) -> float:
    return result["accuracy"]


def get_subset_accuracy(
    result: dict,
    problem_indices: set[int],
) -> float:
    # 일부 문제만 넣은 result는 problem_indices가 있고, 없으면 모든 문제
    result_problem_indices = result.get("problem_indices", range(len(result["corrects"])))
    return mean(correct for problem_index, correct in zip(result_problem_indices, result["corrects"]) if problem_index in problem_indices)


def get_mean_values(
    results: dict[str, list[dict]],
    cases: list[str],
    get_value: Callable[[dict], float],
) -> dict[str, list[float]]:
    # 언어와 case마다 seed 평균
    return {language: [mean(get_value(result[language]) for result in results[case]) for case in cases] for language in LANGUAGES}


def get_losses(
    bf16_results: list[dict],
    results: dict[str, list[dict]],
    cases: list[str],
) -> dict[str, list[float]]:
    # 언어와 case마다 bf16 정확도에서 정확도를 뺀 값
    bf16_accuracies = get_mean_values({"bf16": bf16_results}, ["bf16"], get_accuracy)
    accuracies = get_mean_values(results, cases, get_accuracy)
    return {language: [bf16_accuracies[language][0] - accuracy for accuracy in accuracies[language]] for language in LANGUAGES}


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
    return max(BAR_INCHES * CASE_WIDTH * num_cases + PANEL_MARGIN, MIN_FIGURE_WIDTH), PANEL_HEIGHT * num_panels + 1


def format_value(
    value: float,
    unit: str,
) -> str:
    # 예시) "%p" → "5.2%p", "%" → "5.2%", "kl" → "0.123", "배" → "1.97배"
    # 반올림해 0이 된 음수가 "-0.0"으로 표시되지 않게 0.0을 더함
    if unit in PERCENT_UNITS:
        return f"{round(value * 100, 1) + 0.0:.1f}{unit}"
    if unit == "kl":
        return f"{value:.3f}"
    return f"{value:.2f}{unit}"


def plot_language_bars(
    ax: Axes,
    cases: list[str],
    values: dict[str, list[float]],
    unit: str,
) -> None:
    # case마다 언어 막대를 나란히 그리고 막대 위에 format_value로 값 표시
    for language_index, language in enumerate(LANGUAGES):
        xs = [case_index * CASE_WIDTH + language_index - (len(LANGUAGES) - 1) / 2 for case_index in range(len(cases))]
        ax.bar(xs, values[language], width=1, color=LANGUAGE_COLORS[language])
        for x, value in zip(xs, values[language]):
            ax.annotate(format_value(value, unit), (x, max(value, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    # 배는 1배가 기준
    ax.axhline(1 if unit == "배" else 0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([case_index * CASE_WIDTH for case_index in range(len(cases))], cases)
    ax.set_xlim(-CASE_WIDTH / 2, (len(cases) - 0.5) * CASE_WIDTH)
    if unit in PERCENT_UNITS:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=unit))
    if unit == "배":
        ax.yaxis.set_major_locator(MultipleLocator(RATIO_YTICK))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}배"))


def plot_insert_bars(
    ax: Axes,
    cases: list[str],
    losses: dict[str, list[tuple[float, float]]],
    recovery_cases: list[str],
) -> None:
    # case마다 (삽입 없음, 삽입) 손실을 받아 연한 삽입 없음 막대 위에 삽입 막대를 겹쳐 그림
    # recovery_cases에는 회복 비율도 표시
    # 예시) 막대 위 "13.6%p\n→5.8%p\n57% 회복", 삽입 손실이 음수면 막대 아래 "-6.7%p"
    for language_index, language in enumerate(LANGUAGES):
        for case_index, (no_insert_loss, insert_loss) in enumerate(losses[language]):
            x = case_index * CASE_WIDTH + language_index - (len(LANGUAGES) - 1) / 2
            ax.bar(x, no_insert_loss, width=1, color=LANGUAGE_COLORS[language], alpha=NO_INSERT_ALPHA)
            ax.bar(x, insert_loss, width=1, color=LANGUAGE_COLORS[language])
            # 삽입 손실이 음수면 막대 아래에 따로 표시, 옆 값과 붙지 않게 화살표 생략
            insert_text = format_value(insert_loss, "%p")
            texts = [format_value(no_insert_loss, "%p")]
            if insert_loss >= 0:
                texts.append("→" + insert_text)
            else:
                ax.annotate(insert_text, (x, insert_loss), xytext=(0, -VALUE_OFFSET), textcoords="offset points", ha="center", va="top", fontsize=VALUE_FONTSIZE)
            if cases[case_index] in recovery_cases:
                texts.append(f"{(no_insert_loss - insert_loss) / no_insert_loss:.0%} 회복")
            ax.annotate("\n".join(texts), (x, max(no_insert_loss, insert_loss, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([case_index * CASE_WIDTH for case_index in range(len(cases))], cases)
    ax.set_xlim(-CASE_WIDTH / 2, (len(cases) - 0.5) * CASE_WIDTH)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))


def get_insert_handles(
    no_insert_label: str = "삽입 없음",
    insert_label: str = "삽입",
) -> tuple[Patch, Patch]:
    # plot_insert_bars의 연한 막대와 진한 막대를 설명하는 범례, 삽입이 아닌 비교는 label을 바꿈
    return Patch(color=NEUTRAL_COLOR, alpha=NO_INSERT_ALPHA, label=no_insert_label), Patch(color=NEUTRAL_COLOR, label=insert_label)


def add_language_legend(
    fig: Figure,
    bf16_results: list[dict],
    num_problems: dict[str, int] | None = None,
    extra_handles: tuple[Patch, ...] = (),
) -> None:
    # 언어마다 bf16 정확도의 seed 평균과 표준편차 표시, 일부 문제만 쓰면 문제 수도 표시
    # 예시) "스와힐리어 (bf16 74.0% ±2.1%p, 166문제)"
    handles = []
    for language in LANGUAGES:
        bf16_accuracies = [bf16_result[language]["accuracy"] for bf16_result in bf16_results]
        label = f"{LANGUAGE_NAMES[language]} (bf16 {mean(bf16_accuracies):.1%} ±{stdev(bf16_accuracies):.1%}p"
        label += ")" if num_problems is None else f", {num_problems[language]}문제)"
        handles.append(Patch(color=LANGUAGE_COLORS[language], label=label))
    # 언어 뒤에 plot마다 다른 표시 추가 (예: 삽입 없음 손실)
    fig.legend(handles=handles + list(extra_handles), loc="outside lower center", ncol=3)


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
