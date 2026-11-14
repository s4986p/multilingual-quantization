from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from utils.dataset import LANGUAGES

QUANTIZATION_METHODS = ["gptq", "awq", "nf4", "mxfp4"]
LANGUAGE_COLORS = {"en": "#3c5a9a", "zh": "#de2910", "ja": "#f4a6b7", "sw": "#1eb53a", "te": "#ff9933"}
LANGUAGE_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "sw": "스와힐리어", "te": "텔루구어"}
# bf16과의 차이를 언어 색 빗금으로 표시
BF16_HATCH = "////"
ACCURACY_YMAX = 1.15
NUM_POSITION_BINS = 5
POSITION_LABELS = [f"{index * 100 // NUM_POSITION_BINS}~{(index + 1) * 100 // NUM_POSITION_BINS}%" for index in range(NUM_POSITION_BINS)]
BAR_INCHES = 0.45
BAR_GAP = 1
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


def get_panel_width(
    num_cases: int,
    num_series: int,
) -> float:
    return BAR_INCHES * num_cases * (num_series + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    return sum(panel_widths) + PANEL_MARGIN * len(panel_widths), PANEL_HEIGHT * num_rows + 1


def format_accuracy(
    accuracy: float,
    bf16_accuracy: float,
) -> str:
    return f"{accuracy:.1%}\n{round(accuracy - bf16_accuracy, 3) + 0.0:+.1%}p"


def get_accuracy(
    result: dict,
) -> float:
    return result["accuracy"]


def plot_bars(
    ax: Axes,
    case_labels: list[str],
    series: list[tuple[list[float], str, list[str]]],
    bf16_values: list[float] | None = None,
) -> None:
    case_width = len(series) + BAR_GAP
    for series_index, (values, color, texts) in enumerate(series):
        xs = [case_index * case_width + series_index - (len(series) - 1) / 2 for case_index in range(len(case_labels))]
        if bf16_values is not None:
            ax.bar(xs, [bf16_values[series_index]] * len(xs), width=1, facecolor="white", edgecolor=color, hatch=BF16_HATCH, linewidth=0)
        ax.bar(xs, values, width=1, color=color)
        for x, value, text in zip(xs, values, texts):
            top = max(value, 0 if bf16_values is None else bf16_values[series_index])
            ax.annotate(text, (x, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([case_index * case_width for case_index in range(len(case_labels))], case_labels)
    ax.set_xlim(-case_width / 2, (len(case_labels) - 0.5) * case_width)


def plot_language_bars(
    ax: Axes,
    case_results: dict[str, list[dict]],
    bf16_results: list[dict],
    get_value: Callable[[dict], float],
) -> None:
    series = []
    bf16_values = []
    for language in LANGUAGES:
        bf16_values.append(mean(get_value(bf16_result[language]) for bf16_result in bf16_results))
        values = [mean(get_value(result[language]) for result in results) for results in case_results.values()]
        series.append((values, LANGUAGE_COLORS[language], [format_accuracy(value, bf16_values[-1]) for value in values]))
    plot_bars(ax, list(case_results), series, bf16_values)


def add_language_legend(
    fig: Figure,
    bf16_results: list[dict] | None = None,
) -> None:
    handles = []
    for language in LANGUAGES:
        label = LANGUAGE_NAMES[language]
        if bf16_results is not None:
            bf16_accuracies = [bf16_result[language]["accuracy"] for bf16_result in bf16_results]
            label += f" (bf16 {mean(bf16_accuracies):.1%} ±{stdev(bf16_accuracies):.1%}p)"
        handles.append(Patch(color=LANGUAGE_COLORS[language], label=label))
    if bf16_results is not None:
        handles.append(Patch(facecolor="white", edgecolor="#808080", hatch=BF16_HATCH, linewidth=0, label="bf16 대비 손실"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles) if bf16_results is None else 3)


def fit_bar_width(
    fig: Figure,
    axes: list[Axes],
) -> None:
    for _ in range(2):
        fig.canvas.draw()
        axes_width = sum(ax.get_window_extent().width for ax in axes) / fig.dpi
        num_bars = sum(ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes)
        fig.set_figwidth(fig.get_figwidth() + BAR_INCHES * num_bars - axes_width)


def save_figure(
    fig: Figure,
    model_name: str,
    file_name: str,
) -> None:
    path = Path(f"figures/{model_name.split('/')[-1]}/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
