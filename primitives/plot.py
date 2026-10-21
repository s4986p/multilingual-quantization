from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from primitives.hub import get_repo_name

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "awq": "tab:blue", "qat": "tab:green"}
GRID_COLOR = "#e5e4df"


def get_labels(
    targets: list[str],
) -> list[str]:
    settings = {target: target.split("/")[-1].split(",") for target in targets if "=" in target.split("/")[-1]}
    shared = set.intersection(*[set(target_settings) for target_settings in settings.values()]) if len(settings) > 1 else set()
    return ["\n".join([target.split("/")[0], *[setting for setting in settings.get(target, []) if setting not in shared]]) for target in targets]


def is_loop(
    text: str,
) -> bool:
    return len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def load_results(
    entry_name: str,
    model_name: str,
    file_name: str,
    methods: list[str],
) -> dict[str, list[dict]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/{method}/{model_name}/*{file_name}.json" for method in methods])) / entry_name
    results = defaultdict(list)
    for method in methods:
        for file in sorted(result_dir.glob(f"{method}/{model_name}/**/{file_name}.json")):
            results[str(file.parent.relative_to(result_dir))].append(json.loads(file.read_text()))
    return dict(results)


def style_axis(
    ax: Axes,
    languages: list[str],
    ylabel: str,
) -> None:
    ax.set_xticks(range(len(languages)), languages)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def plot_benchmark(
    results: dict[str, list[dict]],
    bf16_results: list[dict],
    title: str,
) -> Figure:
    targets = list(results)
    languages = list(bf16_results[0])
    width = 0.8 / len(targets)
    fig, axes = plt.subplots(1, 4, figsize=(4 * (2 + 0.75 * len(targets) * len(languages)), 6), layout="constrained")
    for target_index, target in enumerate(targets):
        color = METHOD_COLORS[target.split("/")[0]]
        offset = (target_index - (len(targets) - 1) / 2) * width
        for x, language in enumerate(languages):
            x += offset
            language_results = [result[language] for result in results[target]]
            bf16_language_results = [result[language] for result in bf16_results]

            accuracies = [result["accuracy"] for result in language_results]
            truncated_ratios = [mean(result["truncateds"]) for result in language_results]
            accuracy = mean(accuracies)
            truncated_ratio = mean(truncated_ratios)
            axes[0].bar(x, accuracy, width=width, color=color, zorder=3)
            axes[0].bar(x, truncated_ratio, width=width, bottom=1 - truncated_ratio, color="white", edgecolor="gray", hatch="//", zorder=3)
            axes[0].scatter([x] * len(accuracies), accuracies, color="black", s=12, zorder=4)
            axes[0].annotate(f"{accuracy:.1%}\ntruncated\n{truncated_ratio:.1%}", (x, 1), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

            think_lens = [mean(result["think_lens"]) for result in language_results]
            think_len = mean(think_lens)
            axes[1].bar(x, think_len, width=width, color=color, zorder=3)
            axes[1].scatter([x] * len(think_lens), think_lens, color="black", s=12, zorder=4)
            axes[1].annotate(f"{think_len:.0f}", (x, think_len), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

            ratios = []
            for result, bf16_result in zip(language_results, bf16_language_results):
                problem_ratios = [think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct]
                ratios.append(median(problem_ratios))
            ratio = mean(ratios)
            axes[2].bar(x, ratio, width=width, color=color, zorder=3)
            axes[2].scatter([x] * len(ratios), ratios, color="black", s=12, zorder=4)
            axes[2].annotate(f"{ratio:.2f}x", (x, ratio), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

            loop_ratios = [mean(is_loop(think) for think in result["thinks"]) for result in language_results]
            loop_ratio = mean(loop_ratios)
            axes[3].bar(x, loop_ratio, width=width, color=color, zorder=3)
            axes[3].scatter([x] * len(loop_ratios), loop_ratios, color="black", s=12, zorder=4)
            axes[3].annotate(f"{loop_ratio:.1%}", (x, loop_ratio), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    style_axis(axes[0], languages, "accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("accuracy", fontsize=10, pad=45)
    style_axis(axes[1], languages, "reasoning tokens")
    axes[1].set_title("mean reasoning tokens", fontsize=10, pad=45)
    style_axis(axes[2], languages, "reasoning tokens / bf16")
    axes[2].set_title("reasoning tokens / bf16 per problem\nmedian on both correct", fontsize=10, pad=45)
    style_axis(axes[3], languages, "loop ratio")
    axes[3].set_title("loop ratio", fontsize=10, pad=45)

    handles = [Patch(color=METHOD_COLORS[target.split("/")[0]], label=label.replace("\n", ",")) for target, label in zip(targets, get_labels(targets))]
    handles += [Patch(facecolor="white", edgecolor="gray", hatch="//", label="truncated")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=title)
    return fig
