from dataclasses import dataclass
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse

from utils.hub import list_targets, load_result
from utils.paths import make_bf16_path
from utils.plot import ACCURACY_YMAX, BF16_COLOR, LANGUAGE_COLORS, LANGUAGES, fit_bar_width, format_bf16_accuracy, format_ratio, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def get_label(
    target: str,
) -> str:
    values = dict(value.split("=") for value in target.split("/")[-1].split(","))
    return f"{values['loss']}, {values['think_source'].replace('_', ' ')} {values['think_part']}"


def main() -> None:
    config = parse(Config)
    bf16_file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.bf16_seeds]
    bf16_results = [load_result("mgsm", make_bf16_path(config.model_name), bf16_file_name) for bf16_file_name in bf16_file_names]

    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    targets = list_targets("mgsm", f"finetune/{config.model_name}")
    results = {get_label(target): [load_result("mgsm", target, file_name) for file_name in file_names] for target in targets}

    groups = []
    bf16_values = []
    for language in LANGUAGES:
        bf16_values.append(mean(bf16_result[language]["accuracy"] for bf16_result in bf16_results))
        values = [mean(result[language]["accuracy"] for result in label_results) for label_results in results.values()]
        groups.append((values, LANGUAGE_COLORS[language], [format_ratio(value, bf16_values[-1]) for value in values]))

    fig, ax = plt.subplots(figsize=get_figsize([get_panel_width(len(results), len(LANGUAGES))], 1), layout="constrained")
    plot_bars(ax, list(results), groups, bf16_values)
    ax.set_ylim(0, ACCURACY_YMAX)
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} mgsm finetune\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    handles = [Patch(color=LANGUAGE_COLORS[language], label=f"{language} ({format_bf16_accuracy([bf16_result[language]['accuracy'] for bf16_result in bf16_results])})") for language in LANGUAGES]
    fig.legend(handles=[*handles, Patch(color=BF16_COLOR, label="bf16")], loc="outside lower center", ncol=3)
    fit_bar_width(fig, [ax])
    save_figure(fig, config.model_name, "mgsm_finetune")


if __name__ == "__main__":
    main()
