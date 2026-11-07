from dataclasses import dataclass
from typing import Literal

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from simple_parsing import parse
import torch

from utils.hub import load_tensors
from utils.metrics import split_problems
from utils.paths import make_target_path
from utils.plot import LANGUAGE_COLORS, get_figsize, save_figure

LANGUAGES = list(LANGUAGE_COLORS)
PANELS = {"question": True, "instruction": False}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = "instruction=en"
    entry_name = "hidden_error/mgsm"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    result = load_tensors(entry_name, target, file_name)

    fig, axes = plt.subplots(1, len(PANELS), figsize=get_figsize([6] * len(PANELS), 1), sharey=True, layout="constrained")
    for ax, (panel, question) in zip(axes, PANELS.items()):
        for language in LANGUAGES:
            problem_errors = []
            for errors, is_question in zip(split_problems(result[f"{language}/error"], result[f"{language}/problem_index"]), split_problems(result[f"{language}/is_question"], result[f"{language}/problem_index"])):
                problem_errors.append(errors[is_question == question].mean(dim=0))
            ax.plot(range(1, len(problem_errors[0]) + 1), torch.stack(problem_errors).mean(dim=0), color=LANGUAGE_COLORS[language], linewidth=2)
        ax.set_xlabel("layer")
        ax.set_title(f"{config.method} {panel}")
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("mean hidden error per problem")

    fig.suptitle(f"{config.model_name} mgsm hidden error\n{file_name}")
    fig.legend(handles=[Line2D([], [], color=LANGUAGE_COLORS[language], linewidth=2, label=language) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    save_figure(fig, config.model_name, "mgsm_hidden_error")


if __name__ == "__main__":
    main()
