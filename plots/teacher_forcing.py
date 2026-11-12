from dataclasses import dataclass

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from simple_parsing import parse
import torch
from torch import Tensor

from utils.hub import load_result, load_tensors
from utils.metrics import get_all_correct_mask
from utils.paths import Language, make_target_path, make_uncalibrated_path
from utils.plot import LANGUAGE_COLORS, LANGUAGE_NAMES, LANGUAGES, METHODS, NUM_PROGRESS_BINS, PROGRESS_LABELS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    max_generation_len: int = 16384
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def split_problems(
    values: Tensor,
    token_problem_indices: Tensor,
    used: Tensor,
) -> list[Tensor]:
    problem_indices, counts = token_problem_indices.unique_consecutive(return_counts=True)
    return [problem_values for problem_index, problem_values in zip(problem_indices.tolist(), values.split(counts.tolist())) if used[problem_index]]


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    results = {}
    for method in METHODS:
        target = make_target_path(method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
        results[method] = load_tensors("teacher_forcing/mgsm", target, file_name)
    bf16_results = [load_result("mgsm", make_uncalibrated_path("bf16", config.model_name), f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    used = {language: torch.tensor(get_all_correct_mask(bf16_results, language)) for language in LANGUAGES}

    series = []
    for language in LANGUAGES:
        kl_means = [torch.stack([kl.mean() for kl in split_problems(result[f"{language}/kl"], result[f"{language}/problem_index"], used[language])]).mean().item() for result in results.values()]
        series.append((kl_means, LANGUAGE_COLORS[language], [f"{kl_mean:.3f}" for kl_mean in kl_means]))

    fig, axes = plt.subplot_mosaic([["kl"], ["progress"]], figsize=get_figsize([get_panel_width(len(results), len(LANGUAGES))], 2), layout="constrained")
    plot_bars(axes["kl"], list(results), series)
    axes["kl"].set_ylim(0, max(max(values) for values, _, _ in series) * 1.15)
    axes["kl"].set_ylabel("문제당 평균 kl")
    used_counts = {language: used[language].sum().item() for language in LANGUAGES}
    axes["kl"].set_title(f"kl (사용한 문제 {', '.join(f'{LANGUAGE_NAMES[language]} {used_counts[language]}' for language in LANGUAGES)})")

    for language in LANGUAGES:
        problem_kls = split_problems(results["mxfp4"][f"{language}/kl"], results["mxfp4"][f"{language}/problem_index"], used[language])
        progress_kls = torch.stack([torch.stack([bin_kl.mean() for bin_kl in kl.tensor_split(NUM_PROGRESS_BINS)]) for kl in problem_kls])
        axes["progress"].plot(range(NUM_PROGRESS_BINS), progress_kls.mean(dim=0), color=LANGUAGE_COLORS[language], marker="o", linewidth=2)
    axes["progress"].set_xticks(range(NUM_PROGRESS_BINS), PROGRESS_LABELS)
    axes["progress"].set_xlabel("추론 위치")
    axes["progress"].set_ylabel("문제당 평균 kl")
    axes["progress"].set_title("mxfp4 추론 위치별 kl")

    fig.suptitle(f"{config.model_name} mgsm teacher forcing\n{file_name},bf16_seeds={','.join(map(str, config.bf16_seeds))}")
    fig.legend(handles=[Line2D([], [], color=LANGUAGE_COLORS[language], marker="o", linewidth=2, label=LANGUAGE_NAMES[language]) for language in LANGUAGES], loc="outside lower center", ncol=len(LANGUAGES))
    fit_bar_width(fig, [axes["kl"]])
    save_figure(fig, config.model_name, "mgsm_teacher_forcing")


if __name__ == "__main__":
    main()
