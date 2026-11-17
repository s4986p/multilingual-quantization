from dataclasses import dataclass
from statistics import median

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.metrics import get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path, make_training_path
from utils.plot import VALUE_MARGIN, CASE_WIDTH, RATIO_YMAX, add_language_legend, get_figsize, get_losses, get_mean_values, get_title, plot_language_bars, save_figure

# x축 이름과 (loss, 학습 데이터 수, finetune한 sample 수)
CASES = {
    "ce\n800 sample": ("ce", 20000, 800),
    "ce_kl\n800 sample": ("ce_kl", 20000, 800),
    "ce_kl\n4000 sample": ("ce_kl", 20000, 4000),
    "ce_kl\n20000 sample": ("ce_kl", 20000, 20000),
    "ce_kl\n29492 sample": ("ce_kl", 29492, 29492),
}
# ce는 대부분 잘려 중앙값이 최대 길이라, ce_kl 막대가 보이도록 y축 상한 고정
LEN_RATIO_YMAX = 2.0


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def get_median_reasoning_len(
    result: dict,
) -> float:
    # 잘린 reasoning이 평균을 크게 끌어올리기 때문에 중앙값
    return median(result["reasoning_lens"])


def main() -> None:
    config = parse(Config)
    results = {"bf16": load_results("mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)}
    for case, (loss, num_samples, num_trained_samples) in CASES.items():
        hub_path = make_training_path("finetune", config.model_name, loss, "bf16_en_question_insert", "en_question", num_samples, num_trained_samples)
        results[case] = load_results("mgsm", hub_path, config.max_generation_len, config.seeds)

    # 잘림과 반복은 bf16도 함께 표시
    # reasoning 길이는 언어마다 bf16 중앙값 대비 배
    median_lens = get_mean_values(results, list(results), get_median_reasoning_len)
    len_ratios = {language: [median_len / language_median_lens[0] for median_len in language_median_lens[1:]] for language, language_median_lens in median_lens.items()}
    fig, axes = plt.subplots(4, 1, figsize=get_figsize(len(results), 4), layout="constrained")
    plot_language_bars(axes[0], list(CASES), get_losses(results["bf16"], results, list(CASES)), "%p")
    plot_language_bars(axes[1], list(results), get_mean_values(results, list(results), get_truncated_ratio), "%")
    plot_language_bars(axes[2], list(results), get_mean_values(results, list(results), get_loop_ratio), "%")
    plot_language_bars(axes[3], list(CASES), len_ratios, "배")
    for ax, ylabel in zip(axes, ["손실", "잘림", "반복", "reasoning 길이"]):
        ax.set_ylabel(ylabel)
    # 손실과 reasoning 길이에는 bf16 case가 없어서, 잘림, 반복과 위치를 맞추려고 bf16 자리를 비워 둠
    for ax in [axes[0], axes[3]]:
        ax.set_xlim(-1.5 * CASE_WIDTH, (len(CASES) - 0.5) * CASE_WIDTH)
    axes[0].margins(y=VALUE_MARGIN)
    for ax in axes[1:3]:
        ax.set_ylim(0, RATIO_YMAX)
    axes[3].set_ylim(0, LEN_RATIO_YMAX)

    fig.suptitle(get_title(config.model_name, "mgsm", "영어 question finetune"))
    add_language_legend(fig, results["bf16"])
    save_figure(fig, config.model_name, "mgsm_en_question_finetune")


if __name__ == "__main__":
    main()
