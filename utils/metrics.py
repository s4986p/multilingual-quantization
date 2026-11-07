from statistics import mean

from torch import Tensor


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(think) or is_loop(response) for think, response in zip(result["thinks"], result["responses"]))


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def get_wrong_ratio(
    result: dict,
) -> float:
    return mean(not truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def get_both_correct_mask(
    bf16_results: list[dict],
) -> list[bool]:
    return [all(corrects) for corrects in zip(*[bf16_result["corrects"] for bf16_result in bf16_results])]


def split_problems(
    values: Tensor,
    problem_index: Tensor,
) -> tuple[Tensor, ...]:
    return values.split(problem_index.unique_consecutive(return_counts=True)[1].tolist())
