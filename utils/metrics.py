from statistics import mean


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(len(think) > 200 and think[:-200].count(think[-200:-150]) >= 5 for think in result["thinks"])


def get_both_correct_mask(
    bf16_results: list[dict],
) -> list[bool]:
    return [all(corrects) for corrects in zip(*[bf16_result["corrects"] for bf16_result in bf16_results])]


def get_loss_ratios(
    result: dict,
    both_correct_mask: list[bool],
) -> tuple[float, float]:
    pairs = [(correct, truncated) for both_correct, correct, truncated in zip(both_correct_mask, result["corrects"], result["truncateds"]) if both_correct]
    truncated_ratio = mean(truncated and not correct for correct, truncated in pairs)
    finished_ratio = mean(not truncated and not correct for correct, truncated in pairs)
    return truncated_ratio, finished_ratio


def get_loss_points(
    ratio: float,
    both_correct_mask: list[bool],
) -> float:
    return ratio * sum(both_correct_mask) / len(both_correct_mask)
