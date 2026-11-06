from statistics import mean


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(len(think) > 200 and think[:-200].count(think[-200:-150]) >= 5 for think in result["thinks"])


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
