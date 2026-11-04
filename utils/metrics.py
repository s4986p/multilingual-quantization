from statistics import mean, median


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(len(think) > 200 and think[:-200].count(think[-200:-150]) >= 5 for think in result["thinks"])


def get_length_ratio(
    result: dict,
    bf16_result: dict,
) -> float:
    return median(think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct)



def get_flip_ratios(
    result: dict,
    bf16_result: dict,
) -> tuple[float, float]:
    pairs = list(zip(bf16_result["corrects"], result["corrects"]))
    lost_ratio = mean(bf16_correct and not correct for bf16_correct, correct in pairs)
    gained_ratio = mean(correct and not bf16_correct for bf16_correct, correct in pairs)
    return lost_ratio, gained_ratio
