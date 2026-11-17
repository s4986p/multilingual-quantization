from statistics import mean, median


def get_bf16_solved_mask(
    bf16_results: list[dict],
    language: str,
) -> list[bool]:
    # 모든 bf16 seed가 맞힌 문제
    return [all(corrects) for corrects in zip(*[bf16_result[language]["corrects"] for bf16_result in bf16_results])]


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))


def get_median_reasoning_len(
    result: dict,
) -> float:
    # 잘린 reasoning이 평균을 크게 끌어올리기 때문에 중앙값
    return median(result["reasoning_lens"])
