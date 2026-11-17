from statistics import mean

# 반복: 끝에서 LOOP_TAIL_LEN번째부터 LOOP_PATTERN_LEN 글자가 그 앞에 MIN_LOOP_COUNT번 이상 나옴
LOOP_TAIL_LEN = 200
LOOP_PATTERN_LEN = 50
MIN_LOOP_COUNT = 5


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
    return text is not None and len(text) > LOOP_TAIL_LEN and text[:-LOOP_TAIL_LEN].count(text[-LOOP_TAIL_LEN : -LOOP_TAIL_LEN + LOOP_PATTERN_LEN]) >= MIN_LOOP_COUNT


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))
