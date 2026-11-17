def get_bf16_solved_mask(
    bf16_results: list[dict],
    language: str,
) -> list[bool]:
    # 모든 bf16 seed가 맞힌 문제
    return [all(corrects) for corrects in zip(*[bf16_result[language]["corrects"] for bf16_result in bf16_results])]
