def get_all_correct_mask(
    bf16_results: list[dict],
    language: str,
) -> list[bool]:
    return [all(corrects) for corrects in zip(*[bf16_result[language]["corrects"] for bf16_result in bf16_results])]
