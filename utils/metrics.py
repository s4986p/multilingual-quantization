def get_all_correct_mask(
    bf16_results: list[dict],
) -> list[bool]:
    return [all(corrects) for corrects in zip(*[bf16_result["corrects"] for bf16_result in bf16_results])]
