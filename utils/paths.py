def make_bf16_path(
    model_name: str,
) -> str:
    return f"bf16/{model_name}"


def make_quantized_path(
    method: str,
    model_name: str,
    languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    return f"{method}/{model_name}/languages={'_'.join(languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_target_path(
    method: str,
    model_name: str,
    languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method == "bf16":
        return make_bf16_path(model_name)
    return make_quantized_path(method, model_name, languages, num_bits, group_size, num_samples)
