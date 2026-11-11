def make_bf16_path(
    model_name: str,
) -> str:
    return f"bf16/{model_name}"


def make_nf4_path(
    model_name: str,
) -> str:
    return f"nf4/{model_name}"


def make_mxfp4_path(
    model_name: str,
) -> str:
    return f"mxfp4/{model_name}"


def make_qat_path(
    model_name: str,
    think_source: str,
) -> str:
    return f"qat/{model_name}/think_source={think_source}"


def make_finetune_path(
    model_name: str,
    loss: str,
    think_source: str,
    think_part: str,
) -> str:
    return f"finetune/{model_name}/loss={loss},think_source={think_source},think_part={think_part}"


def make_quantized_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method == "nf4":
        return make_nf4_path(model_name)
    if method == "mxfp4":
        return make_mxfp4_path(model_name)
    return f"{method}/{model_name}/calibration_languages={'_'.join(calibration_languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_target_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method == "bf16":
        return make_bf16_path(model_name)
    return make_quantized_path(method, model_name, calibration_languages, num_bits, group_size, num_samples)
