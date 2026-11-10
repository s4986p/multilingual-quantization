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
    think_language: str,
) -> str:
    return f"qat/{model_name}/think_language={think_language}"


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
    if method in ("qat_en", "qat_native"):
        return make_qat_path(model_name, method.removeprefix("qat_"))
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
