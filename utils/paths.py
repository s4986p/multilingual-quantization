def make_uncalibrated_path(
    method: str,
    model_name: str,
) -> str:
    return f"{method}/{model_name}"


def make_calibrated_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    return f"{method}/{model_name}/calibration_languages={'_'.join(calibration_languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"
