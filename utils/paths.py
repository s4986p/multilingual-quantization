from typing import Literal

Language = Literal["en", "zh", "sw", "ja", "te"]


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


def make_finetune_path(
    model_name: str,
    loss: str,
    think_source: str,
    think_part: str,
) -> str:
    return f"finetune/{model_name}/loss={loss},think_source={think_source},think_part={think_part}"


def make_target_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
) -> str:
    if method in ("bf16", "nf4", "mxfp4"):
        return make_uncalibrated_path(method, model_name)
    return make_calibrated_path(method, model_name, calibration_languages, num_bits, group_size, num_samples)
