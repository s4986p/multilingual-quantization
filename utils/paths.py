def make_hub_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...] | None = None,
    num_bits: int | None = None,
    group_size: int | None = None,
    num_chunks: int | None = None,
) -> str:
    # bf16, nf4, mxfp4는 calibration이 없고 설정이 고정이라 경로에 설정 안 넣음
    if method in ("bf16", "nf4", "mxfp4"):
        return f"{method}/{model_name}"
    return f"{method}/{model_name}/calibration_languages={'_'.join(calibration_languages)},num_bits={num_bits},group_size={group_size},num_chunks={num_chunks}"


def make_finetune_path(
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
) -> str:
    return f"finetune/{model_name}/loss={loss},reasoning_source={reasoning_source},train_part={train_part}"
