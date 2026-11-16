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


def make_training_path(
    method: str,
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
    num_samples: int | None = None,
    num_trained_samples: int | None = None,
) -> str:
    # en_question의 ce는 mcot 영어 question만 학습해 reasoning_source 없음
    # checkpoint를 저장한 학습은 학습 전체 sample 수와 지금까지 학습한 sample 수 추가
    settings = [f"loss={loss}"]
    if not (train_part == "en_question" and loss == "ce"):
        settings.append(f"reasoning_source={reasoning_source}")
    settings.append(f"train_part={train_part}")
    if num_samples is not None:
        settings += [f"num_samples={num_samples}", f"num_trained_samples={num_trained_samples}"]
    return f"{method}/{model_name}/{','.join(settings)}"
