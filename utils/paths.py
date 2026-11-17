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


def make_result_name(
    max_generation_len: int,
    seed: int,
    length_ratio: float | None = None,
) -> str:
    # reasoning 앞부분 길이별 삽입은 앞에 length_ratio 추가
    # 예시) "length_ratio=0.5,max_generation_len=16384,seed=0"
    prefix = "" if length_ratio is None else f"length_ratio={length_ratio},"
    return f"{prefix}max_generation_len={max_generation_len},seed={seed}"


def make_translation_name(
    max_translation_len: int,
    seed: int,
) -> str:
    return f"max_translation_len={max_translation_len},seed={seed}"


def make_training_path(
    method: str,
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
) -> str:
    # 예시) "finetune/google/gemma-4-E2B-it/loss=kl,reasoning_source=on_policy,train_part=understanding"
    return f"{method}/{model_name}/loss={loss},reasoning_source={reasoning_source},train_part={train_part}"
