def make_bf16_path(
    model_name: str,
) -> str:
    return f"bf16/{model_name}"


def make_rtn_path(
    model_name: str,
    num_bits: int,
    group_size: int,
) -> str:
    return f"rtn/{model_name}/num_bits={num_bits},group_size={group_size}"


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
    languages: tuple[str, ...],
    bf16_question: str,
) -> str:
    return f"qat/{model_name}/languages={'_'.join(languages)},bf16_question={bf16_question}"


def make_translate_first_finetune_path(
    model_name: str,
    languages: tuple[str, ...],
) -> str:
    return f"translate_first_finetune/{model_name}/languages={'_'.join(languages)}"


def make_quantized_path(
    method: str,
    model_name: str,
    languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
    bf16_question: str = "same",
) -> str:
    if method == "rtn":
        return make_rtn_path(model_name, num_bits, group_size)
    if method == "nf4":
        return make_nf4_path(model_name)
    if method == "mxfp4":
        return make_mxfp4_path(model_name)
    if method == "qat":
        return make_qat_path(model_name, languages, bf16_question)
    return f"{method}/{model_name}/languages={'_'.join(languages)},num_bits={num_bits},group_size={group_size},num_samples={num_samples}"


def make_target_path(
    method: str,
    model_name: str,
    languages: tuple[str, ...],
    num_bits: int,
    group_size: int,
    num_samples: int,
    bf16_question: str = "same",
) -> str:
    if method == "bf16":
        return make_bf16_path(model_name)
    if method == "translate_first_finetune":
        return make_translate_first_finetune_path(model_name, languages)
    return make_quantized_path(method, model_name, languages, num_bits, group_size, num_samples, bf16_question)
