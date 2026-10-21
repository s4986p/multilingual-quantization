import re

from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from primitives.prompts import MGSM_ANSWER_PREFIXES, MMLU_PROX_LITE_TEMPLATES


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
    think_end_id: int,
) -> tuple[list[str], list[str | None]]:
    thinks = []
    responses = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if think_end_id in token_ids:
            think_len = token_ids.index(think_end_id)
            thinks.append(tokenizer.decode(token_ids[:think_len], skip_special_tokens=True))
            responses.append(tokenizer.decode(token_ids[think_len + 1 :], skip_special_tokens=True))
        else:
            thinks.append(tokenizer.decode(token_ids, skip_special_tokens=True))
            responses.append(None)
    return thinks, responses


def parse_mgsm_answer(
    response: str,
    language: str,
) -> float | None:
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1].replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def parse_mmlu_prox_lite_answer(
    response: str,
    language: str,
) -> str | None:
    _, _, _, answer_format = MMLU_PROX_LITE_TEMPLATES[language]
    answer_prefix = answer_format.split("(")[0].strip()
    letters = re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
    return letters[-1] if letters else None


def make_result(
    outputs: list[CompletionOutput],
    think_end_id: int,
    thinks: list[str],
    responses: list[str | None],
    predictions: list,
    corrects: list[bool],
) -> dict:
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        "think_lens": [list(output.token_ids).index(think_end_id) if think_end_id in output.token_ids else len(output.token_ids) for output in outputs],
        "output_lens": [len(output.token_ids) for output in outputs],
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "thinks": thinks,
        "responses": responses,
    }


def evaluate_mgsm(
    outputs: list[CompletionOutput],
    answers: list[int],
    language: str,
    tokenizer: TokenizerLike,
    think_end_id: int,
) -> dict:
    thinks, responses = split_outputs(outputs, tokenizer, think_end_id)
    predictions = [None if response is None else parse_mgsm_answer(response, language) for response in responses]
    corrects = [prediction is not None and prediction == answer for prediction, answer in zip(predictions, answers)]
    return make_result(outputs, think_end_id, thinks, responses, predictions, corrects)


def evaluate_mmlu_prox_lite(
    outputs: list[CompletionOutput],
    answers: list[str],
    language: str,
    tokenizer: TokenizerLike,
    think_end_id: int,
) -> dict:
    thinks, responses = split_outputs(outputs, tokenizer, think_end_id)
    predictions = [None if response is None else parse_mmlu_prox_lite_answer(response, language) for response in responses]
    corrects = [prediction is not None and prediction == answer for prediction, answer in zip(predictions, answers)]
    return make_result(outputs, think_end_id, thinks, responses, predictions, corrects)
