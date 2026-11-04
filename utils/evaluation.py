import re

from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.dataset import MGSM_ANSWER_PREFIXES, MMLU_PROX_LITE_TEMPLATES
from utils.model import GENERATION_CONFIGS


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    think_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["think_end"])
    thinks = []
    responses = []
    think_lens = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if think_end_id in token_ids:
            think_len = token_ids.index(think_end_id)
            response = tokenizer.decode(token_ids[think_len + 1 :], skip_special_tokens=True)
        else:
            # ran out of tokens while thinking, so no response
            think_len = len(token_ids)
            response = None
        thinks.append(tokenizer.decode(token_ids[:think_len], skip_special_tokens=True))
        responses.append(response)
        think_lens.append(think_len)
    return {
        "think_lens": think_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "thinks": thinks,
        "responses": responses,
    }


def evaluate_mgsm(
    outputs: list[CompletionOutput],
    answers: list[int],
    language: str,
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    predictions = []
    for response in split["responses"]:
        # take last number after last answer prefix
        text = "" if response is None else re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1].replace(",", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        predictions.append(float(numbers[-1]) if numbers else None)
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }


def evaluate_mmlu_prox_lite(
    outputs: list[CompletionOutput],
    answers: list[str],
    language: str,
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    # e.g. "the answer is ({})" -> "the answer is"
    answer_prefix = MMLU_PROX_LITE_TEMPLATES[language][3].split("(")[0].strip()
    predictions = []
    for response in split["responses"]:
        letters = [] if response is None else re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
        predictions.append(letters[-1] if letters else None)
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }
