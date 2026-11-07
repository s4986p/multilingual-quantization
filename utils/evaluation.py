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


def parse_mgsm_prediction(
    response: str | None,
    language: str,
) -> float | None:
    if response is None:
        return None
    # take text after last answer prefix, or whole response if no prefix
    # e.g. "5+3=8，所以答案：8" -> "8"
    #      "Answer :7" -> "7"
    #      "Answer - 7" -> "Answer - 7"
    #      "Answer: 5. Answer: 8" -> " 8"
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    # take last number in text, or in whole response if text has none
    # e.g. "so 366.\n答え:" -> 366.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def parse_mmlu_prox_lite_prediction(
    response: str | None,
    language: str,
) -> str | None:
    if response is None:
        return None
    # drop from "(" to end
    # e.g. "the answer is ({})" -> "the answer is"
    #      "答案是 ({})"       -> "答案是"
    #      "答えは ({}) です"   -> "答えは"
    answer_prefix = MMLU_PROX_LITE_TEMPLATES[language][3].split("(")[0].strip()
    # drop markdown bold
    # e.g. "Jibu ni **E**" -> "Jibu ni E"
    response = response.replace("*", "")
    # take single uppercase letter A-J after prefix
    # e.g. "The answer is C" -> ["C"]
    #      "The answer is K" -> []
    #      "The answer is c" -> []
    #      "The answer is Cat" -> []
    letters = re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
    # if no letter is caught, take any standalone uppercase letter A-J in response
    # e.g. "Jibu: A"      -> ["A"]
    #      "I chose (D)." -> ["I", "D"]
    letters = letters or re.findall(r"\b([A-J])\b", response)
    return letters[-1] if letters else None


def evaluate_mgsm(
    outputs: list[CompletionOutput],
    answers: list[int],
    language: str,
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    predictions = [parse_mgsm_prediction(response, language) for response in split["responses"]]
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
    predictions = [parse_mmlu_prox_lite_prediction(response, language) for response in split["responses"]]
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }
