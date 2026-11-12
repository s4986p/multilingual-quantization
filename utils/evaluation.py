from collections.abc import Callable
import re
from statistics import mean
import time

from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.dataset import MGSM_ANSWER_PREFIXES
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import download_model, load_result, result_exists, save_result
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
            # 추론 중 최대 생성 길이 도달, 따라서 응답 없음
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
    # 마지막 answer prefix 뒤 텍스트 추출, prefix가 없으면 응답 전체 사용
    # 예시) "5+3=8，所以答案：8" → "8"
    # 예시) "Answer :7" → "7"
    # 예시) "Answer - 7" → "Answer - 7"
    # 예시) "Answer: 5. Answer: 8" → " 8"
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    # 텍스트의 마지막 숫자 추출, 숫자가 없으면 응답 전체 사용
    # 예시) "so 366.\n答え:" → 366.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


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


def evaluate_benchmark(
    script_name: str,
    languages: list[str],
    load_problems: Callable[[list[str]], dict],
    evaluate_outputs: Callable[..., dict],
    target: str,
    method: str,
    model_name: str,
    seeds: tuple[int, ...],
    max_generation_len: int,
) -> None:
    file_names = {seed: f"max_generation_len={max_generation_len},seed={seed}" for seed in seeds}

    # 1. seed마다 hub 결과에서 빠진 언어 확인
    results = {}
    missing_languages = {}
    for seed, file_name in file_names.items():
        results[seed] = load_result(script_name, target, file_name) if result_exists(script_name, target, file_name) else {}
        seed_languages = [language for language in languages if language not in results[seed]]
        if seed_languages:
            missing_languages[seed] = seed_languages
        else:
            print(f"{target} seed {seed} already evaluated")
    if not missing_languages:
        return
    problems = load_problems(languages)

    # 2. 빠진 seed와 언어를 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
    model_path = model_name if method == "bf16" else download_model(target)
    model = load_vllm(model_path, model_name)
    tokenizer = model.get_tokenizer()
    prompts = []
    prompt_seeds = []
    for seed, seed_languages in missing_languages.items():
        for language in seed_languages:
            prompts.extend(prompt for prompt, _ in problems[language])
            prompt_seeds.extend([seed] * len(problems[language]))
    start_time = time.perf_counter()
    outputs = generate_vllm(model, model_name, prompts, prompt_seeds, max_generation_len)
    print(f"{target} time {format_duration(time.perf_counter() - start_time)}")

    # 3. seed와 언어마다 채점하고 저장
    start = 0
    for seed, seed_languages in missing_languages.items():
        for language in seed_languages:
            end = start + len(problems[language])
            language_result = evaluate_outputs(outputs[start:end], [answer for _, answer in problems[language]], language, tokenizer, model_name)
            results[seed][language] = language_result
            start = end
            print(f"{target} seed {seed} {language} accuracy {language_result['accuracy']:.2%} reasoning len {mean(language_result['think_lens']):.0f} truncated {mean(language_result['truncateds']):.2%}")
        save_result(script_name, target, file_names[seed], results[seed])
