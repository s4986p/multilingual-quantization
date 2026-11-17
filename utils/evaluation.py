from collections.abc import Callable
import re
from statistics import mean
import time

from torch import Tensor
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.dataset import LANGUAGES
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import GENERATION_CONFIGS
from utils.paths import make_result_name

MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}



def load_vllm(
    model_path: str,
    model_name: str,
    max_num_seqs: int | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, language_model_only=True, max_num_seqs=max_num_seqs)


def get_sampling_params(
    model_name: str,
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[SamplingParams]:
    return [SamplingParams(**GENERATION_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=prompt_seed) for prompt_seed in prompt_seeds]


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = model.chat(messages, get_sampling_params(model_name, prompt_seeds, max_generation_len), use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]


def continue_vllm(
    model: LLM,
    model_name: str,
    input_ids: list[Tensor],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    # chat template을 거치지 않고 이미 만든 token 뒤에 이어서 생성
    # 예시) generate_vllm: "1+2=?" → chat template으로 prompt ids 만듦 → "<|channel>thought\n...<channel|>3" 생성
    # 예시) continue_vllm: 직접 만든 prompt ids + "<|channel>thought\n" + 삽입한 text → "...<channel|>3" 이어서 생성
    prompts = [{"prompt_token_ids": ids.tolist()} for ids in input_ids]
    outputs = model.generate(prompts, get_sampling_params(model_name, prompt_seeds, max_generation_len), use_tqdm=True)
    return [output.outputs[0] for output in outputs]


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])
    reasonings = []
    responses = []
    reasoning_lens = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if reasoning_end_id in token_ids:
            reasoning_len = token_ids.index(reasoning_end_id)
            response = tokenizer.decode(token_ids[reasoning_len + 1 :], skip_special_tokens=True)
        else:
            # reasoning_end token이 없으면 reasoning이 안 끝난 것이라 response 없음
            # 대부분 max_generation_len에 닿은 경우
            reasoning_len = len(token_ids)
            response = None
        reasonings.append(tokenizer.decode(token_ids[:reasoning_len], skip_special_tokens=True))
        responses.append(response)
        reasoning_lens.append(reasoning_len)
    return {
        "reasoning_lens": reasoning_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "reasonings": reasonings,
        "responses": responses,
    }


def parse_mgsm_prediction(
    response: str | None,
    language: str,
) -> float | None:
    if response is None:
        return None
    # 마지막 answer prefix 뒤 text 추출, prefix가 없으면 response 전체 사용
    # 예시) "5+3=8，所以答案：8" → "8"
    # 예시) "Answer :7" → "7"
    # 예시) "Answer - 7" → "Answer - 7"
    # 예시) "Answer: 5. Answer: 8" → " 8"
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    # text의 마지막 숫자 추출, 숫자가 없으면 response 전체에서 추출
    # 예시) "so 366.\n答え:" → 366.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def evaluate_outputs(
    outputs: list[CompletionOutput],
    answers: list[int | str],
    language: str,
    tokenizer: TokenizerLike,
    model_name: str,
    parse_prediction: Callable[[str | None, str], float | str | None],
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    predictions = [parse_prediction(response, language) for response in split["responses"]]
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }


def evaluate_benchmark(
    experiment_name: str,
    load_problems: Callable[[list[str]], dict],
    parse_prediction: Callable[[str | None, str], float | str | None],
    hub_path: str,
    method: str,
    model_name: str,
    seeds: tuple[int, ...],
    max_generation_len: int,
) -> None:
    file_names = {seed: make_result_name(max_generation_len, seed) for seed in seeds}

    # 1. seed마다 hub result에서 빠진 언어 확인, 빠진 것이 없으면 종료
    results = {}
    missing_languages = {}
    for seed, file_name in file_names.items():
        results[seed] = load_result(experiment_name, hub_path, file_name) if result_exists(experiment_name, hub_path, file_name) else {}
        seed_languages = [language for language in LANGUAGES if language not in results[seed]]
        if seed_languages:
            missing_languages[seed] = seed_languages
        else:
            print(f"{hub_path} seed {seed} already evaluated")
    if not missing_languages:
        return
    problems = load_problems(LANGUAGES)

    # 2. vllm은 띄우는 데 오래 걸리기 때문에 빠진 seed와 언어를 모아 한 번에 생성
    model_path = model_name if method == "bf16" else download_model(hub_path)
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
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 3. outputs를 seed와 언어별로 나눠 채점하고 저장
    start = 0
    for seed, seed_languages in missing_languages.items():
        for language in seed_languages:
            end = start + len(problems[language])
            language_result = evaluate_outputs(outputs[start:end], [answer for _, answer in problems[language]], language, tokenizer, model_name, parse_prediction)
            results[seed][language] = language_result
            start = end
            print(f"{hub_path} seed {seed} {language} accuracy {language_result['accuracy']:.2%} reasoning len {mean(language_result['reasoning_lens']):.0f} truncated {mean(language_result['truncateds']):.2%}")
        save_result(experiment_name, hub_path, file_names[seed], results[seed])
