from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse

from utils.dataset import LANGUAGES, get_mgsm_question, load_mgsm
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import result_exists, save_result
from utils.paths import make_hub_path

TRANSLATION_INSTRUCTION = "Translate the following question into English. Output only the translation."


@dataclass
class Config:
    seed: int = 0
    max_translation_len: int = 512
    # translation을 쓸 모델
    model_name: str = "google/gemma-4-E2B-it"


def main() -> None:
    config = parse(Config)
    experiment_name = "translation/mgsm"
    hub_path = make_hub_path("bf16", config.model_name)
    file_name = f"max_translation_len={config.max_translation_len},seed={config.seed}"
    if result_exists(experiment_name, hub_path, file_name):
        print(f"{hub_path} already generated")
        return
    # 영어는 번역할 것이 없어 제외
    languages = LANGUAGES[1:]
    problems = load_mgsm(languages)
    model = load_vllm(config.model_name, config.model_name)

    # 1. 해당 언어 question 앞에 translation instruction을 붙여 thinking 없이 생성
    # 예시) instruction + "\n\n" + 스와힐리어 question → 영어 translation
    prompts = [TRANSLATION_INSTRUCTION + "\n\n" + get_mgsm_question(prompt) for language in languages for prompt, _ in problems[language]]
    start_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, prompts, [config.seed] * len(prompts), config.max_translation_len, enable_thinking=False)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 2. outputs를 language별로 나눠 저장
    result = {}
    start = 0
    for language in languages:
        end = start + len(problems[language])
        result[language] = {
            "translations": [output.text.strip() for output in outputs[start:end]],
            "truncateds": [output.finish_reason == "length" for output in outputs[start:end]],
        }
        start = end
        print(f"{hub_path} {language} truncated {mean(result[language]['truncateds']):.2%} example {result[language]['translations'][0]!r}")
    save_result(experiment_name, hub_path, file_name, result)


if __name__ == "__main__":
    main()
