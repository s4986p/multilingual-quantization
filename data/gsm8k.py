from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import time

from datasets import load_dataset
from simple_parsing import parse
import torch

from utils.dataset import LANGUAGES, MGSM_PROMPTS
from utils.evaluation import evaluate_outputs, parse_mgsm_prediction
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import save_tensors, tensors_exist
from utils.model import get_prompt_ids
from utils.paths import make_hub_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # reasoning을 쓸 모델
    model_name: str = "google/gemma-4-E2B-it"


def load_gsm8k() -> dict[str, list[tuple[str, int]]]:
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    mcot = load_dataset("laihuiyuan/mCoT-MATH", data_files="train_part_*.jsonl", split="train").select_columns(["question", "lang-id"])

    # 1. 같은 문제는 언어마다 mcot id가 같기 때문에 question을 id로 묶음
    # 예시) "en-54530", "sw-54530" → questions["54530"] = {"en": ..., "sw": ...}
    questions = defaultdict(dict)
    for batch in mcot.iter(batch_size=100000):
        for question, language_id in zip(batch["question"], batch["lang-id"]):
            language, mcot_id = language_id.split("-")
            if language in LANGUAGES:
                questions[mcot_id][language] = question.strip()

    # 2. gsm8k train question마다 공백만 다른 mcot 영어 question을 찾아 각 언어 question 연결
    questions_by_en = {" ".join(language_questions["en"].split()): language_questions for language_questions in questions.values()}
    problems = {language: [] for language in LANGUAGES}
    for row in gsm8k:
        language_questions = questions_by_en[" ".join(row["question"].split())]
        # 예시) "...\n#### 72" → 72
        answer = int(row["answer"].split("####")[-1].replace(",", ""))
        for language in LANGUAGES:
            problems[language].append((MGSM_PROMPTS[language].format(question=language_questions[language]), answer))
    return problems


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path("bf16", config.model_name)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    if tensors_exist("gsm8k", hub_path, file_name):
        print(f"{hub_path} already generated")
        return
    problems = load_gsm8k()
    model = load_vllm(config.model_name, config.model_name)
    tokenizer = model.get_tokenizer()

    # 1. 모든 언어 prompt를 한 번에 생성
    prompts = [prompt for language in LANGUAGES for prompt, _ in problems[language]]
    start_time = time.perf_counter()
    all_outputs = generate_vllm(model, config.model_name, prompts, [config.seed] * len(prompts), config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 2. 언어마다 채점
    outputs = {}
    results = {}
    start = 0
    for language in LANGUAGES:
        end = start + len(problems[language])
        outputs[language] = all_outputs[start:end]
        results[language] = evaluate_outputs(outputs[language], [answer for _, answer in problems[language]], language, tokenizer, config.model_name, parse_mgsm_prediction)
        start = end
        print(f"{hub_path} {language} accuracy {results[language]['accuracy']:.2%} reasoning len {mean(results[language]['reasoning_lens']):.0f} truncated {mean(results[language]['truncateds']):.2%}")

    # 3. bf16이 영어로 맞힌 문제만 언어마다 prompt, reasoning, 정답 여부 저장
    # 예시) "<|channel>thought\n...<channel|>Answer: 72" → "<|channel>thought\n..."
    tensors = defaultdict(list)
    for problem_index, en_correct in enumerate(results["en"]["corrects"]):
        if not en_correct:
            continue
        for language in LANGUAGES:
            prompt_ids = get_prompt_ids(tokenizer, problems[language][problem_index][0])
            reasoning_ids = torch.tensor(outputs[language][problem_index].token_ids[: results[language]["reasoning_lens"][problem_index]], dtype=torch.long)
            tensors[f"{language}/prompt_ids"].append(prompt_ids)
            tensors[f"{language}/prompt_lens"].append(torch.tensor([len(prompt_ids)]))
            tensors[f"{language}/reasoning_ids"].append(reasoning_ids)
            tensors[f"{language}/reasoning_lens"].append(torch.tensor([len(reasoning_ids)]))
            tensors[f"{language}/corrects"].append(torch.tensor([results[language]["corrects"][problem_index]]))
    print(f"{hub_path} problems {len(tensors['en/prompt_lens'])} of {len(outputs['en'])}")
    save_tensors("gsm8k", hub_path, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
