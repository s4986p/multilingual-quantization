from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import time

from datasets import load_dataset
from simple_parsing import parse
import torch

from utils.dataset import MGSM_PROMPTS
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import save_tensors, tensors_exist
from utils.model import get_prompt_ids
from utils.paths import make_uncalibrated_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # 추론을 쓸 모델
    model_name: str = "google/gemma-4-E2B-it"


def load_gsm8k() -> dict[str, list[tuple[str, int]]]:
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    mcot = load_dataset("laihuiyuan/mCoT-MATH", data_files="train_part_*.jsonl", split="train").select_columns(["question", "lang-id"])

    # 1. mcot 질문을 id로 묶음, 같은 문제는 언어마다 id가 같기 때문
    # 예시) "en-54530", "sw-54530" → questions["54530"] = {"en": ..., "sw": ...}
    questions = defaultdict(dict)
    for batch in mcot.iter(batch_size=100000):
        for question, language_id in zip(batch["question"], batch["lang-id"]):
            language, mcot_id = language_id.split("-")
            if language in MGSM_PROMPTS:
                questions[mcot_id][language] = question.strip()

    # 2. mcot 영어 질문으로 각 언어 질문 찾음
    questions_by_en = {" ".join(language_questions["en"].split()): language_questions for language_questions in questions.values()}

    # 3. gsm8k train 질문마다 같은 mcot 영어 질문 연결
    problems = {language: [] for language in MGSM_PROMPTS}
    for row in gsm8k:
        language_questions = questions_by_en[" ".join(row["question"].split())]
        # 예시) "...\n#### 72" → 72
        answer = int(row["answer"].split("####")[-1].replace(",", ""))
        for language in MGSM_PROMPTS:
            problems[language].append((MGSM_PROMPTS[language].format(question=language_questions[language]), answer))
    return problems


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    target = make_uncalibrated_path("bf16", config.model_name)
    if tensors_exist("gsm8k", target, file_name):
        print(f"{target} already evaluated")
        return
    problems = load_gsm8k()
    model = load_vllm(config.model_name, config.model_name)
    tokenizer = model.get_tokenizer()

    # 1. 각 언어로 생성하고 채점
    outputs = {}
    results = {}
    for language in MGSM_PROMPTS:
        start_time = time.perf_counter()
        outputs[language] = generate_vllm(model, config.model_name, [prompt for prompt, _ in problems[language]], [config.seed] * len(problems[language]), config.max_generation_len)
        results[language] = evaluate_mgsm(outputs[language], [answer for _, answer in problems[language]], language, tokenizer, config.model_name)
        print(f"{target} {language} accuracy {results[language]['accuracy']:.2%} reasoning len {mean(results[language]['think_lens']):.0f} truncated {mean(results[language]['truncateds']):.2%}")
        print(f"{target} {language} time {format_duration(time.perf_counter() - start_time)}")

    # 2. bf16이 영어로 푼 문제에서 각 언어의 프롬프트, 추론, 정답 여부 저장
    # 예시) "<|channel>thought\n...<channel|>Answer: 72" → "<|channel>thought\n..."
    tensors = defaultdict(list)
    for problem_index, en_correct in enumerate(results["en"]["corrects"]):
        if not en_correct:
            continue
        for language in MGSM_PROMPTS:
            prompt_ids = get_prompt_ids(tokenizer, problems[language][problem_index][0])
            think_ids = torch.tensor(outputs[language][problem_index].token_ids[: results[language]["think_lens"][problem_index]], dtype=torch.long)
            tensors[f"{language}/prompt_ids"].append(prompt_ids)
            tensors[f"{language}/prompt_lens"].append(torch.tensor([len(prompt_ids)]))
            tensors[f"{language}/think_ids"].append(think_ids)
            tensors[f"{language}/think_lens"].append(torch.tensor([len(think_ids)]))
            tensors[f"{language}/corrects"].append(torch.tensor([results[language]["corrects"][problem_index]]))
    print(f"{target} problems {len(tensors['en/prompt_lens'])} of {len(outputs['en'])}")
    save_tensors("gsm8k", target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
