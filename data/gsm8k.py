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
from utils.hub import result_exists, save_tensors
from utils.model import generate_vllm, get_prompt_ids, load_vllm
from utils.paths import make_bf16_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model that writes reasoning
    model_name: str = "google/gemma-4-E2B-it"


def load_gsm8k() -> dict[str, list[tuple[str, int]]]:
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    mcot = load_dataset("laihuiyuan/mCoT-MATH", data_files="train_part_*.jsonl", split="train").select_columns(["question", "lang-id"])

    # 1) group mcot questions by id, because same problem has same id in each language
    # e.g. "en-54530", "sw-54530" -> questions["54530"] = {"en": ..., "sw": ...}
    questions = defaultdict(dict)
    for batch in mcot.iter(batch_size=100000):
        for question, language_id in zip(batch["question"], batch["lang-id"]):
            language, mcot_id = language_id.split("-")
            if language in MGSM_PROMPTS:
                questions[mcot_id][language] = question.strip()

    # 2) map mcot en question to question in each language
    questions_by_en = {" ".join(language_questions["en"].split()): language_questions for language_questions in questions.values()}

    # 3) match each gsm8k train question with mcot en question
    problems = {language: [] for language in MGSM_PROMPTS}
    for row in gsm8k:
        language_questions = questions_by_en[" ".join(row["question"].split())]
        # e.g. "...\n#### 72" -> 72
        answer = int(row["answer"].split("####")[-1].replace(",", ""))
        for language in MGSM_PROMPTS:
            problems[language].append((MGSM_PROMPTS[language].format(question=language_questions[language]), answer))
    return problems


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    target = make_bf16_path(config.model_name)
    if result_exists("gsm8k", target, file_name, "safetensors"):
        print(f"{target} already evaluated")
        return
    problems = load_gsm8k()
    model = load_vllm(config.model_name, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    # 1) generate in en and score it
    log_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, [prompt for prompt, _ in problems["en"]], config.max_generation_len)
    result = evaluate_mgsm(outputs, [answer for _, answer in problems["en"]], "en", tokenizer, config.model_name)
    print(f"{target} en accuracy {result['accuracy']:.2%} reasoning len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")

    # 2) keep prompt ids in each language and en reasoning ids on problems solved by bf16 in en
    # e.g. "<|channel>thought\n...<channel|>Answer: 72" -> "<|channel>thought\n..."
    tensors = defaultdict(list)
    for problem_index, (output, correct, think_len) in enumerate(zip(outputs, result["corrects"], result["think_lens"])):
        if not correct:
            continue
        for language in MGSM_PROMPTS:
            prompt_ids = get_prompt_ids(tokenizer, problems[language][problem_index][0])
            tensors[f"{language}/prompt_ids"].append(prompt_ids)
            tensors[f"{language}/prompt_lens"].append(torch.tensor([len(prompt_ids)]))
        think_ids = torch.tensor(output.token_ids[:think_len])
        tensors["en/think_ids"].append(think_ids)
        tensors["en/think_lens"].append(torch.tensor([len(think_ids)]))
    print(f"{target} problems {len(tensors['en/prompt_lens'])} of {len(outputs)}")
    save_tensors("gsm8k", target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
