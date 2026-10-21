from datasets import load_dataset
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from primitives.prompts import MGSM_INSTRUCTIONS, MMLU_PROX_LITE_SUBJECTS, MMLU_PROX_LITE_TEMPLATES


def load_wikipedia(
    tokenizer: PreTrainedTokenizerBase,
    languages: list[str],
    num_samples: int,
    max_sequence_len: int,
    seed: int,
) -> list[Tensor]:
    prefix_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    chunk_len = max_sequence_len - len(prefix_ids)
    samples = []
    for language_index, language in enumerate(languages):
        num_language_samples = num_samples // len(languages) + (language_index < num_samples % len(languages))
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
        token_ids = []
        language_samples = []
        for row in dataset:
            token_ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
            while len(token_ids) >= chunk_len and len(language_samples) < num_language_samples:
                language_samples.append(torch.tensor(prefix_ids + token_ids[:chunk_len]))
                token_ids = token_ids[chunk_len:]
            if len(language_samples) == num_language_samples:
                break
        samples.extend(language_samples)
    return samples


def load_mgsm(
    language: str,
) -> list[tuple[str, int]]:
    dataset = load_dataset("juletxara/mgsm", language, split="test")
    return [(MGSM_INSTRUCTIONS[language].format(question=row["question"]), row["answer_number"]) for row in dataset]


def load_mmlu_prox_lite(
    language: str,
) -> list[tuple[str, str]]:
    question_label, options_label, description, answer_format = MMLU_PROX_LITE_TEMPLATES[language]
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", language, split="test")
    samples = []
    for row in dataset:
        options = [row[f"option_{index}"] for index in range(10) if row[f"option_{index}"] is not None]
        prompt = description.format(subject=MMLU_PROX_LITE_SUBJECTS[language][row["category"]], answer_format=answer_format.format("X"))
        prompt += f"\n\n{question_label}\n{row['question']}\n{options_label}\n"
        prompt += "".join(f"{chr(ord('A') + index)}. {option}\n" for index, option in enumerate(options))
        samples.append((prompt, row["answer"]))
    return samples
