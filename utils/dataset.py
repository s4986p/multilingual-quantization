from datasets import load_dataset
import torch
from torch import Tensor
from transformers import AutoTokenizer


def load_wikipedia(
    model_name: str,
    languages: tuple[str, ...],
    num_samples: int,
    max_sequence_len: int,
    seed: int,
) -> list[Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    # max_sequence_len includes bos
    chunk_len = max_sequence_len - len(bos_ids)
    samples = []
    for language_index, language in enumerate(languages):
        num_language_samples = num_samples // len(languages) + (language_index < num_samples % len(languages))
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
        token_ids = []
        language_samples = []
        for row in dataset:
            token_ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
            while len(token_ids) >= chunk_len and len(language_samples) < num_language_samples:
                language_samples.append(torch.tensor(bos_ids + token_ids[:chunk_len]))
                token_ids = token_ids[chunk_len:]
            if len(language_samples) == num_language_samples:
                break
        samples.extend(language_samples)
    return samples
