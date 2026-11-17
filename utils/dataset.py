from datasets import load_dataset
import torch
from torch import Tensor
from transformers import AutoTokenizer

LANGUAGES = ["en", "zh", "ja", "sw", "te"]


def load_wikipedia(
    model_name: str,
    languages: tuple[str, ...],
    num_chunks: int,
    chunk_len: int,
    shuffle_seed: int,
) -> list[Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    # chunk_len은 bos 포함, text_len은 bos를 뺀 길이 (bos는 gemma 1개, qwen 0개)
    text_len = chunk_len - len(bos_ids)
    chunks = []
    for language_index, language in enumerate(languages):
        # 나누어떨어지지 않으면 남는 chunk를 앞 언어부터 하나씩 더함
        num_language_chunks = num_chunks // len(languages) + (language_index < num_chunks % len(languages))
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True).shuffle(seed=shuffle_seed, buffer_size=10000)
        token_ids = []
        language_chunks = []
        for row in dataset:
            token_ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
            while len(token_ids) >= text_len and len(language_chunks) < num_language_chunks:
                language_chunks.append(torch.tensor(bos_ids + token_ids[:text_len]))
                token_ids = token_ids[text_len:]
            if len(language_chunks) == num_language_chunks:
                break
        chunks.extend(language_chunks)
    return chunks
