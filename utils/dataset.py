from datasets import load_dataset
import torch
from torch import Tensor
from transformers import AutoTokenizer

MGSM_PROMPTS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
}
MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "sw": "Jibu"}


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


def load_mgsm() -> dict[str, list[tuple[str, int]]]:
    problems = {}
    for language, prompt in MGSM_PROMPTS.items():
        dataset = load_dataset("juletxara/mgsm", language, split="test")
        problems[language] = [(prompt.format(question=row["question"]), row["answer_number"]) for row in dataset]
    return problems
