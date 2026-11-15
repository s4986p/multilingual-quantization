from collections.abc import Callable, Iterator
import math
import random
from statistics import mean
import time

import torch
from torch import Tensor
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from utils.dataset import LANGUAGES
from utils.format import format_duration, format_memory
from utils.hub import load_tensors
from utils.model import GENERATION_CONFIGS, get_reasoning_ids
from utils.paths import make_hub_path
from utils.understanding import get_understanding

# gsm8k 해당 언어 understanding의 약 95%가 들어가는 길이
MAX_UNDERSTANDING_LEN = 512
# gsm8k 해당 언어 reasoning의 약 98%가 들어가는 길이
MAX_REASONING_LEN = 2048

# sample: (학습 중인 모델 prompt_ids, bf16 prompt_ids, 학습할 text_ids)
# ce_kl은 (학습 중인 모델과 bf16 prompt_ids, 영어 question_ids, reasoning_ids)
Sample = tuple[Tensor, Tensor, Tensor]


def load_gsm8k_reasonings(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
) -> dict[str, dict[str, list]]:
    # 언어마다 bf16이 gsm8k에 쓴 prompt_ids, reasoning, 정답 여부
    # data/reasoning.py가 bf16이 영어로 맞힌 문제만 저장했기 때문에 언어끼리 같은 문제 순서
    tensors = load_tensors("reasoning/gsm8k", make_hub_path("bf16", model_name), f"max_generation_len={max_generation_len},seed={seed}")
    return {
        language: {
            "prompt_ids": tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()),
            "reasonings": tokenizer.batch_decode(tensors[f"{language}/reasoning_ids"].split(tensors[f"{language}/reasoning_lens"].tolist()), skip_special_tokens=True),
            "corrects": tensors[f"{language}/corrects"].tolist(),
        }
        for language in LANGUAGES
    }


def generate_reasonings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    train_part: str,
    batch: list[Sample],
) -> list[Sample]:
    # 1. 학습 중인 모델이 prompt에 reasoning 생성
    sampling = GENERATION_CONFIGS[model_name]["sampling"]
    inputs = tokenizer.pad({"input_ids": [prompt_ids for prompt_ids, _, _ in batch]}, padding_side="left", return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"], max_new_tokens=MAX_UNDERSTANDING_LEN if train_part == "understanding" else MAX_REASONING_LEN)

    # 2. reasoning_end에서 자름
    # 예시) "<|channel>thought\n...<channel|>Answer: 72" → "thought\n..."
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])
    reasonings = []
    for ids in output_ids[:, inputs["input_ids"].shape[1] :]:
        end = (ids == reasoning_end_id).nonzero()
        reasonings.append(tokenizer.decode(ids[: end[0, 0]] if len(end) else ids, skip_special_tokens=True))

    # 3. 학습할 text를 생성한 것으로 바꾸고 understanding이 없으면 제외
    generated_batch = []
    for (prompt_ids, bf16_prompt_ids, _), reasoning in zip(batch, reasonings):
        understanding = get_understanding(reasoning)
        if understanding is not None:
            generated_batch.append((prompt_ids, bf16_prompt_ids, get_reasoning_ids(tokenizer, model_name, understanding if train_part == "understanding" else reasoning)))
    return generated_batch


def train(
    weights: list[Tensor],
    samples: list[Sample],
    get_batch_losses: Callable[[list[Sample]], Iterator[Tensor]],
    lr: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
    save_steps: tuple[int, ...] = (),
    save_checkpoint: Callable[[int], None] | None = None,
) -> None:
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=lr, weight_decay=0)
    num_batches = math.ceil(len(samples) / batch_size)
    num_steps = num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio * num_steps), num_steps)

    losses = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), batch_size)):
            step = epoch * num_batches + batch_index + 1
            # sample마다 backward해 메모리 절약
            batch_loss = 0.0
            for loss in get_batch_losses(samples[start : start + batch_size]):
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(batch_loss)
            if step == 1 or step % 16 == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            # save_steps와 마지막 step에서 checkpoint 저장
            if save_checkpoint is not None and (step in save_steps or step == num_steps):
                save_checkpoint(step)
