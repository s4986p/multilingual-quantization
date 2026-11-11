import math
import random
from statistics import mean
import time
from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from utils.format import format_duration, format_memory
from utils.hub import load_tensors
from utils.model import get_logits, get_think_hiddens, get_think_ids
from utils.paths import make_bf16_path
from utils.understanding import get_understanding

TRAIN_LANGUAGES = ["zh", "sw", "ja", "te"]


def load_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    think_source: str,
    think_part: str,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    # keep native prompt, en prompt and en or native reasoning or its understanding of each problem
    tensors = load_tensors("gsm8k", make_bf16_path(model_name), f"max_generation_len={max_generation_len},seed={seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in ["en", *TRAIN_LANGUAGES]}
    language_thinks = {language: tokenizer.batch_decode(tensors[f"{language}/think_ids"].split(tensors[f"{language}/think_lens"].tolist()), skip_special_tokens=True) for language in ["en", *TRAIN_LANGUAGES]}
    samples = []
    # data has only problems solved by bf16 in en
    for problem_index, en_think in enumerate(language_thinks["en"]):
        en_understanding = get_understanding(en_think)
        if en_understanding is None:
            continue
        for language in TRAIN_LANGUAGES:
            native_understanding = get_understanding(language_thinks[language][problem_index])
            # skip if bf16 got it wrong on native prompt
            if native_understanding is None or not tensors[f"{language}/corrects"][problem_index]:
                continue
            think = language_thinks["en" if think_source == "en" else language][problem_index]
            samples.append((language_prompt_ids[language][problem_index], language_prompt_ids["en"][problem_index], get_think_ids(tokenizer, model_name, get_understanding(think) if think_part == "understanding" else think)))
    print(f"samples {len(samples)} of {len(language_thinks['en']) * len(TRAIN_LANGUAGES)}")
    return samples


def get_ce_losses(
    model: PreTrainedModel,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> Iterator[Tensor]:
    # put reasoning after native prompt
    # e.g. sw prompt + "<|channel>thought\n" + understanding
    # train next token predictions on that reasoning only, with cross entropy
    # yield loss of each sample, so train can backward it before next sample
    num_positions = sum(len(think_ids) for _, _, think_ids in batch)
    for prompt_ids, _, think_ids in batch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_think_hiddens(model, torch.cat([prompt_ids, think_ids]), len(prompt_ids)))
        yield cross_entropy(logits, think_ids.to(logits.device), reduction="sum") / num_positions


def train(
    model: PreTrainedModel,
    weights: list[Tensor],
    samples: list[tuple[Tensor, Tensor, Tensor]],
    get_losses: Callable[[list[tuple[Tensor, Tensor, Tensor]]], Iterator[Tensor]],
    lr: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
) -> None:
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=lr, weight_decay=0)
    num_batches = math.ceil(len(samples) / batch_size)
    num_steps = num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio * num_steps), num_steps)

    losses = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), batch_size)):
            step = epoch * num_batches + batch_index + 1
            batch = samples[start : start + batch_size]
            batch_loss = 0.0
            for loss in get_losses(batch):
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(batch_loss)
            if step == 1 or step % 16 == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
