import math
import random
from statistics import mean
import time

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
    think_language: str,
) -> list[tuple[Tensor, int]]:
    # put en or native understanding at start of reasoning, after prompt with native instruction
    # e.g. sw prompt + "<|channel>thought\n" + understanding
    tensors = load_tensors("gsm8k", make_bf16_path(model_name), f"max_generation_len={max_generation_len},seed={seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in TRAIN_LANGUAGES}
    language_thinks = {language: tokenizer.batch_decode(tensors[f"{language}/think_ids"].split(tensors[f"{language}/think_lens"].tolist()), skip_special_tokens=True) for language in ["en", *TRAIN_LANGUAGES]}
    samples = []
    # data has only problems solved by bf16 in en
    for problem_index, en_think in enumerate(language_thinks["en"]):
        en_understanding = get_understanding(en_think)
        if en_understanding is None:
            continue
        for language in TRAIN_LANGUAGES:
            native_understanding = get_understanding(language_thinks[language][problem_index])
            # skip if bf16 got it wrong in native language
            if native_understanding is None or not tensors[f"{language}/corrects"][problem_index]:
                continue
            understanding = en_understanding if think_language == "en" else native_understanding
            prompt_ids = language_prompt_ids[language][problem_index]
            samples.append((torch.cat([prompt_ids, get_think_ids(tokenizer, model_name, understanding)]), len(prompt_ids)))
    print(f"samples {len(samples)} of {len(language_thinks['en']) * len(TRAIN_LANGUAGES)}")
    return samples


def train(
    model: PreTrainedModel,
    weights: list[Tensor],
    samples: list[tuple[Tensor, int]],
    lr: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
) -> None:
    # train next token predictions on understanding only, with cross entropy
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
            num_positions = sum(len(input_ids) - prompt_len for input_ids, prompt_len in batch)
            batch_loss = 0.0

            for input_ids, prompt_len in batch:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = get_logits(model, get_think_hiddens(model, input_ids, prompt_len))
                loss = cross_entropy(logits, input_ids[prompt_len:].to(logits.device), reduction="sum") / num_positions
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
