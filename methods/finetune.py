from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
import math
import random
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import cross_entropy, log_softmax
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup, set_seed

from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_finetune_path, make_uncalibrated_path
from utils.understanding import get_understanding

TRAIN_LANGUAGES = ["zh", "sw", "ja", "te"]
# 생성은 512 token까지, gsm8k 해당 언어 문제 이해의 약 95%가 그 안에 들어가기 때문
MAX_UNDERSTANDING_LEN = 512
# 생성은 2048 token까지, gsm8k 해당 언어 추론의 약 98%가 그 안에 들어가기 때문
MAX_THINK_LEN = 2048


@dataclass
class Config:
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    loss: Literal["ce", "kl"] = "ce"
    model_name: str = "google/gemma-4-E2B-it"
    # 학습할 데이터
    think_source: Literal["en", "on_policy"] = "en"
    think_part: Literal["understanding", "reasoning"] = "understanding"
    num_samples: int = 4000
    seed: int = 0
    max_generation_len: int = 16384


def load_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    think_source: str,
    think_part: str,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    # 문제와 언어마다 해당 언어 프롬프트, 영어 프롬프트, 학습할 텍스트 반환
    tensors = load_tensors("gsm8k", make_uncalibrated_path("bf16", model_name), f"max_generation_len={max_generation_len},seed={seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in ["en", *TRAIN_LANGUAGES]}
    language_thinks = {language: tokenizer.batch_decode(tensors[f"{language}/think_ids"].split(tensors[f"{language}/think_lens"].tolist()), skip_special_tokens=True) for language in ["en", *TRAIN_LANGUAGES]}
    samples = []
    # 데이터에는 bf16이 영어로 푼 문제만 있음
    for problem_index, en_think in enumerate(language_thinks["en"]):
        en_understanding = get_understanding(en_think)
        if en_understanding is None:
            continue
        for language in TRAIN_LANGUAGES:
            understanding = get_understanding(language_thinks[language][problem_index])
            if understanding is None or not tensors[f"{language}/corrects"][problem_index]:
                continue
            think = language_thinks["en" if think_source == "en" else language][problem_index]
            samples.append((language_prompt_ids[language][problem_index], language_prompt_ids["en"][problem_index], get_think_ids(tokenizer, model_name, get_understanding(think) if think_part == "understanding" else think)))
    print(f"samples {len(samples)} of {len(language_thinks['en']) * len(TRAIN_LANGUAGES)}")
    return samples


def generate_thinks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    think_part: str,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    # 1. 해당 언어 프롬프트에 추론 생성
    sampling = GENERATION_CONFIGS[model_name]["sampling"]
    inputs = tokenizer.pad({"input_ids": [prompt_ids for prompt_ids, _, _ in batch]}, padding_side="left", return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"], max_new_tokens=MAX_UNDERSTANDING_LEN if think_part == "understanding" else MAX_THINK_LEN)

    # 2. 추론 끝에서 자름
    # 예시) "<|channel>thought\n...<channel|>Answer: 72" → "thought\n..."
    think_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["think_end"])
    thinks = []
    for ids in output_ids[:, inputs["input_ids"].shape[1] :]:
        end = (ids == think_end_id).nonzero()
        thinks.append(tokenizer.decode(ids[: end[0, 0]] if len(end) else ids, skip_special_tokens=True))

    # 3. 추론을 생성한 것으로 바꾸고 문제 이해가 없으면 제외
    generated_batch = []
    for (prompt_ids, en_prompt_ids, _), think in zip(batch, thinks):
        understanding = get_understanding(think)
        if understanding is not None:
            generated_batch.append((prompt_ids, en_prompt_ids, get_think_ids(tokenizer, model_name, understanding if think_part == "understanding" else think)))
    return generated_batch


def get_ce_losses(
    model: PreTrainedModel,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> Iterator[Tensor]:
    # 해당 언어 프롬프트 뒤에 학습할 텍스트를 넣고 그 텍스트 위치에서만 다음 token 예측을 ce로 학습
    # 예시) 스와힐리어 프롬프트 + "<|channel>thought\n" + 문제 이해
    # sample마다 loss 반환, sample마다 backward해 메모리를 아끼기 때문
    num_positions = sum(len(think_ids) for _, _, think_ids in batch)
    for prompt_ids, _, think_ids in batch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_think_hiddens(model, torch.cat([prompt_ids, think_ids]), len(prompt_ids)))
        yield cross_entropy(logits, think_ids.to(logits.device), reduction="sum") / num_positions


def get_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> Iterator[Tensor]:
    # 같은 텍스트를 학습 중인 모델에는 해당 언어 프롬프트 뒤에, bf16에는 영어 프롬프트 뒤에 넣고 그 텍스트 위치에서만 다음 token 예측을 bf16에 kl로 맞춤
    # 예시) 스와힐리어 프롬프트 + "<|channel>thought\n" + 문제 이해, 영어 프롬프트 + "<|channel>thought\n" + 문제 이해
    num_positions = sum(len(think_ids) for _, _, think_ids in batch)
    for prompt_ids, en_prompt_ids, think_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_think_hiddens(bf16_model, torch.cat([en_prompt_ids, think_ids]), len(en_prompt_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = log_softmax(get_logits(model, get_think_hiddens(model, torch.cat([prompt_ids, think_ids]), len(prompt_ids))), dim=-1)
        yield (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_positions


def get_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    loss: str,
    think_source: str,
    think_part: str,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> Iterator[Tensor]:
    if think_source == "on_policy":
        batch = generate_thinks(model, tokenizer, model_name, think_part, batch)
    if loss == "ce":
        return get_ce_losses(model, batch)
    return get_kl_losses(model, bf16_model, batch)


def train(
    weights: list[Tensor],
    samples: list[tuple[Tensor, Tensor, Tensor]],
    get_batch_losses: Callable[[list[tuple[Tensor, Tensor, Tensor]]], Iterator[Tensor]],
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
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), batch_size)):
            step = epoch * num_batches + batch_index + 1
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


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # kl에는 고정된 bf16 따로 사용, 학습 중 모델이 바뀌기 때문
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)
    samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.think_source, config.think_part)[: config.num_samples]

    # linear weight만 학습, 양자화 대상과 같게 하기 때문
    linears = [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    train([linear.weight for linear in linears], samples, partial(get_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.think_source, config.think_part), config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(make_finetune_path(config.model_name, config.loss, config.think_source, config.think_part), model)


if __name__ == "__main__":
    main()
