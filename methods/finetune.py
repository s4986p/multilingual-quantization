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
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup, set_seed

from utils.dataset import LANGUAGES
from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_hub_path, make_result_name, make_training_path
from utils.understanding import get_understanding

TRAIN_LANGUAGES = [language for language in LANGUAGES if language != "en"]
# gsm8k 해당 언어 understanding의 약 95%가 들어가는 길이
MAX_UNDERSTANDING_LEN = 512
# gsm8k 해당 언어 reasoning의 약 98%가 들어가는 길이
MAX_REASONING_LEN = 2048

# sample: (finetune 중인 모델 prompt_ids, bf16 prompt_ids, finetune할 text_ids)
Sample = tuple[Tensor, Tensor, Tensor]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    loss: Literal["ce", "kl", "hidden"] = "ce"
    # 학습 데이터
    # bf16_en: bf16이 영어 prompt에 쓴 reasoning
    # on_policy: finetune 중인 모델이 해당 언어 prompt에 쓴 reasoning
    reasoning_source: Literal["bf16_en", "on_policy"] = "bf16_en"
    # reasoning_source 중 finetune할 부분
    train_part: Literal["understanding", "reasoning"] = "understanding"
    num_samples: int = 4000
    max_generation_len: int = 16384
    seed: int = 0


def load_gsm8k_reasonings(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
) -> dict[str, dict[str, list]]:
    # 언어마다 bf16이 gsm8k에 쓴 prompt_ids, reasoning, 정답 여부
    # data/reasoning.py가 bf16이 영어로 맞힌 문제만 저장했기 때문에 언어끼리 같은 문제 순서
    tensors = load_tensors("reasoning/gsm8k", make_hub_path("bf16", model_name), make_result_name(max_generation_len, seed))
    return {
        language: {
            "prompt_ids": tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()),
            "reasonings": tokenizer.batch_decode(tensors[f"{language}/reasoning_ids"].split(tensors[f"{language}/reasoning_lens"].tolist()), skip_special_tokens=True),
            "corrects": tensors[f"{language}/corrects"].tolist(),
        }
        for language in LANGUAGES
    }


def load_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    train_part: str,
) -> list[Sample]:
    # 문제와 언어마다 (해당 언어 prompt, 영어 prompt, bf16이 영어 prompt에 쓴 text)
    # bf16이 해당 언어로 맞혔고 영어와 해당 언어 reasoning 모두에 understanding이 있는 문제만 사용
    # on_policy는 finetune 중에 text를 생성한 것으로 바꿈
    reasonings = load_gsm8k_reasonings(tokenizer, model_name, seed, max_generation_len)
    samples = []
    for problem_index, en_reasoning in enumerate(reasonings["en"]["reasonings"]):
        en_understanding = get_understanding(en_reasoning)
        if en_understanding is None:
            continue
        for language in TRAIN_LANGUAGES:
            if get_understanding(reasonings[language]["reasonings"][problem_index]) is None or not reasonings[language]["corrects"][problem_index]:
                continue
            text_ids = get_reasoning_ids(tokenizer, model_name, en_understanding if train_part == "understanding" else en_reasoning)
            samples.append((reasonings[language]["prompt_ids"][problem_index], reasonings["en"]["prompt_ids"][problem_index], text_ids))
    print(f"samples {len(samples)} of {len(reasonings['en']['reasonings']) * len(TRAIN_LANGUAGES)}")
    return samples


def generate_reasonings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    train_part: str,
    batch: list[Sample],
) -> list[Sample]:
    # 1. finetune 중인 모델이 해당 언어 prompt에 reasoning 생성
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

    # 3. finetune할 text를 생성한 것으로 바꾸고 understanding이 없으면 제외
    generated_batch = []
    for (prompt_ids, bf16_prompt_ids, _), reasoning in zip(batch, reasonings):
        understanding = get_understanding(reasoning)
        if understanding is not None:
            generated_batch.append((prompt_ids, bf16_prompt_ids, get_reasoning_ids(tokenizer, model_name, understanding if train_part == "understanding" else reasoning)))
    return generated_batch


def get_ce_losses(
    model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 해당 언어 prompt 뒤에 text를 넣고 text 위치에서만 다음 token 예측을 ce로 finetune
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + understanding
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, _, text_ids in batch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
        yield cross_entropy(logits, text_ids.to(logits.device), reduction="sum") / num_positions


def get_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 같은 text를 finetune 중인 모델에는 해당 언어 prompt 뒤에, bf16에는 영어 prompt 뒤에 넣음
    # text 위치에서만 다음 token 예측을 bf16에 kl로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, en_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_reasoning_hiddens(bf16_model, torch.cat([en_prompt_ids, text_ids]), len(en_prompt_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = log_softmax(get_logits(model, get_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids))), dim=-1)
        yield (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_positions


def get_layer_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # layer마다 text 위치 hidden, embedding은 두 입력에서 같아 제외
    hidden_states = model.model(input_ids=input_ids[None, :].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack(hidden_states[1:])[:, 0, prompt_len:]


def get_hidden_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # kl과 같이 넣고 text 위치에서 layer마다 hidden을 bf16에 cosine 유사도로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, en_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_hiddens = get_layer_hiddens(bf16_model, torch.cat([en_prompt_ids, text_ids]), len(en_prompt_ids))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hiddens = get_layer_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids))
        yield (1 - cosine_similarity(hiddens.float(), bf16_hiddens.float(), dim=-1)).mean(0).sum() / num_positions


def get_finetune_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
    batch: list[Sample],
) -> Iterator[Tensor]:
    if reasoning_source == "on_policy":
        batch = generate_reasonings(model, tokenizer, model_name, train_part, batch)
    if loss == "ce":
        return get_ce_losses(model, batch)
    if loss == "kl":
        return get_kl_losses(model, bf16_model, batch)
    return get_hidden_losses(model, bf16_model, batch)


def train(
    weights: list[Tensor],
    samples: list[Sample],
    get_batch_losses: Callable[[list[Sample]], Iterator[Tensor]],
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


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # finetune 중 모델이 바뀌기 때문에 kl, hidden에는 고정된 bf16 따로 사용
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)
    samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.train_part)[: config.num_samples]

    # 양자화 대상과 같게 linear weight만 finetune
    weights = [module.weight for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    get_batch_losses = partial(get_finetune_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.reasoning_source, config.train_part)
    train(weights, samples, get_batch_losses, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(make_training_path("finetune", config.model_name, config.loss, config.reasoning_source, config.train_part), model)


if __name__ == "__main__":
    main()
