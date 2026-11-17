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
from torch.nn import Linear, Module
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax
from torch.nn.utils import parametrize
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup, set_seed

from utils.dataset import LANGUAGES
from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_hub_path, make_result_name, make_training_path
from utils.quantization import quantize_mxfp4
from utils.understanding import get_understanding

TRAIN_LANGUAGES = [language for language in LANGUAGES if language != "en"]
# gsm8k 해당 언어 understanding의 약 95%가 들어가는 길이
MAX_UNDERSTANDING_LEN = 512
# gsm8k 해당 언어 reasoning의 약 98%가 들어가는 길이
MAX_REASONING_LEN = 2048
# 이 step마다 loss 출력
LOG_STEPS = 16

# sample: (finetune 중인 모델 prompt_ids, bf16 prompt_ids, finetune할 text_ids)
# ce_kl은 (finetune 중인 모델과 bf16 prompt_ids, 영어 question_ids, reasoning_ids)
Sample = tuple[Tensor, Tensor, Tensor]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    # bf16 대신 mxfp4로 양자화한 채로 finetune
    qat: bool = False
    # ce_kl: en_question에서 영어 question은 ce, 뒤 reasoning은 kl
    loss: Literal["ce", "kl", "hidden", "ce_kl"] = "ce"
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    # 학습 데이터
    # bf16_en: bf16이 영어 prompt에 쓴 reasoning
    # on_policy: finetune 중인 모델이 해당 언어 prompt에 쓴 reasoning
    # bf16_en_question_insert: bf16이 해당 언어 prompt + 영어 question 삽입 뒤에 쓴 reasoning
    # bf16_native: bf16이 해당 언어 prompt에 쓴 reasoning, kl은 같은 prompt와 reasoning을 받은 bf16에 맞춤
    reasoning_source: Literal["bf16_en", "on_policy", "bf16_en_question_insert", "bf16_native"] = "bf16_en"
    # reasoning_source 중 finetune할 부분
    # en_question: reasoning_source의 영어 question, ce_kl은 그 뒤 reasoning도 씀
    train_part: Literal["understanding", "reasoning", "en_question"] = "understanding"
    num_samples: int = 4000
    # checkpoint를 저장할 finetune한 sample 수, 마지막은 항상 저장
    save_num_samples: tuple[int, ...] = ()
    max_generation_len: int = 16384
    seed: int = 0


class Mxfp4Ste(Module):
    def forward(
        self,
        weight: Tensor,
    ) -> Tensor:
        # 앞으로는 mxfp4 weight 사용, 뒤로는 gradient를 weight에 그대로 전달
        with torch.no_grad():
            quantized = quantize_mxfp4(weight)
        return weight + (quantized - weight).detach()


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


def load_en_question_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    num_samples: int,
) -> list[Sample]:
    # gsm8k 문제와 언어마다 (해당 언어 prompt, 영어 question, bf16이 영어 question 삽입 뒤에 쓴 reasoning)
    # ce도 같은 sample을 뽑기 때문에 학습 데이터와 순서가 ce_kl과 같음
    # kl은 정답 여부와 상관없이 bf16 분포를 맞추기 때문에 정답 여부로 거르지 않고, reasoning_end를 붙여 MAX_REASONING_LEN보다 길면 제외
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + 영어 question + "\n\n" + reasoning + "<channel|>"
    tensors = load_tensors("en_question_insert_reasoning/gsm8k", make_hub_path("bf16", model_name), make_result_name(max_generation_len, seed))
    reasoning_end_ids = torch.tensor([tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])])
    samples = []
    for language in TRAIN_LANGUAGES:
        parts = [tensors[f"{language}/{part}_ids"].split(tensors[f"{language}/{part}_lens"].tolist()) for part in ("prompt", "question", "reasoning")]
        for prompt_ids, question_ids, reasoning_ids in zip(*parts):
            if len(reasoning_ids) < MAX_REASONING_LEN:
                samples.append((prompt_ids, question_ids, torch.cat([reasoning_ids, reasoning_end_ids])))
    print(f"samples {num_samples} of {len(samples)}")
    return random.sample(samples, num_samples)


def load_native_reasoning_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    num_samples: int,
) -> list[Sample]:
    # gsm8k 문제와 언어마다 (해당 언어 prompt, 같은 prompt, bf16이 해당 언어 prompt에 쓴 reasoning)
    # ce_kl과 같이 정답 여부로 거르지 않고, reasoning_end를 붙여 MAX_REASONING_LEN보다 길면 제외
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + reasoning + "<channel|>"
    tensors = load_tensors("reasoning/gsm8k", make_hub_path("bf16", model_name), make_result_name(max_generation_len, seed))
    reasoning_end_ids = torch.tensor([tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])])
    samples = []
    for language in TRAIN_LANGUAGES:
        parts = [tensors[f"{language}/{part}_ids"].split(tensors[f"{language}/{part}_lens"].tolist()) for part in ("prompt", "reasoning")]
        for prompt_ids, reasoning_ids in zip(*parts):
            if len(reasoning_ids) < MAX_REASONING_LEN:
                samples.append((prompt_ids, prompt_ids, torch.cat([reasoning_ids, reasoning_end_ids])))
    print(f"samples {num_samples} of {len(samples)}")
    return random.sample(samples, num_samples)


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
    # 같은 text를 finetune 중인 모델에는 해당 언어 prompt 뒤에, bf16에는 sample의 bf16 prompt 뒤에 넣음
    # bf16 prompt는 bf16_en, on_policy에서는 영어 prompt, bf16_native에서는 같은 해당 언어 prompt
    # text 위치에서만 다음 token 예측을 bf16에 kl로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, bf16_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_reasoning_hiddens(bf16_model, torch.cat([bf16_prompt_ids, text_ids]), len(bf16_prompt_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = log_softmax(get_logits(model, get_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids))), dim=-1)
        yield (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_positions


def get_ce_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 해당 언어 prompt 뒤에 영어 question과 reasoning을 넣고 영어 question 위치는 ce로 finetune
    # reasoning 위치는 같은 prompt, 영어 question, reasoning을 넣은 bf16에 kl로 맞춤
    # reasoning이 영어 question보다 훨씬 길어 두 loss를 각자의 위치 수로 나눔
    num_question_positions = sum(len(question_ids) for _, question_ids, _ in batch)
    num_reasoning_positions = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
    for prompt_ids, question_ids, reasoning_ids in batch:
        input_ids = torch.cat([prompt_ids, question_ids, reasoning_ids])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_reasoning_hiddens(bf16_model, input_ids, len(prompt_ids) + len(question_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_reasoning_hiddens(model, input_ids, len(prompt_ids)))
        ce_loss = cross_entropy(logits[: len(question_ids)], question_ids.to(logits.device), reduction="sum") / num_question_positions
        logprobs = log_softmax(logits[len(question_ids) :], dim=-1)
        kl_loss = (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_reasoning_positions
        yield ce_loss + kl_loss


def get_layer_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # layer마다 text 위치 hidden, embedding은 같은 text라 두 모델에서 같아 제외
    hidden_states = model.model(input_ids=input_ids[None, :].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack(hidden_states[1:])[:, 0, prompt_len:]


def get_hidden_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # kl과 같이 넣고 text 위치에서 layer마다 hidden을 bf16에 cosine 유사도로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, bf16_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_hiddens = get_layer_hiddens(bf16_model, torch.cat([bf16_prompt_ids, text_ids]), len(bf16_prompt_ids))
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
    if loss == "ce_kl":
        return get_ce_kl_losses(model, bf16_model, batch)
    return get_hidden_losses(model, bf16_model, batch)


def train(
    weights: list[Tensor],
    samples: list[Sample],
    get_batch_losses: Callable[[list[Sample]], Iterator[Tensor]],
    lr: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
    save_steps: tuple[int, ...],
    save_checkpoint: Callable[[int], None],
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
            if step == 1 or step % LOG_STEPS == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            # save_steps와 마지막 step에서 checkpoint 저장
            if step in save_steps or step == num_steps:
                save_checkpoint(step)


def save_checkpoint(
    model: PreTrainedModel,
    linears: list[Linear],
    config: Config,
    step: int,
) -> None:
    # 1. finetune 중인 fp32 weight를 두고, qat는 mxfp4로 양자화한 weight로 바꿈
    # en_question과 qat는 경로에 학습 데이터 수와 지금까지 finetune한 sample 수를 넣음
    num_samples, num_trained_samples = None, None
    if config.train_part == "en_question" or config.qat:
        num_samples, num_trained_samples = config.num_samples, min(step * config.batch_size, config.num_epochs * config.num_samples)
    if config.qat:
        fp32_weights = [linear.parametrizations.weight.original.detach().clone() for linear in linears]
        with torch.no_grad():
            for linear in linears:
                parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
                linear.weight.copy_(quantize_mxfp4(linear.weight))
    else:
        fp32_weights = [linear.weight.detach().clone() for linear in linears]

    # 2. mxfp4 값은 bf16으로 정확히 표현되기 때문에 bf16으로 저장
    model.to(torch.bfloat16)
    save_model(make_training_path("qat" if config.qat else "finetune", config.model_name, config.loss, config.reasoning_source, config.train_part, num_samples, num_trained_samples), model)

    # 3. fp32 weight와 qat의 STE 복원, optimizer가 들고 있는 weight를 그대로 쓰기 때문에 finetune 이어서 가능
    model.float()
    with torch.no_grad():
        for linear, fp32_weight in zip(linears, fp32_weights):
            linear.weight.copy_(fp32_weight)
            if config.qat:
                parametrize.register_parametrization(linear, "weight", Mxfp4Ste())


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # finetune 중 모델이 바뀌기 때문에 kl, hidden, ce_kl에는 고정된 bf16 따로 사용
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)
    if config.train_part == "en_question":
        if config.loss not in ("ce", "ce_kl") or config.reasoning_source != "bf16_en_question_insert":
            raise ValueError("en_question needs loss ce or ce_kl and reasoning_source bf16_en_question_insert")
        samples = load_en_question_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.num_samples)
        # ce는 영어 question만 finetune해 reasoning을 빼고 (해당 언어 prompt, 같은 prompt, 영어 question)으로 바꿈
        if config.loss == "ce":
            samples = [(prompt_ids, prompt_ids, question_ids) for prompt_ids, question_ids, _ in samples]
    elif config.loss == "ce_kl":
        raise ValueError("ce_kl needs train_part en_question")
    elif config.reasoning_source == "bf16_native":
        if config.loss != "kl" or config.train_part != "reasoning":
            raise ValueError("bf16_native needs loss kl and train_part reasoning")
        samples = load_native_reasoning_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.num_samples)
    else:
        samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.train_part)[: config.num_samples]

    # 양자화 대상과 같게 linear weight만 finetune, qat는 mxfp4로 양자화한 채로
    linears = [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    if config.qat:
        for linear in linears:
            parametrize.register_parametrization(linear, "weight", Mxfp4Ste())
    weights = [linear.parametrizations.weight.original if config.qat else linear.weight for linear in linears]
    get_batch_losses = partial(get_finetune_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.reasoning_source, config.train_part)
    # train은 step 단위라 save_num_samples를 step으로 바꿔 전달
    if any(num_trained_samples % config.batch_size for num_trained_samples in config.save_num_samples):
        raise ValueError(f"save_num_samples must be multiples of batch_size {config.batch_size}")
    save_steps = tuple(num_trained_samples // config.batch_size for num_trained_samples in config.save_num_samples)
    train(weights, samples, get_batch_losses, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size, save_steps, partial(save_checkpoint, model, linears, config))


if __name__ == "__main__":
    main()
