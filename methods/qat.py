from dataclasses import dataclass
from functools import partial
import random
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.utils import parametrize
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from methods.finetune import TRAIN_LANGUAGES, get_ce_kl_losses, get_kl_losses, load_en_question_reasoning_samples
from methods.mxfp4 import quantize_weight
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, load_model
from utils.paths import make_hub_path, make_training_path
from utils.training import MAX_REASONING_LEN, Sample, train


@dataclass
class Config:
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    # ce_kl: 영어 question은 ce, 뒤 reasoning은 kl
    # kl: reasoning만 kl
    loss: Literal["ce_kl", "kl"] = "ce_kl"
    model_name: str = "google/gemma-4-E2B-it"
    # 학습 데이터
    # bf16_en_question_insert: bf16이 해당 언어 prompt + 영어 question 삽입 뒤에 쓴 reasoning
    # bf16_native: bf16이 해당 언어 prompt에 쓴 reasoning
    reasoning_source: Literal["bf16_en_question_insert", "bf16_native"] = "bf16_en_question_insert"
    train_part: Literal["en_question", "reasoning"] = "en_question"
    num_samples: int = 20000
    # checkpoint를 저장할 학습한 sample 수, 마지막은 항상 저장
    save_num_samples: tuple[int, ...] = ()
    seed: int = 0
    max_generation_len: int = 16384


class Mxfp4Ste(Module):
    def forward(
        self,
        weight: Tensor,
    ) -> Tensor:
        # 앞으로는 mxfp4 weight 사용, 뒤로는 gradient를 weight에 그대로 전달
        with torch.no_grad():
            quantized = quantize_weight(weight)
        return weight + (quantized - weight).detach()


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
    tensors = load_tensors("reasoning/gsm8k", make_hub_path("bf16", model_name), f"max_generation_len={max_generation_len},seed={seed}")
    reasoning_end_ids = torch.tensor([tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])])
    samples = []
    for language in TRAIN_LANGUAGES:
        parts = [tensors[f"{language}/{part}_ids"].split(tensors[f"{language}/{part}_lens"].tolist()) for part in ("prompt", "reasoning")]
        for prompt_ids, reasoning_ids in zip(*parts):
            if len(reasoning_ids) < MAX_REASONING_LEN:
                samples.append((prompt_ids, prompt_ids, torch.cat([reasoning_ids, reasoning_end_ids])))
    print(f"samples {num_samples} of {len(samples)}")
    return random.sample(samples, num_samples)


def save_checkpoint(
    model: PreTrainedModel,
    linears: list[Linear],
    config: Config,
    step: int,
) -> None:
    # 1. 학습 중인 fp32 weight를 두고 mxfp4로 양자화한 weight로 바꿈
    num_trained_samples = min(step * config.batch_size, config.num_epochs * config.num_samples)
    fp32_weights = [linear.parametrizations.weight.original.detach().clone() for linear in linears]
    with torch.no_grad():
        for linear in linears:
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
            linear.weight.copy_(quantize_weight(linear.weight))

    # 2. mxfp4 값은 bf16으로 정확히 표현되기 때문에 bf16으로 저장
    model.to(torch.bfloat16)
    save_model(make_training_path("qat", config.model_name, config.loss, config.reasoning_source, config.train_part, config.num_samples, num_trained_samples), model)

    # 3. fp32 weight와 STE 복원, optimizer가 들고 있는 weight를 그대로 쓰기 때문에 학습 이어서 가능
    model.float()
    with torch.no_grad():
        for linear, fp32_weight in zip(linears, fp32_weights):
            linear.weight.copy_(fp32_weight)
            parametrize.register_parametrization(linear, "weight", Mxfp4Ste())


def main() -> None:
    config = parse(Config)
    if (config.loss, config.reasoning_source, config.train_part) not in (("ce_kl", "bf16_en_question_insert", "en_question"), ("kl", "bf16_native", "reasoning")):
        raise ValueError("ce_kl needs bf16_en_question_insert and en_question, kl needs bf16_native and reasoning")
    if any(num_trained_samples % config.batch_size for num_trained_samples in config.save_num_samples):
        raise ValueError(f"save_num_samples must be multiples of batch_size {config.batch_size}")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # 학습 중 모델이 바뀌기 때문에 kl에는 고정된 bf16 따로 사용
    bf16_model = load_model(config.model_name)
    # ce_kl은 finetune과 같은 seed와 순서로 뽑아 같은 sample 사용
    if config.loss == "ce_kl":
        samples = load_en_question_reasoning_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.num_samples)
        get_batch_losses = partial(get_ce_kl_losses, model, bf16_model)
    else:
        samples = load_native_reasoning_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.num_samples)
        get_batch_losses = partial(get_kl_losses, model, bf16_model)

    # 양자화 대상과 같게 linear weight만 mxfp4로 양자화해 학습
    linears = [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    for linear in linears:
        parametrize.register_parametrization(linear, "weight", Mxfp4Ste())
    # train은 step 단위라 save_num_samples를 step으로 바꿔 전달
    save_steps = tuple(num_trained_samples // config.batch_size for num_trained_samples in config.save_num_samples)
    weights = [linear.parametrizations.weight.original for linear in linears]
    train(weights, samples, get_batch_losses, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size, save_steps, partial(save_checkpoint, model, linears, config))


if __name__ == "__main__":
    main()
