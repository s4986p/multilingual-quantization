from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.dataset import LANGUAGES
from utils.hub import save_model
from utils.model import get_logits, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_finetune_path
from utils.training import Sample, generate_reasonings, load_gsm8k_reasonings, train
from utils.understanding import get_understanding

TRAIN_LANGUAGES = [language for language in LANGUAGES if language != "en"]


@dataclass
class Config:
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    loss: Literal["ce", "kl", "hidden"] = "ce"
    model_name: str = "google/gemma-4-E2B-it"
    # 학습 데이터
    # bf16_en: bf16이 영어 prompt에 쓴 reasoning
    # on_policy: 학습 중인 모델이 해당 언어 prompt에 생성한 reasoning
    reasoning_source: Literal["bf16_en", "on_policy"] = "bf16_en"
    train_part: Literal["understanding", "reasoning"] = "understanding"
    num_samples: int = 4000
    seed: int = 0
    max_generation_len: int = 16384


def load_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    train_part: str,
) -> list[Sample]:
    # 문제와 언어마다 (해당 언어 prompt, 영어 prompt, bf16이 영어 prompt에 쓴 text)
    # bf16이 해당 언어로 맞혔고 영어와 해당 언어 reasoning 모두에 understanding이 있는 문제만 사용
    # on_policy는 학습 중에 text를 생성한 것으로 바꿈
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


def get_ce_losses(
    model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 해당 언어 prompt 뒤에 text를 넣고 text 위치에서만 다음 token 예측을 ce로 학습
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
    # 같은 text를 학습 중인 모델에는 해당 언어 prompt 뒤에, bf16에는 영어 prompt 뒤에 넣음
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


def get_losses(
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


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # 학습 중 모델이 바뀌기 때문에 kl, hidden에는 고정된 bf16 따로 사용
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)
    samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.train_part)[: config.num_samples]

    # 양자화 대상과 같게 linear weight만 학습
    linears = [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    train([linear.weight for linear in linears], samples, partial(get_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.reasoning_source, config.train_part), config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(make_finetune_path(config.model_name, config.loss, config.reasoning_source, config.train_part), model)


if __name__ == "__main__":
    main()
