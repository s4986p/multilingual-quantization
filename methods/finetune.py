from dataclasses import dataclass
from functools import partial
from typing import Iterator, Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.hub import save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_finetune_path
from utils.training import get_ce_losses, load_samples, train
from utils.understanding import get_understanding

# cap generation at 512 tokens, because about 95% of native understandings on gsm8k fit in it
MAX_UNDERSTANDING_LEN = 512
# cap generation at 2048 tokens, because about 98% of native reasoning on gsm8k fit in it
MAX_THINK_LEN = 2048


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    loss: Literal["ce", "kl"] = "ce"
    # data to train on
    think_source: Literal["en", "native", "on_policy"] = "en"
    think_part: Literal["understanding", "reasoning"] = "understanding"
    num_samples: int = 4000
    seed: int = 0
    max_generation_len: int = 16384


def generate_thinks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    think_part: str,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    # 1) write reasoning on native prompt
    generation_config = GENERATION_CONFIGS[model_name]
    inputs = tokenizer.pad({"input_ids": [prompt_ids for prompt_ids, _, _ in batch]}, padding_side="left", return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, do_sample=True, temperature=generation_config["temperature"], top_p=generation_config["top_p"], top_k=generation_config["top_k"], max_new_tokens=MAX_UNDERSTANDING_LEN if think_part == "understanding" else MAX_THINK_LEN)

    # 2) cut each at end of reasoning
    # e.g. "<|channel>thought\n...<channel|>Answer: 72" -> "thought\n..."
    think_end_id = tokenizer.convert_tokens_to_ids(generation_config["think_end"])
    thinks = []
    for ids in output_ids[:, inputs["input_ids"].shape[1] :]:
        end = (ids == think_end_id).nonzero()
        thinks.append(tokenizer.decode(ids[: end[0, 0]] if len(end) else ids, skip_special_tokens=True))

    # 3) replace reasoning with generated one, and skip if it has no understanding
    generated_batch = []
    for (prompt_ids, en_prompt_ids, _), think in zip(batch, thinks):
        understanding = get_understanding(think)
        if understanding is not None:
            generated_batch.append((prompt_ids, en_prompt_ids, get_think_ids(tokenizer, model_name, understanding if think_part == "understanding" else think)))
    return generated_batch


def get_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> Iterator[Tensor]:
    # put same reasoning after native prompt for model and after en prompt for bf16
    # e.g. sw prompt + "<|channel>thought\n" + understanding, en prompt + "<|channel>thought\n" + understanding
    # match next token predictions to bf16 with kl, only at reasoning positions
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


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # load frozen bf16 for kl, because model changes while training
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)

    samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.think_source, config.think_part)[: config.num_samples]

    # train same linears as qat, because only mxfp4 rounding must differ
    linears = [linear for linear in model.model.language_model.layers.modules() if isinstance(linear, Linear)]
    train(model, [linear.weight for linear in linears], samples, partial(get_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.think_source, config.think_part), config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(make_finetune_path(config.model_name, config.loss, config.think_source, config.think_part), model)


if __name__ == "__main__":
    main()
