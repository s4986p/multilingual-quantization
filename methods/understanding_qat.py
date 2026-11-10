from dataclasses import dataclass
import math
import random
from statistics import mean
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedModel, get_cosine_schedule_with_warmup, set_seed

from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import get_logits, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_bf16_path, make_understanding_qat_path
from utils.quantization import MXFP4_GROUP_SIZE, find_fp4_scale, round_to_fp4
from utils.understanding import get_understanding

TRAIN_LANGUAGES = ["zh", "sw", "ja", "te"]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    # data to train on
    seed: int = 0
    max_generation_len: int = 16384


class STELinear(Module):
    def __init__(
        self,
        linear: Linear,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.scale = find_fp4_scale(self.groups())[..., None]

    def groups(
        self,
    ) -> Tensor:
        return self.weight.reshape(self.weight.shape[0], -1, MXFP4_GROUP_SIZE)

    def quantize(
        self,
    ) -> Tensor:
        # detach only rounding, because gradient of quantized weight must go to weight
        with torch.no_grad():
            quantized = round_to_fp4(self.groups(), self.scale).reshape(self.weight.shape)
        return self.weight + (quantized - self.weight).detach()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return torch.nn.functional.linear(x, self.quantize(), self.bias)


def replace_linears(
    model: PreTrainedModel,
) -> list[tuple[str, Linear, STELinear]]:
    layers = model.model.language_model.layers
    linears = [(name, linear) for name, linear in layers.named_modules() if isinstance(linear, Linear)]
    replaced = []
    for name, linear in linears:
        ste_linear = STELinear(linear)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(layers.get_submodule(parent_name), child_name, ste_linear)
        replaced.append((name, linear, ste_linear))
    return replaced


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()

    # 1) put en understanding at start of reasoning, after prompt with native instruction
    # e.g. sw prompt + "<|channel>thought\n" + en understanding
    tensors = load_tensors("gsm8k", make_bf16_path(config.model_name), f"max_generation_len={config.max_generation_len},seed={config.seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in TRAIN_LANGUAGES}
    en_thinks = tokenizer.batch_decode(tensors["en/think_ids"].split(tensors["en/think_lens"].tolist()), skip_special_tokens=True)
    samples = []
    for problem_index, en_think in enumerate(en_thinks):
        en_understanding = get_understanding(en_think)
        if en_understanding is None:
            continue
        en_understanding_ids = get_think_ids(tokenizer, config.model_name, en_understanding)
        for language in TRAIN_LANGUAGES:
            prompt_ids = language_prompt_ids[language][problem_index]
            samples.append((torch.cat([prompt_ids, en_understanding_ids]), len(prompt_ids)))
    print(f"samples {len(samples)} problems {len(samples) // len(TRAIN_LANGUAGES)} of {len(en_thinks)}")

    # 2) use ste linears, because gradient must go through mxfp4 rounding
    replaced = replace_linears(model)
    ste_linears = [ste_linear for _, _, ste_linear in replaced]
    for ste_linear in ste_linears:
        ste_linear.weight.requires_grad_(True)
    optimizer = AdamW([ste_linear.weight for ste_linear in ste_linears], lr=config.lr, weight_decay=0)
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_steps = config.num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(config.warmup_ratio * num_steps), num_steps)

    # 3) train next token predictions on en understanding only, with cross entropy
    losses = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), config.batch_size)):
            step = epoch * num_batches + batch_index + 1
            batch = samples[start : start + config.batch_size]
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
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(losses):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                losses = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    # 4) put quantized weights back into linears and save
    layers = model.model.language_model.layers
    with torch.no_grad():
        for name, linear, ste_linear in replaced:
            linear.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(layers.get_submodule(parent_name), child_name, linear)
    model.to(torch.bfloat16)
    save_model(make_understanding_qat_path(config.model_name), model)


if __name__ == "__main__":
    main()
