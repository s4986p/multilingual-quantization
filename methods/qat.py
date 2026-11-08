from dataclasses import dataclass
import math
import random
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from transformers import PreTrainedModel, get_cosine_schedule_with_warmup, set_seed

from utils.dataset import MGSM_PROMPTS
from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import get_logits, get_think_hiddens, load_model
from utils.paths import make_bf16_path, make_qat_path
from utils.quantization import MXFP4_GROUP_SIZE, find_fp4_scale, round_to_fp4


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    seed: int = 0
    # languages of training questions
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en", "zh", "sw", "ja", "te")
    # prompt that bf16 reads
    # same means same prompt as quantized model
    # en means en prompt
    # reasoning is bf16 reasoning on that prompt
    bf16_question: Literal["same", "en"] = "same"
    # bf16 gsm8k result to take reasoning from
    max_generation_len: int = 16384
    # cut prompt, reasoning to this length
    max_sequence_len: int = 2048
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8


class STELinear(Module):
    def __init__(
        self,
        linear: Linear,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        # fix scale from bf16 weight, so groups are not rounded again to new scale when group max crosses power of 2
        self.scale = find_fp4_scale(self.groups())[..., None]

    def groups(
        self,
    ) -> Tensor:
        return self.weight.reshape(self.weight.shape[0], -1, MXFP4_GROUP_SIZE)

    def quantize(
        self,
    ) -> Tensor:
        # detach only rounding, so gradient of quantized weight goes to weight
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
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(config.model_name)
    quantized_model.float()

    # 1) pair quantized, bf16 inputs with bf16 reasoning picked by bf16_question, on problems solved by bf16 in en
    # e.g. same  quantized sw prompt + reasoning on sw, bf16 sw prompt + reasoning on sw
    #      en    quantized sw prompt + reasoning on en, bf16 en prompt + reasoning on en
    tensors = load_tensors("gsm8k", make_bf16_path(config.model_name), f"max_generation_len={config.max_generation_len},seed={config.seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in MGSM_PROMPTS}
    language_think_ids = {language: tensors[f"{language}/think_ids"].split(tensors[f"{language}/think_lens"].tolist()) for language in MGSM_PROMPTS}
    samples = []
    for problem_index in range(len(language_prompt_ids["en"])):
        for language in config.languages:
            bf16_language = language if config.bf16_question == "same" else "en"
            quantized_prompt_ids = language_prompt_ids[language][problem_index]
            bf16_prompt_ids = language_prompt_ids[bf16_language][problem_index]
            think_ids = language_think_ids[bf16_language][problem_index][: config.max_sequence_len - max(len(quantized_prompt_ids), len(bf16_prompt_ids))]
            samples.append((torch.cat([quantized_prompt_ids, think_ids]), len(quantized_prompt_ids), torch.cat([bf16_prompt_ids, think_ids]), len(bf16_prompt_ids)))
    print(f"samples {len(samples)} problems {len(samples) // len(config.languages)}")

    # 2) use ste linears, so gradient goes through rounding
    replaced = replace_linears(quantized_model)
    ste_linears = [ste_linear for _, _, ste_linear in replaced]
    for ste_linear in ste_linears:
        ste_linear.weight.requires_grad_(True)
    optimizer = AdamW([ste_linear.weight for ste_linear in ste_linears], lr=config.lr, weight_decay=0)
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_steps = config.num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(config.warmup_ratio * num_steps), num_steps)

    # 3) match quantized next token predictions to bf16 with kl, only at reasoning positions
    kls = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for batch_index, start in enumerate(range(0, len(samples), config.batch_size)):
            step = epoch * num_batches + batch_index + 1
            batch = samples[start : start + config.batch_size]
            num_positions = sum(len(quantized_input_ids) - quantized_prompt_len for quantized_input_ids, quantized_prompt_len, _, _ in batch)
            batch_kl = 0.0

            for quantized_input_ids, quantized_prompt_len, bf16_input_ids, bf16_prompt_len in batch:
                with torch.no_grad():
                    bf16_logprobs = log_softmax(get_logits(bf16_model, get_think_hiddens(bf16_model, bf16_input_ids, bf16_prompt_len)), dim=-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    quantized_logprobs = log_softmax(get_logits(quantized_model, get_think_hiddens(quantized_model, quantized_input_ids, quantized_prompt_len)), dim=-1)
                kl = (bf16_logprobs.exp() * (bf16_logprobs - quantized_logprobs)).sum(dim=-1)
                loss = kl.sum() / num_positions
                loss.backward()
                batch_kl += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            kls.append(batch_kl)
            if step == 1 or step % 16 == 0:
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {batch_index + 1}/{num_batches} kl {mean(kls):.4f} lr {scheduler.get_last_lr()[0]:.2e} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                kls = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    # 4) put quantized weights back into linears, and save
    layers = quantized_model.model.language_model.layers
    with torch.no_grad():
        for name, linear, ste_linear in replaced:
            linear.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(layers.get_submodule(parent_name), child_name, linear)
    quantized_model.to(torch.bfloat16)
    save_model(make_qat_path(config.model_name, config.languages, config.bf16_question), quantized_model)


if __name__ == "__main__":
    main()
