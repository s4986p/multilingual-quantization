from dataclasses import dataclass
import math
import random
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from utils.dataset import load_gsm8k
from utils.format import format_duration, format_memory
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_prompt_ids, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_bf16_path, make_translate_first_finetune_path

# saved gemma reasoning starts with this after think start token
# e.g. "<|channel>thought\nHere's a thinking process ...<channel|>"
THINK_HEADER = "thought\n"


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    seed: int = 0
    # languages of training questions
    languages: tuple[Literal["zh", "sw", "ja", "te"], ...] = ("zh", "sw", "ja", "te")
    # bf16 gsm8k result to take reasoning from
    max_generation_len: int = 16384
    # cut prompt, reasoning to this length
    max_sequence_len: int = 2048
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()

    # 1) pair prompt of each language with en question, then bf16 reasoning on en prompt, on problems solved by bf16 in en
    # e.g. sw prompt + "<|channel>thought\n" + en question + blank line + reasoning on en + think end
    tensors = load_tensors("gsm8k", make_bf16_path(config.model_name), f"max_generation_len={config.max_generation_len},seed={config.seed}")
    language_prompt_ids = {language: tensors[f"{language}/prompt_ids"].split(tensors[f"{language}/prompt_lens"].tolist()) for language in ["en", *config.languages]}
    bf16_think_ids = tensors["en/think_ids"].split(tensors["en/think_lens"].tolist())
    problems = load_gsm8k()
    # saved tensors keep no gsm8k index, so find it by en prompt ids
    gsm8k_indices = {tuple(get_prompt_ids(tokenizer, en_prompt).tolist()): gsm8k_index for gsm8k_index, (en_prompt, _) in enumerate(problems["en"])}
    think_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[config.model_name]["think_end"])
    samples = []
    for problem_index, en_prompt_ids in enumerate(language_prompt_ids["en"]):
        # drop instruction, blank line before question
        en_question = problems["en"][gsm8k_indices[tuple(en_prompt_ids.tolist())]][0].split("\n\n", 1)[1]
        bf16_think = tokenizer.decode(bf16_think_ids[problem_index], skip_special_tokens=True).removeprefix(THINK_HEADER)
        think_ids = torch.cat([get_think_ids(tokenizer, config.model_name, THINK_HEADER + en_question + "\n\n" + bf16_think), torch.tensor([think_end_id])])
        for language in config.languages:
            prompt_ids = language_prompt_ids[language][problem_index]
            samples.append((torch.cat([prompt_ids, think_ids[: config.max_sequence_len - len(prompt_ids)]]), len(prompt_ids)))
    print(f"samples {len(samples)} problems {len(language_prompt_ids['en'])}")

    # 2) train same linears as qat, without rounding
    weights = [linear.weight for linear in model.model.language_model.layers.modules() if isinstance(linear, Linear)]
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=config.lr, weight_decay=0)
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_steps = config.num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(config.warmup_ratio * num_steps), num_steps)

    # 3) train next token predictions on en question with cross entropy
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

    model.to(torch.bfloat16)
    save_model(make_translate_first_finetune_path(config.model_name, config.languages), model)


if __name__ == "__main__":
    main()
