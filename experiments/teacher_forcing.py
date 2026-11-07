from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm, load_mmlu_prox_lite
from utils.format import format_duration, format_memory
from utils.hub import download_model, load_result, result_exists, save_tensors
from utils.metrics import get_both_correct_mask
from utils.model import get_logits, get_prompt_ids, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_bf16_path, make_target_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 32768
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    chunk_len: int = 2048
    # model to compare with bf16
    model_name: str = "Qwen/Qwen3.5-2B"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "awq"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = f"teacher_forcing/{config.benchmark}"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name, "safetensors"):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages) if config.benchmark == "mgsm" else load_mmlu_prox_lite(languages)
    bf16_target = make_bf16_path(config.model_name)
    bf16_results = [load_result(config.benchmark, bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in [config.seed, config.seed + 1]]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(target))

    tensors = defaultdict(list)
    for language in languages:
        # 1) use bf16 reasoning on problems solved by all bf16 seeds
        both_correct_mask = get_both_correct_mask([bf16_result[language] for bf16_result in bf16_results])
        samples = []
        for problem_index, ((prompt, _), think, both_correct) in enumerate(zip(problems[language], bf16_results[0][language]["thinks"], both_correct_mask)):
            if both_correct:
                samples.append((problem_index, get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, think)))
        print(f"{language} samples {len(samples)} positions {sum(len(think_ids) for _, _, think_ids in samples)}")

        # 2) compare bf16, quantized at each reasoning position
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for sample_index, (problem_index, prompt_ids, think_ids) in enumerate(samples):
            input_ids = torch.cat([prompt_ids, think_ids])
            bf16_hiddens = get_think_hiddens(bf16_model, input_ids, len(prompt_ids))
            quantized_hiddens = get_think_hiddens(quantized_model, input_ids, len(prompt_ids))
            for start in range(0, len(think_ids), config.chunk_len):
                bf16_logprobs = log_softmax(get_logits(bf16_model, bf16_hiddens[start : start + config.chunk_len]), dim=-1)
                quantized_logprobs = log_softmax(get_logits(quantized_model, quantized_hiddens[start : start + config.chunk_len]), dim=-1)
                bf16_probs = bf16_logprobs.exp()
                tensors[f"{language}/bf16_entropy"].append(-(bf16_probs * bf16_logprobs).sum(dim=-1).cpu())
                tensors[f"{language}/kl"].append((bf16_probs * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
                tensors[f"{language}/bf16_top1"].append(bf16_logprobs.argmax(dim=-1).int().cpu())
                tensors[f"{language}/quantized_top1"].append(quantized_logprobs.argmax(dim=-1).int().cpu())
            tensors[f"{language}/think_ids"].append(think_ids.int())
            tensors[f"{language}/problem_index"].append(torch.full_like(think_ids, problem_index, dtype=torch.int32))

            if (sample_index + 1) % 50 == 0:
                print(f"{language} sample {sample_index + 1}/{len(samples)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

        kl = torch.cat(tensors[f"{language}/kl"])
        top1_mismatch = torch.cat(tensors[f"{language}/bf16_top1"]) != torch.cat(tensors[f"{language}/quantized_top1"])
        print(f"{target} {language} kl {kl.mean():.4f} top1 mismatch {top1_mismatch.float().mean():.2%}")

    save_tensors(entry_name, target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
