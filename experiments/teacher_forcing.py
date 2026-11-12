from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.format import format_duration, format_memory
from utils.hub import download_model, load_result, save_tensors, tensors_exist
from utils.metrics import get_all_correct_mask
from utils.model import get_logits, get_prompt_ids, get_think_hiddens, get_think_ids, load_model
from utils.paths import make_target_path, make_uncalibrated_path


@dataclass
class Config:
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    logit_chunk_len: int = 2048
    # bf16과 비교할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    script_name = "teacher_forcing/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    if tensors_exist(script_name, target, file_name):
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    bf16_target = make_uncalibrated_path("bf16", config.model_name)
    bf16_result = load_result("mgsm", bf16_target, file_name)
    bf16_results = [load_result("mgsm", bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(target))

    tensors = defaultdict(list)
    for language in MGSM_PROMPTS:
        # 1. 모든 bf16 seed가 푼 문제에서 bf16 추론 추출
        all_correct_mask = get_all_correct_mask(bf16_results, language)
        samples = []
        for problem_index, ((prompt, _), think, all_correct) in enumerate(zip(problems[language], bf16_result[language]["thinks"], all_correct_mask)):
            if all_correct:
                samples.append((problem_index, get_prompt_ids(tokenizer, prompt), get_think_ids(tokenizer, config.model_name, think)))
        print(f"{target} {language} samples {len(samples)} of {len(problems[language])} positions {sum(len(think_ids) for _, _, think_ids in samples)}")

        # 2. 추론 token마다 bf16과 양자화 모델의 다음 token 예측을 kl로 비교
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for sample_index, (problem_index, prompt_ids, think_ids) in enumerate(samples):
            input_ids = torch.cat([prompt_ids, think_ids])
            bf16_hiddens = get_think_hiddens(bf16_model, input_ids, len(prompt_ids))
            quantized_hiddens = get_think_hiddens(quantized_model, input_ids, len(prompt_ids))
            for chunk_start in range(0, len(think_ids), config.logit_chunk_len):
                bf16_logprobs = log_softmax(get_logits(bf16_model, bf16_hiddens[chunk_start : chunk_start + config.logit_chunk_len]), dim=-1)
                quantized_logprobs = log_softmax(get_logits(quantized_model, quantized_hiddens[chunk_start : chunk_start + config.logit_chunk_len]), dim=-1)
                tensors[f"{language}/kl"].append((bf16_logprobs.exp() * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
            tensors[f"{language}/problem_index"].append(torch.full_like(think_ids, problem_index, dtype=torch.int32))

            if (sample_index + 1) % 50 == 0:
                print(f"{target} {language} sample {sample_index + 1}/{len(samples)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

        print(f"{target} {language} kl {torch.cat(tensors[f'{language}/kl']).mean():.4f}")

    save_tensors(script_name, target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
