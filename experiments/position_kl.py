from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, load_mgsm
from utils.format import format_duration, format_memory
from utils.hub import download_model, load_result, load_results, save_tensors, tensors_exist
from utils.metrics import get_bf16_solved_mask
from utils.model import get_logits, get_prompt_ids, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_hub_path, make_result_name

# 이 문제 수마다 진행 상황 출력
LOG_PROBLEMS = 50


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    kl_batch_len: int = 2048
    max_generation_len: int = 16384
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    experiment_name = "position_kl/mgsm"
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    file_name = make_result_name(config.max_generation_len, config.seed)
    if tensors_exist(experiment_name, hub_path, file_name):
        print(f"{hub_path} already evaluated")
        return
    problems = load_mgsm(LANGUAGES)
    bf16_hub_path = make_hub_path("bf16", config.model_name)
    bf16_result = load_result("mgsm", bf16_hub_path, file_name)
    bf16_results = load_results("mgsm", bf16_hub_path, config.max_generation_len, config.bf16_seeds)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(hub_path))

    tensors = defaultdict(list)
    for language in LANGUAGES:
        # 1. 모든 bf16 seed가 맞힌 문제에서 bf16 reasoning 추출
        bf16_solved_mask = get_bf16_solved_mask(bf16_results, language)
        problem_inputs = []
        for problem_index, ((prompt, _), reasoning, solved) in enumerate(zip(problems[language], bf16_result[language]["reasonings"], bf16_solved_mask)):
            if solved:
                problem_inputs.append((problem_index, get_prompt_ids(tokenizer, prompt), get_reasoning_ids(tokenizer, config.model_name, reasoning)))
        print(f"{hub_path} {language} problems {len(problem_inputs)} of {len(problems[language])} positions {sum(len(reasoning_ids) for _, _, reasoning_ids in problem_inputs)}")

        # 2. reasoning token마다 bf16과 양자화 모델의 다음 token 예측을 kl로 비교
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for input_index, (problem_index, prompt_ids, reasoning_ids) in enumerate(problem_inputs):
            input_ids = torch.cat([prompt_ids, reasoning_ids])
            bf16_hiddens = get_reasoning_hiddens(bf16_model, input_ids, len(prompt_ids))
            quantized_hiddens = get_reasoning_hiddens(quantized_model, input_ids, len(prompt_ids))
            # logit은 메모리 때문에 kl_batch_len token씩 나눠 계산
            for batch_start in range(0, len(reasoning_ids), config.kl_batch_len):
                bf16_logprobs = log_softmax(get_logits(bf16_model, bf16_hiddens[batch_start : batch_start + config.kl_batch_len]), dim=-1)
                quantized_logprobs = log_softmax(get_logits(quantized_model, quantized_hiddens[batch_start : batch_start + config.kl_batch_len]), dim=-1)
                tensors[f"{language}/kl"].append((bf16_logprobs.exp() * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
            tensors[f"{language}/problem_indices"].append(torch.full_like(reasoning_ids, problem_index, dtype=torch.int32))

            if (input_index + 1) % LOG_PROBLEMS == 0:
                print(f"{hub_path} {language} problem {input_index + 1}/{len(problem_inputs)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

        print(f"{hub_path} {language} kl {torch.cat(tensors[f'{language}/kl']).mean():.4f}")

    save_tensors(experiment_name, hub_path, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
