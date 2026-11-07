from dataclasses import dataclass
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Literal

from simple_parsing import parse
import torch

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import generate_vllm, load_model, load_vllm
from utils.paths import make_target_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to swap layers with bf16
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # model to swap in for first num_swap_layers layers
    front_method: Literal["bf16", "quantized"] = "bf16"
    num_swap_layers: int = 3


def main() -> None:
    config = parse(Config)
    file_name = f"front_method={config.front_method},num_swap_layers={config.num_swap_layers},max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = "layer_swap/mgsm"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)

    # 1) put question after en instruction
    prompts = [MGSM_PROMPTS["en"].format(question=prompt.split("\n\n", 1)[1]) for language in languages for prompt, _ in problems[language]]

    # 2) swap first layers with front_method model, and load it with vllm
    model_paths = {"bf16": config.model_name, "quantized": download_model(target)}
    back_method = "quantized" if config.front_method == "bf16" else "bf16"
    model = load_model(model_paths[back_method])
    front_model = load_model(model_paths[config.front_method])
    model.model.language_model.layers[: config.num_swap_layers].load_state_dict(front_model.model.language_model.layers[: config.num_swap_layers].state_dict())
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        del model, front_model
        torch.cuda.empty_cache()
        vllm_model = load_vllm(directory, config.model_name, config.seed)
    tokenizer = vllm_model.get_tokenizer()

    # 3) solve and score with en answer prefix
    log_time = time.perf_counter()
    outputs = generate_vllm(vllm_model, config.model_name, prompts, config.max_generation_len)
    result = {}
    start = 0
    for language in languages:
        end = start + len(problems[language])
        result[language] = evaluate_mgsm(outputs[start:end], [answer for _, answer in problems[language]], "en", tokenizer, config.model_name)
        start = end
        print(f"{target} {config.front_method} front {language} accuracy {result[language]['accuracy']:.2%} think len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result(entry_name, target, file_name, result)


if __name__ == "__main__":
    main()
