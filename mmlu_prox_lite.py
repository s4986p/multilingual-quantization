from dataclasses import dataclass, field
from statistics import mean
import time

from simple_parsing import parse

from primitives.dataset import load_mmlu_prox_lite
from primitives.evaluation import evaluate_mmlu_prox_lite
from primitives.hub import download_model, get_pending_targets, save_result
from primitives.model import GENERATION_CONFIGS, generate_vllm, load_vllm
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3.5-2B"
    languages: list[str] = field(default_factory=lambda: ["en", "zh", "sw"])
    max_generation_len: int = 16384
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = get_pending_targets("mmlu_prox_lite", config.model_name, file_name)
    samples = {language: load_mmlu_prox_lite(language) for language in config.languages}

    for target in targets:
        method = target.split("/")[0]
        if method == "bf16":
            model_path = config.model_name
        else:
            model_path = download_model(target)
        model, tokenizer = load_vllm(model_path, config.model_name)
        think_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[config.model_name]["think_end"])

        log_time = time.perf_counter()
        prompts = [prompt for language in config.languages for prompt, _ in samples[language]]
        outputs = generate_vllm(model, config.model_name, prompts, config.max_generation_len, config.seed)
        result = {}
        start = 0
        for language in config.languages:
            end = start + len(samples[language])
            result[language] = evaluate_mmlu_prox_lite(outputs[start:end], [answer for _, answer in samples[language]], language, tokenizer, think_end_id)
            start = end
            print(f"{target} {language} accuracy {result[language]['accuracy']:.2%} think_len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
        print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
        save_result(f"mmlu_prox_lite/{target}", file_name, result)
        model.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
