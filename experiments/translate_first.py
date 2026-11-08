from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import generate_vllm, load_vllm
from utils.paths import make_target_path

TRANSLATE_FIRST_PROMPT = """Solve this math problem. First write the problem in English, then give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{question}"""


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to evaluate
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "rtn", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = "translate_first/mgsm"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)

    # 1) put question after en instruction that asks to write question in en first
    # drop instruction, blank line before question
    # e.g. "Solve this math problem. ...\n\nJames decides to run ..." -> "James decides to run ..."
    prompts = [TRANSLATE_FIRST_PROMPT.format(question=prompt.split("\n\n", 1)[1]) for language in languages for prompt, _ in problems[language]]
    model_path = config.model_name if config.method == "bf16" else download_model(target)
    model = load_vllm(model_path, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    # 2) solve and score with en answer prefix
    log_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, prompts, config.max_generation_len)
    result = {}
    start = 0
    for language in languages:
        end = start + len(problems[language])
        result[language] = evaluate_mgsm(outputs[start:end], [answer for _, answer in problems[language]], "en", tokenizer, config.model_name)
        start = end
        print(f"{target} {language} accuracy {result[language]['accuracy']:.2%} think len {mean(result[language]['think_lens']):.0f} truncated {mean(result[language]['truncateds']):.2%}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result(entry_name, target, file_name, result)


if __name__ == "__main__":
    main()
