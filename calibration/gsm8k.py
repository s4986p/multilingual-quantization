from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse
import torch

from utils.dataset import MGSM_PROMPTS, load_gsm8k
from utils.evaluation import evaluate_mgsm
from utils.format import format_duration
from utils.hub import result_exists, save_tensors
from utils.model import GENERATION_CONFIGS, generate_vllm, get_prompt_ids, load_vllm
from utils.paths import make_bf16_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    model_name: str = "google/gemma-4-E2B-it"


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    target = make_bf16_path(config.model_name)
    if result_exists("gsm8k", target, file_name, "safetensors"):
        print(f"{target} already evaluated")
        return
    problems = load_gsm8k()

    model = load_vllm(config.model_name, config.model_name, config.seed)
    tokenizer = model.get_tokenizer()

    log_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, [prompt for language_problems in problems.values() for prompt, _ in language_problems], config.max_generation_len)
    num_problems = len(problems["en"])
    language_outputs = {language: outputs[language_index * num_problems : (language_index + 1) * num_problems] for language_index, language in enumerate(problems)}
    corrects = {}
    for language, language_problems in problems.items():
        result = evaluate_mgsm(language_outputs[language], [answer for _, answer in language_problems], language, tokenizer, config.model_name)
        print(f"{target} {language} accuracy {result['accuracy']:.2%} think len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
        corrects[language] = result["corrects"]
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")

    # keep prompt, reasoning token ids of each language, on problems solved by bf16 in en
    # reasoning keeps think start token, but drops think end token, response
    think_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[config.model_name]["think_end"])
    tensors = defaultdict(list)
    for problem_index in range(num_problems):
        if not corrects["en"][problem_index]:
            continue
        for language in MGSM_PROMPTS:
            prompt_ids = get_prompt_ids(tokenizer, problems[language][problem_index][0])
            token_ids = list(language_outputs[language][problem_index].token_ids)
            think_ids = torch.tensor(token_ids[: token_ids.index(think_end_id)] if think_end_id in token_ids else token_ids, dtype=torch.long)
            tensors[f"{language}/prompt_ids"].append(prompt_ids)
            tensors[f"{language}/prompt_lens"].append(torch.tensor([len(prompt_ids)]))
            tensors[f"{language}/think_ids"].append(think_ids)
            tensors[f"{language}/think_lens"].append(torch.tensor([len(think_ids)]))
    print(f"{target} problems {len(tensors['en/prompt_lens'])} of {num_problems}")
    save_tensors("gsm8k", target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
