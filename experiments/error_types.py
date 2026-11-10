from dataclasses import dataclass
import re
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import MGSM_PROMPTS, get_mgsm_question, load_mgsm
from utils.evaluation import split_outputs
from utils.format import format_duration
from utils.hub import load_result, result_exists, save_result
from utils.metrics import get_all_correct_mask
from utils.model import generate_vllm, load_vllm
from utils.paths import make_bf16_path, make_target_path

ERROR_TYPES = ["fact", "relation", "target", "doubt", "arithmetic", "other"]
JUDGE_PROMPT = """A small model solved a math word problem and got it wrong. Find the first line where its reasoning goes wrong, and say what kind of error it is.

Problem in English, which is the true meaning:
{en_question}

Problem as the model saw it:
{question}

Correct answer: {answer}

Reasoning of a stronger model that got it right:
<correct>
{bf16_think}
</correct>

Reasoning of the small model that got it wrong, with line numbers:
<wrong>
{numbered_think}
</wrong>

Error types:
- fact: states a number, unit, word or entity of the problem wrong (e.g. "runs 15 miles" becomes "runs 15 minutes")
- relation: states a relation between quantities wrong (e.g. "4 more cats than dogs" becomes "4 cats")
- target: solves for a different quantity than the one asked (e.g. problem asks how many are left, reasoning gives how many were used)
- doubt: calls a given condition missing, ambiguous or irrelevant, and drops it (e.g. "the price per box is not given", though the problem gives it)
- arithmetic: sets up the problem right, but computes wrong (e.g. "7 x 8 = 54")
- other: none of the above

End your answer with exactly these two lines (e.g. for a wrong relation on line 7):
line: 7
type: relation"""


@dataclass
class Config:
    seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # model whose wrong reasoning is judged
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # model that judges
    judge_name: str = "Qwen/Qwen3.5-27B"
    judge_max_generation_len: int = 16384


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    entry_name = "error_types/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)
    bf16_target = make_bf16_path(config.model_name)
    bf16_result = load_result("mgsm", bf16_target, file_name)
    bf16_results = [load_result("mgsm", bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]
    quantized_result = load_result("mgsm", target, file_name)

    # 1) take wrong reasoning on problems solved by all bf16 seeds
    problem_indices = {}
    prompts = []
    for language in languages:
        all_correct_mask = get_all_correct_mask([bf16_result[language] for bf16_result in bf16_results])
        problem_indices[language] = [problem_index for problem_index, (all_correct, correct) in enumerate(zip(all_correct_mask, quantized_result[language]["corrects"])) if all_correct and not correct]
        for problem_index in problem_indices[language]:
            prompt, answer = problems[language][problem_index]
            think = quantized_result[language]["thinks"][problem_index]
            prompts.append(JUDGE_PROMPT.format(
                en_question=get_mgsm_question(problems["en"][problem_index][0]),
                question=get_mgsm_question(prompt),
                answer=answer,
                bf16_think=bf16_result[language]["thinks"][problem_index],
                numbered_think="\n".join(f"{line_index + 1}: {line}" for line_index, line in enumerate(think.split("\n"))),
            ))
        print(f"{target} {language} wrong {len(problem_indices[language])}")

    # 2) read first wrong line and error type from end of judge response
    # 256 sequences, because qwen3.5-27b does not fit default 1024 sequences
    model = load_vllm(config.judge_name, config.judge_name, config.seed, max_num_seqs=256)
    tokenizer = model.get_tokenizer()
    log_time = time.perf_counter()
    outputs = generate_vllm(model, config.judge_name, prompts, config.judge_max_generation_len)
    result = {}
    start = 0
    for language in languages:
        end = start + len(problem_indices[language])
        split = split_outputs(outputs[start:end], tokenizer, config.judge_name)
        error_lines = []
        error_types = []
        for response in split["responses"]:
            lines = re.findall(r"line:\s*(\d+)", response or "")
            types = re.findall(rf"type:\s*({'|'.join(ERROR_TYPES)})\b", response or "")
            error_lines.append(int(lines[-1]) if lines else None)
            error_types.append(types[-1] if types else None)
        result[language] = {"problem_indices": problem_indices[language], "error_lines": error_lines, "error_types": error_types, **split}
        start = end
        print(f"{target} {language} judged {len(error_types)} unparsed {error_types.count(None)}")
    print(f"{target} time {format_duration(time.perf_counter() - log_time)}")
    save_result(entry_name, target, file_name, result)


if __name__ == "__main__":
    main()
