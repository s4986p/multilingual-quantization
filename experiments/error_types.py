from dataclasses import dataclass
import re
import time
from typing import Literal

from simple_parsing import parse

from utils.config import Language
from utils.dataset import MGSM_PROMPTS, get_mgsm_question, load_mgsm
from utils.evaluation import split_outputs
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import load_result, result_exists, save_result
from utils.metrics import get_all_correct_mask
from utils.paths import make_target_path, make_uncalibrated_path

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
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 틀린 추론을 쓴 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # 판정하는 모델
    judge_name: str = "Qwen/Qwen3.5-27B"
    judge_max_generation_len: int = 16384


def main() -> None:
    config = parse(Config)
    script_name = "error_types/mgsm"
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    bf16_target = make_uncalibrated_path("bf16", config.model_name)
    file_names = {seed: f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds}

    # 1. seed마다 hub 결과에서 빠진 것 확인
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(script_name, target, file_name)]
    if not missing_seeds:
        print(f"{target} already evaluated")
        return
    problems = load_mgsm(list(MGSM_PROMPTS))
    bf16_results = [load_result("mgsm", bf16_target, f"max_generation_len={config.max_generation_len},seed={seed}") for seed in config.bf16_seeds]

    # 2. 모든 bf16 seed가 푼 문제에서 틀린 추론 추출
    problem_indices = {}
    prompts = []
    prompt_seeds = []
    for seed in missing_seeds:
        bf16_result = load_result("mgsm", bf16_target, file_names[seed])
        quantized_result = load_result("mgsm", target, file_names[seed])
        for language in MGSM_PROMPTS:
            all_correct_mask = get_all_correct_mask(bf16_results, language)
            problem_indices[seed, language] = [problem_index for problem_index, (all_correct, correct) in enumerate(zip(all_correct_mask, quantized_result[language]["corrects"])) if all_correct and not correct]
            for problem_index in problem_indices[seed, language]:
                prompt, answer = problems[language][problem_index]
                think = quantized_result[language]["thinks"][problem_index]
                prompts.append(JUDGE_PROMPT.format(
                    en_question=get_mgsm_question(problems["en"][problem_index][0]),
                    question=get_mgsm_question(prompt),
                    answer=answer,
                    bf16_think=bf16_result[language]["thinks"][problem_index],
                    numbered_think="\n".join(f"{line_index + 1}: {line}" for line_index, line in enumerate(think.split("\n"))),
                ))
                prompt_seeds.append(seed)
            print(f"{target} seed {seed} {language} wrong {len(problem_indices[seed, language])} of {len(problems[language])}")

    # 3. 판정을 한 번에 생성, vllm을 띄우는 데 오래 걸리기 때문
    # sequence 256개, qwen3.5-27b는 기본 1024개로는 메모리에 안 들어가기 때문
    model = load_vllm(config.judge_name, config.judge_name, max_num_seqs=256)
    tokenizer = model.get_tokenizer()
    start_time = time.perf_counter()
    outputs = generate_vllm(model, config.judge_name, prompts, prompt_seeds, config.judge_max_generation_len)
    print(f"{target} time {format_duration(time.perf_counter() - start_time)}")

    # 4. 판정 끝에서 처음 틀린 줄과 오류 유형 추출하고 저장
    start = 0
    for seed in missing_seeds:
        result = {}
        for language in MGSM_PROMPTS:
            end = start + len(problem_indices[seed, language])
            split = split_outputs(outputs[start:end], tokenizer, config.judge_name)
            error_lines = []
            error_types = []
            for response in split["responses"]:
                lines = re.findall(r"line:\s*(\d+)", response or "")
                types = re.findall(rf"type:\s*({'|'.join(ERROR_TYPES)})\b", response or "")
                error_lines.append(int(lines[-1]) if lines else None)
                error_types.append(types[-1] if types else None)
            result[language] = {"problem_indices": problem_indices[seed, language], "error_lines": error_lines, "error_types": error_types, **split}
            start = end
            print(f"{target} seed {seed} {language} judged {len(error_types)} unparsed {error_types.count(None)}")
        save_result(script_name, target, file_names[seed], result)


if __name__ == "__main__":
    main()
