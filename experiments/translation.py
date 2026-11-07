from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.evaluation import evaluate_mgsm, split_outputs
from utils.format import format_duration
from utils.hub import download_model, load_result, result_exists, save_result
from utils.model import generate_vllm, load_vllm
from utils.paths import make_target_path

TRANSLATE_METHODS = ["bf16", "quantized"]
TRANSLATION_PROMPT = """Translate this math problem into English. Write only the translation, and do not solve it.

{question}"""


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # model to compare with bf16
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    # each stage loads one model, translate loads translate_method, solve loads solve_method
    stage: Literal["translate", "solve"] = "translate"
    # model that translates
    translate_method: Literal["bf16", "quantized"] = "bf16"
    # model that solves
    solve_method: Literal["bf16", "quantized"] = "bf16"


def main() -> None:
    config = parse(Config)
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    entry_name = "translation/mgsm"
    translate_file_names = {translate_method: f"stage=translate,translate_method={translate_method},max_generation_len={config.max_generation_len},seed={config.seed}" for translate_method in TRANSLATE_METHODS}
    solve_file_names = {translate_method: f"stage=solve,translate_method={translate_method},solve_method={config.solve_method},max_generation_len={config.max_generation_len},seed={config.seed}" for translate_method in TRANSLATE_METHODS}
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)

    if config.stage == "translate":
        file_name = translate_file_names[config.translate_method]
        if result_exists(entry_name, target, file_name):
            print(f"{target} already evaluated")
            return

        # 1) translate question with en translation instruction
        # drop instruction, blank line before question
        # e.g. "Solve this math problem. ...\n\nJames decides to run ..." -> "James decides to run ..."
        prompts = [TRANSLATION_PROMPT.format(question=prompt.split("\n\n", 1)[1]) for language in languages for prompt, _ in problems[language]]
        model_path = config.model_name if config.translate_method == "bf16" else download_model(target)
        model = load_vllm(model_path, config.model_name, config.seed)
        tokenizer = model.get_tokenizer()
        log_time = time.perf_counter()
        outputs = generate_vllm(model, config.model_name, prompts, config.max_generation_len)
        result = {}
        start = 0
        for language in languages:
            end = start + len(problems[language])
            result[language] = split_outputs(outputs[start:end], tokenizer, config.model_name)
            start = end
            print(f"{target} {config.translate_method} {language} translated {sum(response is not None for response in result[language]['responses'])}/{len(problems[language])}")
        print(f"{target} {config.translate_method} time {format_duration(time.perf_counter() - log_time)}")
        save_result(entry_name, target, file_name, result)
    else:
        missing_translate_methods = [translate_method for translate_method in TRANSLATE_METHODS if not result_exists(entry_name, target, solve_file_names[translate_method])]
        if not missing_translate_methods:
            print(f"{target} already evaluated")
            return
        translations = {translate_method: load_result(entry_name, target, translate_file_names[translate_method]) for translate_method in missing_translate_methods}
        model_path = config.model_name if config.solve_method == "bf16" else download_model(target)
        model = load_vllm(model_path, config.model_name, config.seed)
        tokenizer = model.get_tokenizer()

        for translate_method in missing_translate_methods:
            # 2) take translations of problems that reached response
            problem_indices = {}
            prompts = []
            for language in languages:
                problem_indices[language] = []
                for problem_index, response in enumerate(translations[translate_method][language]["responses"]):
                    if response is not None:
                        problem_indices[language].append(problem_index)
                        prompts.append(MGSM_PROMPTS["en"].format(question=response.strip()))

            # 3) solve translations with en prompt and score them
            log_time = time.perf_counter()
            outputs = generate_vllm(model, config.model_name, prompts, config.max_generation_len)
            result = {}
            start = 0
            for language in languages:
                end = start + len(problem_indices[language])
                answers = [problems[language][problem_index][1] for problem_index in problem_indices[language]]
                result[language] = {**evaluate_mgsm(outputs[start:end], answers, "en", tokenizer, config.model_name), "problem_indices": problem_indices[language]}
                start = end
                print(f"{target} {translate_method} {config.solve_method} {language} accuracy {result[language]['accuracy']:.2%} used {len(answers)}/{len(problems[language])}")
            print(f"{target} {translate_method} {config.solve_method} time {format_duration(time.perf_counter() - log_time)}")
            save_result(entry_name, target, solve_file_names[translate_method], result)


if __name__ == "__main__":
    main()
