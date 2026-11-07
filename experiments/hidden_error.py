from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import MGSM_PROMPTS, load_mgsm
from utils.format import format_duration, format_memory
from utils.hub import download_model, result_exists, save_tensors
from utils.model import load_model
from utils.paths import make_target_path


@dataclass
class Config:
    # model to compare with bf16
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["rtn", "gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = "instruction=en"
    entry_name = "hidden_error/mgsm"
    target = make_target_path(config.method, config.model_name, config.languages, config.num_bits, config.group_size, config.num_samples)
    if result_exists(entry_name, target, file_name, "safetensors"):
        print(f"{target} already evaluated")
        return
    languages = list(MGSM_PROMPTS)
    problems = load_mgsm(languages)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(target))

    tensors = defaultdict(list)
    for language in languages:
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for problem_index, (prompt, _) in enumerate(problems[language]):
            # 1) put question after en instruction, and find instruction, question tokens
            question = prompt.split("\n\n", 1)[1]
            en_prompt = MGSM_PROMPTS["en"].format(question=question)
            text = tokenizer.apply_chat_template([{"role": "user", "content": en_prompt}], add_generation_prompt=True, enable_thinking=True, tokenize=False)
            encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
            instruction_start = text.index(en_prompt)
            question_start = instruction_start + len(en_prompt) - len(question)
            starts, ends = encoding["offset_mapping"][0].T
            # tokens are chat template, instruction, question, chat template
            # e.g. chat template  "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
            #      instruction    "Solve this math problem. ...\n\n"
            #      question       "James decides to run ..."
            #      chat template  "<turn|>\n<|turn>model\n"
            is_instruction = (starts >= instruction_start) & (ends <= question_start)
            is_question = (starts >= question_start) & (ends <= question_start + len(question))
            used = is_instruction | is_question

            # 2) compare bf16, quantized hiddens of each layer at each token
            input_ids = encoding["input_ids"].to(bf16_model.device)
            # hidden_states[0] is embedding, which is not quantized
            bf16_hiddens = torch.stack(bf16_model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[1:])[:, 0].float()
            quantized_hiddens = torch.stack(quantized_model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[1:])[:, 0].float()
            errors = ((quantized_hiddens - bf16_hiddens).norm(dim=-1) / bf16_hiddens.norm(dim=-1)).T.cpu()
            tensors[f"{language}/errors"].append(errors[used])
            tensors[f"{language}/is_question"].append(is_question[used])
            tensors[f"{language}/problem_index"].append(torch.full((int(used.sum()),), problem_index, dtype=torch.int32))

            if (problem_index + 1) % 50 == 0:
                print(f"{language} problem {problem_index + 1}/{len(problems[language])} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

        errors = torch.cat(tensors[f"{language}/errors"])
        is_question = torch.cat(tensors[f"{language}/is_question"])
        print(f"{target} {language} question error {errors[is_question].mean():.4f} instruction error {errors[~is_question].mean():.4f}")

    save_tensors(entry_name, target, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
