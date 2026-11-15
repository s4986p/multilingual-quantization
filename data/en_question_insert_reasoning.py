from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, get_mgsm_question, load_gsm8k
from utils.evaluation import evaluate_outputs, parse_mgsm_prediction
from utils.format import format_duration
from utils.generation import continue_vllm, load_vllm
from utils.hub import save_tensors, tensors_exist
from utils.model import GENERATION_CONFIGS, get_prompt_ids, get_reasoning_ids
from utils.paths import make_hub_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 16384
    # reasoning을 쓸 모델
    model_name: str = "google/gemma-4-E2B-it"


def main() -> None:
    config = parse(Config)
    experiment_name = "en_question_insert_reasoning/gsm8k"
    hub_path = make_hub_path("bf16", config.model_name)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    if tensors_exist(experiment_name, hub_path, file_name):
        print(f"{hub_path} already generated")
        return
    # 영어는 영어 question을 넣을 필요가 없어 제외
    languages = LANGUAGES[1:]
    problems = load_gsm8k()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # 1. 해당 언어 prompt 뒤, reasoning 시작에 같은 문제의 영어 question 입력
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + 영어 question + "\n\n"
    reasoning_header = GENERATION_CONFIGS[config.model_name]["reasoning_header"]
    question_ids = [get_reasoning_ids(tokenizer, config.model_name, reasoning_header + get_mgsm_question(prompt) + "\n\n") for prompt, _ in problems["en"]]
    prompt_ids = {language: [get_prompt_ids(tokenizer, prompt) for prompt, _ in problems[language]] for language in languages}
    input_ids = [torch.cat([prompt_ids[language][problem_index], question_ids[problem_index]]) for language in languages for problem_index in range(len(question_ids))]

    # 2. vllm은 띄우는 데 오래 걸리기 때문에 모든 언어를 모아 영어 question에 이어서 한 번에 생성
    model = load_vllm(config.model_name, config.model_name)
    start_time = time.perf_counter()
    all_outputs = continue_vllm(model, config.model_name, input_ids, [config.seed] * len(input_ids), config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 3. 언어마다 채점하고 reasoning이 끝난 것만 prompt, 영어 question, reasoning, 정답 여부 저장
    # 예시) 이어서 생성한 "...<channel|>Jibu: 72" → "..."
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[config.model_name]["reasoning_end"])
    tensors = defaultdict(list)
    start = 0
    for language in languages:
        end = start + len(problems[language])
        outputs = all_outputs[start:end]
        result = evaluate_outputs(outputs, [answer for _, answer in problems[language]], language, tokenizer, config.model_name, parse_mgsm_prediction)
        start = end
        print(f"{hub_path} {language} accuracy {result['accuracy']:.2%} reasoning len {mean(result['reasoning_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
        for problem_index, (output, reasoning_len) in enumerate(zip(outputs, result["reasoning_lens"])):
            if reasoning_end_id not in output.token_ids:
                continue
            reasoning_ids = torch.tensor(output.token_ids[:reasoning_len], dtype=torch.long)
            tensors[f"{language}/prompt_ids"].append(prompt_ids[language][problem_index])
            tensors[f"{language}/prompt_lens"].append(torch.tensor([len(prompt_ids[language][problem_index])]))
            tensors[f"{language}/question_ids"].append(question_ids[problem_index])
            tensors[f"{language}/question_lens"].append(torch.tensor([len(question_ids[problem_index])]))
            tensors[f"{language}/reasoning_ids"].append(reasoning_ids)
            tensors[f"{language}/reasoning_lens"].append(torch.tensor([len(reasoning_ids)]))
            tensors[f"{language}/corrects"].append(torch.tensor([result["corrects"][problem_index]]))
        print(f"{hub_path} {language} problems {len(tensors[f'{language}/reasoning_lens'])} of {len(outputs)}")
    save_tensors(experiment_name, hub_path, file_name, {name: torch.cat(values) for name, values in tensors.items()})


if __name__ == "__main__":
    main()
