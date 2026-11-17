import torch
from transformers import AutoModelForImageTextToText, PreTrainedModel

# reasoning_start: reasoning 시작 token
# reasoning_end: reasoning과 response를 나누는 token
# qwen은 chat template이 prompt 끝에 reasoning_start token을 붙여서 비워 둠
# 예시) qwen 출력 "...\n</think>\n\nAnswer: 72"
# 예시) gemma 출력 "<|channel>thought\n...<channel|>Answer: 72"
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "Qwen/Qwen3.5-27B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "reasoning_start": "<|channel>", "reasoning_end": "<channel|>"},
}


def load_model(
    model_path: str,
) -> PreTrainedModel:
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model
