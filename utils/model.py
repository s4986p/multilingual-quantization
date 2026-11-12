import torch
from transformers import AutoModelForImageTextToText, PreTrainedModel

# thinking mode 권장 sampling 설정과 추론 끝 token
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "think_end": "</think>"},
    "google/gemma-4-E2B-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0, "think_end": "<channel|>"},
}


def load_model(
    model_path: str,
) -> PreTrainedModel:
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model
