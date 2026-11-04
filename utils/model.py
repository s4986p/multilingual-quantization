import torch
from transformers import AutoModelForImageTextToText, PreTrainedModel
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

# recommended setting for thinking mode
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


def load_vllm(
    model_path: str,
    model_name: str,
    seed: int,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, seed=seed, language_model_only=True)


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    max_generation_len: int,
) -> list[CompletionOutput]:
    generation_config = GENERATION_CONFIGS[model_name]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = SamplingParams(temperature=generation_config["temperature"], top_p=generation_config["top_p"], top_k=generation_config["top_k"], presence_penalty=generation_config["presence_penalty"], max_tokens=max_generation_len)
    outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]
