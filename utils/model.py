import torch
from torch import Tensor
from transformers import AutoModelForImageTextToText, PreTrainedModel, PreTrainedTokenizerBase

# think_start: 추론 시작 token
# think_end: 추론과 응답 나누는 token
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "think_start": "", "think_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "think_start": "<|channel>", "think_end": "<channel|>"},
}


def load_model(
    model_path: str,
) -> PreTrainedModel:
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model


def get_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tensor:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]


def get_think_ids(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    think: str,
) -> Tensor:
    # 추론 시작 token 다시 추가, 저장된 추론에는 special token이 없기 때문
    return tokenizer(GENERATION_CONFIGS[model_name]["think_start"] + think, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def get_think_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # 마지막 프롬프트 token부터 끝에서 두 번째 추론 token까지 유지, t의 hidden이 t+1 token을 예측하기 때문
    # 예시) 입력 [p0, p1, p2, r0, r1, r2]
    #      유지 [        p2, r0, r1    ] (각각 r0, r1, r2 예측)
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]


def get_logits(
    model: PreTrainedModel,
    hiddens: Tensor,
) -> Tensor:
    logits = model.lm_head(hiddens).float()
    # lm_head를 직접 부를 때 logit softcap 적용, gemma forward가 softcap하기 때문
    softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return logits if softcapping is None else torch.tanh(logits / softcapping) * softcapping
