import torch
from torch import Tensor
from transformers import AutoModelForImageTextToText, PreTrainedModel, PreTrainedTokenizerBase

# reasoning_start: reasoning 시작 token
# reasoning_end: reasoning과 response를 나누는 token
# qwen은 chat template이 prompt 끝에 reasoning_start token을 붙여서 비워 둠
# 예시) qwen 출력 "...\n</think>\n\nAnswer: 72"
# 예시) gemma 출력 "<|channel>thought\n...<channel|>Answer: 72"
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "reasoning_start": "<|channel>", "reasoning_end": "<channel|>"},
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


def get_reasoning_ids(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    reasoning: str,
) -> Tensor:
    # evaluation.py의 split_outputs가 skip_special_tokens로 decode해 저장된 reasoning에는 reasoning_start token이 없음
    # 따라서 reasoning_start token 다시 추가
    return tokenizer(GENERATION_CONFIGS[model_name]["reasoning_start"] + reasoning, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def get_reasoning_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # t의 hidden이 t+1 token을 예측하기 때문에 마지막 prompt token부터 끝에서 두 번째 reasoning token까지 유지
    # 예시) 입력 [p0, p1, p2, r0, r1, r2]
    # 예시) 유지 [        p2, r0, r1    ] (각각 r0, r1, r2 예측)
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]


def get_logits(
    model: PreTrainedModel,
    hiddens: Tensor,
) -> Tensor:
    logits = model.lm_head(hiddens).float()
    # lm_head만 직접 부르면 gemma forward의 logit softcap이 빠지기 때문에 여기서 적용 (qwen은 softcap 없음)
    softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return logits if softcapping is None else torch.tanh(logits / softcapping) * softcapping
