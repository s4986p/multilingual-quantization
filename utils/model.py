import torch
from torch import Tensor
from transformers import AutoModelForImageTextToText, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

# sampling settings recommended for thinking mode, with reasoning start and end tokens
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "think_start": "", "think_end": "</think>"},
    "Qwen/Qwen3.5-27B": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "think_start": "", "think_end": "</think>"},
    "google/gemma-4-E2B-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0, "think_start": "<|channel>", "think_end": "<channel|>"},
}

# saved gemma reasoning starts with this after reasoning start token
# e.g. "<|channel>thought\n...<channel|>"
THINK_HEADER = "thought\n"


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
    max_num_seqs: int | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, seed=seed, language_model_only=True, max_num_seqs=max_num_seqs)


def get_sampling_params(
    model_name: str,
    max_generation_len: int,
) -> SamplingParams:
    generation_config = GENERATION_CONFIGS[model_name]
    return SamplingParams(temperature=generation_config["temperature"], top_p=generation_config["top_p"], top_k=generation_config["top_k"], presence_penalty=generation_config["presence_penalty"], max_tokens=max_generation_len)


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = model.chat(messages, get_sampling_params(model_name, max_generation_len), use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]


def continue_vllm(
    model: LLM,
    model_name: str,
    input_ids: list[Tensor],
    max_generation_len: int,
) -> list[CompletionOutput]:
    prompts = [{"prompt_token_ids": ids.tolist()} for ids in input_ids]
    outputs = model.generate(prompts, get_sampling_params(model_name, max_generation_len), use_tqdm=True)
    return [output.outputs[0] for output in outputs]


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
    # put back reasoning start token, because saved reasoning has no special tokens
    return tokenizer(GENERATION_CONFIGS[model_name]["think_start"] + think, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def get_think_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # keep from last prompt token to second last reasoning token, because hidden at t predicts token t+1
    # e.g. input [p0, p1, p2, r0, r1, r2]
    #      keep  [        p2, r0, r1    ] (each predicts r0, r1, r2)
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]


def get_logits(
    model: PreTrainedModel,
    hiddens: Tensor,
) -> Tensor:
    logits = model.lm_head(hiddens).float()
    # cap logits when calling lm_head directly, because gemma forward caps them
    softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return logits if softcapping is None else torch.tanh(logits / softcapping) * softcapping
