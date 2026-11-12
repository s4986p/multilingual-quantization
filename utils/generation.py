from torch import Tensor
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

from utils.model import GENERATION_CONFIGS


def load_vllm(
    model_path: str,
    model_name: str,
    max_num_seqs: int | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, language_model_only=True, max_num_seqs=max_num_seqs)


def get_sampling_params(
    model_name: str,
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[SamplingParams]:
    return [SamplingParams(**GENERATION_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=prompt_seed) for prompt_seed in prompt_seeds]


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = model.chat(messages, get_sampling_params(model_name, prompt_seeds, max_generation_len), use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]


def continue_vllm(
    model: LLM,
    model_name: str,
    input_ids: list[Tensor],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    prompts = [{"prompt_token_ids": ids.tolist()} for ids in input_ids]
    outputs = model.generate(prompts, get_sampling_params(model_name, prompt_seeds, max_generation_len), use_tqdm=True)
    return [output.outputs[0] for output in outputs]
