from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore

from utils.model import GENERATION_CONFIGS


def load_vllm(
    model_path: str,
    model_name: str,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=40960, language_model_only=True)


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = [SamplingParams(**GENERATION_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=prompt_seed) for prompt_seed in prompt_seeds]
    outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]
