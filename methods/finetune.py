from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.dataset import LANGUAGES, MGSM_PROMPTS, get_mgsm_question, load_mcot, load_mgsm
from utils.hub import load_tensors, save_model
from utils.model import GENERATION_CONFIGS, get_logits, get_prompt_ids, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_finetune_path, make_hub_path
from utils.training import MAX_REASONING_LEN, Sample, generate_reasonings, load_gsm8k_reasonings, train
from utils.understanding import get_understanding

TRAIN_LANGUAGES = [language for language in LANGUAGES if language != "en"]


@dataclass
class Config:
    lr: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    # ce_kl: en_question에서 영어 question은 ce, 뒤 reasoning은 kl
    loss: Literal["ce", "kl", "hidden", "ce_kl"] = "ce"
    model_name: str = "google/gemma-4-E2B-it"
    # 학습 데이터
    # bf16_en: bf16이 영어 prompt에 쓴 reasoning
    # on_policy: 학습 중인 모델이 해당 언어 prompt에 생성한 reasoning
    # bf16_en_question_insert: bf16이 해당 언어 prompt + 영어 question 삽입 뒤에 쓴 reasoning
    reasoning_source: Literal["bf16_en", "on_policy", "bf16_en_question_insert"] = "bf16_en"
    # understanding, reasoning: reasoning_source 중 학습할 부분
    # en_question: 영어 question, ce는 reasoning_source를 안 쓰고 ce_kl은 영어 question 뒤에 reasoning_source를 씀
    train_part: Literal["understanding", "reasoning", "en_question"] = "understanding"
    num_samples: int = 4000
    # en_question에서 checkpoint를 저장할 학습한 sample 수, 마지막은 항상 저장
    save_num_samples: tuple[int, ...] = ()
    seed: int = 0
    max_generation_len: int = 16384


def load_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    train_part: str,
) -> list[Sample]:
    # 문제와 언어마다 (해당 언어 prompt, 영어 prompt, bf16이 영어 prompt에 쓴 text)
    # bf16이 해당 언어로 맞혔고 영어와 해당 언어 reasoning 모두에 understanding이 있는 문제만 사용
    # on_policy는 학습 중에 text를 생성한 것으로 바꿈
    reasonings = load_gsm8k_reasonings(tokenizer, model_name, seed, max_generation_len)
    samples = []
    for problem_index, en_reasoning in enumerate(reasonings["en"]["reasonings"]):
        en_understanding = get_understanding(en_reasoning)
        if en_understanding is None:
            continue
        for language in TRAIN_LANGUAGES:
            if get_understanding(reasonings[language]["reasonings"][problem_index]) is None or not reasonings[language]["corrects"][problem_index]:
                continue
            text_ids = get_reasoning_ids(tokenizer, model_name, en_understanding if train_part == "understanding" else en_reasoning)
            samples.append((reasonings[language]["prompt_ids"][problem_index], reasonings["en"]["prompt_ids"][problem_index], text_ids))
    print(f"samples {len(samples)} of {len(reasonings['en']['reasonings']) * len(TRAIN_LANGUAGES)}")
    return samples


def load_en_question_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    num_samples: int,
) -> list[Sample]:
    # mcot에서 모든 언어 question이 있고 mgsm과 겹치지 않는 문제를 뽑아 언어마다 (해당 언어 prompt, 영어 prompt, 영어 question)
    # 학습할 text가 영어 원문이라 bf16 정답 여부로 거르지 않음
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + 영어 question + "\n\n"
    mgsm_en_questions = {" ".join(get_mgsm_question(prompt).split()) for prompt, _ in load_mgsm(["en"])["en"]}
    problems = [language_questions for language_questions in load_mcot(LANGUAGES).values() if len(language_questions) == len(LANGUAGES) and " ".join(language_questions["en"].split()) not in mgsm_en_questions]
    reasoning_header = GENERATION_CONFIGS[model_name]["reasoning_header"]
    samples = []
    for language_questions in random.sample(problems, num_samples // len(TRAIN_LANGUAGES)):
        en_prompt_ids = get_prompt_ids(tokenizer, MGSM_PROMPTS["en"].format(question=language_questions["en"]))
        text_ids = get_reasoning_ids(tokenizer, model_name, reasoning_header + language_questions["en"] + "\n\n")
        for language in TRAIN_LANGUAGES:
            samples.append((get_prompt_ids(tokenizer, MGSM_PROMPTS[language].format(question=language_questions[language])), en_prompt_ids, text_ids))
    print(f"samples {len(samples)} from {len(problems)} problems")
    return samples


def load_en_question_reasoning_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    seed: int,
    max_generation_len: int,
    num_samples: int,
) -> list[Sample]:
    # gsm8k 문제와 언어마다 (해당 언어 prompt, 영어 question, bf16이 영어 question 삽입 뒤에 쓴 reasoning)
    # kl은 정답 여부와 상관없이 bf16 분포를 맞추기 때문에 정답 여부로 거르지 않고, reasoning_end를 붙여 MAX_REASONING_LEN보다 길면 제외
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + 영어 question + "\n\n" + reasoning + "<channel|>"
    tensors = load_tensors("en_question_insert_reasoning/gsm8k", make_hub_path("bf16", model_name), f"max_generation_len={max_generation_len},seed={seed}")
    reasoning_end_ids = torch.tensor([tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])])
    samples = []
    for language in TRAIN_LANGUAGES:
        parts = [tensors[f"{language}/{part}_ids"].split(tensors[f"{language}/{part}_lens"].tolist()) for part in ("prompt", "question", "reasoning")]
        for prompt_ids, question_ids, reasoning_ids in zip(*parts):
            if len(reasoning_ids) < MAX_REASONING_LEN:
                samples.append((prompt_ids, question_ids, torch.cat([reasoning_ids, reasoning_end_ids])))
    print(f"samples {num_samples} of {len(samples)}")
    return random.sample(samples, num_samples)


def get_ce_losses(
    model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 해당 언어 prompt 뒤에 text를 넣고 text 위치에서만 다음 token 예측을 ce로 학습
    # 예시) 스와힐리어 prompt + "<|channel>thought\n" + understanding
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, _, text_ids in batch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
        yield cross_entropy(logits, text_ids.to(logits.device), reduction="sum") / num_positions


def get_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 같은 text를 학습 중인 모델에는 해당 언어 prompt 뒤에, bf16에는 영어 prompt 뒤에 넣음
    # text 위치에서만 다음 token 예측을 bf16에 kl로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, en_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_reasoning_hiddens(bf16_model, torch.cat([en_prompt_ids, text_ids]), len(en_prompt_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = log_softmax(get_logits(model, get_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids))), dim=-1)
        yield (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_positions


def get_ce_kl_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # 해당 언어 prompt 뒤에 영어 question과 reasoning을 넣고 영어 question 위치는 ce로 학습
    # reasoning 위치는 같은 입력을 넣은 bf16에 kl로 맞춤
    # reasoning이 영어 question보다 훨씬 길어 두 loss를 각자의 위치 수로 나눔
    num_question_positions = sum(len(question_ids) for _, question_ids, _ in batch)
    num_reasoning_positions = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
    for prompt_ids, question_ids, reasoning_ids in batch:
        input_ids = torch.cat([prompt_ids, question_ids, reasoning_ids])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logprobs = log_softmax(get_logits(bf16_model, get_reasoning_hiddens(bf16_model, input_ids, len(prompt_ids) + len(question_ids))), dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = get_logits(model, get_reasoning_hiddens(model, input_ids, len(prompt_ids)))
        ce_loss = cross_entropy(logits[: len(question_ids)], question_ids.to(logits.device), reduction="sum") / num_question_positions
        logprobs = log_softmax(logits[len(question_ids) :], dim=-1)
        kl_loss = (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum() / num_reasoning_positions
        yield ce_loss + kl_loss


def get_layer_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # layer마다 text 위치 hidden, embedding은 두 입력에서 같아 제외
    hidden_states = model.model(input_ids=input_ids[None, :].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack(hidden_states[1:])[:, 0, prompt_len:]


def get_hidden_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    batch: list[Sample],
) -> Iterator[Tensor]:
    # kl과 같이 넣고 text 위치에서 layer마다 hidden을 bf16에 cosine 유사도로 맞춤
    num_positions = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, en_prompt_ids, text_ids in batch:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_hiddens = get_layer_hiddens(bf16_model, torch.cat([en_prompt_ids, text_ids]), len(en_prompt_ids))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hiddens = get_layer_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids))
        yield (1 - cosine_similarity(hiddens.float(), bf16_hiddens.float(), dim=-1)).mean(0).sum() / num_positions


def get_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
    batch: list[Sample],
) -> Iterator[Tensor]:
    if reasoning_source == "on_policy":
        batch = generate_reasonings(model, tokenizer, model_name, train_part, batch)
    if loss == "ce":
        return get_ce_losses(model, batch)
    if loss == "kl":
        return get_kl_losses(model, bf16_model, batch)
    if loss == "ce_kl":
        return get_ce_kl_losses(model, bf16_model, batch)
    return get_hidden_losses(model, bf16_model, batch)


def save_checkpoint(
    model: PreTrainedModel,
    weights: list[Tensor],
    config: Config,
    step: int,
) -> None:
    # 학습 중인 weight는 fp32이기 때문에 bf16으로 저장한 뒤 fp32 값 복원
    num_trained_samples = min(step * config.batch_size, config.num_epochs * config.num_samples)
    fp32_weights = [weight.detach().clone() for weight in weights]
    model.to(torch.bfloat16)
    save_model(make_finetune_path("finetune", config.model_name, config.loss, config.reasoning_source, config.train_part, config.num_samples, num_trained_samples), model)
    model.float()
    with torch.no_grad():
        for weight, fp32_weight in zip(weights, fp32_weights):
            weight.copy_(fp32_weight)


def main() -> None:
    config = parse(Config)
    if config.loss == "ce_kl" and (config.train_part != "en_question" or config.reasoning_source != "bf16_en_question_insert"):
        raise ValueError("ce_kl needs train_part en_question and reasoning_source bf16_en_question_insert")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    # 학습 중 모델이 바뀌기 때문에 kl, hidden, ce_kl에는 고정된 bf16 따로 사용
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)
    if config.loss == "ce_kl":
        samples = load_en_question_reasoning_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.num_samples)
    elif config.train_part == "en_question":
        samples = load_en_question_samples(tokenizer, config.model_name, config.num_samples)
    else:
        samples = load_samples(tokenizer, config.model_name, config.seed, config.max_generation_len, config.train_part)[: config.num_samples]

    # 양자화 대상과 같게 linear weight만 학습
    linears = [module for module in model.model.language_model.layers.modules() if isinstance(module, Linear)]
    weights = [linear.weight for linear in linears]
    get_batch_losses = partial(get_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.reasoning_source, config.train_part)
    if config.train_part == "en_question":
        # train은 step 단위라 save_num_samples를 step으로 바꿔 전달
        if any(num_trained_samples % config.batch_size for num_trained_samples in config.save_num_samples):
            raise ValueError(f"save_num_samples must be multiples of batch_size {config.batch_size}")
        save_steps = tuple(num_trained_samples // config.batch_size for num_trained_samples in config.save_num_samples)
        train(weights, samples, get_batch_losses, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size, save_steps, partial(save_checkpoint, model, weights, config))
        return
    train(weights, samples, get_batch_losses, config.lr, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(make_finetune_path("finetune", config.model_name, config.loss, config.reasoning_source, config.train_part), model)


if __name__ == "__main__":
    main()
