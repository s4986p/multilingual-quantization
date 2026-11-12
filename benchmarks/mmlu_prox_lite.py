from dataclasses import dataclass
import re
from typing import Literal

from datasets import load_dataset
from simple_parsing import parse
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.evaluation import evaluate_benchmark, split_outputs
from utils.paths import make_target_path

MMLU_PROX_LITE_TEMPLATES = {
    "en": ("Question:", "Options:", 'The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "{answer_format}" where X is the correct letter choice.', "the answer is ({})"),
    "zh": ("问题：", "选项：", '以下是关于{subject}的选择题（带有答案）。请逐步思考，然后以"{answer_format}"结束您的回答，其中X是正确的选项字母。', "答案是 ({})"),
    "sw": ("Swali:", "Chaguo:", 'Yafuatayo ni maswali ya chaguo-nyingi (yenye majibu) kuhusu {subject}. Fikiria hatua kwa hatua kisha malizia jibu lako kwa "{answer_format}" ambapo X ni herufi ya chaguo sahihi.', "Jibu ni ({})"),
    "ja": ("質問：", "選択肢：", "以下は{subject}に関する選択問題（解答付き）です。段階的に考え、最後に「{answer_format}」と回答を締めくくってください。Xは正解の選択肢を示す文字です。", "答えは ({}) です"),
    "te": ("ప్రశ్న:", "ఎంపికలు:", 'క్రింది {subject}కి సంబంధించిన బహుళఎంపిక ప్రశ్న (సమాధానాలతో). దయచేసి దశలవారీగా ఆలోచించి, మీ సమాధానాన్ని "{answer_format}"తో ముగించండి, ఇక్కడ X సరైన ఎంపిక అక్షరం.', "సమాధానం ({})"),
}
MMLU_PROX_LITE_SUBJECTS = {
    "en": {"biology": "biology", "business": "business", "chemistry": "chemistry", "computer science": "computer_science", "economics": "economics", "engineering": "engineering", "health": "health", "history": "history", "law": "law", "math": "math", "other": "other", "philosophy": "philosophy", "physics": "physics", "psychology": "psychology"},
    "zh": {"biology": "生物学", "business": "商业", "chemistry": "化学", "computer science": "计算机科学", "economics": "经济学", "engineering": "工程学", "health": "健康", "history": "历史", "law": "法律", "math": "数学", "other": "其他", "philosophy": "哲学", "physics": "物理学", "psychology": "心理学"},
    "sw": {"biology": "biolojia", "business": "biashara", "chemistry": "kemia", "computer science": "sayansi ya kompyuta", "economics": "uchumi", "engineering": "uhandisi", "health": "afya", "history": "historia", "law": "sheria", "math": "hisabati", "other": "nyingine", "philosophy": "falsafa", "physics": "fizikia", "psychology": "saikolojia"},
    "ja": {"biology": "生物学", "business": "ビジネス", "chemistry": "化学", "computer science": "コンピュータサイエンス", "economics": "経済学", "engineering": "工学", "health": "健康科学", "history": "歴史", "law": "法律", "math": "数学", "other": "その他", "philosophy": "哲学", "physics": "物理学", "psychology": "心理学"},
    "te": {"biology": "జీవశాస్త్రం", "business": "వ్యాపారం", "chemistry": "రసాయన శాస్త్రం", "computer science": "కంప్యూటర్ సైన్స్", "economics": "ఆర్థిక శాస్త్రం", "engineering": "ఇంజనీరింగ్", "health": "ఆరోగ్యం", "history": "చరిత్ర", "law": "న్యాయశాస్త్రం", "math": "గణితం", "other": "ఇతరమైన", "philosophy": "తత్వవేత్త", "physics": "భౌతిక శాస్త్రం", "psychology": "మనోవిజ్ఞానశాస్త్రం"},
}


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    max_generation_len: int = 16384
    # 평가할 모델
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "sw", "ja", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128


def load_mmlu_prox_lite(
    languages: list[str],
) -> dict[str, list[tuple[str, str]]]:
    problems = {}
    for language in languages:
        question_label, options_label, instruction, answer_format = MMLU_PROX_LITE_TEMPLATES[language]
        dataset = load_dataset("li-lab/MMLU-ProX-Lite", language, split="test")
        problems[language] = []
        for row in dataset:
            options = [row[f"option_{index}"] for index in range(10) if row[f"option_{index}"] is not None]
            prompt = instruction.format(subject=MMLU_PROX_LITE_SUBJECTS[language][row["category"]], answer_format=answer_format.format("X"))
            prompt += f"\n\n{question_label}\n{row['question']}\n{options_label}\n"
            prompt += "".join(f"{chr(ord('A') + index)}. {option}\n" for index, option in enumerate(options))
            problems[language].append((prompt, row["answer"]))
    return problems


def parse_mmlu_prox_lite_prediction(
    response: str | None,
    language: str,
) -> str | None:
    if response is None:
        return None
    _, _, _, answer_format = MMLU_PROX_LITE_TEMPLATES[language]
    # "("부터 끝까지 제거
    # 예시) "the answer is ({})" → "the answer is"
    #      "答案是 ({})"          → "答案是"
    #      "答えは ({}) です"      → "答えは"
    answer_prefix = answer_format.split("(")[0].strip()
    # markdown 굵은 글씨 제거
    # 예시) "Jibu ni **E**" → "Jibu ni E"
    response = response.replace("*", "")
    # prefix 뒤 A~J 대문자 한 글자 추출
    # 예시) "The answer is C"   → ["C"]
    #      "The answer is K"   → []
    #      "The answer is c"   → []
    #      "The answer is Cat" → []
    letters = re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
    # 못 찾으면 응답 전체에서 따로 떨어진 A~J 대문자 추출
    # 예시) "Jibu: A"      → ["A"]
    #      "I chose (D)." → ["I", "D"]
    letters = letters or re.findall(r"\b([A-J])\b", response)
    return letters[-1] if letters else None


def evaluate_mmlu_prox_lite(
    outputs: list[CompletionOutput],
    answers: list[str],
    language: str,
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    predictions = [parse_mmlu_prox_lite_prediction(response, language) for response in split["responses"]]
    corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": predictions,
        **split,
    }


def main() -> None:
    config = parse(Config)
    target = make_target_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_samples)
    evaluate_benchmark("mmlu_prox_lite", list(MMLU_PROX_LITE_TEMPLATES), load_mmlu_prox_lite, evaluate_mmlu_prox_lite, target, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
