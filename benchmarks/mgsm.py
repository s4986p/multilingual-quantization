from dataclasses import dataclass
import re
from typing import Literal

from datasets import load_dataset
from simple_parsing import parse

from utils.evaluation import evaluate_benchmark
from utils.paths import make_hub_path

MGSM_PROMPTS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。\n\n{question}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.\n\n{question}""",
}
MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def load_mgsm(
    languages: list[str],
) -> dict[str, list[tuple[str, int]]]:
    problems = {}
    for language in languages:
        dataset = load_dataset("juletxara/mgsm", language, split="test")
        problems[language] = [(MGSM_PROMPTS[language].format(question=row["question"]), row["answer_number"]) for row in dataset]
    return problems


def parse_mgsm_prediction(
    response: str | None,
    language: str,
) -> float | None:
    if response is None:
        return None
    # 마지막 answer prefix 뒤 text 추출, prefix가 없으면 response 전체 사용
    # 예시) "5+3=8，所以答案：8" → "8"
    # 예시) "Answer :7" → "7"
    # 예시) "Answer - 7" → "Answer - 7"
    # 예시) "Answer: 5. Answer: 8" → " 8"
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    # text의 마지막 숫자 추출, 숫자가 없으면 response 전체에서 추출
    # 예시) "so 366.\n答え:" → 366.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    evaluate_benchmark("mgsm", load_mgsm, parse_mgsm_prediction, hub_path, config.method, config.model_name, config.seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
