from collections import defaultdict

from datasets import load_dataset
import torch
from torch import Tensor
from transformers import AutoTokenizer

LANGUAGES = ["en", "zh", "ja", "sw", "te"]
# wikipedia streaming shuffle에 쓰는 row 수
SHUFFLE_BUFFER_SIZE = 10000
# mcot를 한 번에 읽는 row 수
MCOT_BATCH_SIZE = 100000
MGSM_PROMPTS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。\n\n{question}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.\n\n{question}""",
}


def load_wikipedia(
    model_name: str,
    languages: tuple[str, ...],
    num_chunks: int,
    chunk_len: int,
    shuffle_seed: int,
) -> list[Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    # chunk_len은 bos 포함, text_len은 bos를 뺀 길이 (bos는 gemma 1개, qwen 0개)
    text_len = chunk_len - len(bos_ids)
    chunks = []
    for language_index, language in enumerate(languages):
        # 나누어떨어지지 않으면 남는 chunk를 앞 언어부터 하나씩 더함
        num_language_chunks = num_chunks // len(languages) + (language_index < num_chunks % len(languages))
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True).shuffle(seed=shuffle_seed, buffer_size=SHUFFLE_BUFFER_SIZE)
        token_ids = []
        language_chunks = []
        for row in dataset:
            token_ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
            while len(token_ids) >= text_len and len(language_chunks) < num_language_chunks:
                language_chunks.append(torch.tensor(bos_ids + token_ids[:text_len]))
                token_ids = token_ids[text_len:]
            if len(language_chunks) == num_language_chunks:
                break
        chunks.extend(language_chunks)
    return chunks


def load_mgsm(
    languages: list[str],
) -> dict[str, list[tuple[str, int]]]:
    problems = {}
    for language in languages:
        dataset = load_dataset("juletxara/mgsm", language, split="test")
        problems[language] = [(MGSM_PROMPTS[language].format(question=row["question"]), row["answer_number"]) for row in dataset]
    return problems


def get_mgsm_question(
    prompt: str,
) -> str:
    # instruction과 question 사이 빈 줄로 나눠 question만 추출
    # 예시) instruction + "\n\n" + question → question
    return prompt.split("\n\n", 1)[1]


def load_mcot(
    languages: list[str],
) -> dict[str, dict[str, str]]:
    # 같은 문제는 언어마다 mcot id가 같기 때문에 question을 id로 묶음
    # 예시) "en-54530", "sw-54530" → questions["54530"] = {"en": ..., "sw": ...}
    mcot = load_dataset("laihuiyuan/mCoT-MATH", data_files="train_part_*.jsonl", split="train").select_columns(["question", "lang-id"])
    questions = defaultdict(dict)
    for batch in mcot.iter(batch_size=MCOT_BATCH_SIZE):
        for question, language_id in zip(batch["question"], batch["lang-id"]):
            language, mcot_id = language_id.split("-")
            if language in languages:
                questions[mcot_id][language] = question.strip()
    return questions


def load_gsm8k() -> dict[str, list[tuple[str, int]]]:
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    questions = load_mcot(LANGUAGES)

    # gsm8k train question마다 공백만 다른 mcot 영어 question을 찾아 각 언어 question 연결
    questions_by_en = {" ".join(language_questions["en"].split()): language_questions for language_questions in questions.values()}
    problems = {language: [] for language in LANGUAGES}
    for row in gsm8k:
        language_questions = questions_by_en[" ".join(row["question"].split())]
        # 예시) "...\n#### 72" → 72
        answer = int(row["answer"].split("####")[-1].replace(",", ""))
        for language in LANGUAGES:
            problems[language].append((MGSM_PROMPTS[language].format(question=language_questions[language]), answer))
    return problems
