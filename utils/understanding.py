import re

# 제목에 calculate, calculation처럼 품사가 다른 단어가 나오기 때문에 마지막 e 제거
SOLVING_WORDS = ["calculat", "determin", "comput", "solv", "perform", "set up", "formulat"]
STEP_TITLE = re.compile(r"^[ \t]*\d+\.\s+\*\*(.+?)\*\*", re.M)


def get_understanding(
    reasoning: str,
) -> str | None:
    # 제목에 SOLVING_WORDS가 있는 첫 단계 앞까지를 understanding으로 추출, 그런 단계가 없으면 None
    # 예시) "...\n\n3.  **Calculate Blue Fiber:** ..." → "...\n\n"
    for match in STEP_TITLE.finditer(reasoning):
        if any(word in match.group(1).lower() for word in SOLVING_WORDS):
            return reasoning[: match.start()]
    return None


def get_understanding_problem_indices(
    bf16_result: dict,
    language: str,
) -> set[int]:
    # 해당 언어와 영어 bf16 reasoning 모두에 understanding이 있는 문제
    # 해당 언어 understanding 삽입과 영어 understanding 삽입을 같은 문제로 비교하는 데 사용
    return {problem_index for problem_index, (reasoning, en_reasoning) in enumerate(zip(bf16_result[language]["reasonings"], bf16_result["en"]["reasonings"])) if get_understanding(reasoning) is not None and get_understanding(en_reasoning) is not None}
