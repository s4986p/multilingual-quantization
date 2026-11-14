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
