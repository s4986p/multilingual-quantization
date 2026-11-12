import re

# 마지막 e 제거, 제목에 calculate, calculation처럼 품사가 다른 단어가 나오기 때문
SOLVING_WORDS = ["calculat", "determin", "comput", "solv", "perform", "set up", "formulat"]
STEP_TITLE = re.compile(r"^[ \t]*\d+\.\s+\*\*(.+?)\*\*", re.M)


def get_understanding(
    think: str,
) -> str | None:
    # 제목에 풀이 단어가 있는 첫 단계 앞까지 추출, 그런 단계가 없으면 None
    # 예시) "...\n\n3.  **Calculate Blue Fiber:** ..." → "...\n\n"
    for match in STEP_TITLE.finditer(think):
        if any(word in match.group(1).lower() for word in SOLVING_WORDS):
            return think[: match.start()]
    return None
