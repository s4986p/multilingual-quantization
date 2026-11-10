import re

SOLVING_WORDS = ["calculat", "determin", "comput", "solv", "perform", "set up", "formulat"]
STEP_TITLE = re.compile(r"^[ \t]*\d+\.\s+\*\*(.+?)\*\*", re.M)


def get_understanding(
    think: str,
) -> str | None:
    # cut before first step whose title has a solving word, or none if no such step
    # e.g. "... 2.  **Deconstruct the Recipe:** ...\n\n3.  **Calculate Blue Fiber:** ..." -> "... 2.  **Deconstruct the Recipe:** ...\n\n"
    for match in STEP_TITLE.finditer(think):
        if any(word in match.group(1).lower() for word in SOLVING_WORDS):
            return think[: match.start()]
    return None
