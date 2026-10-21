MGSM_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
}
MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "sw": "Jibu"}

MMLU_PROX_LITE_TEMPLATES = {
    "en": ("Question:", "Options:", 'The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "{answer_format}" where X is the correct letter choice.', "the answer is ({})"),
    "zh": ("问题：", "选项：", '以下是关于{subject}的选择题（带有答案）。请逐步思考，然后以"{answer_format}"结束您的回答，其中X是正确的选项字母。', "答案是 ({})"),
    "sw": ("Swali:", "Chaguo:", 'Yafuatayo ni maswali ya chaguo-nyingi (yenye majibu) kuhusu {subject}. Fikiria hatua kwa hatua kisha malizia jibu lako kwa "{answer_format}" ambapo X ni herufi ya chaguo sahihi.', "Jibu ni ({})"),
}
MMLU_PROX_LITE_SUBJECTS = {
    "en": {"biology": "biology", "business": "business", "chemistry": "chemistry", "computer science": "computer_science", "economics": "economics", "engineering": "engineering", "health": "health", "history": "history", "law": "law", "math": "math", "other": "other", "philosophy": "philosophy", "physics": "physics", "psychology": "psychology"},
    "zh": {"biology": "生物学", "business": "商业", "chemistry": "化学", "computer science": "计算机科学", "economics": "经济学", "engineering": "工程学", "health": "健康", "history": "历史", "law": "法律", "math": "数学", "other": "其他", "philosophy": "哲学", "physics": "物理学", "psychology": "心理学"},
    "sw": {"biology": "biolojia", "business": "biashara", "chemistry": "kemia", "computer science": "sayansi ya kompyuta", "economics": "uchumi", "engineering": "uhandisi", "health": "afya", "history": "historia", "law": "sheria", "math": "hisabati", "other": "nyingine", "philosophy": "falsafa", "physics": "fizikia", "psychology": "saikolojia"},
}
