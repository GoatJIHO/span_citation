import re

YEAR_TOKEN = r'(?:19|20)\d{2}(?:[a-z])?(?:\s*(?:[–-]|,\s*)[a-z])?'
YEAR_HARD_BOUND = rf'(?<!\d){YEAR_TOKEN}(?!\d)'

BRACKET_NUM_CIT = re.compile(
    r'\[(?:\s*\d+\s*(?:[-–]\s*\d+\s*)?)(?:\s*[,;]\s*\d+\s*(?:[-–]\s*\d+\s*)?)*\s*\]'
)

TAG_FOLLOWS_AUTHOR_YEAR_PAREN = re.compile(
    rf'(#{0,2}CITATION_TAG)\s*\([^()]*{YEAR_HARD_BOUND}[^()]*\)'
)

PAREN_AUTHOR_YEAR_CIT = re.compile(
    rf'\([^()]*{YEAR_HARD_BOUND}[^()]*\)'
)

BROKEN_OPEN_PAREN_TO_YEAR_BEFORE_TAG = re.compile(
    rf'\([^()\n]*?{YEAR_HARD_BOUND}[^()\n]*?(?=#{0,2}CITATION_TAG)'
)

def unwrap_citation_tag(text: str) -> str:
    if text is None:
        return text

    tag = r'(?:#{0,2}CITATION_TAG)'

    text = re.sub(
        rf'[\(\[\{{]\s*({tag})\s*[\'"`’”]*\s*[.?!,:;]*\s*[\)\]\}}]?',
        r'\1',
        text
    )

    text = re.sub(
        rf'({tag})[\'"`’”]*[.?!,:;]+(?=\s|$|[\)\]\}}])',
        r'\1',
        text
    )

    text = re.sub(rf'({tag})\s*\d+\s*[\]\)]', r'\1', text)
    text = re.sub(rf'({tag})\s*\d+\b', r'\1', text)

    return text


def preserve_tag_inside_parentheses(text: str) -> str:
    if text is None:
        return text

    tag_pat = r'(?:#{0,2}CITATION_TAG)'

    def normalize_and_dedupe_tags(s: str) -> str:
        s = re.sub(tag_pat, '#CITATION_TAG', s)
        s = re.sub(r'(?:\s*#CITATION_TAG\s*){2,}', ' #CITATION_TAG ', s)
        return s

    def paren_repl(m: re.Match) -> str:
        inner = m.group(1)
        if not re.search(tag_pat, inner):
            return m.group(0)

        if re.search(YEAR_HARD_BOUND, inner):
            return " #CITATION_TAG "

        inner2 = normalize_and_dedupe_tags(inner)
        return f"({inner2})"

    def bracket_repl(m: re.Match) -> str:
        inner = m.group(1)
        if re.search(tag_pat, inner):
            return " #CITATION_TAG "
        return m.group(0)

    text = re.sub(r'\(([^()]*)\)', paren_repl, text)
    text = re.sub(r'\[([^\[\]]*)\]', bracket_repl, text)
    return text


def ensure_spaces_around_tag(text: str) -> str:
    TAG = r'(?:#CITATION_TAG)'
    if text is None:
        return text

    text = re.sub(rf'(\S)({TAG})', r'\1 \2', text)
    text = re.sub(rf'({TAG})(\S)', r'\1 \2', text)
    return text


def remove_citations_keep_tag(text: str) -> str:
    if text is None:
        return text

    text = preserve_tag_inside_parentheses(text)

    text = BRACKET_NUM_CIT.sub('', text)

    text = TAG_FOLLOWS_AUTHOR_YEAR_PAREN.sub(r'\1', text)

    text = BROKEN_OPEN_PAREN_TO_YEAR_BEFORE_TAG.sub('', text)

    text = PAREN_AUTHOR_YEAR_CIT.sub('', text)

    text = unwrap_citation_tag(text)

    text = re.sub(r'\s+', ' ', text).strip()
    text = ensure_spaces_around_tag(text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def pre_process(context):
    clean_context = remove_citations_keep_tag(context)
    return clean_context
