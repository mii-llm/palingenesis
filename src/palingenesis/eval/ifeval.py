"""IFEval instruction checks (Zhou et al. 2023, "Instruction-Following Evaluation for Large Language Models").

A port of the checks of the benchmark's reference implementation (google-research/instruction_following_eval,
Apache License 2.0), one function per instruction id, with the reference's regular expressions and comparisons.
When the reference implementation is installed (the `instruction_following_eval` package, or lm-eval's copy), it is
used instead (`checker: auto`); `checker: builtin` forces this port, `checker: official` requires the reference.

Where the port differs from the reference:

    sentences        the reference counts sentences with NLTK's punkt model; the port uses NLTK when it is installed
                     with its data, else a rule-based splitter (abbreviations, decimals, initials). Counts can differ
                     by one on unusual punctuation (length_constraints:number_sentences).
    capital words    the reference tokenises with nltk.word_tokenize; the port uses NLTK when available, else words
                     of letters and digits (change_case:capital_word_frequency).
    language         language:response_language, and the English test of change_case:english_capital /
                     english_lowercase, need the langdetect package (as the reference does). Without it, items with
                     language:response_language cannot be checked (follows() returns None and the item is left out
                     of the score), and the two case checks test the case only.

    follows(response, ["punctuation:no_comma", "length_constraints:number_words"],
            [{}, {"relation": "at least", "num_words": 300}])    ->    [True, False]
"""

from __future__ import annotations

import collections
import json
import re
from typing import Any, Callable

_LESS, _AT_LEAST = "less than", "at least"


def _compare(count: int, relation: str, target: int) -> bool:
    if relation == _LESS:
        return count < target
    if relation == _AT_LEAST:
        return count >= target
    raise ValueError(f"unknown relation {relation!r}")


# ------------------------------------------------------------------- utilities

_ALPHABETS = r"([A-Za-z])"
_PREFIXES = r"(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = r"(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Rule-based sentence splitting (the reference's instructions_util.split_into_sentences)."""
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    for closer in ("”", '"'):
        for mark in ".!?":
            text = text.replace(mark + closer, closer + mark)
    for mark in ".?!":
        text = text.replace(mark, mark + "<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def _nltk():
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        return nltk
    except Exception:  # noqa: BLE001 — not installed, or without its tokenizer data
        return None


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))  # the reference's RegexpTokenizer(r"\w+")


def count_sentences(text: str) -> int:
    nltk = _nltk()
    if nltk is not None:
        return len(nltk.tokenize.sent_tokenize(text))
    return len(split_into_sentences(text))


def _words(text: str) -> list[str]:
    nltk = _nltk()
    if nltk is not None:
        return nltk.word_tokenize(text)
    return re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", text)


def has_langdetect() -> bool:
    try:
        import langdetect  # noqa: F401

        return True
    except ImportError:
        return False


def _is_language(text: str, language: str) -> bool:
    """Whether langdetect reads `text` as `language`; True when it cannot tell (as the reference)."""
    import langdetect

    try:
        return langdetect.detect(text) == language
    except langdetect.LangDetectException:
        return True


# ---------------------------------------------------------------------- checks


def _existence(response: str, keywords: list, **_) -> bool:
    return all(re.search(keyword, response, flags=re.IGNORECASE) for keyword in keywords)


def _frequency(response: str, keyword: str, frequency: int, relation: str, **_) -> bool:
    return _compare(len(re.findall(keyword, response, flags=re.IGNORECASE)), relation, frequency)


def _forbidden_words(response: str, forbidden_words: list, **_) -> bool:
    return not any(re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE) for word in forbidden_words)


def _letter_frequency(response: str, letter: str, let_frequency: int, let_relation: str, **_) -> bool:
    return _compare(collections.Counter(response.lower())[letter.lower()], let_relation, let_frequency)


def _response_language(response: str, language: str, **_) -> bool:
    return _is_language(response, language)


def _number_sentences(response: str, num_sentences: int, relation: str, **_) -> bool:
    return _compare(count_sentences(response), relation, num_sentences)


def _number_paragraphs(response: str, num_paragraphs: int, **_) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


def _number_words(response: str, num_words: int, relation: str, **_) -> bool:
    return _compare(count_words(response), relation, num_words)


def _nth_paragraph_first_word(response: str, num_paragraphs: int, nth_paragraph: int, first_word: str, **_) -> bool:
    paragraphs = re.split(r"\n\n", response)
    count = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            count -= 1
    if nth_paragraph > count:
        return False
    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    found = ""
    for letter in word:
        if letter in (".", ",", "?", "!", "'", '"'):
            break
        found += letter.lower()
    return count == num_paragraphs and found == first_word.lower()


def _number_placeholders(response: str, num_placeholders: int, **_) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= num_placeholders


def _postscript(response: str, postscript_marker: str, **_) -> bool:
    text = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, text, flags=re.MULTILINE))


def _number_bullet_lists(response: str, num_bullets: int, **_) -> bool:
    stars = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(stars) + len(dashes) == num_bullets


def _constrained_response(response: str, **_) -> bool:
    return any(
        option in response.strip() for option in ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    )


def _number_highlighted_sections(response: str, num_highlights: int, **_) -> bool:
    count = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", response):
        if highlight.strip("*").strip():
            count += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if highlight.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def _multiple_sections(response: str, section_spliter: str, num_sections: int, **_) -> bool:
    sections = re.split(r"\s?" + section_spliter + r"\s?\d+\s?", response)
    return len(sections) - 1 >= num_sections


def _json_format(response: str, **_) -> bool:
    text = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _title(response: str, **_) -> bool:
    return any(title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", response))


def _two_responses(response: str, **_) -> bool:
    valid = []
    parts = response.split("******")
    for index, part in enumerate(parts):
        if not part.strip():
            if index != 0 and index != len(parts) - 1:
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(response: str, prompt_to_repeat: str, **_) -> bool:
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


def _end_checker(response: str, end_phrase: str, **_) -> bool:
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def _capital_word_frequency(response: str, capital_frequency: int, capital_relation: str, **_) -> bool:
    return _compare(sum(word.isupper() for word in _words(response)), capital_relation, capital_frequency)


def _english_capital(response: str, **_) -> bool:
    return response.isupper() and (not has_langdetect() or _is_language(response, "en"))


def _english_lowercase(response: str, **_) -> bool:
    return response.islower() and (not has_langdetect() or _is_language(response, "en"))


def _no_comma(response: str, **_) -> bool:
    return not re.search(r"\,", response)


def _quotation(response: str, **_) -> bool:
    text = response.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


CHECKS: dict[str, Callable[..., bool]] = {
    "keywords:existence": _existence,
    "keywords:frequency": _frequency,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:letter_frequency": _letter_frequency,
    "language:response_language": _response_language,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:number_words": _number_words,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "punctuation:no_comma": _no_comma,
    "startend:quotation": _quotation,
}
NEEDS_LANGDETECT = ("language:response_language",)


# ------------------------------------------------------------------ evaluation


def loose_variants(response: str) -> list[str]:
    """The reference's loose criterion: the response, without its first and / or last line, and without asterisks."""
    lines = response.split("\n")
    first, last, both = "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()
    plain = response.replace("*", "")
    return [response, plain, first, last, both, first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]


def official_registry() -> dict | None:
    """The reference implementation's {instruction id: checker class}, when it is installed."""
    for module in ("instruction_following_eval.instructions_registry", "lm_eval.tasks.ifeval.instructions_registry"):
        try:
            import importlib

            return importlib.import_module(module).INSTRUCTION_DICT
        except Exception:  # noqa: BLE001 — not installed, or its own dependencies (nltk data) are missing
            continue
    return None


def _official_check(registry: dict, instruction_id: str, arguments: dict) -> Callable[[str], bool]:
    instruction = registry[instruction_id](instruction_id)
    instruction.build_description(**arguments)
    return instruction.check_following


def follows(
    response: str,
    instruction_ids: list[str],
    kwargs: list[dict | None],
    loose: bool = False,
    checker: str = "auto",
) -> list[bool] | None:
    """Whether `response` follows each instruction. kwargs: one mapping per instruction (null values ignored, as
    the dataset stores them). None when an instruction cannot be checked in this environment (see the module
    docstring). An empty response follows nothing."""
    if checker not in ("auto", "builtin", "official"):
        raise ValueError(f"ifeval: checker must be auto, builtin or official, got {checker!r}")
    registry = official_registry() if checker != "builtin" else None
    if checker == "official" and registry is None:
        raise RuntimeError(
            "ifeval: checker official needs the reference implementation (pip install "
            "git+https://github.com/josejg/instruction_following_eval, or lm-eval)"
        )
    if len(instruction_ids) != len(kwargs):
        raise ValueError("ifeval: instruction_id_list and kwargs must have one entry per instruction")
    variants = [v for v in (loose_variants(response) if loose else [response]) if v.strip()]
    results = []
    for instruction_id, arguments in zip(instruction_ids, kwargs):
        arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
        if registry is not None:
            check: Any = _official_check(registry, instruction_id, arguments)
        else:
            if instruction_id not in CHECKS:
                raise ValueError(f"ifeval: unknown instruction {instruction_id!r}")
            if instruction_id in NEEDS_LANGDETECT and not has_langdetect():
                return None

            def check(text: str, fn=CHECKS[instruction_id], arguments=arguments) -> bool:
                return fn(text, **arguments)

        results.append(any(bool(check(variant)) for variant in variants))
    return results
