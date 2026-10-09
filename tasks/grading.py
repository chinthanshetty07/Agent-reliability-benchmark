"""
Correctness checking for numeric benchmark tasks.

The idea in plain words:
1. Find the part of the model's reply that is its FINAL answer
   (a "Final answer:" line, or a \\boxed{...} value, or as a last resort
   the last line that contains a number).
2. Pull the number out of that part, understanding "$3,000", "-1,200",
   "(1,200)" and phrases like "a loss of $1,200" (which means -1200).
3. Compare that number with the task's expected_answer.

We only look at the final answer part, so numbers that appear in the
question or in intermediate working (like "8,000 - 9,200") are ignored.
"""

import re
from decimal import Decimal, InvalidOperation

# Matches numbers such as 3000, 3,000, 3,000.50, 1200.5
# An optional minus sign and/or $ sign may come before the digits.
NUMBER_RE = re.compile(
    r"(?P<paren>\()?"                 # optional "(" for accounting-style negatives
    r"(?P<minus1>[-−])?\s*"           # optional minus before the $ sign
    r"\$?\s*"                         # optional dollar sign
    r"(?P<minus2>[-−])?"              # optional minus after the $ sign
    r"(?<![\w.,])"                    # don't start in the middle of a word or number
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<close>\))?"
)

# "Final answer:" with optional markdown bold, e.g. "**Final Answer:**"
FINAL_ANSWER_RE = re.compile(r"\**\s*final\s+answer\s*\**\s*[:：]", re.IGNORECASE)

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

# Words that mean the amount is a loss (a negative profit).
# "no loss" is excluded so "profit of 3000 (no loss)" stays positive.
LOSS_WORD_RE = re.compile(r"(?<!no )\b(loss|deficit)\b", re.IGNORECASE)


def strip_thinking(text):
    """Remove a <think>...</think> block if the model put its reasoning in the answer."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def find_final_answer_text(answer):
    """
    Return (text, method): the part of the reply that holds the final answer,
    and how we found it. Returns (None, "not_found") if nothing usable.
    """
    text = strip_thinking(answer)

    # 1. Last "Final answer:" marker -> the rest of that line.
    markers = list(FINAL_ANSWER_RE.finditer(text))
    if markers:
        rest = text[markers[-1].end():]
        line = rest.strip().splitlines()[0] if rest.strip() else ""
        if NUMBER_RE.search(line):
            return line, "final_answer_line"

    # 2. Last \boxed{...} value (common in math-style answers).
    boxes = BOXED_RE.findall(text)
    if boxes:
        return boxes[-1], "boxed"

    # 3. Fallback: the last non-empty line that contains a number.
    for line in reversed(text.strip().splitlines()):
        if NUMBER_RE.search(line):
            return line, "last_line_with_number"

    return None, "not_found"


def _to_decimal(match, text):
    """Turn one NUMBER_RE match into a signed Decimal."""
    value = Decimal(match.group("digits").replace(",", ""))

    has_minus = bool(match.group("minus1") or match.group("minus2"))
    if has_minus:
        # A "-" right after another number is subtraction ("8000 - 9200"),
        # not a negative sign, so don't treat it as negative.
        before = text[:match.start()].rstrip()
        if before and before[-1].isdigit():
            has_minus = bool(match.group("minus2"))

    is_parenthesised = bool(match.group("paren") and match.group("close"))
    if has_minus or is_parenthesised:
        value = -value
    return value


def parse_number(text):
    """
    Return the LAST number in `text` as a Decimal, or None.
    If the text says it is a loss and the number has no sign, make it negative.
    """
    if text is None:
        return None
    cleaned = text.replace("*", "").replace("`", "")
    matches = list(NUMBER_RE.finditer(cleaned))
    if not matches:
        return None

    value = _to_decimal(matches[-1], cleaned)
    if value > 0 and LOSS_WORD_RE.search(cleaned):
        value = -value
    return value


def grade_numeric_answer(model_answer, expected_answer, tolerance="0.005"):
    """
    Grade one answer. Returns a dict with:
      correct            True / False
      extracted_answer   the number we read from the model (string) or None
      extraction_method  how we found it
      grading_note       short explanation, useful when something is off
    """
    try:
        expected = Decimal(str(expected_answer).replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return {
            "correct": False,
            "extracted_answer": None,
            "extraction_method": "not_graded",
            "grading_note": f"expected_answer {expected_answer!r} is not a number; "
                            "numeric grading is not possible for this task.",
        }

    if not model_answer or not model_answer.strip():
        return {
            "correct": False,
            "extracted_answer": None,
            "extraction_method": "not_found",
            "grading_note": "Model answer is empty.",
        }

    final_text, method = find_final_answer_text(model_answer)
    value = parse_number(final_text)
    if value is None:
        return {
            "correct": False,
            "extracted_answer": None,
            "extraction_method": method,
            "grading_note": "Could not find a final numeric answer in the reply.",
        }

    correct = abs(value - expected) <= Decimal(tolerance)
    # normalize() turns Decimal("3000.00") into "3E+3", so format manually.
    shown = format(value.normalize(), "f")
    return {
        "correct": correct,
        "extracted_answer": shown,
        "extraction_method": method,
        "grading_note": f"Read {shown} from: {final_text.strip()[:120]!r}",
    }
