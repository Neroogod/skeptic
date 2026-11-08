"""Evidence verifier: every number in a final answer must appear in a tool result (or in
the question). An LLM that 'remembers' a plausible value instead of measuring it is
caught here, before the answer is accepted.

Matching rule: a stated number may be a ROUNDED tool value but not a changed one, i.e.
it must lie within half a unit of its own last written digit of some tool value
('12.3' matches 12.348; '5,110' does not match 5,113). Derived numbers (differences,
ratios) must come from the calculate tool so they too appear in a tool result."""
import re

NUM = re.compile(r"(?<![A-Za-z0-9_])-?\d[\d,]*(?:\.(\d+))?")
DATEISH = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")


def numbers(text, with_decimals=False):
    """Numbers in free text, ignoring dates, clock times and ids like c12 / H2 / T00031."""
    text = DATEISH.sub(" ", str(text))
    out = []
    for m in NUM.finditer(text):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        out.append((v, len(m.group(1) or "")) if with_decimals else v)
    return out


def matches(x, decimals, pool):
    tol = 0.5 * 10 ** (-decimals) * 1.0001 + 1e-9
    return any(abs(x - p) <= tol for p in pool)


def unverified_numbers(answer, tool_texts, question=""):
    """Numbers in the answer (explanation, value, wrong_value) not found in any tool output.
    Small integers (0-10) are allowed: they are counts ('two sources'), not measurements."""
    pool = set()
    for t in list(tool_texts) + [question]:
        pool.update(numbers(t))
    claimed = numbers(answer.get("explanation", ""), with_decimals=True)
    for k in ("value", "wrong_value"):
        v = answer.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            claimed += numbers(repr(float(v)).rstrip("0").rstrip(".") if float(v) != int(v) else str(int(v)), with_decimals=True)
    return sorted({x for x, d in claimed if not (x.is_integer() and 0 <= x <= 10) and not matches(x, d, pool)})
