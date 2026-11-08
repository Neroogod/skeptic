"""Benchmark task schema.

A task = a question about one well + (optional) injection spec + answer key + scope.
  scope: the structured version of what the question asks (window, quantity, channels).
         The fixed-pipeline baseline reads the scope instead of parsing English; the
         LLM configurations only see the question.
  key:   what a correct answer must contain (never shown to any configuration):
         verdicts     accepted verdicts, e.g. ["fault_found"]
         fault_types  accepted fault types (if the verdict is fault_found)
         value, tol   the correct value and tolerance (if one is expected)
         cause_any    keywords, one of which must appear in `cause` (explanation tasks)
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Task:
    id: str
    well: str
    slice: str                    # injected | control | real | identify | explain
    family: str                   # stuck | clock | depth | mud_weight | survey | drift | unit | planted | ...
    question: str
    scope: dict
    key: dict
    injection: Optional[dict] = None
    note: str = ""


# Tasks used while developing the harness (the live smoke test on 2026-11-07). Prompt and
# tool fixes were made after seeing these, so results are reported with and without them.
DEV_TASKS = {"forge-stuck-1", "forge-friction-answer", "forge-mw-unit-1"}


def save(tasks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in tasks], indent=1, default=str))


def load(path):
    return [Task(**d) for d in json.loads(path.read_text())]
