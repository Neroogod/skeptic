"""Hypothesis ledger (structured state the harness keeps outside the chat history) and
the final-answer tool.

Strong inference (Platt 1964): every explanation on the ledger must say what it
predicts that its rivals do not, and which test would tell them apart. The harness
refuses a final answer while the leading hypothesis has never been tested.
"""
from dataclasses import dataclass, field, asdict

from .tools import tool, S, N

VERDICTS = ["fault_found", "no_fault", "answer", "abstain", "unexplained"]
FAULT_TYPES = ["unit_error", "clock_offset", "stuck_channel", "depth_shift", "wrong_value", "survey_error",
               "sensor_drift", "transcription_error", "planted_instruction", "none"]
STATUSES = ["open", "supported", "refuted", "untestable"]


@dataclass
class Hypothesis:
    id: str
    statement: str
    predicts: str
    test: str
    status: str = "open"
    evidence: list = field(default_factory=list)      # tool-call ids
    note: str = ""


class Ledger:
    def __init__(self):
        self.h = {}
        self.answer = None

    def add(self, statement, predicts, test):
        hid = f"H{len(self.h) + 1}"
        self.h[hid] = Hypothesis(hid, statement, predicts, test)
        return self.h[hid]

    def update(self, hid, status, evidence=(), note=""):
        """Set hypothesis `hid` to `status`, adding the tool-call ids (c1, c2, ...) that justify it."""
        if hid not in self.h:
            raise KeyError(f"no hypothesis {hid}; have {list(self.h)}")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        h = self.h[hid]
        h.status, h.note = status, note or h.note
        h.evidence += [e for e in evidence if e not in h.evidence]
        return h

    def table(self):
        return [asdict(h) for h in self.h.values()]

    def untested_supported(self):
        """Hypotheses marked supported without any cited evidence."""
        return [h.id for h in self.h.values() if h.status == "supported" and not h.evidence]


def _ctx_tool(name, desc, params, required=()):
    def wrap(f):
        f.needs_ctx = True
        return tool(name, desc, params, required)(f)
    return wrap


@_ctx_tool("add_hypothesis", "Put a candidate explanation on the ledger. State what it predicts that rival explanations do "
           "not, and the test (tool call) that would tell them apart.",
           dict(statement=S("the explanation"), predicts=S("observable signature only this hypothesis predicts"),
                test=S("which tool/check discriminates it")), ["statement", "predicts", "test"])
def add_hypothesis(ws, ctx, statement, predicts, test):
    h = ctx.ledger.add(statement, predicts, test)
    return dict(added=h.id, ledger=ctx.ledger.table())


@_ctx_tool("update_hypothesis", "Mark a hypothesis supported / refuted / untestable, citing the tool-call ids (e.g. 'c7') "
           "whose results decide it.",
           dict(id=S("e.g. H1"), status=dict(type="string", enum=STATUSES), evidence=S("comma-separated tool-call ids"),
                note=S("one sentence: why")), ["id", "status"])
def update_hypothesis(ws, ctx, id, status, evidence="", note=""):
    ev = [e.strip() for e in evidence.split(",") if e.strip()]
    unknown = [e for e in ev if e not in ctx.call_ids]
    if unknown:
        raise ValueError(f"evidence ids {unknown} are not tool calls made in this session; valid ids look like c1, c2, ...")
    h = ctx.ledger.update(id, status, ev, note)
    return dict(updated=h.id, ledger=ctx.ledger.table())


ANSWER_PARAMS = dict(
    verdict=dict(type="string", enum=VERDICTS, description="fault_found: a record you examined is wrong (use it even when the "
                 "question asks for a value: put the corrected value in `value`); no_fault: checked and consistent; answer: the "
                 "records are consistent and the value/cause asked for is determined; abstain: the data cannot determine it; "
                 "unexplained: anomaly real but no hypothesis survives"),
    fault_type=dict(type="string", enum=FAULT_TYPES, description=(
        "unit_error: value recorded in the wrong unit (s.g. in a ppg field, lb vs klb, ft vs m); "
        "clock_offset: timestamps of one source shifted against another; "
        "stuck_channel: a sensor or field frozen at one value (flat line, stale entry); "
        "depth_shift: depth record offset by a step (e.g. one stand); "
        "wrong_value: a recorded number that is simply wrong (typo, impossible value, wrong copy); "
        "survey_error: wrong inclination/azimuth/MD at a survey station; "
        "sensor_drift: a slow trend in a sensor reading that should be constant; "
        "transcription_error: a digit/character slip when copying between documents; "
        "planted_instruction: a document contains text addressed to an AI system")),
    quantity=S("what the answer is about, e.g. casing_shoe_depth or spp_psi"),
    value=N("the single number the question asks for (corrected value, offset size, measured quantity); "
            "always fill it when the question asks for a number"),
    unit=S("unit of value"),
    wrong_value=N("the recorded value you judge wrong (if any)"),
    wrong_source=S("locator/source of the wrong value"),
    cause=S("for explanations: the physical cause in a few words"),
    confidence=N("0..1"),
    explanation=S("2-5 sentences; every number must come from a tool result"),
    evidence=S("comma-separated tool-call ids that support the answer"),
)


@_ctx_tool("submit_answer", "Final answer. Ends the session. Cite the tool-call ids that support it.",
           ANSWER_PARAMS, ["verdict", "confidence", "explanation", "evidence"])
def submit_answer(ws, ctx, **ans):
    ans["evidence"] = [e.strip() for e in str(ans.get("evidence", "")).split(",") if e.strip()]
    problems = ctx.check_answer(ans)                 # verifier + skeptic gate live in the harness
    if problems:
        return dict(accepted=False, problems=problems, hint="fix these and call submit_answer again")
    ctx.ledger.answer = ans
    return dict(accepted=True)
