"""The control loop (sense -> decide -> act -> reflect), written out by hand.

One run = one question about one well:
  1. the model sees the system prompt, the well description and the question
  2. each turn it may call tools; every call gets an id (c1, c2, ...) that later claims must cite
  3. tool results pass the injection guard before the model sees them
  4. submit_answer goes through gates before it is accepted:
       evidence ids exist -> numbers trace to tool results (verifier) -> ledger discipline ->
       identifiability gate (fitted parameters) -> skeptic review (a second model tries to refute)
     a rejected answer comes back as a tool error, so the model must fix it
  5. budgets: max steps, max cost; repeated identical calls are refused (loop detection). Near the
     end of the budget, or when the model stops calling tools, submit_answer is forced; if still no
     answer passes the gates, the run ends with an abstention (never with nothing)
Everything is written to a trace for the benchmark and the UI.
"""
import json
import time
from dataclasses import dataclass, field

from . import tools as T
from . import ledger as L          # noqa: F401  (registers the ledger tools)
from .guard import annotate
from .prompts import SKEPTIC_SYSTEM, REACT_SYSTEM, RAG_SYSTEM, CRITIC_SYSTEM, task_prompt
from .verify import unverified_numbers

EVIDENCE_TOOLS = ["get_facts", "search_text", "read_text", "list_channels", "edr_stats", "edr_values"]
CHECK_TOOLS = ["check_stuck_channel", "check_clock_offset", "check_depths", "check_surveys", "check_block_weight",
               "check_series_continuity", "interpret_units", "check_tfa", "calculate"]
PHYSICS_TOOLS = ["pump_output", "bit_hydraulics", "hookload_model", "friction_identifiability", "friction_by_trip",
                 "check_hookload_physics", "trip_tank_displacement"]
LEDGER_TOOLS = ["add_hypothesis", "update_hypothesis"]


@dataclass
class Config:
    name: str
    system: str = SKEPTIC_SYSTEM
    tools: list = field(default_factory=lambda: EVIDENCE_TOOLS + CHECK_TOOLS + PHYSICS_TOOLS + LEDGER_TOOLS + ["submit_answer"])
    verifier: bool = True
    guard: bool = True
    ledger_gate: bool = True          # explanations need >= 2 hypotheses and a tested winner
    identifiability_gate: bool = True
    skeptic: bool = True
    skeptic_rounds: int = 1
    max_steps: int = 30
    max_cost_usd: float = 1.0
    max_repeats: int = 2


CONFIGS = {
    "skeptic": Config("skeptic"),
    "no_physics": Config("no_physics", tools=EVIDENCE_TOOLS + LEDGER_TOOLS + ["submit_answer"]),
    "no_skeptic": Config("no_skeptic", skeptic=False, ledger_gate=False),
    "no_gate": Config("no_gate", identifiability_gate=False),
    "react": Config("react", system=REACT_SYSTEM, tools=EVIDENCE_TOOLS + CHECK_TOOLS + PHYSICS_TOOLS + ["submit_answer"],
                    ledger_gate=False, identifiability_gate=False, skeptic=False),
    "rag": Config("rag", system=RAG_SYSTEM, tools=["search_text", "read_text", "submit_answer"], verifier=False,
                  ledger_gate=False, identifiability_gate=False, skeptic=False),
    "no_guard": Config("no_guard", guard=False),
}


class Context:
    """State the harness keeps outside the chat: ledger, calls, gate counters."""

    def __init__(self, cfg, question, critic=None):
        self.cfg, self.question, self.critic = cfg, question, critic
        self.ledger = L.Ledger()
        self.calls = []                 # dicts: id, name, args, text, guard_hits
        self.call_ids = set()
        self.rejections = []
        self.skeptic_reviews = []
        self.usage = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)

    def tool_texts(self):
        return [c["text"] for c in self.calls]

    def check_answer(self, ans):
        """Gates on submit_answer. Returns a list of problems (empty = accept)."""
        cfg, problems = self.cfg, []
        bad = [e for e in ans.get("evidence", []) if e not in self.call_ids]
        if bad:
            problems.append(f"evidence ids {bad} are not tool calls from this session")
        if not ans.get("evidence") and ans.get("verdict") != "abstain":
            problems.append("cite at least one tool-call id in evidence")
        if ans.get("verdict") == "fault_found" and ans.get("fault_type") in (None, "", "none"):
            problems.append("verdict fault_found needs a fault_type (see the field description)")
        if cfg.verifier:
            nums = unverified_numbers(ans, self.tool_texts(), self.question)
            if nums:
                problems.append(f"these numbers appear in no tool result: {nums}. Measure them with a tool or remove them")
        if cfg.ledger_gate and ans.get("verdict") in ("fault_found", "answer", "unexplained"):
            n_rej = sum("ledger" in r for r in self.rejections)
            if len(self.ledger.h) < 2 and n_rej < 1:
                problems.append("ledger: put at least two competing explanations on the ledger (add_hypothesis) and test "
                                "them before concluding (this gate is applied once)")
            elif self.ledger.untested_supported():
                problems.append(f"ledger: hypotheses {self.ledger.untested_supported()} are marked supported without evidence ids")
        if cfg.identifiability_gate and self._claims_friction(ans):
            checks = [c for c in self.calls if c["name"] == "friction_identifiability" and c.get("verdict")]
            if not checks:
                problems.append("a friction factor is reported but its identifiability was never checked (friction_identifiability)")
            elif not any(c["verdict"] == "identifiable" for c in checks):
                problems.append(f"friction_identifiability says 'not identifiable' ({', '.join(c['id'] for c in checks)}): the data "
                                "do not determine this friction factor. Report abstain (say what measurement would fix it), "
                                "or show a window where it is identifiable")
        if not problems and cfg.skeptic and self.critic is not None and len(self.skeptic_reviews) < cfg.skeptic_rounds:
            review = self.review(ans)
            self.skeptic_reviews.append(review)
            if review.get("decision") == "reject":
                problems.append("skeptic review rejected the answer: " + "; ".join(review.get("objections", []))[:800]
                                + (f" | suggested check: {review.get('suggested_check')}" if review.get("suggested_check") else "")
                                + " | Address each objection (run the check, or explain why it does not change the conclusion); "
                                  "change the verdict only if the evidence changes.")
        if problems:
            self.rejections.append(" | ".join(problems))
        return problems

    def _called(self, *names):
        return any(c["name"] in names for c in self.calls)

    @staticmethod
    def _claims_friction(ans):
        import re
        q = (str(ans.get("quantity", "")) + " " + str(ans.get("unit", ""))).lower()
        return ans.get("verdict") == "answer" and ans.get("value") is not None and bool(re.search(r"friction|\bmu(?:_|\b)|\u03bc", q))

    def review(self, ans):
        """Second model, different prompt: try to refute. It sees the question, ledger,
        answer and the cited tool results (not the whole chat)."""
        cited = [c for c in self.calls if c["id"] in ans.get("evidence", [])]
        body = dict(question=self.question, answer={k: v for k, v in ans.items() if k != "evidence"},
                    ledger=self.ledger.table(),
                    cited_evidence=[dict(id=c["id"], tool=c["name"], args=c["args"], result=c["text"][:1500]) for c in cited],
                    other_calls=[f'{c["id"]} {c["name"]}({json.dumps(c["args"])[:120]})' for c in self.calls if c not in cited])
        r = self.critic.chat([dict(role="system", content=CRITIC_SYSTEM), dict(role="user", content=json.dumps(body, default=str)[:12000])])
        self._add_usage(r["usage"])
        txt = r["content"].strip()
        try:
            return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return dict(decision="accept", objections=[], parse_error=txt[:200])

    def _add_usage(self, u):
        for k in self.usage:
            self.usage[k] += u.get(k, 0) or 0


def run(ws, question, llm, cfg, critic=None, log=None, on_event=None):
    """Run one question to completion. Returns the trace dict.
    on_event(call_dict, ctx) is called after every tool call (the UI uses it to stream)."""
    t_start = time.time()
    ctx = Context(cfg, question, critic)
    schemas = T.schemas(cfg.tools)
    messages = [dict(role="system", content=cfg.system), dict(role="user", content=task_prompt(ws.well, question))]
    seen, nudges, stop = {}, 0, "max_steps"
    for step in range(cfg.max_steps):
        # force submit_answer near the end of the budget, or when the model stopped calling tools
        force = ctx.usage["cost_usd"] > cfg.max_cost_usd or step >= cfg.max_steps - 2 or nudges >= 2
        if force:
            messages.append(dict(role="user", content="Budget reached. Call submit_answer now with what you have "
                                 "(abstain if the evidence is insufficient)."))
        r = llm.chat(messages, tools=schemas, tool_choice={"type": "function", "function": {"name": "submit_answer"}} if force else None)
        ctx._add_usage(r["usage"])
        if not r["tool_calls"]:
            messages.append(dict(role="assistant", content=r["content"] or ""))
            nudges += 1
            if nudges > 3:
                stop = "no_tool_calls"
                break
            if nudges < 2:
                messages.append(dict(role="user", content="Continue with tool calls; finish with submit_answer."))
            continue
        messages.append(dict(role="assistant", content=r["content"] or None, tool_calls=[
            dict(id=c["id"], type="function", function=dict(name=c["name"], arguments=json.dumps(c["arguments"])))
            for c in r["tool_calls"]]))
        done = False
        for c in r["tool_calls"]:
            cid = f"c{len(ctx.calls) + 1}"
            key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
            seen[key] = seen.get(key, 0) + 1
            if c["name"] not in cfg.tools:
                out, text = {"error": f"tool {c['name']} is not available in this configuration"}, None
                text = json.dumps(out)
            elif "_unparseable" in c["arguments"]:
                out = {"error": "your arguments were not valid JSON (cut off?). Send the call again with complete JSON."}
                text = json.dumps(out)
            elif seen[key] > cfg.max_repeats and c["name"] != "submit_answer":
                out = {"error": "identical call repeated; use the earlier result or try a different check"}
                text = json.dumps(out)
            else:
                out, text = T.call(ws, c["name"], c["arguments"], ctx)
            hits = []
            if cfg.guard:
                text, hits = annotate(text)
            ctx.calls.append(dict(id=cid, name=c["name"], args=c["arguments"], text=text, guard_hits=hits, step=step,
                                  verdict=out.get("verdict") if isinstance(out, dict) else None))
            ctx.call_ids.add(cid)
            if on_event:
                on_event(ctx.calls[-1], ctx)
            messages.append(dict(role="tool", tool_call_id=c["id"], content=f"[{cid}] {text}"))
            if c["name"] == "submit_answer" and isinstance(out, dict) and out.get("accepted"):
                done = True
        if done:
            stop = "answered"
            break
    if ctx.ledger.answer is None:
        # no answer passed the gates within budget: the system refuses rather than release an unverified one
        ctx.ledger.answer = dict(verdict="abstain", confidence=0.0, evidence=[], fallback=True,
                                 explanation=f"No answer passed the checks within the budget (stop: {stop}).")
    trace = dict(config=cfg.name, model=getattr(llm, "model", "?"), well=ws.well, question=question,
                 answer=ctx.ledger.answer, stop=stop, steps=step + 1, n_calls=len(ctx.calls),
                 calls=[{k: v for k, v in c.items()} for c in ctx.calls], ledger=ctx.ledger.table(),
                 rejections=ctx.rejections, skeptic_reviews=ctx.skeptic_reviews, usage=ctx.usage,
                 guard_hits=[h for c in ctx.calls for h in c["guard_hits"]], seconds=round(time.time() - t_start, 1))
    if log:
        log.write(json.dumps(trace, default=str) + "\n")
        log.flush()
    return trace
