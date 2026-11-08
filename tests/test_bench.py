"""Phase 3/5 gate: the benchmark is valid (every planted fault is detectable, every
control is clean, nothing leaks the key) and the scorer/baseline run end to end."""
import pandas as pd
import pytest

from skeptic import config
from skeptic.bench import build, inject, score as S
from skeptic.bench.baselines import fixed_pipeline
from skeptic.bench.tasks import Task, load
from skeptic.evidence import workspace

FORGE_TASKS = config.BENCH / "public" / "forge_tasks.json"
pytestmark = pytest.mark.skipif(not (config.forge_available() and FORGE_TASKS.exists()), reason="run python -m skeptic.bench.build")


@pytest.fixture(scope="module")
def tasks():
    return load(FORGE_TASKS)


@pytest.fixture(scope="module")
def ws():
    return workspace.load("forge")


def test_benchmark_shape(tasks):
    df = pd.DataFrame([dict(slice=t.slice, family=t.family) for t in tasks])
    assert len(tasks) >= 40 and df.slice.value_counts()["control"] >= 10
    assert len({t.id for t in tasks}) == len(tasks)


def test_every_injected_fault_is_detectable_and_every_control_clean(tasks, ws):
    for t in tasks:
        if t.family in ("audit", "friction", "tfa", "date"):
            continue                          # audits validated at build time per family; natural tasks have no oracle
        if t.slice == "injected":
            assert build.oracle(t, inject.apply(ws, t.injection)), t.id
        elif t.slice == "control":
            assert not build.oracle(t, ws), t.id


def test_injection_never_touches_the_clean_workspace(tasks, ws):
    before_facts = ws.facts.df.value.astype(str).tolist()
    before_t = ws.edr["edr"].t.iloc[[0, -1]].tolist()
    before_spp = ws.edr["edr"].spp_psi.sum()
    for t in tasks:
        if t.injection:
            inject.apply(ws, t.injection)
    assert ws.facts.df.value.astype(str).tolist() == before_facts
    assert ws.edr["edr"].t.iloc[[0, -1]].tolist() == before_t
    assert ws.edr["edr"].spp_psi.sum() == pytest.approx(before_spp)


def test_questions_do_not_leak_the_key(tasks):
    for t in tasks:
        v = t.key.get("value")
        if v is not None and t.slice == "injected":
            assert f"{v:g}" not in t.question, t.id
        assert "inject" not in t.question.lower() and "planted" not in t.question.lower()


def test_scorer():
    t = Task("t", "forge", "injected", "clock", "q", {}, dict(verdicts=["fault_found"], fault_types=["clock_offset"], value=90, tol=15))
    assert S.score(t, dict(verdict="fault_found", fault_type="clock_offset", value=100, confidence=0.9))["correct"]
    r = S.score(t, dict(verdict="fault_found", fault_type="clock_offset", value=130, confidence=0.9))
    assert not r["correct"] and r["confident_wrong"] and r["detected"]
    r = S.score(t, dict(verdict="abstain", confidence=0.2))
    assert not r["correct"] and r["abstained"] and not r["confident_wrong"]
    c = Task("c", "forge", "control", "clock", "q", {}, dict(verdicts=["no_fault"]))
    assert S.score(c, dict(verdict="fault_found", fault_type="clock_offset", confidence=0.8))["false_alarm"]
    e = Task("e", "lsu", "explain", "float", "q", {}, dict(verdicts=["answer"], cause_any=["float", "empty"]))
    assert S.score(e, dict(verdict="answer", cause="pipe running empty behind the float valve", confidence=0.8))["correct"]
    assert not S.score(e, dict(verdict="answer", cause="high friction in open hole", confidence=0.8))["correct"]
    assert S.score(t, None)["correct"] is False


def test_risk_coverage_and_aurc():
    rows = [dict(correct=True, abstained=False, confidence=0.9), dict(correct=False, abstained=False, confidence=0.8),
            dict(correct=True, abstained=True, confidence=0.1)]
    curve = S.risk_coverage(rows)
    assert curve[0] == dict(coverage=round(1 / 3, 3), risk=0.0) and curve[-1]["risk"] == 0.5
    assert S.aurc(curve) is not None


def test_fixed_pipeline_end_to_end(tasks, ws):
    t = next(x for x in tasks if x.id.startswith("forge-stuck-") and x.slice == "injected")
    ans, calls = fixed_pipeline(inject.apply(ws, t.injection), t)
    assert S.score(t, ans)["correct"] and calls
