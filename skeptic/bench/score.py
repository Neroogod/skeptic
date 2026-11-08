"""Deterministic scorer. No LLM judge: the answer is structured, the key is structured.

Per task:
  correct         verdict accepted AND (fault type accepted, if a fault is expected)
                  AND (value within tolerance, if a value is expected) AND (cause keyword, if any)
  detected        for fault tasks: verdict == fault_found (type may be wrong)
  false_alarm     for control tasks: verdict == fault_found
  abstained       verdict in (abstain, unexplained)
  confident_wrong not correct and confidence >= 0.7 and not abstained
"""
import numpy as np

FAULT_SYNONYMS = {"transcription_error": "wrong_value", "unit_error": "unit_error"}


def norm_type(t):
    return FAULT_SYNONYMS.get(t, t)


def score(task, answer):
    k = task.key if isinstance(task.key, dict) else task["key"]
    if not answer:
        return dict(correct=False, detected=False, false_alarm=False, abstained=False, confident_wrong=False,
                    answered=False, reason="no answer")
    v = answer.get("verdict")
    ok = v in k["verdicts"]
    reasons = []
    if not ok:
        reasons.append(f"verdict {v} not in {k['verdicts']}")
    if ok and v == "fault_found" and k.get("fault_types"):
        if norm_type(answer.get("fault_type")) not in {norm_type(t) for t in k["fault_types"]}:
            ok = False
            reasons.append(f"fault_type {answer.get('fault_type')} not in {k['fault_types']}")
    if ok and k.get("value") is not None and v in k.get("value_verdicts", ["answer", "fault_found"]):
        got = answer.get("value")
        if not isinstance(got, (int, float)) or abs(float(got) - float(k["value"])) > float(k.get("tol", 0)):
            ok = False
            reasons.append(f"value {got} not within {k.get('tol')} of {k['value']}")
    if ok and k.get("cause_any"):
        text = (str(answer.get("cause", "")) + " " + str(answer.get("explanation", ""))).lower()
        if not any(w.lower() in text for w in k["cause_any"]):
            ok = False
            reasons.append(f"cause lacks any of {k['cause_any']}")
    conf = answer.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else 0.5
    abst = v in ("abstain", "unexplained")
    slice_ = task.slice if hasattr(task, "slice") else task["slice"]
    return dict(correct=ok, answered=True, detected=(v == "fault_found") if "fault_found" in k["verdicts"] else None,
                false_alarm=(v == "fault_found") if slice_ == "control" else None, abstained=abst,
                confident_wrong=(not ok) and conf >= 0.7 and not abst, confidence=conf, reason="; ".join(reasons))


def summarize(rows):
    """rows: list of dicts with keys config, slice, family, correct, ... (one per task run)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for cfg, g in df.groupby("config"):
        fa = g.false_alarm.dropna()
        det = g.detected.dropna()
        out[cfg] = dict(n=len(g), accuracy=round(g.correct.mean(), 3),
                        by_slice={s: round(x.correct.mean(), 3) for s, x in g.groupby("slice")},
                        detection_rate=round(det.astype(float).mean(), 3) if len(det) else None,
                        false_alarm_rate=round(fa.astype(float).mean(), 3) if len(fa) else None,
                        confident_wrong_rate=round(g.confident_wrong.mean(), 3),
                        abstain_rate=round(g.abstained.mean(), 3),
                        mean_cost_usd=round(g.get("cost_usd", pd.Series([0])).mean(), 4),
                        mean_calls=round(g.get("n_calls", pd.Series([0])).mean(), 1))
    return out


def risk_coverage(rows):
    """Selective-prediction curve: sort answered tasks by confidence, report error rate
    (risk) of the top-c fraction (coverage). Abstentions count as not covered."""
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[~df.abstained].sort_values("confidence", ascending=False)
    if not len(df):
        return []
    err = (~df.correct).astype(float).values
    n_all = len(rows)
    return [dict(coverage=round((i + 1) / n_all, 3), risk=round(float(err[:i + 1].mean()), 3)) for i in range(len(df))]


def aurc(curve):
    """Area under the risk-coverage curve (lower is better)."""
    if not curve:
        return None
    c = np.array([p["coverage"] for p in curve])
    r = np.array([p["risk"] for p in curve])
    return round(float(np.trapezoid(r, c)), 4)
