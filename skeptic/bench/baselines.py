"""Fixed-pipeline baseline (the 'fixed workflow' comparison from the approved proposal).

No LLM. It reads the task's structured scope (not the English question), runs EVERY
applicable check in a fixed order, and reports the first one that flags. For value
questions it takes the most common recorded value (majority vote across sources).
It cannot choose checks, weigh sources, explain causes or abstain: the comparison with
the agent shows what that judgement is worth.
"""
from collections import Counter

import pandas as pd

from ..agent import tools as T
from ..physics import units


def _flag(ans, fault, explanation, value=None):
    ans.update(verdict="fault_found", fault_type=fault, explanation=explanation, confidence=0.9)
    if value is not None:
        ans["value"] = value
    return ans


def fixed_pipeline(ws, task):
    s = task.scope
    ans = dict(verdict="no_fault", confidence=0.6, explanation="all checks passed", evidence=[])
    calls = []

    def run(name, **args):
        out, _ = T.call(ws, name, args)
        calls.append((name, args, out))
        return out

    if s.get("start") and s.get("end") and "edr" in ws.edr and not s.get("quantity"):
        long_window = (pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).days >= 3
        for ch in ("spp_psi", "hookload_klb"):
            if run("check_stuck_channel", channel=ch, start=s["start"], end=s["end"]).get("verdict") == "stuck":
                return _flag(ans, "stuck_channel", f"{ch} is flat while the rig is active"), calls
        if not long_window:
            r = run("check_clock_offset", start=s["start"], end=s["end"], against=s.get("against", "reports"), max_lag_min=180)
            if r.get("verdict") == "offset":
                return _flag(ans, "clock_offset", f"best lag {r['best_lag_min']} min", r["best_lag_min"]), calls
            d = run("check_depths", start=s["start"], end=s["end"])
            jumps = d.get("jumps", {})
            if jumps.get("verdict") == "jumps found":
                return _flag(ans, "depth_shift", "hole depth steps", abs(jumps["events"][0]["change_ft"])), calls
        else:
            if run("check_block_weight", start=s["start"], end=s["end"]).get("verdict") == "drift":
                return _flag(ans, "sensor_drift", "in-slips hook load trends"), calls

    if s.get("depth_min") is not None:
        if run("check_surveys", depth_min=s["depth_min"], depth_max=s["depth_max"]).get("verdict") == "inconsistent":
            return _flag(ans, "survey_error", "survey stations disagree"), calls

    if s.get("quantity"):
        q = s["quantity"]
        facts = run("get_facts", quantity=q, qualifier=s.get("qualifier"), start=s.get("start"), end=s.get("end"), limit=200)
        cands = facts.get("candidates", [])
        vals = []
        for c in cands:
            v = c["value"]
            if q.endswith("_weight") and q != "mud_weight" and isinstance(v, (int, float)):
                kv, why = units.read_hook_load(v, c.get("unit") or "")
                if c.get("unit") == "lb" and v < 1000:
                    return _flag(ans, "unit_error", f"{v} lb is not a physical hook load", v), calls
                v = kv
            if q == "mud_weight" and isinstance(v, (int, float)) and not 6.5 <= v <= 20:
                return _flag(ans, "unit_error", f"mud weight {v} out of range", round(v * 8.345, 2)), calls
            if v is not None:
                vals.append(v)
        if q == "mud_weight" and s.get("start"):
            r = run("check_series_continuity", quantity="mud_weight", max_step=0.5, qualifier="mud check",
                    start=str(pd.Timestamp(s["start"]) - pd.Timedelta("4D")), end=str(pd.Timestamp(s["end"]) + pd.Timedelta("4D")))
            for o in r.get("isolated_outliers", []):
                if pd.Timestamp(s["start"]) <= pd.Timestamp(o["time"]) <= pd.Timestamp(s["end"]):
                    return _flag(ans, "wrong_value", "isolated outlier", sum(o["neighbours"]) / 2), calls
        if vals:
            v, n = Counter(map(str, vals)).most_common(1)[0]
            try:
                v = float(v)
            except ValueError:
                pass
            ans.update(verdict="answer", value=v, explanation=f"most common value ({n} of {len(vals)} records)", confidence=0.7)
            return ans, calls

    if s.get("parameter"):
        r = run("hookload_model", model="zone_mu_and_fill", start=s.get("start"), end=s.get("end"))
        if "params" in r:
            ans.update(verdict="answer", value=r["params"][s["parameter"]]["value"], quantity=s["parameter"], confidence=0.7,
                       explanation="zone friction fit")
        return ans, calls

    if not s:
        ans.update(verdict="unexplained", confidence=0.3, explanation="no structured checks apply")
    return ans, calls
