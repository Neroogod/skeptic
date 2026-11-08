"""Merge benchmark runs into result tables (markdown) for the report and slides.

  .venv/bin/python -m skeptic.bench.report runs/20261107-1830 runs/20261107-1905

If a (config, task, seed) appears in several runs, the latest run wins. The held-out test
set (task ids test-*) is the headline; the development set (FORGE + LSU tasks used while
building the agent) is reported apart. Paired comparisons average each task over seeds,
then bootstrap over tasks (95% CI) and use a sign-flip permutation test (p).
Only aggregates are printed, so the output may mention LSU slices without exposing LSU values.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import score as S
from .tasks import DEV_TASKS

ORDER = ["skeptic", "no_skeptic", "no_gate", "no_guard", "no_physics", "react", "rag", "fixed"]


def load_rows(dirs, rescore=True):
    """Rows from saved runs. rescore=True re-scores every saved answer against the CURRENT
    task keys, so a corrected key applies to old runs without new LLM calls."""
    from .run import load_tasks
    keys = {t.id: t for t in load_tasks(["forge", "lsu", "test"])} if rescore else {}
    rows = {}
    for d in sorted(dirs):
        f = Path(d) / "traces.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            rec = json.loads(line)
            r = rec["row"]
            if r["task"] in keys and "answer" in rec["trace"]:
                r.update(S.score(keys[r["task"]], rec["trace"]["answer"]))
            rows[(r["config"], r["task"], r["seed"])] = r
    return pd.DataFrame(rows.values())


def table(df):
    out = []
    for cfg in [c for c in ORDER if c in set(df.config)] + sorted(set(df.config) - set(ORDER)):
        g = df[df.config == cfg]
        det, fa = g.detected.dropna(), g.false_alarm.dropna()
        curve = S.risk_coverage(g.to_dict("records"))
        out.append({"config": cfg, "runs": len(g), "accuracy": g.correct.mean(),
                    **{f"acc_{s}": g[g.slice == s].correct.mean() for s in ("injected", "control", "real", "identify", "explain")},
                    "detection": det.astype(float).mean() if len(det) else None,
                    "false_alarm": fa.astype(float).mean() if len(fa) else None,
                    "confident_wrong": g.confident_wrong.mean(), "abstain": g.abstained.mean(), "aurc": S.aurc(curve),
                    "usd_per_task": g.cost_usd.mean(), "calls_per_task": g.n_calls.mean()})
    return pd.DataFrame(out)


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def paired(df, a, b, metric="correct", n=10000, seed=0):
    """a vs b on the same tasks: per-task mean over seeds, bootstrap CI and sign-flip p for the mean difference."""
    g = df[df.config.isin([a, b])].assign(m=lambda x: x[metric].astype(float)).groupby(["task", "config"]).m.mean().unstack().dropna()
    if a not in g or b not in g or not len(g):
        return None
    d = (g[a] - g[b]).values
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (n, len(d))).mean(1)
    flips = rng.choice([-1.0, 1.0], (n, len(d)))
    p = float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return {"compare": f"{a} - {b}", "tasks": len(d), a: g[a].mean(), b: g[b].mean(), "diff": d.mean(),
            "ci95_low": np.percentile(boot, 2.5), "ci95_high": np.percentile(boot, 97.5), "p": p,
            f"{a}_better": int((d > 0).sum()), f"{b}_better": int((d < 0).sum())}


def paired_table(df, a="skeptic", b="react"):
    rows = [dict(slice="all", **r) for r in [paired(df, a, b)] if r]
    for s_ in ("injected", "control", "identify"):
        r = paired(df[df.slice == s_], a, b)
        if r:
            rows.append(dict(slice=s_, **r))
    return pd.DataFrame(rows)


def section(title, d):
    parts = [f"## {title}\n", fmt(table(d)), "\n\n### Accuracy by family\n",
             d.pivot_table(index="family", columns="config", values="correct", aggfunc="mean").round(2).to_markdown()]
    pt = paired_table(d)
    if len(pt):
        parts += ["\n\n### Skeptic vs ReAct, paired by task\n", fmt(pt)]
    return parts


def main(dirs):
    df = load_rows(dirs)
    if df.empty:
        print("no runs found")
        return
    test, dev = df[df.task.str.startswith("test-")], df[~df.task.str.startswith("test-")]
    parts = []
    if len(test):
        parts += section("Held-out test set (headline)", test) + ["\n"]
    if len(dev):
        parts += section("Development set (first smoke-test tasks excluded)", dev[~dev.task.isin(DEV_TASKS)])
    parts.append(f"\n\nTotal cost: ${df.cost_usd.sum():.3f} over {len(df)} runs.")
    text = "\n".join(parts)
    out = Path(dirs[0]).parent / "report.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in Path("runs").iterdir() if p.is_dir()])
