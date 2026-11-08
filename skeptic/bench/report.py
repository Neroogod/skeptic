"""Merge benchmark runs into result tables (markdown) for the report and slides.

  .venv/bin/python -m skeptic.bench.report runs/20261107-1830 runs/20261107-1905

If a (config, task, seed) appears in several runs, the latest run wins. Development
tasks (bench.tasks.DEV_TASKS) are excluded from the headline table and reported apart.
Only aggregates are printed, so the output may mention LSU slices without exposing LSU values.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from . import score as S
from .tasks import DEV_TASKS

ORDER = ["skeptic", "no_skeptic", "no_gate", "no_guard", "no_physics", "react", "rag", "fixed"]


def load_rows(dirs, rescore=True):
    """Rows from saved runs. rescore=True re-scores every saved answer against the CURRENT
    task keys, so a corrected key applies to old runs without new LLM calls."""
    from .run import load_tasks
    keys = {t.id: t for t in load_tasks(["forge", "lsu"])} if rescore else {}
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


def main(dirs):
    df = load_rows(dirs)
    if df.empty:
        print("no runs found")
        return
    head = df[~df.task.isin(DEV_TASKS)]
    parts = ["## Headline (development tasks excluded)\n", fmt(table(head)),
             "\n\n## Accuracy by fault family\n",
             head.pivot_table(index="family", columns="config", values="correct", aggfunc="mean").round(2).to_markdown(),
             "\n\n## Development tasks only\n", fmt(table(df[df.task.isin(DEV_TASKS)])),
             f"\n\nTotal cost: ${df.cost_usd.sum():.3f} over {len(df)} runs."]
    text = "\n".join(parts)
    out = Path(dirs[0]).parent / "report.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in Path("runs").iterdir() if p.is_dir()])
