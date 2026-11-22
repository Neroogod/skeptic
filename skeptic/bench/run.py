"""Run configurations over benchmark tasks and score them.

  .venv/bin/python -m skeptic.bench.run --configs fixed                  # no API key needed
  .venv/bin/python -m skeptic.bench.run --configs skeptic,react,rag --seeds 3 --wells forge
  .venv/bin/python -m skeptic.bench.run --configs skeptic --tasks forge-stuck-1,lsu-shoe-20
  .venv/bin/python -m skeptic.bench.run --configs skeptic,react --wells test --seeds 3   # held-out test set

Writes runs/<stamp>/traces.jsonl (one trace + score per run; LSU traces stay local:
runs/ is git-ignored) and runs/<stamp>/summary.json.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .. import config
from ..agent import harness
from ..evidence import workspace
from . import inject, score as S
from .baselines import fixed_pipeline
from .tasks import load


TASK_FILES = {"forge": "public/forge_tasks.json",      # development set (FORGE)
              "lsu": "private/lsu_tasks.json",         # development set (LSU, private)
              "test": "public/forge_test.json"}        # held-out test set (FORGE), frozen before any run


def load_tasks(wells, ids=None):
    """Tasks from the named task sets (forge, lsu, test), optionally only the given ids."""
    tasks = []
    for w in wells:
        f = config.BENCH / TASK_FILES[w]
        if f.exists():
            tasks += load(f)
    if ids:
        tasks = [t for t in tasks if t.id in ids]
    return tasks


def task_workspace(task):
    ws = workspace.load(task.well)
    return inject.apply(ws, task.injection) if task.injection else ws


def run_one(task, cfg_name, seed=0, model=None):
    """Run one task with one configuration and seed; returns (score row, trace)."""
    ws = task_workspace(task)
    if cfg_name == "fixed":
        t0 = time.time()
        ans, calls = fixed_pipeline(ws, task)
        trace = dict(config="fixed", answer=ans, n_calls=len(calls), usage=dict(cost_usd=0.0), seconds=round(time.time() - t0, 1),
                     calls=[dict(name=n, args=a) for n, a, _ in calls])
    else:
        from ..agent.llm import OpenRouterLLM
        llm = OpenRouterLLM(model=model, seed=seed, temperature=0.0 if seed == 0 else 0.7)
        critic = OpenRouterLLM(model=config.CRITIC_MODEL or model, seed=seed) if harness.CONFIGS[cfg_name].skeptic else None
        trace = harness.run(ws, task.question, llm, harness.CONFIGS[cfg_name], critic=critic)
    sc = S.score(task, trace["answer"])
    return dict(task=task.id, well=task.well, slice=task.slice, family=task.family, config=cfg_name, seed=seed,
                cost_usd=trace.get("usage", {}).get("cost_usd", 0.0), n_calls=trace.get("n_calls", 0),
                seconds=trace.get("seconds"), **sc), trace


def main(argv=None):
    """Command line: run configurations x task sets x seeds in parallel; writes runs/<stamp>/traces.jsonl
    and summary.json, and stops if the total cost passes --max-cost."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="fixed")
    ap.add_argument("--wells", default="forge,lsu", help="task sets: forge, lsu (development), test (held out)")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=3.0, help="stop the whole run above this total cost (USD)")
    a = ap.parse_args(argv)
    tasks = load_tasks(a.wells.split(","), set(filter(None, a.tasks.split(","))))
    out = config.RUNS / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows, total = [], 0.0
    jobs = [(t, cfg, seed) for cfg in a.configs.split(",") for seed in range(a.seeds if cfg != "fixed" else 1) for t in tasks]
    lock = threading.Lock()

    def work(job):
        nonlocal total
        t, cfg, seed = job
        with lock:
            if total > a.max_cost:
                return None
        try:
            row, trace = run_one(t, cfg, seed, a.model)
        except Exception as e:                       # noqa: BLE001 - one failed run must not kill the batch
            row, trace = dict(task=t.id, well=t.well, slice=t.slice, family=t.family, config=cfg, seed=seed, cost_usd=0.0,
                              n_calls=0, seconds=None, **S.score(t, None)), dict(error=repr(e))
            row["reason"] = f"run error: {e!r}"[:200]
        with lock:
            total += row["cost_usd"] or 0.0
            rows.append(row)
            log.write(json.dumps(dict(row=row, trace=trace), default=str) + "\n")
            log.flush()
            print(f"{cfg:10s} s{seed} {t.id:28s} {'OK ' if row['correct'] else 'BAD'} {row['reason'][:70]}  ${total:.3f}", flush=True)
        return row

    with open(out / "traces.jsonl", "w") as log, ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, jobs))
    if total > a.max_cost:
        print(f"cost cap {a.max_cost} USD reached; some runs skipped")
    configs = {r["config"] for r in rows}
    aurc = {c: S.aurc(S.risk_coverage([r for r in rows if r["config"] == c])) for c in configs}
    summ = dict(summary=S.summarize(rows), aurc=aurc,
                total_cost_usd=round(total, 3), n_runs=len(rows))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
