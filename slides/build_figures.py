"""Figures for the slide deck.

Test-set figures come from the committed runs in results/test_set (FORGE only). The
physics-check figure needs the FORGE torque-and-drag export (cache/tnd_forge.pkl,
scripts/export_tnd.py); it is skipped if the export is missing.

    .venv/bin/python slides/build_figures.py      ->  slides/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skeptic import config                          # noqa: E402
from skeptic.bench import report, score as S        # noqa: E402

OUT = ROOT / "slides" / "figures"
RUNS = [ROOT / "results" / "test_set" / "llm", ROOT / "results" / "test_set" / "fixed"]

# One colour per system, checked for colour-blind separation (dataviz validator, all checks pass).
SYSTEMS = {"skeptic": ("Skeptic", "#6B3FA8"), "react": ("ReAct, same tools", "#C27C0E"), "fixed": ("Fixed pipeline, no LLM", "#0095B0")}
INK, MUTED, GRID, FLAG = "#2A1050", "#6E6680", "#E4E0EA", "#B3261E"

plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def test_rows():
    df = report.load_rows([str(d) for d in RUNS])
    return df[df.task.str.startswith("test-")]


def risk_coverage(df):
    """Error rate among the answers a system is most confident about (risk) against the share
    of questions it answers (coverage). Abstentions are never covered, so a curve stops early."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for cfg, (name, color) in SYSTEMS.items():
        rows = df[df.config == cfg].to_dict("records")
        c = S.risk_coverage(rows)
        x, y = [p["coverage"] for p in c], [p["risk"] for p in c]
        ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=9, color=color, mec="white", mew=2)
        ax.annotate(f"{name}\nAURC {S.aurc(c):.3f}", (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=13, color=INK)
    ax.set_xlim(0, 1.32)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, None)
    ax.set_xlabel("Coverage: share of questions answered, most confident first")
    ax.set_ylabel("Risk: share of those answers that are wrong")
    ax.spines["bottom"].set_bounds(0, 1)
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)


def headline(df):
    """Three measures as small multiples (one scale each): accuracy ties, confident errors halve."""
    t = report.table(df).set_index("config")
    panels = [("accuracy", "Accuracy"), ("confident_wrong", "Wrong with confidence ≥ 0.7"), ("abstain", "Abstained")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    order = list(SYSTEMS)[::-1]
    for ax, (col, title) in zip(axes, panels):
        vals = [t.loc[c, col] for c in order]
        ax.barh(range(len(order)), vals, height=0.62, color=[SYSTEMS[c][1] for c in order])
        top = max(vals) * 1.35 if max(vals) > 0 else 1
        for i, v in enumerate(vals):
            ax.text(v + top * 0.02, i, f"{v:.0%}" if col != "accuracy" else f"{v:.2f}", va="center", fontsize=14, color=INK)
        ax.set_xlim(0, top)
        ax.set_title(title, fontsize=15, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([SYSTEMS[c][0] for c in order], fontsize=14, color=INK)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "headline.png")
    plt.close(fig)


def physics_check():
    """FORGE pick-ups against the friction envelope: a deep trip out that friction explains, and
    the 27-28 Dec logging trips, one of which weighs less than the frictionless string."""
    from skeptic.physics import tnd
    ds = tnd.dataset("forge")
    ev, th = ds["ev"], ds["base_theta"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    cases = [("13 Dec, deep trip out: inside the envelope", "2020-12-12", "2020-12-14", [73], (100, 420)),
             ("27-28 Dec, logging trips: trip 133 is lighter than zero friction", "2020-12-26", "2020-12-29", [133, 136], (60, 300))]
    for ax, (title, t0, t1, trips, ylim) in zip(axes, cases):
        idx = tnd.select(ds, t0, t1)
        chk = tnd.physics_check(ds, idx)
        flagged = {(r["trip"], r["time"]) for r in chk["worst_readings"]}
        for c in trips:
            i = idx[ev.cluster.values[idx] == c]
            i = i[np.argsort(ev.bit.values[i])]
            bit, y = ev.bit.values[i], ev.hook.values[i]
            free = tnd.predict(ds, dict(th, c_s=0.0, mu_cased=0.0, mu_open=0.0), i)
            most = tnd.predict(ds, dict(th, c_s=0.0, mu_cased=tnd.MU_MAX, mu_open=tnd.MU_MAX), i)
            ax.fill_between(bit, free, most, color="#EFEAF6", lw=0)
            ax.plot(bit, free, color=MUTED, lw=2, ls="--")
            ax.plot(bit, most, color=MUTED, lw=2, ls=":")
            out = np.array([(c, str(t)[:16].replace(" ", "T")) in flagged for t in ev.t.values[i]])
            ax.plot(bit[~out], y[~out], "o", ms=8, color=SYSTEMS["skeptic"][1], mec="white", mew=1.2, ls="none")
            if out.any():
                ax.plot(bit[out], y[out], "o", ms=10, mfc="white", mec=FLAG, mew=2.2, ls="none")
        ax.set_ylim(*ylim)                       # the friction-0.8 edge runs off the top in the deep hole
        ax.set_title(title, fontsize=14, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Bit depth (ft MD)")
        ax.set_ylabel("Pick-up hook load (klb)")
    h = [plt.Line2D([], [], color=MUTED, lw=2, ls="--"), plt.Line2D([], [], color=MUTED, lw=2, ls=":"),
         plt.Line2D([], [], marker="o", ms=8, color=SYSTEMS["skeptic"][1], ls="none"),
         plt.Line2D([], [], marker="o", ms=10, mfc="white", mec=FLAG, mew=2.2, ls="none")]
    fig.legend(h, ["zero friction", f"friction {tnd.MU_MAX}", "reading", "outside the envelope (flagged)"],
               loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUT / "physics_check.png")
    plt.close(fig)


def friction_windows():
    """The physics data check and the identifiability advice on the 7 friction windows of the test
    set (post-test addition, PLAN.md v4) -> results/v4_friction_windows.json for the deck."""
    import json
    from skeptic.bench.build import IDENTIFY
    from skeptic.physics import tnd
    ds = tnd.dataset("forge")
    rows = []
    for t0, t1, expected, why in IDENTIFY:
        idx = tnd.select(ds, t0, t1)
        chk = tnd.physics_check(ds, idx)
        idn = tnd.identifiability(ds, "mu_open", idx, t0=t0, t1=t1)
        rows.append(dict(start=t0, end=t1, key="answer" if expected == "identifiable" else "abstain", why=why,
                         physics_check=chk["verdict"], readings_outside=chk["readings_outside_envelope"],
                         by_reason=chk["by_reason"], identifiability=idn["verdict"], estimate=idn["estimate"],
                         next_measurement=idn["next_measurement"]))
    (ROOT / "results" / "v4_friction_windows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = test_rows()
    risk_coverage(df)
    headline(df)
    if (config.CACHE / "tnd_forge.pkl").exists():
        physics_check()
        friction_windows()
    else:
        print("cache/tnd_forge.pkl missing: physics_check.png not rebuilt (run scripts/export_tnd.py)")
    print("figures in", OUT)
