"""Figures for the slide deck (dark theme, transparent backgrounds so they sit on the slides).

Test-set figures come from the committed runs in results/test_set (FORGE only). The rig-log
art and the physics-check figure need the local FORGE caches (cache/forge_edr.parquet from the
evidence loader, cache/tnd_forge.pkl from scripts/export_tnd.py); they are skipped if missing.

    .venv/bin/python slides/build_figures.py      ->  slides/figures/*.png
"""
import json
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

# One colour per system, checked with the dataviz validator against the dark slide surface (#120B20):
# lightness band, chroma, colour-blind separation and contrast all pass.
SYSTEMS = {"skeptic": ("Skeptic", "#BA8A00"), "react": ("ReAct, same tools", "#8F6CF0"), "fixed": ("Fixed pipeline, no LLM", "#109AAE")}
TEXT, MUTED, GRID, GOLD, FLAG = "#F3EEFB", "#A89CC8", "#2C2244", "#FDD023", "#FF6B5E"

plt.rcParams.update({"font.family": "Arial", "font.size": 15, "text.color": TEXT, "axes.edgecolor": MUTED,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "savefig.transparent": True, "legend.labelcolor": TEXT})


def test_rows():
    df = report.load_rows([str(d) for d in RUNS])
    return df[df.task.str.startswith("test-")]


def risk_coverage(df):
    """Error rate among the answers a system is most confident about (risk) against the share of
    questions it answers (coverage). Abstentions are never covered, so a curve can stop early."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for cfg, (name, color) in SYSTEMS.items():
        c = S.risk_coverage(df[df.config == cfg].to_dict("records"))
        x, y = [p["coverage"] for p in c], [p["risk"] for p in c]
        ax.plot(x, y, color=color, lw=2.4, solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", ms=10, color=color, mec="#120B20", mew=2)
        ax.annotate(f"{name}\nAURC {S.aurc(c):.3f}", (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=13, color=TEXT)
    ax.set_xlim(0, 1.36)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, None)
    ax.set_xlabel("Share of questions answered, most confident first")
    ax.set_ylabel("Share of those answers that are wrong")
    ax.spines["bottom"].set_bounds(0, 1)
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)


def headline(df):
    """Three measures as small multiples, one scale each."""
    t = report.table(df).set_index("config")
    panels = [("accuracy", "Accuracy"), ("confident_wrong", "Wrong while ≥ 70% sure"), ("abstain", "Abstained")]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.8), sharey=True)
    order = list(SYSTEMS)[::-1]
    for ax, (col, title) in zip(axes, panels):
        vals = [t.loc[c, col] for c in order]
        ax.barh(range(len(order)), vals, height=0.6, color=[SYSTEMS[c][1] for c in order])
        top = max(vals) * 1.38
        for i, v in enumerate(vals):
            ax.text(v + top * 0.02, i, f"{v:.2f}" if col == "accuracy" else f"{v:.0%}", va="center", fontsize=13, color=TEXT)
        ax.set_xlim(0, top)
        ax.set_title(title, fontsize=13, color=TEXT, loc="left", fontweight="bold")
        ax.grid(False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([SYSTEMS[c][0].replace(", ", "\n") for c in order], fontsize=12, color=TEXT)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "headline.png")
    plt.close(fig)


def depth_art():
    """Title-slide art from the real FORGE 16A rig log: bit depth (every trip in and out) and hole depth over the
    72 days, depth downward. No axes: it is a picture of the data, captioned on the slide."""
    import pandas as pd
    e = pd.read_parquet(config.CACHE / "forge_edr.parquet", columns=["t", "bit_depth_ft", "hole_depth_ft"])
    e = e.set_index("t").resample("5min").median().dropna()
    fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.plot(e.index, e.bit_depth_ft, color="#8F6CF0", lw=0.45, alpha=0.55)
    for lw, a in ((9, 0.06), (5, 0.12), (2.2, 0.9)):                 # soft glow under the hole-depth line
        ax.plot(e.index, e.hole_depth_ft.cummax(), color=GOLD, lw=lw, alpha=a, solid_capstyle="round")
    ax.invert_yaxis()
    ax.axis("off")
    fig.savefig(OUT / "depth_art.png", pad_inches=0)
    plt.close(fig)


def physics_check():
    """FORGE pick-ups against the friction envelope: a deep trip out that friction explains, and the 27-28 Dec
    logging trips, one of which weighs less than the frictionless string."""
    from skeptic.physics import tnd
    ds = tnd.dataset("forge")
    ev, th = ds["ev"], ds["base_theta"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
    cases = [("13 Dec: inside the envelope", "2020-12-12", "2020-12-14", [73], (100, 420)),
             ("27-28 Dec: lighter than zero friction", "2020-12-26", "2020-12-29", [133, 136], (60, 300))]
    for ax, (title, t0, t1, trips, ylim) in zip(axes, cases):
        idx = tnd.select(ds, t0, t1)
        flagged = {(r["trip"], r["time"]) for r in tnd.physics_check(ds, idx)["worst_readings"]}
        for c in trips:
            i = idx[ev.cluster.values[idx] == c]
            i = i[np.argsort(ev.bit.values[i])]
            bit, y = ev.bit.values[i], ev.hook.values[i]
            free = tnd.predict(ds, dict(th, c_s=0.0, mu_cased=0.0, mu_open=0.0), i)
            most = tnd.predict(ds, dict(th, c_s=0.0, mu_cased=tnd.MU_MAX, mu_open=tnd.MU_MAX), i)
            ax.fill_between(bit, free, most, color="#8F6CF0", alpha=0.16, lw=0)
            ax.plot(bit, free, color=MUTED, lw=2, ls="--")
            ax.plot(bit, most, color=MUTED, lw=2, ls=":")
            out = np.array([(c, str(t)[:16].replace(" ", "T")) in flagged for t in ev.t.values[i]])
            ax.plot(bit[~out], y[~out], "o", ms=7, color=GOLD, mec="#120B20", mew=0.8, ls="none")
            if out.any():
                ax.plot(bit[out], y[out], "o", ms=10, mfc="none", mec=FLAG, mew=2.4, ls="none")
        ax.set_ylim(*ylim)                       # the friction-0.8 edge runs off the top in the deep hole
        ax.set_title(title, fontsize=15, color=TEXT, loc="left", fontweight="bold")
        ax.set_xlabel("Bit depth (ft)")
    axes[0].set_ylabel("Pick-up hook load (klb)")
    h = [plt.Line2D([], [], color=MUTED, lw=2, ls="--"), plt.Line2D([], [], color=MUTED, lw=2, ls=":"),
         plt.Line2D([], [], marker="o", ms=7, color=GOLD, ls="none"),
         plt.Line2D([], [], marker="o", ms=10, mfc="none", mec=FLAG, mew=2.4, ls="none")]
    fig.legend(h, ["zero friction", f"friction {tnd.MU_MAX}", "reading", "flagged: no friction explains it"],
               loc="lower center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(OUT / "physics_check.png")
    plt.close(fig)


def friction_windows():
    """The physics data check and the identifiability advice on the 7 friction windows of the test set
    (post-test addition, PLAN.md v4) -> results/v4_friction_windows.json for the deck."""
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
    if (config.CACHE / "forge_edr.parquet").exists():
        depth_art()
    if (config.CACHE / "tnd_forge.pkl").exists():
        physics_check()
        friction_windows()
    print("figures in", OUT)
