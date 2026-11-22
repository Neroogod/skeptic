"""One-off export of the torque-and-drag inputs from the author's PETE 7103 pipeline
(Drilling_Project/pete_td) into skeptic/cache/tnd_<well>.pkl.

What is exported per well: steady events (pick-up / slack-off / rotating, with fill
state), the element-grid geometry, and the per-event element arrays the soft-string
solver needs. Skeptic's own solver (skeptic/physics/tnd.py) runs on these arrays, so
skeptic does not import pete_td at run time.

Run:  .venv/bin/python scripts/export_tnd.py
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PETE = Path(os.environ.get("SKEPTIC_PETE_SRC", ROOT.parent / "Drilling_Project" / "pete_td" / "src"))
os.environ.setdefault("PETE_TD_DATA", str(ROOT.parent / "Drilling_Project"))
sys.path.insert(0, str(PETE))

import json               # noqa: E402
import pandas as pd       # noqa: E402
import tdcore as tc       # noqa: E402


def base_theta(fits_file, *keys):
    """PETE's global fit (W_block, w_dp, mu_cased, mu_open, eta, c_s) used as the
    starting point for per-trip friction tracking."""
    f = json.load(open(Path(tc.RES_DIR) / fits_file))
    for k in keys:
        f = f[k]
    th = dict(zip(f["free"], f["x"]))
    if "mu" in th:
        th["mu_cased"] = th["mu_open"] = th.pop("mu")
    return {k: float(th.get(k, 0.0)) for k in ("W_block", "w_dp", "mu_cased", "mu_open", "eta", "c_s")}


def export_lsu():
    ev = pd.read_pickle(Path(tc.CACHE_DIR) / "events_fill.pkl")
    md, inc, az, _ = tc.load_survey()
    geo = tc.geometry(md, inc, az)
    bha, runs = tc.load_bha(), tc.load_runs()
    P = tc.prepare(ev, geo, bha, runs, tc.load_mud_checks(), tc.load_caliper())
    return dict(well="lsu", ev=ev, geo=geo, P={k: np.asarray(v) for k, v in P.items()},
                w_dp_nominal=21.9, w_block0=44.5, shoe_by_event=ev.run.map(tc.SHOE_BY_RUN).values,
                cap_gal_ft=tc.CAP_DP_GAL_FT, dp="5-1/2in 21.9 ppf, ID 4.778 in",
                base_theta=base_theta("stage4_fits.json", "F2S"),
                note="LSU H-001: LSU-only data")


def export_forge():
    import forge_data as fd
    ev, geo, by_run, P = fd.load_all()
    return dict(well="forge", ev=ev, geo=geo, P={k: np.asarray(v) for k, v in P.items()},
                w_dp_nominal=19.5, w_block0=41.0, shoe_by_event=np.full(len(ev), fd.SHOE_FT),
                cap_gal_ft=fd.CAP_GAL_FT, dp="5in 19.5 ppf, ID 4.276 in",
                base_theta=base_theta("forge_fits.json", "ALL", "F2S"),
                note="FORGE 16A(78)-32, DOI 10.15121/1776602 (CC BY 4.0)")


if __name__ == "__main__":
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from skeptic import config
    for f in (export_lsu, export_forge):
        if f is export_lsu and not config.lsu_available():
            print("LSU files not found (LSU-only data): skipping the LSU export")
            continue
        d = f()
        (out / f"tnd_{d['well']}.pkl").write_bytes(pickle.dumps(d))
        print(d["well"], len(d["ev"]), "events;", d["P"]["L"].shape, "element arrays;",
              d["ev"].state.value_counts().to_dict())
