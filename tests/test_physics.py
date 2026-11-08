"""Phase 2 gate: physics against closed-form cases and against the author's PETE
results; each QC check fires on a synthetic fault and stays quiet on clean data."""
import math

import numpy as np
import pandas as pd
import pytest

from skeptic import config
from skeptic.physics import hydraulics as hy, qc, tnd, units

has_tnd = lambda w: (config.CACHE / f"tnd_{w}.pkl").exists()


# ---------------------------------------------------------------- soft string
def straight_geo(n, dl=20.0, inc_deg=0.0):
    c = math.cos(math.radians(inc_deg))
    g = np.array([0, 0, 1.0])
    t = np.array([math.sin(math.radians(inc_deg)), 0, c])
    gp = g - c * t
    return dict(dl=dl, cos=np.full(n, c), uu=np.zeros(n), uv=np.zeros(n), vv=np.full(n, gp @ gp), s=np.arange(n) * dl)


def test_vertical_string_weighs_its_buoyed_weight_and_has_no_drag():
    geo = straight_geo(100)
    L = np.full((1, 100), 20.0)
    wL = np.full((1, 100), 20.0 * 18.0)          # 18 lb/ft buoyed
    for sgn in (1.0, -1.0, 0.0):
        F = tnd.softstring(L, wL, np.full((1, 100), 0.3), np.array([sgn]), geo)
        assert F[0] == pytest.approx(2000 * 18.0, rel=1e-9)


def test_inclined_straight_hole_matches_incline_formula():
    """Straight hole at inclination I: F = W (cos I +/- mu sin I)."""
    geo = straight_geo(50, inc_deg=60.0)
    L, wL = np.full((2, 50), 20.0), np.full((2, 50), 20.0 * 25.0)
    F = tnd.softstring(L, wL, np.full((2, 50), 0.3), np.array([1.0, -1.0]), geo)
    W = 1000 * 25.0
    I = math.radians(60)
    assert F[0] == pytest.approx(W * (math.cos(I) + 0.3 * math.sin(I)), rel=1e-6)
    assert F[1] == pytest.approx(W * (math.cos(I) - 0.3 * math.sin(I)), rel=1e-6)


# ----------------------------------------------------------------- hydraulics
def test_pump_output_triplex():
    p = hy.pump_output(6.0, 12.0, 95, 95)           # FORGE BOMCO 1600: report lists 4.19 gal/stk, ~400 gpm
    assert p["gal_per_stroke"] == pytest.approx(4.19, abs=0.01)
    assert p["gpm"] == pytest.approx(398, abs=1)


def test_tfa_from_nozzle_sizes():
    assert hy.tfa_from_nozzles("9x13") == pytest.approx(1.167, abs=0.001)
    assert hy.tfa_from_nozzles("9x15") == pytest.approx(1.553, abs=0.001)
    assert hy.tfa_from_nozzles("1x24, 3x20") == pytest.approx(1.362, abs=0.001)
    assert hy.tfa_from_nozzles("1X22 3X20") == pytest.approx(1.292, abs=0.001)


def test_bit_pressure_drop_formula():
    # 8.311e-5 * 12 * 400^2 / (0.95^2 * 0.5^2) = 159.57 / 0.225625 = 707.2 psi
    assert hy.bit_pressure_drop(12.0, 400.0, 0.5) == pytest.approx(707.2, abs=0.2)
    # plugging nozzles raises the drop; 4x12 -> plug 2 -> TFA halves -> dP x4
    r = hy.nozzles_to_explain(0, 10.0, 500, "4x12")
    assert r["scenarios"][1]["tfa_in2"] == pytest.approx(r["tfa_in2"] / 2, abs=1e-3)
    assert r["scenarios"][1]["spp_rise_psi"] == pytest.approx(3 * r["base_bit_dp_psi"], rel=0.01)


def test_annulus_regime_switches():
    lo = hy.annulus_loss(100, 10, 20, 10, 12.25, 5.5, 1000)
    hi = hy.annulus_loss(1500, 10, 20, 10, 8.75, 6.5, 1000)
    assert lo["regime"] == "laminar" and hi["regime"] == "turbulent"
    assert hi["dp_psi"] > lo["dp_psi"] > 0


# ---------------------------------------------------------------------- units
def test_undeclared_hook_load_units():
    assert units.read_hook_load(93)[0] == 93
    assert units.read_hook_load(93000)[0] == 93
    assert units.read_hook_load(220000, "lb")[0] == 220
    assert units.read_mud_weight(1.44)[0] == pytest.approx(12.02, abs=0.01)
    assert units.read_mud_weight(35)[0] is None
    with pytest.raises(ValueError):
        units.convert(1, "ppg", "furlongs")


# ------------------------------------------------------------------------- QC
def edr_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-12-01", periods=n, freq="1min")
    hook = 150 + 30 * np.sin(np.arange(n) / 37.0) + rng.normal(0, 1.5, n)
    return pd.DataFrame(dict(t=t, hookload_klb=hook, spp_psi=2500 + rng.normal(0, 20, n),
                             hole_depth_ft=np.linspace(8000, 8300, n), in_slips=(np.arange(n) % 30 < 3).astype(float),
                             on_bottom=(np.arange(n) % 200 < 120).astype(float), flow_gpm=550.0))


def test_stuck_channel_fires_only_on_frozen_values():
    e = edr_frame()
    assert qc.stuck_channel(e, "spp_psi")["verdict"] == "not stuck"
    spp = e.spp_psi.values.copy()
    spp[500:800] = spp[500]
    e2 = e.assign(spp_psi=spp)
    r = qc.stuck_channel(e2, "spp_psi")
    assert r["verdict"] == "stuck" and r["flat_runs"][0]["minutes"] >= 290


def test_clock_offset_recovers_injected_lag():
    a = edr_frame(seed=1)
    b = a.assign(t=a.t + pd.Timedelta("47min"))
    r = qc.clock_offset_series(a, b, "hookload_klb", a.t.iloc[0], a.t.iloc[-1], max_lag_min=120)
    assert r["best_lag_min"] == 47 and r["verdict"] == "offset"
    r0 = qc.clock_offset_series(a, a, "hookload_klb", a.t.iloc[0], a.t.iloc[-1], max_lag_min=60)
    assert r0["best_lag_min"] == 0 and r0["verdict"] == "aligned"


def test_depth_jumps():
    e = edr_frame()
    assert qc.depth_jumps(e)["verdict"] == "no jumps"
    h = e.hole_depth_ft.values.copy()
    h[1000:] += 93.0                                  # one stand
    r = qc.depth_jumps(e.assign(hole_depth_ft=h))
    assert r["verdict"] == "jumps found" and r["events"][0]["change_ft"] == pytest.approx(93.15, abs=0.5)


def test_block_weight_drift():
    n = 8 * 1440
    e = edr_frame(n)
    hook = np.where(e.in_slips > 0.5, 45 + np.random.default_rng(2).normal(0, 0.5, n), e.hookload_klb)
    assert qc.block_weight_trend(e.assign(hookload_klb=hook))["verdict"] == "stable"
    drift = hook + np.where(e.in_slips > 0.5, np.arange(n) / 1440.0 * 1.5, 0)     # +1.5 klb/day
    r = qc.block_weight_trend(e.assign(hookload_klb=drift))
    assert r["verdict"] == "drift" and r["slope_klb_per_day"] == pytest.approx(1.5, abs=0.2)


def test_survey_check_catches_transposed_digits():
    rows = []
    for k, (md, inc, az) in enumerate([(1000, 10.0, 45.0), (1100, 12.0, 46.0), (1200, 14.0, 47.0), (1300, 16.0, 48.0)]):
        for src in ("a", "b"):
            rows += [dict(quantity="survey_inclination", depth=md, value=inc, source=src, locator=f"{src}{k}"),
                     dict(quantity="survey_azimuth", depth=md, value=az, source=src, locator=f"{src}{k}")]
    df = pd.DataFrame(rows)
    assert qc.survey_check(df)["verdict"] == "consistent"
    df.loc[(df.locator == "b2") & (df.quantity == "survey_inclination"), "value"] = 41.0     # 14 -> 41
    r = qc.survey_check(df)
    assert r["verdict"] == "inconsistent" and r["cross_source_disagreements"][0]["md"] == 1200


def test_continuity_and_tfa():
    assert qc.continuity([8.8, 8.8, 9.8, 8.9, 8.9], list(range(5)), 0.5)["isolated_outliers"][0]["value"] == 9.8
    assert qc.continuity([8.8, 8.9, 9.0, 9.1], list(range(4)), 0.5)["verdict"] == "continuous"
    assert qc.tfa_consistency("4x12, 4x11", 0.910)["verdict"] == "inconsistent"      # FORGE bit 9: nozzles give 0.813
    assert qc.tfa_consistency("3x20", 0.920)["verdict"] == "consistent"




@pytest.mark.skipif(not has_tnd("forge"), reason="run scripts/export_tnd.py")
def test_forge_open_hole_friction_is_identifiable():
    forge = tnd.dataset("forge")
    rows = tnd.track_open_friction(forge, t0="2020-12-13", t1="2020-12-20")
    deep = [r for r in rows if r["state"] == "POOH" and r["sens_klb_per_mu"] > 100]
    assert deep and all(0.30 <= r["mu_open"] <= 0.40 and r["se"] < 0.05 for r in deep)


@pytest.mark.skipif(not config.forge_available(), reason="FORGE data not found")
def test_flat_depth_channels_are_explained_by_physics():
    """Bit depth flat in slips and hole depth flat off bottom are normal; hole depth frozen while drilling is not."""
    from skeptic.bench import inject
    from skeptic.evidence import workspace
    ws = workspace.load("forge")
    e = ws.edr["edr"]
    bit = qc.stuck_channel(e, "bit_depth_ft", "2020-11-10 06:00", "2020-11-11 06:00")
    assert bit["verdict"] == "not stuck" and bit["n_explained"] >= 3 and all(r["expected"] for r in bit["flat_runs"])
    hole = qc.stuck_channel(e, "hole_depth_ft", "2020-11-27 06:00", "2020-11-28 06:00")
    assert hole["verdict"] == "not stuck" and hole["n_explained"] >= 1
    frozen = inject.apply(ws, dict(type="stuck_channel", feed="edr", channel="hole_depth_ft", start="2020-11-20 08:00", end="2020-11-20 11:00"))
    assert qc.stuck_channel(frozen.edr["edr"], "hole_depth_ft", "2020-11-20 06:00", "2020-11-21 06:00")["verdict"] == "stuck"


@pytest.mark.skipif(not has_tnd("forge"), reason="run scripts/export_tnd.py")
def test_identifiability_needs_consistent_physical_trips():
    """Deep trips 13-20 Dec agree (identifiable, ~0.35); logging trips 26-29 Dec give negative mu (not identifiable);
    shallow trips 24-27 Nov barely see open hole (not identifiable)."""
    ds = tnd.dataset("forge")
    ok = tnd.identifiability(ds, "mu_open", tnd.select(ds, "2020-12-13", "2020-12-20"), t0="2020-12-13", t1="2020-12-20")
    assert ok["verdict"] == "identifiable" and abs(ok["estimate"] - 0.35) < 0.03
    bad = tnd.identifiability(ds, "mu_open", tnd.select(ds, "2020-12-26", "2020-12-29"), t0="2020-12-26", t1="2020-12-29")
    assert bad["verdict"] == "not identifiable" and any("physical range" in r for r in bad["reasons"])
    weak = tnd.identifiability(ds, "mu_open", tnd.select(ds, "2020-11-24", "2020-11-27"), t0="2020-11-24", t1="2020-11-27")
    assert weak["verdict"] == "not identifiable" and "sensitivity" in weak["reasons"][0]
