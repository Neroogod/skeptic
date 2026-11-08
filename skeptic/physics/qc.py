"""Data-quality checks on rig time series and reported facts. Each check returns a
plain dict that says what was compared, the numbers, and a verdict, so the agent can
cite it. None of these decide anything on their own: they produce evidence."""
import numpy as np
import pandas as pd

from .hydraulics import tfa_from_nozzles


def _window(df, t0=None, t1=None):
    m = pd.Series(True, index=df.index)
    if t0 is not None:
        m &= df.t >= pd.Timestamp(t0)
    if t1 is not None:
        m &= df.t <= pd.Timestamp(t1)
    return df[m]


# ------------------------------------------------------------------ stuck channel
def _depth_activity(x, channel):
    """Should a depth channel be moving? Hole depth changes only while drilling (on bottom
    with flow); bit depth follows the block only out of slips. Returns (active, why)."""
    if channel == "hole_depth_ft" and {"on_bottom", "flow_gpm"} <= set(x):
        drilling = float(((x.on_bottom.fillna(0) > 0.5) & (x.flow_gpm.fillna(0) > 50)).mean())
        return drilling >= 0.2, f"drilling (on bottom with flow) {drilling:.0%} of the run; hole depth must grow while drilling"
    if channel == "bit_depth_ft" and {"in_slips", "block_ft"} <= set(x):
        free = x[x.in_slips.fillna(1) < 0.5]
        moving = len(free) >= 3 and float(free.block_ft.std()) > 1.0
        return moving, (f"out of slips {len(free) / max(len(x), 1):.0%} of the run, block std there "
                        f"{float(free.block_ft.std()) if len(free) > 2 else 0:.1f} ft; in slips the bit cannot move")
    return None, None


def stuck_channel(df, channel, t0=None, t1=None, min_minutes=60, tol=1e-9, reference=None):
    """Runs where `channel` repeats exactly while the rig is active (the reference channel
    moves). A healthy sensor always shows noise, so a flat line during activity means a
    frozen value. Flat zeros are skipped: pumps off (SPP = 0) or no reading is physical.
    Depth channels use physics instead of a reference channel: a flat hole depth off bottom,
    or a flat bit depth in slips, is expected and reported as explained, not flagged."""
    reference = reference or ("block_ft" if channel == "hookload_klb" else "hookload_klb")
    depth = channel in ("hole_depth_ft", "bit_depth_ft")
    extra = [c for c in ("on_bottom", "flow_gpm", "in_slips", "block_ft") if depth and c in df and c != channel]
    cols = ["t", channel] + ([reference] if reference in df and reference != channel and reference not in extra else []) + extra
    x = _window(df, t0, t1)[cols].dropna(subset=[channel])
    if len(x) < 3:
        return dict(check="stuck_channel", channel=channel, verdict="insufficient data", n=len(x))
    v = x[channel].values
    same = np.r_[False, np.abs(np.diff(v)) <= tol]
    run_id = np.cumsum(~same)
    runs = []
    for rid in np.unique(run_id):
        idx = np.where(run_id == rid)[0]
        if len(idx) < 3 or abs(v[idx[0]]) < 1e-6:
            continue
        a, b = x.t.iloc[idx[0]], x.t.iloc[idx[-1]]
        minutes = (b - a).total_seconds() / 60.0
        if minutes < min_minutes:
            continue
        if depth:
            should_move, why = _depth_activity(x.iloc[idx], channel)
            if should_move is not None:
                runs.append(dict(start=str(a), end=str(b), minutes=round(minutes), value=float(v[idx[0]]),
                                 expected=not should_move, physics=why))
                continue
        active = None
        if reference in x and reference != channel:
            r = x[reference].iloc[idx]
            active = float(r.std()) if r.notna().sum() > 2 else None
        runs.append(dict(start=str(a), end=str(b), minutes=round(minutes), value=float(v[idx[0]]),
                         reference_std_during=None if active is None else round(active, 2),
                         expected=active is not None and active <= 1.0))
    flagged = [r for r in runs if not r["expected"]]
    return dict(check="stuck_channel", channel=channel, window=[str(x.t.iloc[0]), str(x.t.iloc[-1])],
                n_rows=len(x), channel_std=round(float(np.nanstd(v)), 4), flat_runs=runs[:10],
                n_flagged=len(flagged), n_explained=len(runs) - len(flagged),
                verdict="stuck" if flagged else "not stuck",
                note="expected=true: the flat run has a physical explanation (rig idle, off bottom, or in slips)")


# ------------------------------------------------------------------ clock offset
def best_lag(a, b, max_lag_min=240, step_min=1):
    """Lag (minutes) that best aligns series b to series a (both on a 1-min grid, indexed
    by time): b(t + lag) ~ a(t). Score = correlation of the overlapping parts."""
    a = a.astype(float)
    b = b.astype(float)
    best = (0, -2.0)
    scores = {}
    for lag in range(-max_lag_min, max_lag_min + 1, step_min):
        bs = b.shift(-lag, freq="min")
        j = pd.concat([a, bs], axis=1, join="inner").dropna()
        if len(j) < 30 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
            continue
        c = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
        scores[lag] = c
        if c > best[1]:
            best = (lag, c)
    return best, scores


def clock_offset_activity(edr, activity, t0, t1, max_lag_min=240):
    """Compare a report's activity log with the rig record: 'drilling' minutes from the
    report (1/0) against on-bottom-with-flow minutes in the EDR. Returns the lag that
    maximises agreement. activity: list of (start, end, is_drilling)."""
    x = _window(edr, t0, t1).set_index("t")
    drill = ((x.on_bottom.fillna(0) > 0.5) & (x.get("flow_gpm", pd.Series(1, index=x.index)).fillna(0) > 50)).astype(float)
    a = drill.resample("1min").mean().fillna(0)
    rep = pd.Series(0.0, index=a.index)
    for s, e, is_drill in activity:
        if is_drill:
            rep[(rep.index >= pd.Timestamp(s)) & (rep.index < pd.Timestamp(e))] = 1.0
    (lag, c), scores = best_lag(rep, a, max_lag_min)
    if not scores:                                    # no drilling in the report or the rig record: nothing to align
        return dict(check="clock_offset", compared="report activity log vs EDR on-bottom drilling", window=[str(t0), str(t1)],
                    verdict="insufficient data", note="no drilling intervals to compare in this window (report or rig)")
    zero = scores.get(0)
    near = [k for k, s in scores.items() if s >= c - 0.02]          # lags almost as good as the best one
    return dict(check="clock_offset", compared="report activity log vs EDR on-bottom drilling", window=[str(t0), str(t1)],
                best_lag_min=lag, corr_at_best=round(c, 3), corr_at_zero=None if zero is None else round(zero, 3),
                lag_range_min=[min(near), max(near)] if near else None,
                verdict="offset" if abs(lag) >= 15 and zero is not None and c - zero > 0.1 else "aligned",
                note="positive lag: EDR events happen later than the report says. lag_range_min: lags that fit almost as well "
                     "(report times are rounded, so the offset is only known to about this range)")


def clock_offset_series(a_df, b_df, channel, t0, t1, max_lag_min=240):
    """Two independent feeds of the same quantity (e.g. Pason vs mud-logger WITS hook load):
    lag of feed b relative to feed a."""
    a = _window(a_df, t0, t1).set_index("t")[channel].resample("1min").mean()
    b = _window(b_df, t0, t1).set_index("t")[channel].resample("1min").mean()
    (lag, c), scores = best_lag(a.interpolate(limit=5), b.interpolate(limit=5), max_lag_min)
    if not scores:
        return dict(check="clock_offset", compared=f"{channel}: feed A vs feed B", window=[str(t0), str(t1)],
                    verdict="insufficient data", note="the feeds do not overlap or do not vary in this window")
    zero = scores.get(0)
    return dict(check="clock_offset", compared=f"{channel}: feed A vs feed B", window=[str(t0), str(t1)],
                best_lag_min=lag, corr_at_best=round(c, 3), corr_at_zero=None if zero is None else round(zero, 3),
                verdict="offset" if abs(lag) >= 3 and zero is not None and c - zero > 0.05 else "aligned",
                note="positive lag: feed B is late (its clock runs behind)")


# ------------------------------------------------------------------ depth checks
def depth_vs_reports(edr, report_depths):
    """Reported hole depth at report time vs the EDR hole-depth channel at that time.
    report_depths: DataFrame with columns time, value, locator."""
    e = edr[["t", "hole_depth_ft"]].dropna()
    rows = []
    for r in report_depths.itertuples():
        k = np.searchsorted(e.t.values, np.datetime64(pd.Timestamp(r.time)))
        if k == 0 or k >= len(e):
            continue
        edr_v = float(e.hole_depth_ft.iloc[k - 1])
        rows.append(dict(time=str(r.time), reported_ft=float(r.value), edr_ft=round(edr_v, 1),
                         diff_ft=round(edr_v - float(r.value), 1), locator=r.locator))
    bad = [x for x in rows if abs(x["diff_ft"]) > 30]
    return dict(check="depth_vs_reports", n=len(rows), mismatches_over_30ft=bad[:15],
                median_abs_diff_ft=round(float(np.median([abs(x["diff_ft"]) for x in rows])), 1) if rows else None,
                verdict="mismatch" if bad else "consistent")


def depth_jumps(edr, t0=None, t1=None, min_jump_ft=30.0, min_drop_ft=10.0):
    """Hole depth can only grow by drilling (a few ft per 10-s sample) and never decreases.
    A step of tens of feet (e.g. one 93-ft stand) or a drop points to a depth reset or shift.
    Small corrections (+/- 2 ft, the depth tracker re-syncing) are ignored."""
    x = _window(edr, t0, t1)[["t", "hole_depth_ft"]].dropna()
    dt = x.t.diff().dt.total_seconds().values / 60.0
    dh = x.hole_depth_ft.diff().values
    events = []
    for i in np.where((dh < -min_drop_ft) | ((dh > min_jump_ft) & (dt <= 1.0)))[0]:
        events.append(dict(time=str(x.t.iloc[i]), from_ft=round(float(x.hole_depth_ft.iloc[i - 1]), 1),
                           to_ft=round(float(x.hole_depth_ft.iloc[i]), 1), change_ft=round(float(dh[i]), 1)))
    return dict(check="depth_jumps", n_rows=len(x), events=events[:15], n_events=len(events),
                verdict="jumps found" if events else "no jumps")


# ------------------------------------------------------------------ surveys
def dogleg_severity(md, inc_deg, az_deg):
    """Minimum-curvature dogleg severity between consecutive stations (deg/100 ft)."""
    i, a = np.radians(inc_deg), np.radians(az_deg)
    c = np.cos(i[1:] - i[:-1]) - np.sin(i[1:]) * np.sin(i[:-1]) * (1 - np.cos(a[1:] - a[:-1]))
    dl = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return dl / np.maximum(np.diff(md), 1e-9) * 100


def survey_check(facts_df, dls_limit=8.0, tol_inc=0.2, tol_az=2.0):
    """Cross-check survey stations: (1) the same MD reported by different sources must
    agree; (2) DLS spikes inside one source flag a typo."""
    inc = facts_df[facts_df.quantity == "survey_inclination"][["depth", "value", "source", "locator"]].rename(columns={"value": "inc"})
    az = facts_df[facts_df.quantity == "survey_azimuth"][["depth", "value", "locator"]].rename(columns={"value": "az"})
    s = inc.merge(az, on=["depth", "locator"]).astype({"depth": float, "inc": float, "az": float})
    disagreements = []
    for md, g in s.groupby(s.depth.round(0)):
        if g.source.nunique() < 2 and len(g) < 2:
            continue
        di, da = g.inc.max() - g.inc.min(), (g.az.max() - g.az.min())
        da = min(da, 360 - da)
        if di > tol_inc or (da > tol_az and g.inc.min() > 3):     # azimuth is noise at near-zero inclination
            disagreements.append(dict(md=md, values=g[["source", "inc", "az", "locator"]].to_dict("records")))
    spikes = []
    for src, g in s.drop_duplicates(["source", "depth"]).groupby("source"):
        g = g.sort_values("depth")
        if len(g) < 3:
            continue
        dls = dogleg_severity(g.depth.values, g.inc.values, g.az.values)
        for k in np.where(dls > dls_limit)[0]:
            spikes.append(dict(source=src, md_from=float(g.depth.iloc[k]), md_to=float(g.depth.iloc[k + 1]), dls=round(float(dls[k]), 1),
                               locator=g.locator.iloc[k + 1]))
    return dict(check="survey_check", stations=int(len(s)), cross_source_disagreements=disagreements[:10],
                dls_spikes=spikes[:10], verdict="inconsistent" if disagreements or spikes else "consistent")


# ------------------------------------------------------------------ sensor drift
def block_weight_trend(edr, t0=None, t1=None):
    """Hook load while the string is in slips = weight of the travelling block and top
    drive alone: a constant of the rig. A trend in it is load-cell drift, not physics."""
    x = _window(edr, t0, t1)
    s = x[(x.in_slips > 0.5)][["t", "hookload_klb"]].dropna()
    s = s[(s.hookload_klb > 5) & (s.hookload_klb < 120)]
    if len(s) < 50:
        return dict(check="block_weight_trend", verdict="insufficient in-slips data", n=len(s))
    daily = s.set_index("t").hookload_klb.resample("1D").median().dropna()
    days = (daily.index - daily.index[0]).days.values.astype(float)
    slope = float(np.polyfit(days, daily.values, 1)[0]) if len(daily) >= 3 else 0.0
    span = float(daily.max() - daily.min())
    return dict(check="block_weight_trend", n_in_slips_rows=len(s), daily_median_klb={str(k.date()): round(float(v), 1) for k, v in daily.items()},
                slope_klb_per_day=round(slope, 2), range_klb=round(span, 1),
                verdict="drift" if abs(slope) > 0.5 and span > 4 else "stable")


# ------------------------------------------------------------------ reported numbers
def tfa_consistency(nozzles, tfa_reported):
    tfa = tfa_from_nozzles(nozzles)
    return dict(check="tfa_consistency", nozzles=nozzles, tfa_from_nozzles=tfa, tfa_reported=tfa_reported,
                diff_in2=round(tfa_reported - tfa, 3), verdict="consistent" if abs(tfa_reported - tfa) < 0.02 else "inconsistent")


def continuity(values, times, max_step):
    """Isolated outliers in a reported series (e.g. one mud weight that differs from both
    neighbours by more than max_step while the neighbours agree)."""
    v = np.asarray(values, float)
    out = []
    for k in range(1, len(v) - 1):
        if abs(v[k] - v[k - 1]) > max_step and abs(v[k] - v[k + 1]) > max_step and abs(v[k - 1] - v[k + 1]) <= max_step:
            out.append(dict(time=str(times[k]), value=float(v[k]), neighbours=[float(v[k - 1]), float(v[k + 1])]))
    return dict(check="continuity", n=len(v), isolated_outliers=out, verdict="outliers" if out else "continuous")


def trip_tank_displacement(edr, t0, t1, cap_closed_bbl_ft=None, cap_open_bbl_ft=None):
    """Median trip-tank gain per foot of pipe run in, between consecutive slip settings.
    Closed-end pipe (float valve holding, pipe not filled) displaces its full outside
    volume; open-end pipe only its steel volume. (Method of the author's PETE tdcore.)"""
    x = _window(edr, t0, t1)
    if "trip_tank_bbl" not in x or x.trip_tank_bbl.notna().sum() < 20:
        return dict(check="trip_tank_displacement", verdict="no trip-tank record in this window")
    slip = x.in_slips > 0.5
    grp = (slip != slip.shift()).cumsum()
    s = x[slip].groupby(grp[slip]).agg(bit=("bit_depth_ft", "median"), tt=("trip_tank_bbl", "median"))
    dz, dv = s.bit.diff(), s.tt.diff()
    ok = (dz > 60) & (dz < 110) & (dv > -2)
    if ok.sum() < 3:
        return dict(check="trip_tank_displacement", verdict="fewer than 3 stands run in this window", stands=int(ok.sum()))
    val = float((dv[ok] / dz[ok]).median())
    out = dict(check="trip_tank_displacement", stands=int(ok.sum()), measured_bbl_per_ft=round(val, 4))
    if cap_closed_bbl_ft and cap_open_bbl_ft:
        out.update(closed_end_theory=cap_closed_bbl_ft, open_end_theory=cap_open_bbl_ft,
                   verdict="closed end (pipe running empty behind a float)" if abs(val - cap_closed_bbl_ft) < abs(val - cap_open_bbl_ft)
                   else "open end (pipe filling through the bit)")
    return out
