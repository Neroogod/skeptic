"""Torque-and-drag physics: soft-string hook-load model, fitting, identifiability.

Inputs come from cache/tnd_<well>.pkl (scripts/export_tnd.py, built with the author's
PETE 7103 pipeline): steady trip events and per-event element arrays
  L    pipe length in each 20-ft element (ft)
  Wb   buoyed BHA weight in the element (lb)
  Ldp  buoyed drill-pipe 'length' (ft x buoyancy factor): DP weight = w_dp * Ldp
  Wfill  weight of the mud missing from empty pipe above the float valve (lb, eta = 1)
  cased  1 if the element is inside casing

Measurement model (PETE stage 4, without the surge and indicator-bias terms):
  HL = W_block + F_surface(mu, w_dp, eta) / 1000 + sgn * c_s * |v| * D / 1000     [klb]
  c_s: closed-end surge/swab drag, linear in pipe speed v (ft/min) and bit depth D (kft)
  F_surface: soft-string (Johancsik 1984) tension at surface, solved by Picard
  iteration on the element equations (same discrete equations as PETE tdcore).

physics_check() and next_measurement() adapt the author's PETE 7103 stage P8 (physics-based
error control) and P6 (hole-condition monitor) to these arrays.
"""
import pickle
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from .. import config

STEEL_PPG = 65.5
PARAMS = ("W_block", "w_dp", "mu_cased", "mu_open", "eta", "c_s")
# Model ladder: which parameters are free (others fixed at FIXED). 'mu' ties cased = open.
MODELS = {
    "weight_only": ["W_block", "w_dp"],
    "single_mu": ["W_block", "w_dp", "mu"],               # PETE N1: the common practice
    "fill_only": ["W_block", "w_dp", "eta"],
    "mu_and_fill": ["W_block", "w_dp", "mu", "eta"],      # PETE F1
    "zone_mu_and_fill": ["W_block", "w_dp", "mu_cased", "mu_open", "eta"],
}
FIXED = dict(W_block=44.5, w_dp=22.0, mu_cased=0.0, mu_open=0.0, eta=0.0, c_s=0.0)
HUBER_KLB = 2.0


@lru_cache(maxsize=2)
def dataset(well):
    f = config.CACHE / f"tnd_{well}.pkl"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/export_tnd.py")
    return pickle.loads(f.read_bytes())


def softstring(L, wL, mu, sgn, geo, n_iter=6):
    """Surface tension (lb) for each event. Arrays (n_event, n_elem); sgn +1 pick-up
    (friction adds), -1 slack-off (friction subtracts), 0 rotating.
    Picard iteration: guess tension, compute the side force N in every element from it,
    add friction mu*N, re-integrate tension from the bit up; repeat (converges because
    mu * total dogleg << 1)."""
    dl = geo["dl"]
    cos, uu, uv, vv = geo["cos"], geo["uu"], geo["uv"], geo["vv"]
    base = wL * cos[None, :]                      # axial weight component per element
    inc = base
    sg = sgn[:, None]
    for _ in range(n_iter):
        Ftop = np.cumsum(inc[:, ::-1], axis=1)[:, ::-1]          # tension at element tops
        Fbot = np.concatenate([Ftop[:, 1:], np.zeros((len(L), 1))], axis=1)
        a = Fbot * (L / dl)
        N = np.sqrt(a * a * uu + 2 * a * wL * uv + wL * wL * vv + 1e-12)   # side force (lb)
        inc = base + sg * mu * N
    return np.cumsum(inc[:, ::-1], axis=1)[:, ::-1][:, 0]


def theta_dict(model, x):
    th = dict(FIXED)
    for name, v in zip(MODELS[model], x):
        if name == "mu":
            th["mu_cased"] = th["mu_open"] = v
        else:
            th[name] = v
    return th


def predict(ds, th, idx, mw=None):
    """Hook load (klb) for events idx under parameters th. mw (ppg, scalar or per event)
    overrides the mud weight used in the export: buoyed weights scale with (1 - mw/65.5)
    and the fill deficit with mw."""
    P = ds["P"]
    Wb, Ldp, Wfill = P["Wb"][idx], P["Ldp"][idx], P["Wfill"][idx]
    if mw is not None:
        mw0 = P["mw"][idx][:, None]
        mw1 = np.broadcast_to(np.asarray(mw, float), (len(idx),))[:, None]
        r = (1 - mw1 / STEEL_PPG) / (1 - mw0 / STEEL_PPG)
        Wb, Ldp, Wfill = Wb * r, Ldp * r, Wfill * mw1 / mw0
    wL = Wb + th["w_dp"] * Ldp - th["eta"] * Wfill
    mu = th["mu_cased"] * P["cased"][idx] + th["mu_open"] * (1 - P["cased"][idx])
    sgn = P["sgn"][idx].astype(float)
    F = softstring(P["L"][idx], wL, mu, sgn, ds["geo"])
    ev = ds["ev"]
    surge = sgn * th.get("c_s", 0.0) * ev.speed.abs().values[idx] * ev.bit.values[idx] / 1000.0
    return th["W_block"] + F / 1000.0 + surge / 1000.0


def select(ds, t0=None, t1=None, states=("POOH", "RIH"), known_fill_only=True):
    """Event indices: trips in the window; slack-offs only where the fill state is known."""
    ev = ds["ev"]
    m = ev.state.isin(states).values.copy()
    if t0 is not None:
        m &= (ev.t >= np.datetime64(t0)).values
    if t1 is not None:
        m &= (ev.t <= np.datetime64(t1)).values
    if known_fill_only and "fill_known" in ev:
        m &= ev.fill_known.values.astype(bool) | (ev.state != "RIH").values
    return np.where(m)[0]


def jacobian(f, x, h=1e-4):
    """Forward-difference Jacobian of the residual vector (n_event x n_param)."""
    f0 = f(x)
    J = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        x1 = np.array(x, float)
        step = h * max(1.0, abs(x1[j]))
        x1[j] += step
        J[:, j] = (f(x1) - f0) / step
    return J


def sandwich_cov(J, r, clusters, delta=HUBER_KLB):
    """Cluster-robust covariance of a Huber M-estimator (as in PETE stage 4).
    Events from the same trip share errors (same string, same hole, same day), so
    scores are summed within a trip before the outer product:
        bread = sum_i psi'(r_i) J_i^T J_i,   meat = sum_c g_c g_c^T,  g_c = sum_{i in c} psi(r_i) J_i
        cov   = bread^-1 meat bread^-1 * G/(G-1)."""
    psi = np.clip(r, -delta, delta)
    dpsi = (np.abs(r) <= delta).astype(float)
    bread = (J * dpsi[:, None]).T @ J
    meat = np.zeros_like(bread)
    groups = np.unique(clusters)
    for c in groups:
        g = (J[clusters == c] * psi[clusters == c, None]).sum(axis=0)
        meat += np.outer(g, g)
    B = np.linalg.pinv(bread)
    G = len(groups)
    return B @ meat @ B * (G / max(G - 1, 1))


def fit(ds, model, idx, mw=None):
    """Huber least squares (2 klb). Returns estimates, standard errors, the parameter
    correlation matrix, residuals by state, and physical-plausibility flags."""
    y = ds["ev"].hook.values[idx]
    names = MODELS[model]
    x0 = [ds["w_block0"] if n == "W_block" else ds["w_dp_nominal"] if n == "w_dp" else 1.0 if n == "eta" else 0.2 for n in names]
    resid = lambda x: predict(ds, theta_dict(model, x), idx, mw) - y
    res = least_squares(resid, x0, loss="huber", f_scale=HUBER_KLB, x_scale="jac")
    r = res.fun
    J = jacobian(resid, res.x)
    cov = sandwich_cov(J, r, ds["ev"].cluster.values[idx])
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(se, se).clip(1e-30)
    states = ds["ev"].state.values[idx]
    out = dict(model=model, n_events=int(len(idx)),
               params={n: dict(value=round(float(v), 4), se=round(float(e), 4), ci95=[round(float(v - 1.96 * e), 4), round(float(v + 1.96 * e), 4)])
                       for n, v, e in zip(names, res.x, se)},
               rmse_klb=round(float(np.sqrt(np.mean(r ** 2))), 3),
               residual_mean_by_state={s: round(float(r[states == s].mean()), 2) for s in np.unique(states)},
               correlations={f"{names[i]}~{names[j]}": round(float(corr[i, j]), 3)
                             for i in range(len(names)) for j in range(i + 1, len(names)) if abs(corr[i, j]) > 0.7})
    out["flags"] = plausibility(ds, out["params"])
    return out


def plausibility(ds, params):
    """Physics limits a fitted value must respect."""
    flags = []
    w = params.get("w_dp")
    if w and w["ci95"][1] < ds["w_dp_nominal"]:
        flags.append(f"w_dp = {w['value']} lb/ft is below the nominal {ds['w_dp_nominal']} lb/ft "
                     f"(tool joints only add weight): physically impossible, the model is misspecified")
    for k in ("mu", "mu_cased", "mu_open"):
        p = params.get(k)
        if p and (p["value"] > 0.6 or p["value"] < 0):
            flags.append(f"{k} = {p['value']} is outside the physical range 0-0.6 for steel on rock/casing in mud")
    e = params.get("eta")
    if e and not (0.5 <= e["value"] <= 1.5):
        flags.append(f"eta = {e['value']} far from 1 (hydrostatic fill deficit)")
    return flags


def sensitivity(ds, idx, th=None, h=0.01):
    """d(hook load)/d(mu) split by zone: how much of the friction signal comes from
    cased hole and how much from open hole, for these events (finite differences)."""
    th = dict(th or dict(FIXED, W_block=ds["w_block0"], w_dp=ds["w_dp_nominal"], mu_cased=0.25, mu_open=0.25, eta=1.0))
    base = predict(ds, th, idx)
    d = {}
    for k in ("mu_cased", "mu_open"):
        t2 = dict(th)
        t2[k] += h
        d[k] = (predict(ds, t2, idx) - base) / h                  # klb per unit mu, per event
    tot = np.abs(d["mu_cased"]).sum() + np.abs(d["mu_open"]).sum()
    return dict(n_events=int(len(idx)),
                dHL_dmu_cased_klb=round(float(np.abs(d["mu_cased"]).mean()), 2),
                dHL_dmu_open_klb=round(float(np.abs(d["mu_open"]).mean()), 2),
                share_open_hole=round(float(np.abs(d["mu_open"]).sum() / max(tot, 1e-12)), 3),
                share_cased_hole=round(float(np.abs(d["mu_cased"]).sum() / max(tot, 1e-12)), 3))


MIN_TRIP_SENS = 20.0      # klb of hook load per unit mu: below this a trip barely constrains mu_open
MU_RANGE = (0.0, 0.6)     # steel on rock or casing in mud


def identifiability(ds, parameter, idx, model="zone_mu_and_fill", tolerance=0.1, t0=None, t1=None):
    """Is `parameter` pinned down by these events? Three tests, all required for mu_open:

      precise     at least one trip whose hook loads are sensitive to mu_open (>= 20 klb per
                  unit mu), with the weight terms held at the well's global fit
      consistent  the trips agree: sensitivity-weighted spread of their estimates <= tolerance/2
                  (one friction factor must explain every trip; disagreement means the model
                  is missing something: tight hole, a different string, changing mud)
      physical    every informative trip gives 0 <= mu <= 0.6

    A narrow confidence interval alone is not enough: a mis-specified model can be precise
    and wrong. The free zone-model fit (all parameters fitted, 95% half-width) is reported
    as a second opinion. mu_cased has no trip-by-trip estimate, so it uses the zone fit alone."""
    fr = fit(ds, model, idx)
    names = MODELS[model]
    key = parameter if parameter in names else ("mu" if parameter.startswith("mu") and "mu" in names else None)
    if key is None:
        return dict(error=f"{parameter} is not a parameter of model {model}; choose from {names}")
    p = fr["params"][key]
    half = 1.96 * p["se"]
    zone = dict(model=model, estimate=p["value"], ci95=p["ci95"], half_width=round(half, 3), fit_flags=fr["flags"],
                strong_correlations=fr["correlations"], n_events=fr["n_events"])
    out = dict(parameter=parameter, tolerance=tolerance, sensitivity=sensitivity(ds, idx), zone_fit=zone)
    if parameter != "mu_open":
        ok = half <= tolerance and MU_RANGE[0] <= p["value"] <= MU_RANGE[1]
        return out | dict(estimate=p["value"], verdict="identifiable" if ok else "not identifiable",
                          reasons=[] if ok else [f"zone-fit 95% half-width {half:.3f} vs tolerance {tolerance}, estimate {p['value']}"])
    if t0 is None and len(idx):
        t0, t1 = str(ds["ev"].t.values[idx].min())[:19], str(ds["ev"].t.values[idx].max())[:19]
    trips = track_open_friction(ds, t0, t1)
    inf = [r for r in trips if r["sens_klb_per_mu"] >= MIN_TRIP_SENS]
    reasons = []
    pooled = spread = None
    if not inf:
        reasons.append(f"no trip with open-hole sensitivity >= {MIN_TRIP_SENS:g} klb per unit mu (too little open hole below the shoe)")
    else:
        w = np.array([r["sens_klb_per_mu"] ** 2 for r in inf])
        m = np.array([r["mu_open"] for r in inf])
        pooled = float(np.sum(w * m) / np.sum(w))
        spread = float(np.sqrt(np.sum(w * (m - pooled) ** 2) / np.sum(w)))
        bad = [r for r in inf if not MU_RANGE[0] <= r["mu_open"] <= MU_RANGE[1]]
        if bad:
            reasons.append(f"trip(s) {[r['trip'] for r in bad]} give mu_open {[r['mu_open'] for r in bad]}, outside the "
                           f"physical range {MU_RANGE}: the model does not describe these trips")
        if spread > tolerance / 2:
            reasons.append(f"trips disagree: weighted spread {spread:.3f} > {tolerance / 2:g} (estimates {sorted(m.round(3).tolist())})")
    return out | dict(estimate=None if pooled is None else round(pooled, 3), trip_spread=None if spread is None else round(spread, 3),
                      informative_trips=[dict(trip=r["trip"], state=r["state"], start=r["start"], mu_open=r["mu_open"],
                                              sens_klb_per_mu=r["sens_klb_per_mu"]) for r in inf],
                      n_weak_trips=len(trips) - len(inf), verdict="not identifiable" if reasons else "identifiable",
                      reasons=reasons, next_measurement=next_measurement(ds, reasons, inf),
                      method="trip-by-trip mu_open with weight terms held at the well's global fit")


def track_open_friction(ds, t0=None, t1=None, min_events=6, open_ft=None):
    """Open-hole friction factor trip by trip (method of the author's PETE forge_tracking):
    the weight terms are held at the well's global fit (ds['base_theta']) and only mu_open
    is fitted to each trip's events whose bit is below `open_ft` (default: 500 ft past the
    deepest shoe). Returns one row per trip with the estimate, its SE (within-trip M-estimator
    sandwich: a lower bound) and the sensitivity d(HL)/d(mu_open) in klb per unit mu."""
    from scipy.optimize import minimize_scalar
    th0 = dict(ds["base_theta"])
    ev = ds["ev"]
    open_ft = open_ft or float(np.max(ds["shoe_by_event"])) + 500.0
    m = ev.state.isin(["POOH", "RIH"]).values & (ev.bit.values > open_ft)
    if t0 is not None:
        m &= (ev.t >= np.datetime64(t0)).values
    if t1 is not None:
        m &= (ev.t <= np.datetime64(t1)).values
    rows = []
    for c in np.unique(ev.cluster.values[m]):
        idx = np.where(m & (ev.cluster.values == c))[0]
        if len(idx) < min_events:
            continue
        y = ev.hook.values[idx]
        hl = lambda mu: predict(ds, dict(th0, mu_open=mu), idx)
        loss = lambda mu: float(np.sum(np.where(np.abs(y - hl(mu)) <= HUBER_KLB, 0.5 * (y - hl(mu)) ** 2,
                                                HUBER_KLB * (np.abs(y - hl(mu)) - 0.5 * HUBER_KLB))))
        mu = minimize_scalar(loss, bounds=(-0.5, 2.0), method="bounded", options=dict(xatol=1e-4)).x
        r = y - hl(mu)
        J = (hl(mu + 1e-3) - hl(mu - 1e-3)) / 2e-3
        psi, dpsi = np.clip(r, -HUBER_KLB, HUBER_KLB), (np.abs(r) <= HUBER_KLB)
        A = float(np.sum(dpsi * J * J))
        rows.append(dict(trip=int(c), state=str(ev.state.values[idx[0]]), start=str(ev.t.values[idx[0]])[:16],
                         n=int(len(idx)), bit_ft=[round(float(ev.bit.values[idx].min())), round(float(ev.bit.values[idx].max()))],
                         mu_open=round(float(mu), 3), se=round(float(np.sqrt(np.sum((psi * J) ** 2)) / A), 3) if A > 0 else None,
                         rms_klb=round(float(np.sqrt(np.mean(r ** 2))), 2), sens_klb_per_mu=round(float(np.mean(np.abs(J))), 1)))
    return rows


@lru_cache(maxsize=2)
def visible_depth(well):
    """Shallowest bit depth (ft MD) at which any trip of the well constrains mu_open
    (sensitivity >= MIN_TRIP_SENS), or None if no trip does. Used to say which measurement
    would make friction identifiable in a window that has none."""
    rows = [r for r in track_open_friction(dataset(well)) if r["sens_klb_per_mu"] >= MIN_TRIP_SENS]
    return min(r["bit_ft"][1] for r in rows) if rows else None


def next_measurement(ds, reasons, informative):
    """What would settle a 'not identifiable' verdict (the author's PETE 7103 P8 idea, in words):
    more data only helps when the problem is too little friction signal. When the trips disagree
    or give impossible values, more of the same trips will not help; the records must be checked."""
    if not reasons:
        return None
    if not informative:
        d = visible_depth(ds["well"])
        if d is None:
            return ("no trip anywhere in this well moves the hook load enough for friction to show: the hole is too "
                    "close to vertical, so open-hole friction cannot be measured from hook loads here; it must come "
                    "from another well or a prior")
        return (f"a pick-up/slack-off series with the bit below about {d} ft MD (the shallowest trip in this well "
                f"whose hook loads move at least {MIN_TRIP_SENS:g} klb per unit mu) would make mu_open measurable")
    return ("more trips like these will not settle it: run check_hookload_physics on this window. Readings that no "
            "friction factor explains point to a string, fill-state or sensor record problem, not friction")


# ============================================================ physics data check
MU_MAX = 0.8                                        # largest friction the check allows (PETE P8)
K_GRID = np.round(np.arange(0.0, 2.51, 0.05), 2)    # trip friction multipliers k (PETE P6)
Z_EVENT, Z_TRIP = 3.0, 2.0                          # flag thresholds, in units of sigma_r
WHY = {
    "pick-up below zero-friction load": "lighter than the string would weigh with no friction at all: a hook-load "
                                        "offset or sensor fault, a lighter string than recorded, or a wrong bit depth",
    "slack-off above zero-friction load": "heavier than the string would weigh with no friction: more mud inside the "
                                          "pipe than the fill record says, a heavier string, or a sensor offset",
    "beyond the largest friction": f"would need friction above {MU_MAX}: overpull or sticking (tight hole), a heavier "
                                   "string than recorded, or a sensor fault",
}


def _robust_sigma(r):
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _trip_curves(ds, idx, th):
    """Hook loads of events idx for every multiplier k in K_GRID (mu = k x the global cased and
    open values), weight terms at the global fit, surge term left out. Row 0 (k = 0) is the
    frictionless string."""
    th = dict(th, c_s=0.0)
    return np.array([predict(ds, dict(th, mu_cased=k * th["mu_cased"], mu_open=k * th["mu_open"]), idx) for k in K_GRID])


def _best_k(y, curves, clusters):
    """Per trip: the friction multiplier that fits best, the fitted loads and the RMS misfit left."""
    k, fit, rms = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for c in np.unique(clusters):
        i = np.where(clusters == c)[0]
        sse = ((y[i][None, :] - curves[:, i]) ** 2).sum(axis=1)
        b = int(np.argmin(sse))
        k[i], fit[i], rms[i] = K_GRID[b], curves[b, i], np.sqrt(sse[b] / len(i))
    return k, fit, rms


@lru_cache(maxsize=2)
def noise_scale(well):
    """sigma_r (klb): robust scale (1.4826 x median absolute deviation) of the hook-load
    residuals left after each trip's best friction multiplier, over every trip of the well.
    It is the scatter that no friction value can absorb."""
    ds = dataset(well)
    idx = select(ds)
    y = ds["ev"].hook.values[idx]
    _, fit, _ = _best_k(y, _trip_curves(ds, idx, ds["base_theta"]), ds["ev"].cluster.values[idx])
    return _robust_sigma(y - fit)


def physics_check(ds, idx, sigma_r=None, max_rows=10):
    """Readings that no friction factor can explain (the author's PETE 7103 P8 data check, adapted).

    Whatever the friction, a pick-up cannot weigh less than the string with zero friction, a
    slack-off cannot weigh more, and no reading can need more friction than MU_MAX. The edges of
    this envelope come from the soft-string model with the well's global weight fit (W_block,
    w_dp, eta) and no surge term. A reading outside it by more than 3 sigma_r is flagged.

    A trip is flagged when no friction multiplier k brings its RMS misfit below 2 sigma_r and the
    physics can tell: the trip has at least 2 pick-ups and 2 slack-offs (friction moves them in
    opposite directions, a zero offset moves both the same way), or its best k is 0 (the whole
    trip is lighter than the frictionless string). An offset on a one-direction trip looks like
    a friction change, so it cannot be detected; passing the check does not prove the data right.

    PETE's second tier (a reading far from its own trip's curve) is left out: the friction profile
    here is two constants, not a fitted mu(depth), so that distance mixes profile error with
    data error."""
    ev, th = ds["ev"], ds["base_theta"]
    sigma_r = sigma_r or noise_scale(ds["well"])
    tol = Z_EVENT * sigma_r
    y = ev.hook.values[idx]
    pu = ds["P"]["sgn"][idx] > 0
    curves = _trip_curves(ds, idx, th)
    free = curves[0]
    most = predict(ds, dict(th, c_s=0.0, mu_cased=MU_MAX, mu_open=MU_MAX), idx)
    reason = np.where(pu & (y < free - tol), "pick-up below zero-friction load",
             np.where(~pu & (y > free + tol), "slack-off above zero-friction load",
             np.where((pu & (y > most + tol)) | (~pu & (y < most - tol)), "beyond the largest friction", "")))
    outside = np.where(reason == "beyond the largest friction", np.abs(y - most), np.abs(y - free))

    clusters = ev.cluster.values[idx]
    k, _, rms = _best_k(y, curves, clusters)
    trips = []
    for c in np.unique(clusters):
        i = np.where(clusters == c)[0]
        n_pu, n_so = int(pu[i].sum()), int((~pu[i]).sum())
        testable = (n_pu >= 2 and n_so >= 2) or k[i[0]] == 0
        trips.append(dict(trip=int(c), start=str(ev.t.values[idx[i[0]]])[:16], pickups=n_pu, slackoffs=n_so,
                          bit_ft=[round(float(ev.bit.values[idx[i]].min())), round(float(ev.bit.values[idx[i]].max()))],
                          best_k=float(k[i[0]]), rms_klb=round(float(rms[i[0]]), 1),
                          readings_outside=int((reason[i] != "").sum()),
                          flagged=bool(testable and rms[i[0]] > Z_TRIP * sigma_r)))
    bad = np.where(reason != "")[0]
    rows = [dict(time=str(ev.t.values[idx[j]])[:16], trip=int(clusters[j]), direction="pick-up" if pu[j] else "slack-off",
                 bit_ft=round(float(ev.bit.values[idx[j]])), hook_klb=round(float(y[j]), 1),
                 zero_friction_klb=round(float(free[j]), 1), max_friction_klb=round(float(most[j]), 1),
                 outside_by_klb=round(float(outside[j]), 1), reason=str(reason[j]))
            for j in bad[np.argsort(-outside[bad])][:max_rows]]
    flagged = [t for t in trips if t["flagged"]]
    return dict(verdict="readings no friction factor explains" if len(bad) or flagged else "consistent with friction",
                n_readings=int(len(idx)), n_trips=len(trips), sigma_r_klb=round(sigma_r, 2), tolerance_klb=round(tol, 1),
                readings_outside_envelope=int(len(bad)),
                by_reason={r: int((reason == r).sum()) for r in WHY if (reason == r).any()},
                what_it_means={r: WHY[r] for r in WHY if (reason == r).any()},
                worst_readings=rows,
                trips_with_problems=[t for t in trips if t["readings_outside"] or t["flagged"]],
                trips_flagged=[t["trip"] for t in flagged],
                global_fit={p: round(float(th[p]), 3) for p in ("W_block", "w_dp", "eta", "mu_cased", "mu_open")},
                limits="a zero offset on a trip that only picks up (or only slacks off) looks like a friction change and "
                       "is not detected; passing does not prove the readings right")
