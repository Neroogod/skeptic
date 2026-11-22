"""Tool registry. Every tool is a plain function f(ws, **args) -> dict plus a JSON schema.
The same functions serve the LLM agent, the fixed-pipeline baseline and the tests, so
all configurations see identical evidence.

Rules (enforced in call()):
  * arguments are validated against the schema before the function runs
  * any exception becomes {"error": "..."} returned to the model as text (it can retry)
  * outputs are truncated to a token budget so one call cannot flood the context
"""
import json
import re
import traceback

import numpy as np
import pandas as pd

from ..evidence.store import QUANTITIES
from ..physics import hydraulics as hy, qc, tnd, units

MAX_CHARS = 4000
TOOLS = {}


def tool(name, description, params, required=()):
    """Register f under `name` with a JSON-schema for its arguments."""
    def wrap(f):
        TOOLS[name] = dict(fn=f, schema=dict(type="function", function=dict(
            name=name, description=description,
            parameters=dict(type="object", properties=params, required=list(required)))))
        return f
    return wrap


# JSON-schema helpers for tool arguments: S = string, N = number, I = integer
S = lambda d: dict(type="string", description=d)
N = lambda d: dict(type="number", description=d)
I = lambda d: dict(type="integer", description=d)
TIME = "ISO time, e.g. 2020-12-14 06:00"


def _clean(x):
    """Make tool output JSON-friendly (numpy, Timestamps, NaN)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(x) else str(pd.Timestamp(x))
    if x is pd.NaT:
        return None
    return x


def validate(name, args):
    """Check a call against its schema: known tool, required arguments present, no unknown arguments,
    right types and enum values. Raises ValueError with a message the model can act on."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
    p = TOOLS[name]["schema"]["function"]["parameters"]
    missing = [r for r in p["required"] if r not in args or args[r] in (None, "")]
    if missing:
        raise ValueError(f"missing required argument(s) {missing}")
    extra = [k for k in args if k not in p["properties"]]
    if extra:
        raise ValueError(f"unknown argument(s) {extra}; allowed: {sorted(p['properties'])}")
    for k, v in args.items():
        t = p["properties"][k].get("type")
        if v is None:
            continue
        if t == "number" and not isinstance(v, (int, float)):
            raise ValueError(f"{k} must be a number, got {v!r}")
        if t == "integer" and not isinstance(v, int):
            raise ValueError(f"{k} must be an integer, got {v!r}")
        if t == "string" and not isinstance(v, str):
            raise ValueError(f"{k} must be a string, got {v!r}")
        if "enum" in p["properties"][k] and v not in p["properties"][k]["enum"]:
            raise ValueError(f"{k} must be one of {p['properties'][k]['enum']}, got {v!r}")


def call(ws, name, args, ctx=None):
    """Run one tool. Never raises: errors come back as {'error': ...}."""
    try:
        args = {k: v for k, v in (args or {}).items() if v is not None}
        validate(name, args)
        f = TOOLS[name]["fn"]
        out = f(ws, ctx, **args) if getattr(f, "needs_ctx", False) else f(ws, **args)
        out = _clean(out)
    except Exception as e:                     # noqa: BLE001 - the model gets the message
        out = {"error": f"{type(e).__name__}: {e}"}
        if not isinstance(e, (ValueError, KeyError)):
            out["trace"] = traceback.format_exc(limit=2)[-400:]
    text = json.dumps(out, default=str)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f' ... [truncated {len(text) - MAX_CHARS} chars; narrow the query]'
    return out, text


def schemas(names=None):
    return [TOOLS[n]["schema"] for n in (names or TOOLS)]


def _feed(ws, feed):
    if feed not in ws.edr:
        raise ValueError(f"feed {feed!r} not available for this well; feeds: {sorted(ws.edr)}")
    return ws.edr[feed]


def _channel(df, channel, feed="edr"):
    """Reject unknown channel names with the list of valid ones (so the model can retry)."""
    if channel not in df.columns or channel == "t":
        raise ValueError(f"no channel {channel!r} in feed {feed!r}; channels: {[c for c in df.columns if c != 't']}")
    return channel


def _win(df, start=None, end=None):
    m = pd.Series(True, index=df.index)
    if start:
        m &= df.t >= pd.Timestamp(start)
    if end:
        m &= df.t <= pd.Timestamp(end)
    return df[m]


# ============================================================ evidence tools
@tool("get_facts", "All values of one quantity recorded anywhere in the well's documents (daily reports, tour "
      "sheets, contractor workbooks, rig memos), with source, location, time and depth. Conflicting values are "
      "NOT resolved: every candidate is returned. Quantities: " + "; ".join(f"{k} = {v}" for k, v in QUANTITIES.items()),
      dict(quantity=dict(type="string", enum=list(QUANTITIES)), qualifier=S("substring filter, e.g. '20 in', 'bit 3', 'pump 2', 'narrative'"),
           start=S(TIME), end=S(TIME), depth_min=N("ft MD"), depth_max=N("ft MD"), source=S("substring filter on source name"),
           limit=I("max rows (default 40)")), ["quantity"])
def get_facts(ws, quantity, qualifier=None, start=None, end=None, depth_min=None, depth_max=None, source=None, limit=40):
    x = ws.facts.get(quantity, qualifier, start, end, depth_min, depth_max, source, limit=limit)
    rows = x[["id", "value", "unit", "source", "qualifier", "time", "depth", "locator", "raw"]].to_dict("records")
    for r in rows:
        r["raw"] = str(r["raw"])[:120]
    vals = pd.to_numeric(x.value, errors="coerce").dropna()
    summary = dict(n=len(x), distinct_values=sorted(set(map(str, x.value)))[:20], sources=sorted(x.source.unique()))
    if len(vals):
        summary.update(min=float(vals.min()), max=float(vals.max()))
    return dict(summary=summary, candidates=rows)


@tool("search_text", "BM25 search over free text: daily-report time logs, rig memos, tour-sheet activity lines, "
      "contractor daily summaries and the end-of-well report. Returns the best chunks (truncated; use read_text for full text).",
      dict(query=S("keywords"), start=S(TIME), end=S(TIME), source=S("substring filter on source"), k=I("number of hits (default 6)")), ["query"])
def search_text(ws, query, start=None, end=None, source=None, k=6):
    hits = ws.text.search(query, k=min(k, 15), t0=start, t1=end, source=source)
    return dict(hits=[dict(id=c.id, score=s, source=c.source, time=c.time, locator=c.locator, text=c.text[:350]) for s, c in hits])


@tool("read_text", "Full text of one chunk returned by search_text.", dict(chunk_id=S("e.g. T00123")), ["chunk_id"])
def read_text(ws, chunk_id):
    c = ws.text.get(chunk_id)
    return dict(id=c.id, source=c.source, time=c.time, locator=c.locator, text=c.text[:4000])


@tool("list_channels", "Rig time-series feeds and channels available for this well, with time range and % valid.",
      dict(feed=S("'edr' (main rig record) or 'wits' (LSU mud-logger feed)")))
def list_channels(ws, feed=None):
    out = {}
    for name, df in ws.edr.items():
        if feed and name != feed:
            continue
        out[name] = dict(start=df.t.min(), end=df.t.max(), rows=len(df),
                         channels={c: round(100 * float(df[c].notna().mean()), 1) for c in df.columns if c != "t"})
    return out


@tool("edr_stats", "Summary statistics of one rig channel over a time window, optionally only while on bottom / in slips / "
      "off bottom. Includes the longest run of identical consecutive values (minutes).",
      dict(channel=S("canonical channel name, see list_channels"), start=S(TIME), end=S(TIME), feed=S("edr (default) or wits"),
           condition=dict(type="string", enum=["all", "on_bottom", "off_bottom", "in_slips", "out_of_slips"])), ["channel", "start", "end"])
def edr_stats(ws, channel, start, end, feed="edr", condition="all"):
    df = _win(_feed(ws, feed), start, end)
    _channel(df, channel, feed)
    cond = {"on_bottom": df.get("on_bottom", 0) > 0.5, "off_bottom": df.get("on_bottom", 1) < 0.5,
            "in_slips": df.get("in_slips", 0) > 0.5, "out_of_slips": df.get("in_slips", 1) < 0.5}.get(condition)
    if cond is not None:
        df = df[cond]
    v = df[channel].dropna()
    if not len(v):
        return dict(channel=channel, n=0, note="no valid values in this window/condition")
    same = np.r_[False, np.abs(np.diff(v.values)) < 1e-9]
    run, best = 0, 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    step_s = float(df.t.diff().dt.total_seconds().median() or 10)
    return dict(channel=channel, feed=feed, condition=condition, n=int(len(v)), missing_pct=round(100 * (1 - len(v) / max(len(df), 1)), 1),
                mean=float(v.mean()), median=float(v.median()), std=float(v.std()), min=float(v.min()), max=float(v.max()),
                p05=float(v.quantile(.05)), p95=float(v.quantile(.95)), longest_identical_run_min=round(best * step_s / 60, 1))


@tool("edr_values", "Rig channels resampled (median) on a coarse grid over a window: at most 48 rows.",
      dict(channels=S("comma-separated channel names"), start=S(TIME), end=S(TIME), every_min=I("grid step in minutes (default auto)"),
           feed=S("edr (default) or wits")), ["channels", "start", "end"])
def edr_values(ws, channels, start, end, every_min=None, feed="edr"):
    df = _win(_feed(ws, feed), start, end)
    cols = [c.strip() for c in channels.split(",") if c.strip()]
    for c in cols:
        _channel(df, c, feed)
    span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60
    every = every_min or max(1, int(np.ceil(span / 48)))
    r = df.set_index("t")[cols].resample(f"{every}min").median().head(48)
    rows = []
    for t, row in r.iterrows():
        rows.append(dict(t=str(t), **{c: None if pd.isna(v) else round(float(v), 2) for c, v in row.items()}))
    return dict(every_min=every, rows=rows)


# ============================================================ data-quality checks
@tool("check_stuck_channel", "Find stretches where a rig channel repeats exactly while the rig is active (hook load moving): "
      "a frozen sensor or a stuck value.",
      dict(channel=S("channel"), start=S(TIME), end=S(TIME), feed=S("edr or wits"), min_minutes=I("shortest flat run reported (default 60)")),
      ["channel", "start", "end"])
def check_stuck_channel(ws, channel, start, end, feed="edr", min_minutes=60):
    df = _feed(ws, feed)
    return qc.stuck_channel(df, _channel(df, channel, feed), start, end, min_minutes=min_minutes)


def report_activity(ws, start, end):
    """Drilling intervals from the report time logs (FORGE daily reports / LSU tour log)."""
    out = []
    for c in ws.text.chunks:
        if c.time is None or pd.isna(c.time) or not (pd.Timestamp(start) - pd.Timedelta("1D") <= c.time <= pd.Timestamp(end)):
            continue
        if ws.well == "forge" and c.source == "forge_daily_report":
            m = re.match(r"\[[\d-]+ [\d:]+-[\d:]+, ([\d.]+) h\] (.*)", c.text)
            if m:
                drilling = bool(re.match(r"(Surface|Intermediate|Production) Drilling Drilling\b", m[2]))
                out.append((c.time, c.time + pd.Timedelta(hours=float(m[1])), drilling))
        elif ws.well == "lsu" and c.source == "lsu_tour_sheet":
            m = re.match(r"\[[\d-]+ ([\d:]+)-([\d:]+), IADC ([\d.]+)", c.text)
            if m:
                h0, h1 = (pd.Timedelta(hours=int(x[:2]), minutes=int(x[3:])) for x in (m[1], m[2]))
                dur = (h1 - h0) if h1 > h0 else h1 - h0 + pd.Timedelta("1D")
                out.append((c.time, c.time + dur, float(m[3]) == 2.0))
    return out


@tool("check_clock_offset", "Estimate a clock offset. against='reports': rig on-bottom drilling minutes vs the drilling "
      "intervals written in the report time log. against='wits' (LSU only): Pason vs mud-logger WITS feed of the same channel.",
      dict(start=S(TIME), end=S(TIME), against=dict(type="string", enum=["reports", "wits"]), channel=S("for wits: channel (default hookload_klb)"),
           max_lag_min=I("search range in minutes (default 240)")), ["start", "end"])
def check_clock_offset(ws, start, end, against="reports", channel="hookload_klb", max_lag_min=240):
    if against == "wits":
        _channel(_feed(ws, "wits"), channel, "wits")
        return qc.clock_offset_series(_feed(ws, "edr"), _feed(ws, "wits"), channel, start, end, max_lag_min)
    act = report_activity(ws, start, end)
    if not act:
        raise ValueError("no report time-log entries in this window")
    return qc.clock_offset_activity(_feed(ws, "edr"), act, start, end, max_lag_min)


@tool("check_depths", "Hole depth consistency: (1) depths written in reports vs the rig hole-depth channel at that time, "
      "(2) jumps or drops in the hole-depth channel that drilling cannot explain.",
      dict(start=S(TIME), end=S(TIME)), ["start", "end"])
def check_depths(ws, start, end):
    rep = ws.facts.get("hole_depth", t0=start, t1=end, limit=500)
    e = _feed(ws, "edr")
    return dict(reports_vs_rig=qc.depth_vs_reports(e, rep) if len(rep) else "no reported depths in window",
                jumps=qc.depth_jumps(e, start, end))


@tool("check_surveys", "Survey consistency: the same station in different documents must agree, and dogleg severity spikes "
      "inside one document suggest a typo.", dict(depth_min=N("ft MD"), depth_max=N("ft MD"), dls_limit=N("deg/100 ft (default 8)")))
def check_surveys(ws, depth_min=None, depth_max=None, dls_limit=8.0):
    df = ws.facts.df[ws.facts.df.quantity.isin(["survey_inclination", "survey_azimuth"])]
    d = pd.to_numeric(df.depth, errors="coerce")
    if depth_min is not None:
        df = df[d >= depth_min]
        d = d[d >= depth_min]
    if depth_max is not None:
        df = df[d <= depth_max]
    return qc.survey_check(df, dls_limit=dls_limit)


@tool("check_block_weight", "Hook load while in slips (= travelling block + top drive only, a rig constant) per day. "
      "A trend means hook-load sensor drift.", dict(start=S(TIME), end=S(TIME)), ["start", "end"])
def check_block_weight(ws, start, end):
    return qc.block_weight_trend(_feed(ws, "edr"), start, end)


@tool("check_series_continuity", "Isolated outliers in a reported series (one value that disagrees with both neighbours).",
      dict(quantity=dict(type="string", enum=list(QUANTITIES)), start=S(TIME), end=S(TIME), max_step=N("allowed change between neighbours"),
           qualifier=S("filter"), source=S("filter")), ["quantity", "max_step"])
def check_series_continuity(ws, quantity, max_step, start=None, end=None, qualifier=None, source=None):
    x = ws.facts.get(quantity, qualifier, start, end, source=source, limit=1000)
    x = x[pd.to_numeric(x.value, errors="coerce").notna()]
    out = qc.continuity(x.value.astype(float).values, list(x.time), max_step)
    lookup = {str(t): loc for t, loc in zip(x.time, x.locator)}
    for o in out["isolated_outliers"]:
        o["locator"] = lookup.get(o["time"])
    return out


@tool("interpret_units", "Read a value whose unit may be missing or wrong, using the physical range of the quantity "
      "(hook load: klb vs lb; torque: kft-lb vs ft-lb; mud weight: ppg vs s.g.). Also converts between units.",
      dict(kind=dict(type="string", enum=["hook_load", "torque", "mud_weight", "convert"]), value=N("value"), unit=S("unit as written ('' if none)"),
           to_unit=S("for kind=convert")), ["kind", "value"])
def interpret_units(ws, kind, value, unit="", to_unit=None):
    if kind == "convert":
        return dict(value=units.convert(value, unit, to_unit), unit=to_unit)
    f = dict(hook_load=units.read_hook_load, torque=units.read_torque, mud_weight=units.read_mud_weight)[kind]
    v, why = f(value, unit)
    return dict(input=value, unit_written=unit, value=v, unit={"hook_load": "klb", "torque": "kft-lb", "mud_weight": "ppg"}[kind], reading=why)


# ============================================================ physics tools
@tool("pump_output", "Triplex pump flow from liner, stroke, SPM and efficiency.",
      dict(liner_in=N("liner ID, in"), stroke_in=N("stroke length, in"), spm=N("strokes/min"), efficiency_pct=N("volumetric efficiency % (default 95)")),
      ["liner_in", "stroke_in", "spm"])
def pump_output(ws, liner_in, stroke_in, spm, efficiency_pct=95.0):
    return hy.pump_output(liner_in, stroke_in, spm, efficiency_pct)


@tool("bit_hydraulics", "Bit nozzle hydraulics: TFA from nozzle sizes, pressure drop across the bit, and (if spp_rise_psi is "
      "given) how many plugged nozzles would explain that standpipe-pressure rise.",
      dict(mud_weight_ppg=N("ppg"), flow_gpm=N("gpm"), nozzles=S("e.g. '3x20, 1x24' (32nds)"), tfa_in2=N("TFA if nozzles unknown"),
           spp_rise_psi=N("observed SPP rise to explain")), ["mud_weight_ppg", "flow_gpm"])
def bit_hydraulics(ws, mud_weight_ppg, flow_gpm, nozzles=None, tfa_in2=None, spp_rise_psi=None):
    tfa = hy.tfa_from_nozzles(nozzles) if nozzles else tfa_in2
    if not tfa:
        raise ValueError("give nozzles or tfa_in2")
    out = dict(tfa_in2=tfa, bit_pressure_drop_psi=hy.bit_pressure_drop(mud_weight_ppg, flow_gpm, tfa))
    if spp_rise_psi is not None and nozzles:
        out["plugged_nozzle_scenarios"] = hy.nozzles_to_explain(spp_rise_psi, mud_weight_ppg, flow_gpm, nozzles)
    return out


@tool("check_tfa", "Does a reported total flow area match the reported nozzle sizes?",
      dict(nozzles=S("e.g. '4x12, 4x11'"), tfa_reported=N("in^2")), ["nozzles", "tfa_reported"])
def check_tfa(ws, nozzles, tfa_reported):
    return qc.tfa_consistency(nozzles, tfa_reported)


def _tnd(ws):
    try:
        return tnd.dataset(ws.well)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e


@tool("hookload_model", "Fit a soft-string torque-and-drag model to the steady trip hook loads (pick-ups and slack-offs) in a "
      "window. Models: weight_only, single_mu (one friction factor, common practice), fill_only (empty pipe above a float "
      "valve), mu_and_fill, zone_mu_and_fill (cased/open friction + fill). Returns estimates with trip-clustered 95% CIs, "
      "mean residual by state (POOH = pick-up, RIH = slack-off), strong parameter correlations, and physical-plausibility flags.",
      dict(model=dict(type="string", enum=list(tnd.MODELS)), start=S(TIME), end=S(TIME),
           mud_weight_ppg=N("override the mud weight (buoyancy) to test a candidate value")), ["model"])
def hookload_model(ws, model, start=None, end=None, mud_weight_ppg=None):
    ds = _tnd(ws)
    idx = tnd.select(ds, start, end)
    if len(idx) < 10:
        raise ValueError(f"only {len(idx)} steady trip events in this window; widen it")
    out = tnd.fit(ds, model, idx, mw=mud_weight_ppg)
    out["string"] = ds["dp"]
    out["nominal_dp_weight_lb_ft"] = ds["w_dp_nominal"]
    return out


@tool("friction_identifiability", "Can a friction factor be determined from the data in this window? mu_open needs (1) at "
      "least one trip whose hook loads are sensitive to it, (2) trips that agree with each other, (3) a physical value "
      "(0-0.6). Also reports the free zone-model fit and the cased/open sensitivity split. Use BEFORE reporting a friction factor.",
      dict(parameter=dict(type="string", enum=["mu_open", "mu_cased"]), start=S(TIME), end=S(TIME), tolerance=N("acceptable +/- (default 0.1)")),
      ["parameter"])
def friction_identifiability(ws, parameter, start=None, end=None, tolerance=0.1):
    ds = _tnd(ws)
    idx = tnd.select(ds, start, end)
    if len(idx) < 10:
        raise ValueError(f"only {len(idx)} steady trip events in this window; widen it")
    return tnd.identifiability(ds, parameter, idx, tolerance=tolerance, t0=start, t1=end)


@tool("friction_by_trip", "Open-hole friction factor fitted trip by trip (weight terms held at the well's global fit). "
      "sens_klb_per_mu says how strongly each trip constrains mu (small = weak).", dict(start=S(TIME), end=S(TIME)))
def friction_by_trip(ws, start=None, end=None):
    rows = tnd.track_open_friction(_tnd(ws), start, end)
    return dict(trips=rows, note="se is a lower bound (ignores within-trip correlation)")


@tool("check_hookload_physics", "Tells friction from data errors in trip hook loads. Whatever the friction, a pick-up cannot "
      "weigh less than the string with zero friction, a slack-off cannot weigh more, and no reading can need friction above "
      "0.8. Flags readings outside that envelope (by more than 3 sigma of the well's scatter) and trips that no single "
      "friction value fits. A flagged reading is a sensor, string, fill-state, depth or sticking problem, not friction.",
      dict(start=S(TIME), end=S(TIME)), ["start", "end"])
def check_hookload_physics(ws, start, end):
    ds = _tnd(ws)
    idx = tnd.select(ds, start, end)
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} steady trip events in this window; widen it")
    return tnd.physics_check(ds, idx)


@tool("trip_tank_displacement", "Trip-tank gain per foot of pipe run in during a trip in: closed-end (float holding, pipe "
      "running empty) vs open-end (pipe filling through the bit).", dict(start=S(TIME), end=S(TIME)), ["start", "end"])
def trip_tank_displacement(ws, start, end):
    ds = _tnd(ws)
    od, idn = (5.5, 4.778) if ws.well == "lsu" else (5.0, 4.276)
    closed = round(od ** 2 / 1029.4, 4)                      # bbl/ft displaced by closed-end pipe
    open_ = round((od ** 2 - idn ** 2) / 1029.4, 4)          # steel only
    return qc.trip_tank_displacement(_feed(ws, "edr"), start, end, closed, open_) | dict(pipe=ds["dp"])


# ============================================================ arithmetic
_OPS = None


@tool("calculate", "Evaluate arithmetic (+ - * / ** parentheses, abs, min, max, round, sqrt) so that derived numbers "
      "(differences, ratios, unit products) are traceable. Example: '(5113 - 5110) / 93'.",
      dict(expression=S("arithmetic expression")), ["expression"])
def calculate(ws, expression):
    import ast
    import math
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}
    funcs = dict(abs=abs, min=min, max=max, round=round, sqrt=math.sqrt)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs:
            return funcs[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in expression: {ast.dump(n)[:60]}")
    if len(expression) > 200:
        raise ValueError("expression too long")
    return dict(expression=expression, result=ev(ast.parse(expression.replace(",", ""), mode="eval")))
