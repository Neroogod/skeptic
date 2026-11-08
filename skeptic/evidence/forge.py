"""FORGE 16A(78)-32 (Utah FORGE, DOI 10.15121/1776602, CC BY 4.0).

Two sources:
  * daily drilling reports (WellEz form, text extracted from PDF) -> facts + narrative chunks
  * 10-s rig time data (EDR) -> one DataFrame with canonical channel names

Report period: a report dated D covers 06:00 on D-1 to 06:00 on D, so a time-log or
mud-check entry at hh:mm belongs to D-1 if hh >= 6, else to D.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .store import Fact
from .text import Chunk

WELL = "forge"
SRC = "forge_daily_report"
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def num(s):
    """'1,020.67' -> 1020.67 ; None/'' -> nan."""
    if s is None:
        return np.nan
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# Clocks: the rig clock fell back from MDT to MST at 02:00 on 1 Nov 2020 (the EDR repeats
# 01:00-01:59). Everything here is put on MST: earlier times move back one hour.
DST_END = pd.Timestamp("2020-11-01 02:00")


def to_mst(t):
    return t - pd.Timedelta("1h") if t < DST_END else t


def entry_time(report_date, hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    day = report_date - pd.Timedelta("1D") if h >= 6 else report_date
    return to_mst(day + pd.Timedelta(hours=h % 24, minutes=m))


def section(lines, start_pat, stop_pats):
    """Lines strictly between the line matching start_pat and the next stop line."""
    out, on = [], False
    for ln in lines:
        if on and any(re.match(p, ln) for p in stop_pats):
            break
        if on:
            out.append(ln)
        if re.match(start_pat, ln):
            on = True
    return out


# ------------------------------------------------------------------ narrative numbers
NARR = [  # (quantity, regex, unit); group 'v' is the value, optional group 'q' the qualifier
    ("spp", re.compile(r"\b(?:SPP|SSP)\s*[:=-]?\s*(?P<v>\d{3,4})\b", re.I), "psi"),
    ("flow_rate", re.compile(r"\b(?P<v>\d{3,4})\s*gpm\b", re.I), "gpm"),
    ("flow_rate", re.compile(r"\bGPM\s*[:=-]\s*(?P<v>\d{3,4})\b", re.I), "gpm"),
    ("mud_weight", re.compile(r"\bMW\s*(?:IN\s*)?[:=]?\s*(?P<v>\d{1,2}\.\d)\b", re.I), "ppg"),
    ("pickup_weight", re.compile(r"\bP/?U\b(?P<q>[^.;\d]{0,25}?)[,\s]*(?<![A-Za-z])(?P<v>\d{2,3}(?:,\d{3})?)\s*(?P<k>K|klb)?\b"), ""),        # case-sensitive: crew writes PU/SO
    ("slackoff_weight", re.compile(r"\bS/?O\b(?P<q>[^.;\d]{0,25}?)[,\s]*(?<![A-Za-z])(?P<v>\d{2,3}(?:,\d{3})?)\s*(?P<k>K|klb)?\b"), ""),
]


def narrative_facts(text, well, source, locator, time):
    """Numbers stated in free text. Hook-load units are kept exactly as written:
    '210K' -> 210 klb, '230' -> 230 with no unit (the crew left it off)."""
    out = []
    for q, pat, unit in NARR:
        for m in pat.finditer(text):
            v = num(m.group("v"))
            u = unit
            qual = "narrative"
            if q in ("pickup_weight", "slackoff_weight"):
                k = m.groupdict().get("k")
                u = "klb" if k else ("lb" if v >= 1000 else "")
                ql = (m.group("q") or "").lower()
                if "out" in ql and "pump" in ql:
                    qual = "narrative, pumps off"
                elif "pump" in ql:
                    qual = "narrative, pumps on"
            out.append(Fact(well, q, v, u, source, locator, time=time, qualifier=qual, raw=m.group(0).strip()))
    return out


# ----------------------------------------------------------------------- one report
def parse_report(path):
    """-> (facts, chunks) for one daily-report text file."""
    path = Path(path)
    txt = path.read_text(errors="replace")
    lines = [ln.rstrip() for ln in txt.splitlines()]
    m = re.search(r"RPT DATE:\s*([\d/]+)", txt)
    if not m:
        return [], []
    rdate = pd.Timestamp(m[1])
    t_rep = to_mst(rdate + pd.Timedelta("6h"))              # end of the report period
    rpt = re.search(r"RPT NUM\.?:\s*(\d+)", txt)
    loc = f"{path.name}"
    F = []

    def fact(q, v, unit, line, **kw):
        kw.setdefault("time", t_rep)
        F.append(Fact(WELL, q, v, unit, SRC, f"{loc}:{line}", **kw))

    def lineno(pat):
        for i, ln in enumerate(lines, 1):
            if re.search(pat, ln):
                return i
        return 0

    fact("report_date", rdate.strftime("%Y-%m-%d"), "", lineno(r"RPT DATE"), raw=m.group(0),
         qualifier=f"report {rpt[1] if rpt else '?'}")

    m = re.search(r"MD/TVD:\s*(" + NUM + r")?(?:\s*/\s*(" + NUM + r"))?\s*24 HR FTG:", txt)
    if m and m[1]:
        fact("hole_depth", num(m[1]), "ft", lineno(r"MD/TVD"), depth=num(m[1]), qualifier="report header", raw=m.group(0))

    for which in ("LAST", "NEXT"):
        m = re.search(which + r" BOP TEST:\s*([\d/]+)", txt)
        if m:
            fact("bop_test_date", m[1], "", lineno(which + " BOP TEST"), qualifier=which.lower(), raw=m.group(0))

    # casing table: "Surface 13.38 68 L-80 1,629 1,629"
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^(Conductor|Surface|Intermediate|Production|Liner)\s+(\d+\.\d+)\s+(\S+)\s+(\S+)\s+(" + NUM + r")\s+(" + NUM + r")\s*$", ln)
        if m:
            fact("casing_shoe_depth", num(m[5]), "ft", i, qualifier=f"{num(m[2]):g} in {m[1].lower()}", depth=num(m[5]), raw=ln)

    # mud table: "22:00 8.8 43.0 7.8 ..." -> time, MW, funnel viscosity
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^(\d{1,2}:\d{2})\s+(\d{1,2}\.\d+)\s+(\d+(?:\.\d+)?)\s", ln)
        if m and 6.0 < num(m[2]) < 20.0:
            t = entry_time(rdate, m[1])
            fact("mud_weight", num(m[2]), "ppg", i, time=t, qualifier="mud check", raw=ln)
            fact("funnel_viscosity", num(m[3]), "s/qt", i, time=t, qualifier="mud check", raw=ln)

    # bits: "9 8.75 SMITH PDC JV2705 4x12, 4x11 0.910 N 5,200 5,345 145 ..."
    for i, ln in enumerate(section(lines, r"^NO\. SIZE MANUFACTURER", [r"^BIT NO\.", r"^PUMPS"]), 0):
        m = re.match(r"^(\w+)\s+(\d+(?:\.\d+)?)\s+(.*?)\s+(\d?\.\d{2,3})\s+([YN])\b\s*(" + NUM + r")?\s*(" + NUM + r")?", ln)
        if not m:
            continue
        bit, li = m[1], lineno(re.escape(ln))
        fact("bit_size", num(m[2]), "in", li, qualifier=f"bit {bit}", raw=ln)
        fact("tfa", num(m[4]), "in2", li, qualifier=f"bit {bit}", raw=ln)
        nz = re.search(r"\d+x\d+(?:\s*,\s*\d+x\d+)*", m[3])
        if nz:
            fact("nozzles", nz[0].replace(" ", ""), "32nds", li, qualifier=f"bit {bit}", raw=ln)
        if m[6]:
            fact("bit_depth_in", num(m[6]), "ft", li, qualifier=f"bit {bit}", raw=ln)
        if m[7]:
            fact("bit_depth_out", num(m[7]), "ft", li, qualifier=f"bit {bit}", raw=ln)

    # pumps: "2 BOMCO 1600 6.000 12.000 95 2800 400 [gps] 95 ..."
    for ln in section(lines, r"^PUMP MAKE", [r"^BHA", r"^SURVEY", r"^# COMPONENT"]):
        tok = ln.split()
        if len(tok) < 5 or not tok[0].isdigit():
            continue
        li, pump = lineno(re.escape(ln)), f"pump {tok[0]}"
        vals = [num(x) for x in tok[3:]]
        names = ["liner_size", "stroke_length", "spm", "spp", "flow_rate"]
        units = ["in", "in", "spm", "psi", "gpm"]
        for q, u, v in zip(names, units, vals):
            if np.isfinite(v):
                fact(q, v, u, li, qualifier=f"{pump} table", raw=ln)
        rest = vals[5:]
        if rest and rest[0] < 50 and len(rest) > 1:          # strokes-per-gallon column present
            rest = rest[1:]
        if rest and 50 <= rest[0] <= 100:
            fact("pump_efficiency", rest[0], "%", li, qualifier=f"{pump} table", raw=ln)

    # BHA summary
    m = re.search(r"DESCRIPTION:\s*(.*)", txt)
    bha = m[1].strip() if m else ""
    mb = re.search(r"#\s*(\d+)", bha)
    bq = f"BHA #{mb[1]}" if mb else "BHA"
    for q, pat, unit in (("bha_length", r"LENGTH:\s*(" + NUM + ")", "ft"),
                         ("bha_weight_air", r"BHA WT\. AIR:\s*(" + NUM + ")", "klb"),
                         ("bha_weight_mud", r"BHA WT\. MUD:\s*(" + NUM + ")", "klb")):
        m = re.search(pat, txt)
        if m:
            fact(q, num(m[1]), unit, lineno(pat.split(":")[0]), qualifier=bq, raw=m.group(0))
    for q, pat in (("rotating_weight", r"ROT\. WT\.:\s*(" + NUM + ")"), ("pickup_weight", r"P\.U\. WT\.:\s*(" + NUM + ")"),
                   ("slackoff_weight", r"S\.O\. WT\.:\s*(" + NUM + ")")):
        m = re.search(pat, txt)
        if m:
            fact(q, num(m[1]), "lb", lineno(r"ROT\. WT"), qualifier="report summary", raw=m.group(0))
    m = re.search(r"TORQUE:\s*(" + NUM + ")", txt)
    if m:
        fact("torque", num(m[1]), "ft-lb", lineno(r"TORQUE:"), qualifier="report summary", raw=m.group(0))

    # surveys: "5,294 1.67 51.93 5,293 -7.1 8.3 9.9 1.1"
    for ln in section(lines, r"^MD INCLINATION", [r"^CONSUMABLES", r"^TIME BREAKDOWN"]):
        tok = ln.split()
        if len(tok) >= 4:
            md, inc, az = num(tok[0]), num(tok[1]), num(tok[2])
            if np.isfinite(md) and 0 <= inc <= 180 and 0 <= az <= 360:
                li = lineno(re.escape(ln))
                fact("survey_inclination", inc, "deg", li, depth=md, qualifier="report survey table", raw=ln)
                fact("survey_azimuth", az, "deg", li, depth=md, qualifier="report survey table", raw=ln)

    # time breakdown -> one chunk per entry, plus the numbers stated in it
    chunks = []
    m = re.search(r"PRESENT OPERATIONS:\s*(.*)", txt)
    if m:
        chunks.append(Chunk(WELL, SRC, f"{loc}:{lineno('PRESENT OPERATIONS')}", f"[{rdate:%Y-%m-%d} present operations] {m[1]}", time=t_rep))
    body = section(lines, r"^FROM TO HRS", [r"^TOTAL HRS", r"^www\.wellez"])
    entries, cur = [], None
    for ln in body:
        mt = re.match(r"^(\d{2}:\d{2})\s+(\d{2}:\d{2})\s+([\d.]+)\s*$", ln)
        if mt:
            cur = dict(start=mt[1], end=mt[2], hrs=mt[3], lines=[], line0=lineno(re.escape(ln)))
            entries.append(cur)
        elif cur is not None and ln.strip():
            cur["lines"].append(ln.strip())
    for e in entries:
        t = entry_time(rdate, e["start"])
        text = " ".join(e["lines"])
        where = f"{loc}:{e['line0']}"
        chunks.append(Chunk(WELL, SRC, where, f"[{t:%Y-%m-%d %H:%M}-{e['end']}, {e['hrs']} h] {text}", time=t))
        F += narrative_facts(text, WELL, SRC, where, t)
    return F, chunks


def parse_reports(folder=None):
    """All reports. The folder holds two copies of many reports (two download bundles);
    byte-identical copies (after whitespace) are parsed once, revisions are kept."""
    import hashlib
    folder = Path(folder or config.FORGE_FILES["reports"])
    facts, chunks, seen = [], [], set()
    for p in sorted(folder.glob("*.txt")):
        h = hashlib.md5(" ".join(p.read_text(errors="replace").split()).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        f, c = parse_report(p)
        facts += f
        chunks += c
    return facts, chunks


# ----------------------------------------------------------------------------- EDR
EDR_COLS = {"Time": "t", "Block Position": "block_ft", "Weight on Bit": "wob_klb", "Hookload": "hookload_klb",
            "Slips Set": "in_slips", "On Bottom": "on_bottom", "Top Drive RPM": "rpm",
            "Top Drive Torque (ft-lbs)": "torque_ftlb", "Flow In": "flow_gpm", "Pump Pressure": "spp_psi",
            "SPM Total": "spm_total", "Pit Volume Active": "pit_bbl", "Trip Volume Active": "trip_tank_bbl",
            "MWD Inclination": "inc_deg", "MWD Azimuth": "azi_deg", "Total Depth": "hole_depth_ft",
            "Depth Hole TVD": "hole_tvd_ft", "Differential Pressure": "diff_psi", "ROP Depth/Hour": "rop_fph"}

# bit-depth rebuild constants (from the author's PETE 7103 forge_bitdepth.py)
W_EMPTY = 55.0       # klb: below this nothing hangs on the block
MAX_STEP = 120.0     # ft per 10-s row: larger block jumps are glitches
HOLE_MAX = 11000.0   # ft: hole-depth readings above this are glitches (TD 10,987 ft)


def load_edr(use_cache=True):
    """10-s record with canonical names, on MST. The raw stamps are truncated to the
    minute (six rows per minute, in time order), so rows get +0/10/.../50 s. Bit depth
    is rebuilt from block position."""
    cache = config.CACHE / "forge_edr.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    d = pd.read_csv(config.FORGE_FILES["edr"], usecols=list(EDR_COLS), low_memory=False).rename(columns=EDR_COLS)
    d["t"] = pd.to_datetime(d["t"], format="%m/%d/%y %H:%M", errors="coerce")
    d = d.dropna(subset=["t"]).reset_index(drop=True)
    back = np.where(d.t.diff() < -pd.Timedelta("30min"))[0]        # the DST fall-back step
    if len(back):
        d.loc[:back[0] - 1, "t"] -= pd.Timedelta("1h")
    for c in d.columns.drop("t"):
        d[c] = pd.to_numeric(d[c], errors="coerce").replace(-999.25, np.nan)
    d.loc[(d.block_ft < -20) | (d.block_ft > 130), "block_ft"] = np.nan
    d["t"] = d["t"] + pd.to_timedelta(10 * d.groupby("t").cumcount().clip(upper=5), unit="s")
    d["torque_kftlb"] = d.pop("torque_ftlb") / 1000.0
    d["hole_depth_ft"] = d.hole_depth_ft.where(d.hole_depth_ft < HOLE_MAX).ffill()
    d["bit_depth_ft"] = rebuild_bit_depth(d)
    config.CACHE.mkdir(exist_ok=True)
    d.to_parquet(cache)
    return d


def rebuild_bit_depth(d):
    """Bit depth from block movement (method of the author's PETE forge_bitdepth.rebuild):
    on the hook the bit moves with the block; in slips it stays; on bottom bit = hole.
    Each off-bottom excursion is integrated forward from the depth it left, up to its
    shallowest point, and backward from the depth it returns to after that."""
    n = len(d)
    db = np.diff(d.block_ft.values, prepend=np.nan)
    db[~np.isfinite(db) | (np.abs(db) > MAX_STEP)] = 0.0
    hook = d.hookload_klb.values
    in_slips = np.where(np.isnan(d.in_slips.values), hook < W_EMPTY, d.in_slips.values > 0.5)
    on_hook = ~in_slips & (hook > W_EMPTY)
    step = np.where(on_hook | np.r_[False, on_hook[:-1]], -db, 0.0)
    hole = d.hole_depth_ft.values
    onb = d.on_bottom.fillna(0).values > 0.5
    bit = np.where(onb, hole, np.nan)
    starts = np.where(~onb & np.r_[True, onb[:-1]])[0]
    ends = np.where(~onb & np.r_[onb[1:], True])[0]
    for s, e in zip(starts, ends):
        a0 = hole[s - 1] if s > 0 else np.nan
        a1 = hole[e + 1] if e + 1 < n else np.nan
        C = np.cumsum(step[s:min(e + 2, n)])
        fwd = a0 + C[:e - s + 1]
        bwd = a1 - (C[-1] - C[:e - s + 1])
        if np.isfinite(a0) and np.isfinite(a1):
            k = int(np.argmin(fwd))
            b = np.r_[fwd[:k + 1], bwd[k + 1:]]
        else:
            b = fwd if np.isfinite(a0) else bwd
        bit[s:e + 1] = b
    return np.clip(bit, 0.0, hole)


def load_survey():
    """Definitive survey: (md, inc_deg, azi_deg) arrays."""
    x = pd.read_excel(config.FORGE_FILES["survey"], header=None).apply(pd.to_numeric, errors="coerce")
    s = x[x[1].notna() & x[2].notna() & x[3].notna() & x[4].notna()]
    return s[2].values.astype(float), s[3].values.astype(float), s[4].values.astype(float)


def survey_facts():
    md, inc, az = load_survey()
    loc = config.FORGE_FILES["survey"].name
    out = []
    for i, (a, b, c) in enumerate(zip(md, inc, az)):
        out.append(Fact(WELL, "survey_inclination", b, "deg", "forge_survey_file", f"{loc}:station {i}", depth=a, qualifier="definitive survey"))
        out.append(Fact(WELL, "survey_azimuth", c, "deg", "forge_survey_file", f"{loc}:station {i}", depth=a, qualifier="definitive survey"))
    return out
