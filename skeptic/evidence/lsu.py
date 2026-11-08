"""LSU H-001 (LSU-only data: process locally, never commit or publish).

Sources parsed into facts / chunks / EDR frames:
  Pason 10-s CSV            -> EDR frame 'edr' (+ flow/SPP minute-matched from Pason_1min); memos -> chunks + facts
  Mudlog WITS 1-min         -> EDR frame 'wits' (a second, independently clocked feed)
  Tour sheets (IADC)        -> facts (mud checks, pumps, hook loads, bits, BHA, casing, surveys); time log -> chunks
  Halliburton workbook      -> facts (casing, runs, BHA, bits, surveys, mud); daily summaries -> chunks
  Directional EOW report    -> page chunks

HELD OUT (never parsed): README sheets of every workbook and the Halliburton
'Checks vs Pason' sheet. They hold the instructor's QC conclusions, which are the
benchmark's answer key.
"""
import re

import numpy as np
import pandas as pd

from .. import config
from .store import Fact
from .text import Chunk
from .forge import narrative_facts, num

WELL = "lsu"
HELD_OUT_SHEETS = {"README", "Checks vs Pason"}


def _xl(key, sheet, header):
    if sheet in HELD_OUT_SHEETS:
        raise ValueError(f"sheet {sheet!r} is held out (benchmark answer key)")
    return pd.read_excel(config.LSU_FILES[key], sheet_name=sheet, header=header)


def ts(t, f="%Y-%m-%d %H:%M"):
    """Timestamp text that survives missing times."""
    t = pd.to_datetime(t, errors="coerce")
    return "?" if pd.isna(t) else t.strftime(f)


def tour_time(report_date, tour):
    """Midpoint of a 12-h tour. Day = 06-18 on D, Night = 18:00 D to 06:00 D+1."""
    d = pd.Timestamp(report_date)
    return d + pd.Timedelta("12h") if str(tour).startswith("Day") else d + pd.Timedelta("24h")


# ---------------------------------------------------------------- narrative extras
MEMO = [
    ("mud_weight", re.compile(r"\bMW\s*(?P<io>IN|OUT|I|O)\b\s*(?P<v>\d{1,2}(?:\.\d+)?)", re.I)),
    ("mud_weight", re.compile(r"\bMW\s*(?P<io>IN/OUT)\s*(?P<v>\d{1,2}(?:\.\d+)?)", re.I)),
]
SURVEY_MEMO = re.compile(r"Survey @ (?P<md>[\d.]+) \| INC (?P<inc>[\d.]+)° \| AZM (?P<az>[\d.]+)°")
CASING_TEXT = re.compile(
    r"(?P<size>\b(?:30|20|13[ -]3/8|9[ -]5/8))\s*(?:[”\"]|-?in\.?)?\s*(?:\w+\s+){0,2}casing\b[^.;]{0,60}?"
    r"(?:to depth|landing point at|landed at|set at|at|to)\s*(?P<v>\d,?\d{3})\s*(?:['’]|ft)", re.I)
LANDING = re.compile(r"landing point at\s*(?P<v>\d,?\d{3})\s*['’]", re.I)
DPID_TEXT = re.compile(r"\bDP\s+ID\s+(?P<v>\d\.\d+)", re.I)


def size_in(s):
    s = s.replace("-", " ")
    if " " in s:
        whole, frac = s.split()
        a, b = frac.split("/")
        return float(whole) + float(a) / float(b)
    return float(s)


def text_facts(text, source, locator, time):
    """Numbers in LSU free text: SPP/GPM/MW (shared patterns), memo MW in/out,
    casing set depths, drill-pipe ID."""
    out = narrative_facts(text, WELL, source, locator, time)
    out = [f for f in out if f.quantity != "mud_weight"]          # LSU MW handled below (in/out)
    for q, pat in MEMO:
        for m in pat.finditer(text):
            io = {"IN/OUT": "in and out"}.get(m["io"].upper(), "in" if m["io"].upper().startswith("I") else "out")
            out.append(Fact(WELL, q, num(m["v"]), "ppg", source, locator, time=time, qualifier=f"{source}, {io}", raw=m.group(0)))
    for m in CASING_TEXT.finditer(text):
        out.append(Fact(WELL, "casing_shoe_depth", num(m["v"]), "ft", source, locator, time=time,
                        depth=num(m["v"]), qualifier=f"{size_in(m['size']):g} in (text)", raw=m.group(0)))
    for m in LANDING.finditer(text):
        if not CASING_TEXT.search(text[max(0, m.start() - 80):m.end()]):
            out.append(Fact(WELL, "casing_shoe_depth", num(m["v"]), "ft", source, locator, time=time,
                            depth=num(m["v"]), qualifier="size not stated (text)", raw=m.group(0)))
    for m in DPID_TEXT.finditer(text):
        out.append(Fact(WELL, "dp_id", num(m["v"]), "in", source, locator, time=time, qualifier="text", raw=m.group(0)))
    return out


# ---------------------------------------------------------------------- Pason EDR
PASON = {"Date Time": "t", "Hole Depth (feet)": "hole_depth_ft", "Bit Depth (feet)": "bit_depth_ft",
         "Hook Load (klbs)": "hookload_klb", "Rotary RPM (RPM)": "rpm", "Weight on Bit (klbs)": "wob_klb",
         "Block Height (feet)": "block_ft", "Convertible Torque (kft_lb)": "torque_kftlb",
         "In Slip (unitless)": "in_slips", "On Bottom (unitless)": "on_bottom",
         "Trip Tank Mud Volume (barrels)": "trip_tank_bbl", "Inclination (degrees)": "inc_deg",
         "Azimuth (degrees)": "azi_deg", "True Vertical Depth (feet)": "tvd_ft", "Realtime TVD (feet)": "rt_tvd_ft",
         "Over Pull (klbs)": "overpull_klb", "Calculated String Weight (klbs)": "calc_string_weight_klb",
         "PT Hole Diameter (in)": "pt_hole_diameter_in", "PT Inside Diameter (in)": "pt_id_in",
         "PT Outside Diameter (in)": "pt_od_in", "Rig Super State (unitless)": "rig_super_state",
         "Trip Speed (ft_per_min)": "trip_speed_ftmin", "Memos": "memo"}


def load_edr(use_cache=True):
    cache = config.CACHE / "lsu_edr.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    d = pd.read_csv(config.LSU_FILES["pason"], usecols=list(PASON), low_memory=False).rename(columns=PASON)
    d["t"] = pd.to_datetime(d["t"], errors="coerce")
    d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
    for c in d.columns.drop(["t", "memo"]):
        d[c] = pd.to_numeric(d[c], errors="coerce").replace(-999.25, np.nan)
    d.loc[d.hookload_klb < 0, "hookload_klb"] = np.nan
    # flow-in and SPP exist only in the 1-min extract: match by minute (as in PETE tdcore)
    p1 = _xl("supplement", "Pason_1min", 1)
    p1["tm"] = pd.to_datetime(p1["datetime"])
    d["tm"] = d.t.dt.floor("min")
    d = d.merge(p1[["tm", "flow_in_gpm", "spp_psi"]].rename(columns={"flow_in_gpm": "flow_gpm"}), on="tm", how="left").drop(columns="tm")
    config.CACHE.mkdir(exist_ok=True)
    d.to_parquet(cache)
    return d


def load_wits():
    w = _xl("supplement", "Mudlog_WITS_1min", 1)
    w = w.rename(columns={"datetime": "t", "hole_depth_ft": "hole_depth_ft", "bit_depth_ft": "bit_depth_ft",
                          "hookload_klb": "hookload_klb", "wob_klb": "wob_klb", "torque_kftlb": "torque_kftlb",
                          "spp_psi": "spp_psi", "total_spm": "spm_total", "rop_ft_hr": "rop_fph"})
    w["t"] = pd.to_datetime(w["t"])
    return w.drop(columns=["rig_state_auto"]).sort_values("t").reset_index(drop=True)


def memo_evidence(edr):
    facts, chunks = [], []
    m = edr.dropna(subset=["memo"])
    for i, r in m.iterrows():
        loc = f"Pason CSV memo row {i}"
        text = str(r.memo)
        chunks.append(Chunk(WELL, "lsu_pason_memo", loc, f"[{r.t:%Y-%m-%d %H:%M}] {text}", time=r.t, depth=r.bit_depth_ft))
        facts += text_facts(text, "lsu_pason_memo", loc, r.t)
        s = SURVEY_MEMO.search(text)
        if s:
            for q, v in (("survey_inclination", s["inc"]), ("survey_azimuth", s["az"])):
                facts.append(Fact(WELL, q, num(v), "deg", "lsu_pason_memo", loc, time=r.t, depth=num(s["md"]), qualifier="WITS survey memo", raw=s.group(0)))
    return facts, chunks


# --------------------------------------------------------------------- tour sheets
def tour_evidence():
    F, C = [], []
    S = "lsu_tour_sheet"
    fn = config.LSU_FILES["tour"].name

    mc = _xl("tour", "Mud Checks", 0)
    for i, r in mc.iterrows():
        if pd.notna(r["Mud Weight (ppg)"]):
            F.append(Fact(WELL, "mud_weight", float(r["Mud Weight (ppg)"]), "ppg", S, f"{fn}:Mud Checks row {i + 2}",
                          time=pd.Timestamp(r["Check Time"]), qualifier="crew mud check", raw=str(r.get("Other entry", ""))))
        if pd.notna(r["Funnel Viscosity (s/qt)"]):
            F.append(Fact(WELL, "funnel_viscosity", float(r["Funnel Viscosity (s/qt)"]), "s/qt", S, f"{fn}:Mud Checks row {i + 2}",
                          time=pd.Timestamp(r["Check Time"]), qualifier="crew mud check"))

    sv = _xl("tour", "Surveys", 0)
    for i, r in sv.iterrows():
        t = tour_time(r["Report Date"], r["Tour"])
        for q, col in (("survey_inclination", "Inclination (°)"), ("survey_azimuth", "Azimuth (°)")):
            F.append(Fact(WELL, q, float(r[col]), "deg", S, f"{fn}:Surveys row {i + 2}", time=t, depth=float(r["MD (ft)"]), qualifier="tour survey"))

    pu = _xl("tour", "Pumps", 0)
    for i, r in pu.iterrows():
        t = tour_time(r["Report Date"], r["Tour"])
        loc = f"{fn}:Pumps row {i + 2}"
        if pd.notna(r["Pressure (psi)"]):
            F.append(Fact(WELL, "spp", float(r["Pressure (psi)"]), "psi", S, loc, time=t, qualifier="tour pumps"))
        if pd.notna(r["Total Flow (gpm)"]):
            F.append(Fact(WELL, "flow_rate", float(r["Total Flow (gpm)"]), "gpm", S, loc, time=t, qualifier="tour pumps total"))
        for p in (1, 2, 3):
            if pd.notna(r[f"Pump {p} Liner (in)"]):
                F.append(Fact(WELL, "liner_size", float(r[f"Pump {p} Liner (in)"]), "in", S, loc, time=t, qualifier=f"pump {p}"))
            if pd.notna(r[f"Pump {p} SPM"]):
                F.append(Fact(WELL, "spm", float(r[f"Pump {p} SPM"]), "spm", S, loc, time=t, qualifier=f"pump {p}"))

    hk = _xl("tour", "Hookload & Torque", 1)
    for i, r in hk.iterrows():
        t = tour_time(r["Report Date"], r["Tour"])
        loc = f"{fn}:Hookload & Torque row {i + 3}"
        dep = num(r["Depth recorded (ft)"])
        for q, col in (("pickup_weight", "Pick-up Weight"), ("slackoff_weight", "Slack-off Weight"),
                       ("rotating_weight", "Rotating Weight"), ("string_weight", "String Weight")):
            if pd.notna(r[col]):     # units vary by tour (klb or lb) exactly as typed: unit left undeclared
                F.append(Fact(WELL, q, float(r[col]), "", S, loc, time=t, depth=dep, qualifier="tour, units as typed", raw=str(r[col])))
        if pd.notna(r["Rotary Torque"]):
            F.append(Fact(WELL, "torque", float(r["Rotary Torque"]), "", S, loc, time=t, depth=dep, qualifier="tour, units as typed"))

    bt = _xl("tour", "Bits", 0)
    for i, r in bt.iterrows():
        loc, q = f"{fn}:Bits row {i + 2}", f"bit {r['Bit No']}"
        t = pd.Timestamp(r["First Report"]) + pd.Timedelta("12h") if pd.notna(r["First Report"]) else None
        F.append(Fact(WELL, "bit_size", float(r["Size (in)"]), "in", S, loc, time=t, qualifier=q, raw=f"{r['Size (in)']} {r['Make']} {r['Type']}"))
        for qq, col, u in (("bit_depth_in", "Depth In (ft)", "ft"), ("bit_depth_out", "Depth Out (ft)", "ft"), ("tfa", "TFA (in²)", "in2")):
            if pd.notna(r[col]):
                F.append(Fact(WELL, qq, float(r[col]), u, S, loc, time=t, qualifier=q))
        if pd.notna(r["Jets"]):
            F.append(Fact(WELL, "nozzles", str(r["Jets"]), "32nds", S, loc, time=t, qualifier=q))

    bh = _xl("tour", "BHA", 0)
    for i, r in bh[bh["Total BHA Length (ft)"].notna()].iterrows():
        F.append(Fact(WELL, "bha_length", float(r["Total BHA Length (ft)"]), "ft", S, f"{fn}:BHA row {i + 2}",
                      qualifier=f"BHA {r['BHA #']} ({r['First Used']} to {r['Last Used']})"))

    cs = _xl("tour", "Casing", 0)
    for i, r in cs.iterrows():
        F.append(Fact(WELL, "casing_shoe_depth", float(r["Set Depth (ft)"]), "ft", S, f"{fn}:Casing row {i + 2}",
                      depth=float(r["Set Depth (ft)"]), qualifier=f"{float(r['Size (in)']):g} in {str(r['String']).lower()}"))

    di = _xl("tour", "Drilling Intervals", 0)
    for i, r in di.iterrows():
        if pd.notna(r["To (ft)"]):
            F.append(Fact(WELL, "hole_depth", float(r["To (ft)"]), "ft", S, f"{fn}:Drilling Intervals row {i + 2}",
                          time=tour_time(r["Report Date"], r["Tour"]), depth=float(r["To (ft)"]), qualifier=f"end of tour, type {r['Type']}"))

    tl = _xl("tour", "Time Log", 0)
    for i, r in tl.iterrows():
        loc = f"{fn}:Time Log row {i + 2}"
        t = pd.to_datetime(r["Start"], errors="coerce")
        t = None if pd.isna(t) else t
        text = f"[{ts(t)}-{ts(r['End'], '%H:%M')}, IADC {r['IADC Code']} {r['IADC Activity']}] {r['Description']}"
        C.append(Chunk(WELL, S, loc, text, time=t))
        F += text_facts(str(r["Description"]), S, loc, t)
    return F, C


# ---------------------------------------------------------------------- Halliburton
def halliburton_evidence():
    F, C = [], []
    S = "lsu_halliburton"
    fn = config.LSU_FILES["halliburton"].name

    wc = _xl("halliburton", "Well & Casing", None)
    hdr = wc.index[wc[0].astype(str).str.startswith("String / section")][0]
    for i in range(hdr + 1, len(wc)):
        r = wc.loc[i]
        name, od, shoe = str(r[0]), r[3], str(r[7])
        if name in ("nan", "") or str(od) in ("nan", "—"):
            continue
        for v in re.findall(r"(\d[\d,]*)\s*(?:\(([^)]*)\))?", shoe):
            depth = num(v[0])
            if np.isfinite(depth) and depth > 50 and "plan" not in shoe:
                F.append(Fact(WELL, "casing_shoe_depth", depth, "ft", S, f"{fn}:Well & Casing row {i + 1}", depth=depth,
                              qualifier=f"{float(od):g} in {name.lower()}" + (f" ({v[1]})" if v[1] else ""), raw=f"{name}: {shoe}; notes: {r[9]}"))

    rs = _xl("halliburton", "Run Summary", 1)
    rs = rs[pd.to_numeric(rs["Run"], errors="coerce").notna()]
    for i, r in rs.iterrows():
        run, loc = int(r["Run"]), f"{fn}:Run Summary row {i + 3}"
        t = pd.Timestamp(r["Time in"])
        q = f"run {run}"
        for qq, col, u in (("mud_weight", "MW (ppg)", "ppg"), ("plastic_viscosity", "PV (cP)", "cP"),
                           ("yield_point", "YP (lb/100ft²)", "lb/100ft2"), ("bit_size", "Bit size (in)", "in"),
                           ("tfa", "TFA (in²)", "in2"), ("flow_rate", "Flow – Halliburton (gpm)", "gpm"),
                           ("spp", "SPP – Halliburton (psi)", "psi")):
            if pd.notna(r[col]):
                F.append(Fact(WELL, qq, float(r[col]), u, S, loc, time=t, qualifier=f"{q} (Halliburton run average)" if qq in ("flow_rate", "spp") else q))
        F.append(Fact(WELL, "nozzles", str(r["Nozzles"]), "32nds", S, loc, time=t, qualifier=q))
        F.append(Fact(WELL, "bit_depth_in", float(r["MD in (ft)"]), "ft", S, loc, time=t, qualifier=f"{q}, bit {r['Bit #']}"))
        F.append(Fact(WELL, "bit_depth_out", float(r["MD out (ft)"]), "ft", S, loc, time=t, qualifier=f"{q}, bit {r['Bit #']}"))

    bc = _xl("halliburton", "BHA Components", 1)
    bc = bc[pd.to_numeric(bc["Run / BHA #"], errors="coerce").notna()]
    for run, g in bc.groupby("Run / BHA #"):
        dp = g[g["Component"].str.contains("Drill Pipe to Surface", case=False)]
        rest = g.drop(dp.index)
        loc = f"{fn}:BHA Components run {int(run)}"
        F.append(Fact(WELL, "bha_length", round(float(rest["Length (ft)"].sum()), 2), "ft", S, loc, qualifier=f"run {int(run)} (sum of components)"))
        F.append(Fact(WELL, "bha_weight_air", round(float((rest["Length (ft)"] * rest["Weight (lb/ft)"]).sum()) / 1000, 2), "klb", S, loc,
                      qualifier=f"run {int(run)} (sum of components)"))
        for _, d in dp.iterrows():
            for qq, col, u in (("dp_od", "OD (in)", "in"), ("dp_id", "ID (in)", "in"), ("dp_weight", "Weight (lb/ft)", "lb/ft")):
                F.append(Fact(WELL, qq, float(d[col]), u, S, loc, qualifier=f"run {int(run)} drill pipe", raw=str(d["Component"])))

    bt = _xl("halliburton", "Bits", 1)
    for i, r in bt[bt["Bit #"].notna()].iterrows():
        loc, q = f"{fn}:Bits row {i + 3}", f"bit {r['Bit #']}"
        F.append(Fact(WELL, "bit_size", float(r["Size (in)"]), "in", S, loc, qualifier=q, raw=f"{r['Maker']} {r['Model']}"))
        F.append(Fact(WELL, "tfa", float(r["TFA (in²)"]), "in2", S, loc, qualifier=q))
        F.append(Fact(WELL, "nozzles", str(r["Nozzles (32nds)"]), "32nds", S, loc, qualifier=q))
        F.append(Fact(WELL, "bit_depth_in", float(r["MD in (ft)"]), "ft", S, loc, qualifier=q))
        F.append(Fact(WELL, "bit_depth_out", float(r["MD out (ft)"]), "ft", S, loc, qualifier=q))

    sv = _xl("halliburton", "Surveys", 1)
    for i, r in sv.iterrows():
        for q, col in (("survey_inclination", "Inclination (°)"), ("survey_azimuth", "Azimuth (°, grid)")):
            F.append(Fact(WELL, q, float(r[col]), "deg", S, f"{fn}:Surveys row {i + 3}", depth=float(r["MD (ft)"]), qualifier="definitive survey"))

    md = _xl("halliburton", "Mud", 1)
    for i, r in md.iterrows():
        if not str(r["Source"]).startswith("Daily mud report"):
            continue
        loc, t = f"{fn}:Mud row {i + 3}", pd.Timestamp(r["Time"])
        F.append(Fact(WELL, "mud_weight", float(r["MW (ppg)"]), "ppg", S, loc, time=t, depth=num(r["Depth (ft)"]), qualifier="Baroid daily mud report"))
        notes = str(r["Notes"])
        F += text_facts(notes, S, loc, t)
        C.append(Chunk(WELL, S, loc, f"[{t:%Y-%m-%d %H:%M} Baroid mud report] {notes}", time=t))

    ds = _xl("halliburton", "Daily Summary", 1)
    last_casing = None
    for i, r in ds.iterrows():
        t = pd.to_datetime(r["Date"], errors="coerce")
        if pd.isna(t):
            continue
        loc = f"{fn}:Daily Summary row {i + 3}"
        text = f"[{t:%Y-%m-%d} Halliburton daily summary] Depth {r['Depth']}. Casing: {r['Casing']}. Last 24 h: {r['Activity last 24 h']} Current: {r['Current activity']}"
        C.append(Chunk(WELL, S, loc, text, time=t))
        F += text_facts(f"{r['Activity last 24 h']} {r['Current activity']}", S, loc, t)
        cas = str(r["Casing"])
        if cas != last_casing:                                 # the casing column repeats daily
            F += text_facts(cas, S, loc, t)
            last_casing = cas
        dm = re.match(r"\s*([\d,.]+)\s*['’]\s*MD", str(r["Depth"]))
        if dm:
            F.append(Fact(WELL, "hole_depth", num(dm[1]), "ft", S, loc, time=t, depth=num(dm[1]), qualifier="daily summary"))
    return F, C


# ---------------------------------------------------------------------------- EOW
def eow_chunks(max_chars=1200):
    import fitz
    doc = fitz.open(config.LSU_FILES["eow"])
    out = []
    for p, page in enumerate(doc, 1):
        paras = [x.strip() for x in page.get_text().split("\n\n") if x.strip()]
        text = " ".join(" ".join(paras).split())
        for k in range(0, len(text), max_chars):
            out.append(Chunk(WELL, "lsu_eow_report", f"{config.LSU_FILES['eow'].name}:p{p}", text[k:k + max_chars]))
    return out


def survey_arrays():
    """Definitive survey (Halliburton): md, inc_deg, azi_deg."""
    s = _xl("halliburton", "Surveys", 1).apply(pd.to_numeric, errors="coerce").dropna(subset=["MD (ft)"])
    return s["MD (ft)"].values, s["Inclination (°)"].values, s["Azimuth (°, grid)"].values
