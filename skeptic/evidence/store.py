"""Fact store: every number the agent can see, with where it came from.

A Fact is one value of one quantity as written in one source. Conflicts are not
resolved here: get() returns *every* candidate so that disagreements stay visible.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional

import pandas as pd

# Canonical quantity names (the agent sees this list in the tool description).
QUANTITIES = {
    "mud_weight": "drilling-fluid density (ppg)",
    "casing_shoe_depth": "casing shoe / set depth (ft MD); qualifier = casing OD in inches",
    "dp_id": "drill-pipe inside diameter (in)",
    "dp_od": "drill-pipe outside diameter (in)",
    "dp_weight": "drill-pipe nominal weight (lb/ft)",
    "bit_size": "bit diameter (in); qualifier = bit number",
    "bit_depth_in": "depth a bit run started (ft MD); qualifier = bit number",
    "bit_depth_out": "depth a bit run ended (ft MD); qualifier = bit number",
    "tfa": "bit total flow area (in^2); qualifier = bit number or run",
    "nozzles": "bit nozzle sizes (32nds), text; qualifier = bit number or run",
    "hole_depth": "measured depth of the hole (ft MD) as reported",
    "pickup_weight": "pick-up hook load; unit as written (klb or lb)",
    "slackoff_weight": "slack-off hook load; unit as written (klb or lb)",
    "rotating_weight": "rotating off-bottom hook load; unit as written (klb or lb)",
    "string_weight": "string weight as reported; unit as written",
    "torque": "rotary / top-drive torque as reported",
    "spp": "standpipe pressure (psi)",
    "flow_rate": "pump flow rate (gpm)",
    "spm": "pump strokes per minute; qualifier = pump",
    "liner_size": "pump liner diameter (in); qualifier = pump",
    "stroke_length": "pump stroke length (in); qualifier = pump",
    "pump_efficiency": "pump volumetric efficiency (%); qualifier = pump",
    "survey_inclination": "survey inclination (deg) at depth = MD",
    "survey_azimuth": "survey azimuth (deg) at depth = MD",
    "bha_length": "BHA length (ft); qualifier = BHA/run",
    "bha_weight_air": "BHA weight in air (klb); qualifier = BHA/run",
    "bha_weight_mud": "BHA weight in mud (klb); qualifier = BHA/run",
    "plastic_viscosity": "mud plastic viscosity (cP)",
    "yield_point": "mud yield point (lb/100 ft^2)",
    "funnel_viscosity": "mud funnel viscosity (s/qt)",
    "bop_test_date": "BOP test date (text); qualifier = last / next",
    "report_date": "date printed on a report",
}


@dataclass
class Fact:
    well: str                       # "forge" | "lsu"
    quantity: str                   # key of QUANTITIES
    value: object                   # float, or text (nozzles, dates)
    unit: str                       # unit as declared by the source ("" = not declared)
    source: str                     # e.g. "forge_daily_report", "lsu_tour_sheet"
    locator: str                    # file + sheet/row or page/line, for provenance
    time: Optional[pd.Timestamp] = None
    depth: Optional[float] = None   # ft MD the value refers to (survey station, report depth)
    qualifier: str = ""             # casing OD, bit no, pump no, "with pumps", run ...
    raw: str = ""                   # verbatim text the value was read from
    id: str = field(default="")


class FactStore:
    def __init__(self, facts=None):
        self.df = pd.DataFrame(columns=[f for f in Fact.__dataclass_fields__])
        if facts:
            self.add(facts)

    def add(self, facts):
        rows = [asdict(f) for f in facts]
        new = pd.DataFrame(rows)
        start = len(self.df)
        new["id"] = [f"F{start + i:05d}" for i in range(len(new))]
        self.df = pd.concat([self.df, new], ignore_index=True) if len(self.df) else new
        return self

    def copy(self):
        out = FactStore()
        out.df = self.df.copy()
        return out

    def get(self, quantity, qualifier=None, t0=None, t1=None, d0=None, d1=None, source=None, limit=60):
        """All candidate values, oldest first. Filters are optional and inclusive.
        qualifier matches as a case-insensitive substring."""
        x = self.df[self.df.quantity == quantity]
        if qualifier:
            x = x[x.qualifier.astype(str).str.contains(str(qualifier), case=False, regex=False)]
        if source:
            x = x[x.source.str.contains(source, case=False, regex=False)]
        if t0 is not None or t1 is not None:      # facts without a time drop out here
            t = pd.to_datetime(x.time)
            keep = pd.Series(True, index=x.index)
            if t0 is not None:
                keep &= t >= pd.Timestamp(t0)
            if t1 is not None:
                keep &= t <= pd.Timestamp(t1)
            x = x[keep]
        if d0 is not None:
            x = x[pd.to_numeric(x.depth, errors="coerce") >= d0]
        if d1 is not None:
            x = x[pd.to_numeric(x.depth, errors="coerce") <= d1]
        x = x.sort_values(["time", "depth"], na_position="first")
        return x.head(limit)

    def set_value(self, fact_id, value, raw=None):
        """Used only by the fault injector."""
        i = self.df.index[self.df.id == fact_id]
        if len(i) != 1:
            raise KeyError(fact_id)
        self.df.loc[i, "value"] = value
        if raw is not None:
            self.df.loc[i, "raw"] = raw

    def __len__(self):
        return len(self.df)
