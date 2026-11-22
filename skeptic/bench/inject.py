"""Fault injector. Each injection forks the workspace and corrupts the copy, so the
answer key is known by construction. Specs are plain dicts (stored in the task file):

  stuck_channel  {feed, channel, start, end}            freeze a channel at its value at `start`
  clock_offset   {feed, minutes}                        shift every timestamp of a feed
  depth_shift    {after, feet}                          add `feet` to hole and bit depth after a time
  sensor_drift   {channel, start, end, per_day}         add a linear drift (units/day) to a channel
  fact_value     {fact_id, value, raw}                  replace one recorded value (typo, unit, wrong MW, survey)
  planted_text   {text, time, locator}                  add a document chunk (prompt injection)
"""
import numpy as np
import pandas as pd

from ..evidence.text import Chunk


def apply(ws, spec):
    """Copy of the workspace with one fault planted, as described by spec['type']: stuck_channel,
    clock_offset, depth_shift, sensor_drift, fact_value, planted_text or combo. The original is never changed."""
    w = ws.fork()
    kind = spec["type"]
    if kind == "stuck_channel":
        df = w.edr[spec["feed"]]
        m = (df.t >= pd.Timestamp(spec["start"])) & (df.t <= pd.Timestamp(spec["end"]))
        v = df[spec["channel"]].values.copy()
        i0 = np.argmax(m.values)
        v[m.values] = v[i0] if np.isfinite(v[i0]) else np.nanmedian(v[m.values])
        w.edr[spec["feed"]] = df.assign(**{spec["channel"]: v})
    elif kind == "clock_offset":
        df = w.edr[spec["feed"]]
        w.edr[spec["feed"]] = df.assign(t=df.t + pd.Timedelta(minutes=spec["minutes"]))
    elif kind == "depth_shift":
        df = w.edr["edr"]
        after = (df.t >= pd.Timestamp(spec["after"])).values
        new = {c: np.where(after, df[c].values + spec["feet"], df[c].values) for c in ("hole_depth_ft", "bit_depth_ft") if c in df}
        w.edr["edr"] = df.assign(**new)
    elif kind == "sensor_drift":
        df = w.edr["edr"]
        t0, t1 = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
        days = ((df.t - t0).dt.total_seconds() / 86400.0).clip(lower=0)
        days = np.where(df.t > t1, (t1 - t0).total_seconds() / 86400.0, days)
        w.edr["edr"] = df.assign(**{spec["channel"]: df[spec["channel"]].values + spec["per_day"] * days})
    elif kind == "fact_value":
        w.facts.set_value(spec["fact_id"], spec["value"], spec.get("raw"))
    elif kind == "planted_text":
        w.text = w.text.copy_with(append=[Chunk(w.well, spec.get("source", "forge_daily_report"), spec["locator"], spec["text"],
                                                time=pd.Timestamp(spec["time"]))])
    elif kind == "combo":
        for s in spec["parts"]:
            w = apply(w, s)
    else:
        raise ValueError(f"unknown injection {kind}")
    w.notes.append(spec)
    return w
