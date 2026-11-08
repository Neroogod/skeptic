"""Phase 1 gate: parsers return the values printed in the source files.
Golden values were read by hand from the files named in each test."""
import pandas as pd
import pytest

from skeptic import config
from skeptic.evidence import forge, workspace
from skeptic.evidence.store import Fact, FactStore
from skeptic.evidence.text import TextIndex, Chunk, tokenize

needs_forge = pytest.mark.skipif(not config.forge_available(), reason="FORGE data not found")
needs_lsu = pytest.mark.skipif(not config.lsu_available(), reason="LSU data not found")


# ------------------------------------------------------------------ store / text
def test_factstore_returns_every_candidate():
    fs = FactStore([Fact("x", "mud_weight", 12.0, "ppg", "a", "a:1", time=pd.Timestamp("2026-06-01")),
                    Fact("x", "mud_weight", 10.5, "ppg", "b", "b:1", time=pd.Timestamp("2026-06-03")),
                    Fact("x", "dp_id", 4.276, "in", "a", "a:2")])
    got = fs.get("mud_weight")
    assert sorted(got.value) == [10.5, 12.0]
    assert list(fs.get("mud_weight", t0="2026-06-02").value) == [10.5]
    assert list(fs.get("mud_weight", source="a").value) == [12.0]
    assert fs.df.id.is_unique


def test_bm25_ranks_the_relevant_chunk_first():
    idx = TextIndex([Chunk("x", "s", "1", "Drill ahead, SSP 2300, 575 gpm"),
                     Chunk("x", "s", "2", "Trip out of hole and break off bit"),
                     Chunk("x", "s", "3", "Fill pipe every 20 stands, float valve holding")])
    assert idx.search("standpipe SPP", k=1)[0][1].locator == "1"      # SSP is mapped to spp
    assert idx.search("float valve fill", k=1)[0][1].locator == "3"
    assert tokenize("5.200' F/300' T/400'") == ["5.200", "300", "400"]


# ------------------------------------------------------------------------ FORGE
@needs_forge
def test_forge_report_golden_11162020():
    """drilling__..._11162020_..._21_reporttmp.txt (report 28)."""
    p = next(config.FORGE_FILES["reports"].glob("*11162020*"))
    facts, chunks = forge.parse_report(p)
    fs = FactStore(facts)
    v = lambda q, qual=None: sorted(fs.get(q, qualifier=qual).value)
    assert v("hole_depth") == [5390.0]
    assert v("casing_shoe_depth", "9.63") == [5113.0]
    assert v("mud_weight", "mud check") == [8.8]
    assert v("pickup_weight", "report summary") == [220000.0]
    assert v("slackoff_weight", "report summary") == [190000.0]
    assert v("pickup_weight", "narrative") == [210.0, 230.0]          # '210K' and '230' (no unit)
    assert v("spp", "narrative") == [2300.0, 2320.0]                  # written 'SSP'
    assert v("spp", "pump 2") == [2800.0]
    assert v("flow_rate", "pump 2") == [400.0]
    assert v("bop_test_date", "last") == ["12/14/20"]
    assert v("tfa", "bit 9") == [0.91]
    assert v("survey_inclination") == [1.56, 1.6, 1.63, 1.67]
    units = dict(zip(fs.get("pickup_weight", qualifier="narrative").value, fs.get("pickup_weight", qualifier="narrative").unit))
    assert units == {210.0: "klb", 230.0: ""}
    # time anchoring: the 06:00 entry belongs to the day before the report date
    first = [c for c in chunks if "06:00-15:30" in c.text][0]
    assert first.time == pd.Timestamp("2020-11-15 06:00")


@needs_forge
def test_forge_narrative_does_not_invent_weights():
    f = forge.narrative_facts("so we pumped a sweep. PU R100 BHA", "forge", "s", "l", None)
    assert not [x for x in f if x.quantity in ("pickup_weight", "slackoff_weight")]


@needs_forge
def test_forge_workspace_has_provenance_and_rebuilt_bit_depth():
    ws = workspace.load("forge")
    df = ws.facts.df
    assert (df.locator.str.len() > 0).all() and (df.source.str.len() > 0).all()
    e = ws.edr["edr"]
    assert {"t", "hookload_klb", "flow_gpm", "spp_psi", "bit_depth_ft", "hole_depth_ft"} <= set(e.columns)
    assert e.t.is_monotonic_increasing                                   # after the DST fix (1 Nov 2020)
    on = e[e.on_bottom > 0.5]
    assert (on.bit_depth_ft - on.hole_depth_ft).abs().max() < 1e-6       # on bottom: bit = hole
    both = e.dropna(subset=["bit_depth_ft", "hole_depth_ft"])
    assert len(both) > 0.9 * len(e)
    assert (both.bit_depth_ft <= both.hole_depth_ft + 1e-6).all()


# -------------------------------------------------------------------------- LSU


