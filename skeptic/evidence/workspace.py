"""A Workspace is everything the agent may look at for one well: facts, text chunks
and rig time series. The fault injector forks a workspace and corrupts the copy, so
every benchmark task gets its own view of the data and the original stays clean."""
import pickle
from dataclasses import dataclass, field

import pandas as pd

from .. import config
from .store import FactStore
from .text import TextIndex


@dataclass
class Workspace:
    well: str
    facts: FactStore
    text: TextIndex
    edr: dict                                  # name -> DataFrame with column 't' + canonical channels
    meta: dict = field(default_factory=dict)   # survey arrays etc.
    notes: list = field(default_factory=list)  # what the injector changed (never shown to the agent)

    def fork(self):
        """Copy that can be corrupted without touching this one. EDR frames are copied
        shallowly: the injector must assign new column arrays, never write in place."""
        return Workspace(self.well, self.facts.copy(), self.text.copy_with(),
                         {k: v.copy(deep=False) for k, v in self.edr.items()}, dict(self.meta), list(self.notes))


def _build(well):
    if well == "forge":
        from . import forge
        facts, chunks = forge.parse_reports()
        facts += forge.survey_facts()
        edr = {"edr": forge.load_edr()}
        md, inc, az = forge.load_survey()
    elif well == "lsu":
        from . import lsu
        edr_df = lsu.load_edr()
        f1, c1 = lsu.memo_evidence(edr_df)
        f2, c2 = lsu.tour_evidence()
        f3, c3 = lsu.halliburton_evidence()
        facts, chunks = f1 + f2 + f3, c1 + c2 + c3 + lsu.eow_chunks()
        edr = {"edr": edr_df.drop(columns=["memo"]), "wits": lsu.load_wits()}
        md, inc, az = lsu.survey_arrays()
    else:
        raise ValueError(well)
    return facts, chunks, edr, dict(survey=(md, inc, az))


_CACHE = {}


def load(well, use_cache=True):
    """Workspace for 'forge' or 'lsu' (parsed once, then cached on disk and in memory)."""
    if use_cache and well in _CACHE:
        return _CACHE[well].fork()
    f = config.CACHE / f"{well}_evidence.pkl"
    if use_cache and f.exists():
        facts, chunks, meta = pickle.loads(f.read_bytes())
        if well == "forge":
            from . import forge
            edr = {"edr": forge.load_edr()}
        else:
            from . import lsu
            edr = {"edr": lsu.load_edr().drop(columns=["memo"]), "wits": lsu.load_wits()}
    else:
        facts, chunks, edr, meta = _build(well)
        config.CACHE.mkdir(exist_ok=True)
        f.write_bytes(pickle.dumps((facts, chunks, meta)))
    ws = Workspace(well, FactStore(facts), TextIndex(chunks), edr, meta)
    _CACHE[well] = ws
    return ws.fork()
