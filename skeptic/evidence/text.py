"""Text evidence: report narratives, rig memos, EOW pages, as time/depth-anchored chunks,
searched with BM25 (Robertson & Zaragoza 2009), written out here rather than imported.

Drilling text is full of codes and numbers ("SPP-2200", "5.200'", "P/U 210K"), so the
tokenizer keeps digits and splits on anything that is not a letter, digit or dot.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("the a an and or of to in at on for with from by is was were be as it this that "
           "f t w".split())        # 'f'/'t' come from crew shorthand F/ 300' T/ 400'
SYNONYMS = {"ssp": "spp", "pu": "pickup", "so": "slackoff", "mw": "mudweight", "mwi": "mudweight",
            "mwo": "mudweight", "tih": "rih", "trip": "trip", "pooh": "pooh", "tooh": "pooh",
            "fv": "floatvalve", "xfer": "transfer"}


def tokenize(text):
    toks = TOKEN.findall(str(text).lower())
    return [SYNONYMS.get(t, t) for t in toks if t not in STOP]


@dataclass
class Chunk:
    well: str
    source: str
    locator: str
    text: str
    time: Optional[pd.Timestamp] = None
    depth: Optional[float] = None
    id: str = ""


class TextIndex:
    """BM25 with the usual k1 = 1.5, b = 0.75."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        for i, c in enumerate(self.chunks):
            c.id = c.id or f"T{i:05d}"
        self.k1, self.b = k1, b
        self._build()

    def _build(self):
        self.tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}

    def copy_with(self, replace=None, append=None):
        """New index with some chunk texts replaced and/or chunks appended (fault injector)."""
        chunks = [Chunk(**asdict(c)) for c in self.chunks]
        for cid, text in (replace or {}).items():
            for c in chunks:
                if c.id == cid:
                    c.text = text
        chunks += list(append or [])
        return TextIndex(chunks, self.k1, self.b)

    def score(self, query):
        """BM25 score of every chunk for the query."""
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out.append(s)
        return out

    def search(self, query, k=8, t0=None, t1=None, source=None):
        """Top-k chunks by BM25 score, optionally limited to a time window or a source."""
        scores = self.score(query)
        hits = []
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if source and source.lower() not in c.source.lower():
                continue
            if t0 is not None and (c.time is None or pd.isna(c.time) or c.time < pd.Timestamp(t0)):
                continue
            if t1 is not None and (c.time is None or pd.isna(c.time) or c.time > pd.Timestamp(t1)):
                continue
            hits.append((round(scores[i], 2), c))
            if len(hits) >= k:
                break
        return hits

    def get(self, chunk_id):
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        raise KeyError(chunk_id)
