"""Build the CSC 7700 final-report deck (10 slides, in the order the course asks for).

Every result number is computed here from the committed runs (results/test_set, FORGE only), the
task files (bench/public) and results/v4_friction_windows.json, so the deck cannot drift from the
code. The few development-set numbers come from RESULTS.md (those runs include LSU data and stay
local). Figures come from slides/build_figures.py (run it first).

    .venv/bin/python slides/build_figures.py
    .venv/bin/python slides/build_slides.py      ->  slides/Skeptic_CSC7700_Oghale.pptx

Dark theme with LSU gold; Arial / Arial Narrow so it renders the same on any machine.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skeptic.agent import harness, tools as T         # noqa: E402
from skeptic.bench import report                      # noqa: E402

FIG = ROOT / "slides" / "figures"
OUT = ROOT / "slides" / "Skeptic_CSC7700_Oghale.pptx"
REPO = "github.com/Neroogod/skeptic"
rgb = RGBColor.from_string
BG1, BG2, PANEL, EDGE = rgb("1D1233"), rgb("0B0714"), rgb("1A1130"), rgb("3A2C5A")
WHITE, TEXT, MUTED, GOLD = rgb("FFFFFF"), rgb("F3EEFB"), rgb("A89CC8"), rgb("FDD023")
FX, CHIP = rgb("109AAE"), rgb("4B3A78")                            # teal accent (data / evaluation), tool chips
OK, PART, BAD = rgb("2FA36B"), rgb("C98A00"), rgb("D9483B")         # status chips, always with a word
OK_TXT, BAD_TXT = rgb("6EE7A8"), rgb("FF8A80")
HEAD, BODY, MONO = "Arial Narrow", "Arial", "Menlo"
FOOTER = "VICTORY OGHALE  ·  CSC 7700  ·  LSU  ·  FALL 2026"


# =========================================================================== numbers
def critic_replies(path):
    """How the critic's replies went in the scored Skeptic runs: all, cut off, cut off while rejecting."""
    c = Counter()
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        if r["row"]["config"] != "skeptic":
            continue
        for rv in r["trace"].get("skeptic_reviews") or []:
            cut = bool(rv.get("parse_error"))
            c["all"] += 1
            c["cut"] += cut
            c["cut_reject"] += cut and '"reject"' in rv["parse_error"]
    return c


def numbers():
    """Everything the slides quote, from the committed test-set runs and task files."""
    runs = [ROOT / "results" / "test_set" / "llm", ROOT / "results" / "test_set" / "fixed"]
    df = report.load_rows([str(d) for d in runs])
    df = df[df.task.str.startswith("test-")]
    t = report.table(df).set_index("config")
    sure = {c: df[(df.config == c) & ~df.abstained & (df.confidence >= 0.7)] for c in ("skeptic", "react")}
    tasks = json.loads((ROOT / "bench" / "public" / "forge_test.json").read_text())
    return dict(
        t=t, pair={m: report.paired(df, "skeptic", "react", m) for m in ("correct", "confident_wrong", "abstained")},
        sure={c: (g.correct.mean(), len(g)) for c, g in sure.items()},
        fam=df.pivot_table(index="family", columns="config", values="correct", aggfunc="mean"),
        without_flawed=df[~df.task.isin(["test-clock-control-1", "test-clock-control-3"])].groupby("config").correct.mean().to_dict(),
        critic=critic_replies(runs[0] / "traces.jsonl.gz"),
        runs=int((df.config == "skeptic").sum()), n_test=len(tasks), slices=Counter(x["slice"] for x in tasks),
        n_dev_forge=len(json.loads((ROOT / "bench" / "public" / "forge_tasks.json").read_text())), n_dev_lsu=14,
        dev_acc=dict(skeptic=0.759, react=0.810),                  # RESULTS.md, development set (local runs)
        n_tools=len(T.TOOLS), limits=harness.Config("x"),
        groups={k: len(v) for k, v in (("evidence", harness.EVIDENCE_TOOLS), ("checks", harness.CHECK_TOOLS),
                                       ("physics", harness.PHYSICS_TOOLS), ("ledger", harness.LEDGER_TOOLS))})


# =========================================================================== drawing helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def gold_links():
    """PowerPoint draws hyperlinks in the theme's hyperlink colour, not the run colour: set it to gold."""
    from lxml import etree
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    theme = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(theme.blob)
    for tag in ("hlink", "folHlink"):
        el = root.find(f".//{qn('a:clrScheme')}/{qn('a:' + tag)}")
        for child in list(el):
            el.remove(child)
        etree.SubElement(el, qn("a:srgbClr"), val="FDD023")
    theme._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


gold_links()


def text(s, x, y, w, h, content, font=BODY, size=16, bold=False, color=TEXT, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space=6, spacing=None, link=None):
    """Text box. content: a string or a list of lines; a line may be a list of (text, bold[, colour]) runs."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    for i, line in enumerate(content if isinstance(content, list) else [content]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(space)
        for part in (line if isinstance(line, list) else [(line, bold)]):
            r = p.add_run()
            r.text = part[0]
            r.font.size, r.font.bold = Pt(size), part[1]
            r.font.color.rgb = part[2] if len(part) > 2 else color
            r.font.name = font
            if spacing:                                    # letter spacing, in 1/100 pt
                r._r.get_or_add_rPr().set("spc", str(spacing))
            if link and len(part) > 3:                     # (text, bold, colour, True): this run is the link
                r.hyperlink.address = link
                r.font.underline = False
    return tb


def glow(tb, color="FDD023", radius=120000, alpha=35000):
    """Soft glow behind every run of a text box (DrawingML effectLst, placed before <a:latin>)."""
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            rpr = r._r.get_or_add_rPr()
            eff = rpr.makeelement(qn("a:effectLst"), {})
            g = eff.makeelement(qn("a:glow"), {"rad": str(radius)})
            c = g.makeelement(qn("a:srgbClr"), {"val": color})
            c.append(c.makeelement(qn("a:alpha"), {"val": str(alpha)}))
            g.append(c)
            eff.append(g)
            latin = rpr.find(qn("a:latin"))
            if latin is not None:
                latin.addprevious(eff)
            else:
                rpr.append(eff)


def shape(s, kind, x, y, w, h, fill=PANEL, line=EDGE, weight=1.0, radius=0.06):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb, sh.line.width = line, Pt(weight)
    sh.shadow.inherit = False
    return sh


def panel(s, x, y, w, h, line=EDGE, fill=PANEL, weight=1.0):
    return shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, line, weight)


def bar(s, x, y, w, h, color):
    return shape(s, MSO_SHAPE.RECTANGLE, x, y, w, h, color, None)


def chip(s, x, y, w, label, color, size=11, h=0.3):
    sh = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, color, None, radius=0.5)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, WHITE
    return sh


def arrow(s, x1, y1, x2, y2, color=MUTED, weight=2.0):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb, c.line.width = color, Pt(weight)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def picture(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def background(s):
    f = s.background.fill
    f.gradient()
    f.gradient_angle = 125
    stops = f.gradient_stops
    stops[0].color.rgb, stops[0].position = BG1, 0.0
    stops[1].color.rgb, stops[1].position = BG2, 1.0


def fade(s):
    """Fade transition (appended after <p:clrMapOvr>, which is where <p:transition> belongs)."""
    tr = s._element.makeelement(qn("p:transition"), {"spd": "med"})
    tr.append(tr.makeelement(qn("p:fade"), {}))
    s._element.append(tr)


def slide(kicker, headline):
    s = prs.slides.add_slide(BLANK)
    background(s)
    bar(s, 0.7, 0.52, 0.42, 0.045, GOLD)
    text(s, 1.22, 0.43, 10, 0.3, kicker.upper(), HEAD, 13, True, GOLD, spacing=250)
    if headline:
        text(s, 0.7, 0.78, 12.0, 0.8, headline, HEAD, 32, True, WHITE)
    text(s, 0.7, 7.08, 8, 0.25, FOOTER, HEAD, 10, color=MUTED, spacing=150)
    return s


def notes(s, msg):
    s.notes_slide.notes_text_frame.text = msg


pct = lambda v: f"{100 * v:.1f}%"


def stat(s, x, y, w, h, big, label, sub, color=GOLD):
    """Hero number: big value, a bold label and a muted line under it."""
    panel(s, x, y, w, h)
    bar(s, x, y + 0.25, 0.06, h - 0.5, color)
    text(s, x + 0.3, y + 0.22, w - 0.45, 0.95, big, HEAD, 40 if len(big) > 10 else 50, True, color)
    text(s, x + 0.3, y + 1.15, w - 0.45, 0.4, label, BODY, 15, True, WHITE)
    if sub:
        text(s, x + 0.3, y + 1.52, w - 0.45, 0.5, sub, BODY, 12, color=MUTED)


# =========================================================================== slides
def title(N):
    s = prs.slides.add_slide(BLANK)
    background(s)
    picture(s, "depth_art.png", 6.55, 0.15, h=7.2)
    text(s, 0.7, 0.8, 7, 0.3, "CSC 7700 · AI & LLM DEVELOPMENT · FINAL REPORT", HEAD, 13, True, GOLD, spacing=250)
    glow(text(s, 0.62, 1.35, 6.5, 1.6, "SKEPTIC", HEAD, 100, True, WHITE, spacing=300))
    text(s, 0.7, 3.05, 5.7, 1.5, "A falsification-first LLM agent that checks drilling records against physics, "
         "and abstains when the data cannot decide", BODY, 22, color=TEXT, space=4)
    bar(s, 0.7, 4.75, 1.2, 0.05, GOLD)
    text(s, 0.7, 4.98, 6, 0.45, "Victory Oghale", HEAD, 26, True, WHITE)
    text(s, 0.7, 5.48, 6, 0.35, "MS, Petroleum Engineering  ·  Louisiana State University", BODY, 16, color=MUTED)
    text(s, 0.7, 5.95, 6, 0.35, [[("Code  ", False, MUTED), (REPO, True, GOLD, True)]], BODY, 16, link="https://" + REPO)
    text(s, 6.9, 7.05, 6.1, 0.3, "FORGE 16A rig log, 72 days: hole depth (gold) and every bit trip (violet)",
         BODY, 10, color=MUTED, align=PP_ALIGN.RIGHT)
    notes(s, "Skeptic is an LLM agent for drilling records. It treats every answer as a hypothesis to refute, uses physics "
             "to decide between conflicting sources, and is allowed to say the data cannot decide. The picture is the real "
             "rig log of the public well used for the test set.")


def bluf(N):
    t, p = N["t"], N["pair"]
    s = slide("Bottom line up front", None)
    panel(s, 0.7, 0.95, 11.95, 1.95, line=GOLD, weight=1.25)
    text(s, 1.0, 1.12, 11.4, 1.7, [[
        ("Goal  ", True, GOLD), ("An LLM agent that answers questions about drilling records only as far as the evidence supports, "
                                 "and says so when the data cannot decide.   ", False),
        ("Method  ", True, GOLD), (f"A hand-written agent harness with {N['n_tools']} tools (parsers, data checks, torque-and-drag "
                                   f"physics) and six gates that make it try to refute its own answer, tested on {N['n_test']} "
                                   "held-out tasks with planted faults.   ", False),
        ("Result  ", True, GOLD), (f"Same accuracy as a standard tool-using agent ({t.loc['skeptic', 'accuracy']:.2f} vs "
                                   f"{t.loc['react', 'accuracy']:.2f}), but confidently wrong half as often "
                                   f"({pct(t.loc['skeptic', 'confident_wrong'])} vs {pct(t.loc['react', 'confident_wrong'])}, "
                                   f"p = {p['confident_wrong']['p']:.3f}).", False)]], BODY, 17, space=0)
    panel(s, 0.7, 3.15, 7.05, 3.75)
    text(s, 1.0, 3.3, 6.5, 0.3, "PRIOR WORK", HEAD, 13, True, GOLD, spacing=200)
    rows = [("RAG", "Lewis et al., NeurIPS 2020", "grounds answers in retrieved text; never checks the number"),
            ("ReAct", "Yao et al., ICLR 2023", "reasons and calls tools; no rule for conflicting sources (my baseline)"),
            ("Reflexion · Debate", "Shinn et al. 2023 · Du et al. 2024", "a model critiques a model; no outside referee"),
            ("Strong inference", "Platt, Science 1964", "rival hypotheses and a test that separates them"),
            ("Selective prediction", "Geifman & El-Yaniv, NeurIPS 2017", "abstain when unsure; risk-coverage curves"),
            ("Torque and drag", "Johancsik et al., JPT 1984", "hook-load physics; never asks whether friction is knowable")]
    for i, (name, cite, gap) in enumerate(rows):
        text(s, 1.0, 3.72 + i * 0.52, 6.6, 0.5, [[(name + "  ", True, WHITE), (cite, False, MUTED)], [(gap, False, TEXT)]],
             BODY, 12, space=1)
    panel(s, 8.0, 3.15, 4.65, 3.75)
    text(s, 8.3, 3.3, 4.2, 0.3, "WHY IT MATTERS", HEAD, 13, True, GOLD, spacing=200)
    text(s, 8.3, 3.72, 4.15, 3.1, [
        [("Friction and record errors set the limits ", True, WHITE), ("for running casing, overpull margins and stuck-pipe calls.", False)],
        [("A confident wrong number is worse than \"I can't tell\": ", True, WHITE), ("it gets acted on.", False)],
        [("If it works, ", True, WHITE), ("engineers review only what the agent flags or declines, and can trust its "
                                          "confidence for the rest.", False)],
        [("Real errors it found in public FORGE records: ", True, GOLD), ("a report clock 18 min off, two shoe depths, a wrong "
                                                                           "nozzle area, a trip lighter than physics allows.", False)]],
        BODY, 13, space=9)
    fade(s)
    notes(s, "BLUF in one breath: the goal, the method and the result. Prior work: RAG and ReAct give grounding and tool use, "
             "self-critique gives a model checking a model, strong inference gives the method, selective prediction gives "
             "abstention and the risk-coverage metric, and Johancsik gives the physics. None of them referees conflicting "
             "records with physics or decides when a quantity cannot be known.\n\nReferences: Lewis et al. 2020, RAG for "
             "knowledge-intensive NLP, NeurIPS. Yao et al. 2023, ReAct, ICLR. Shinn et al. 2023, Reflexion, NeurIPS. Du et al. "
             "2024, multiagent debate, ICML. Platt 1964, Strong inference, Science 146. Geifman & El-Yaniv 2017, Selective "
             "classification for deep neural networks, NeurIPS. Johancsik, Friesen & Dawson 1984, Torque and drag in "
             "directional wells, JPT 36(6).")


def card(s, x, y, w, h, head, lines, color=EDGE, mod=None, size=11):
    """Diagram box: bold heading, body lines, optional module path at the bottom."""
    panel(s, x, y, w, h, line=color, weight=1.25)
    text(s, x + 0.14, y + 0.1, w - 0.26, h - 0.2, [[(head, True, WHITE)]] + [[(ln, False, TEXT)] for ln in lines],
         BODY, size, space=2)
    if mod:
        text(s, x + 0.14, y + h - 0.3, w - 0.2, 0.25, mod, MONO, 8, color=MUTED)


def diagram(N):
    g, L = N["groups"], N["limits"]
    s = slide("Technical approach", "From raw rig data to an answer that survived refutation")
    for x, w, name, color in ((0.6, 2.3, "DATA", FX), (3.2, 2.55, "EVIDENCE LAYER", FX), (6.05, 3.25, "AGENT LOOP", GOLD),
                              (9.6, 3.1, "GATES", GOLD)):
        text(s, x, 1.68, w, 0.3, name, HEAD, 12, True, color, spacing=200)
    card(s, 0.6, 2.05, 2.3, 2.05, "FORGE 16A (public)", ["10-s rig log: 608,676 rows, 20 channels", "124 daily-report files",
                                                          "definitive survey"], FX, "evidence/forge.py")
    card(s, 0.6, 4.25, 2.3, 1.45, "LSU H-001 (private)", ["Pason log, tour sheets, contractor workbook, EOW; stays local"],
         EDGE, "evidence/lsu.py")
    card(s, 3.2, 2.05, 2.55, 1.15, "Fact store", ["3,575 facts, 26 quantities; source and line on every value"], FX, "evidence/store.py")
    card(s, 3.2, 3.32, 2.55, 1.0, "BM25 text index", ["333 report chunks, hand-written"], FX, "evidence/text.py")
    card(s, 3.2, 4.44, 2.55, 1.26, "Torque-and-drag export", ["2,620 trip events x 550 elements of 20 ft"], FX, "scripts/export_tnd.py")
    card(s, 6.05, 2.05, 3.25, 1.0, "LLM: Claude Haiku 5.5", ["OpenRouter, OpenAI-format function calls"], GOLD, "agent/llm.py")
    text(s, 10.3, 1.7, 2.45, 0.3, "failed gate: reason goes back", BODY, 10, color=BAD_TXT, align=PP_ALIGN.RIGHT)
    panel(s, 6.05, 3.4, 3.25, 2.3, line=GOLD, weight=1.25)
    text(s, 6.19, 3.5, 3.0, 0.9, [[("Tool registry", True, WHITE)], [("schema-checked arguments, errors returned as text, "
                                                                      "output capped at 4,000 characters", False, TEXT)]], BODY, 11, space=2)
    for i, (lab, n) in enumerate((("Evidence", g["evidence"]), ("Checks", g["checks"]), ("Physics", g["physics"]), ("Ledger", g["ledger"]))):
        chip(s, 6.2 + (i % 2) * 1.5, 4.42 + (i // 2) * 0.38, 1.4, f"{lab}  {n}", CHIP, 11)
    text(s, 6.19, 5.2, 3.0, 0.25, f"budgets: {L.max_steps} steps · ${L.max_cost_usd:.0f} · repeat limit {L.max_repeats}", BODY, 10, color=MUTED)
    text(s, 6.19, 5.43, 3.0, 0.25, "agent/harness.py · agent/tools.py", MONO, 8, color=MUTED)
    arrow(s, 7.45, 3.05, 7.45, 3.4, GOLD)
    arrow(s, 7.9, 3.4, 7.9, 3.05, GOLD)
    gates = ["Cited evidence ids exist", "Every number appears in a tool result", "Two rival hypotheses, tested",
             "Friction only if 'identifiable'", "A critic call tries to refute it", "Document text screened for instructions"]
    panel(s, 9.6, 2.05, 3.1, 2.75, line=GOLD, weight=1.25)
    text(s, 9.75, 2.13, 2.9, 0.3, "submit_answer must pass", MONO, 9, color=GOLD)
    for i, gate in enumerate(gates):
        text(s, 9.75, 2.42 + i * 0.35, 2.9, 0.4, [[(f"{i + 1}  ", True, GOLD), (gate, False, TEXT)]], BODY, 11)
    text(s, 9.75, 4.52, 2.9, 0.25, "agent/verify.py · ledger.py · guard.py", MONO, 8, color=MUTED)
    panel(s, 9.6, 5.0, 3.1, 0.7, line=GOLD, fill=rgb("2A1D45"), weight=1.25)
    text(s, 9.75, 5.08, 2.85, 0.6, [[("Answer  ", True, GOLD), ("verdict, value, confidence, cited evidence, or abstain", False)]], BODY, 11)
    arrow(s, 2.9, 3.0, 3.2, 3.0, FX)
    arrow(s, 2.9, 4.95, 3.2, 4.95, MUTED)
    arrow(s, 5.75, 4.4, 6.05, 4.4, FX)
    arrow(s, 9.3, 2.4, 9.6, 2.4, GOLD)
    arrow(s, 9.6, 2.8, 9.3, 2.8, BAD)
    arrow(s, 11.15, 4.8, 11.15, 5.0, GOLD)
    panel(s, 0.6, 6.0, 12.1, 0.85)
    text(s, 0.8, 6.08, 1.7, 0.3, "EVALUATION", HEAD, 12, True, FX, spacing=200)
    steps = ["plant a fault in a copy", "oracle: the fault is findable", "Skeptic · ReAct · fixed",
             "deterministic answer key", "paired bootstrap, sign-flip"]
    mods = ["bench/inject.py", "bench/build.py", "bench/run.py", "bench/score.py", "bench/report.py"]
    for i, (st, mod) in enumerate(zip(steps, mods)):
        x = 0.8 + i * 2.4
        text(s, x, 6.36, 2.2, 0.3, st, BODY, 11, color=TEXT)
        text(s, x, 6.6, 2.2, 0.22, mod, MONO, 8, color=MUTED)
        if i:
            arrow(s, x - 0.22, 6.47, x - 0.06, 6.47, MUTED, 1.5)
    fade(s)
    notes(s, "Data on the left: the public FORGE well (10-second rig log, daily reports, survey) and the private LSU well, "
             "which stays on my laptop. Parsers turn it into a fact store where every value keeps its source, a hand-written "
             "BM25 index, and the torque-and-drag arrays. The agent loop is my own Python: the model asks for a tool, the "
             "registry validates the arguments and runs it, and the result goes back. When the model submits, six gates check "
             "the answer; a failed gate sends the reason back. The bottom band is how it was evaluated.")


def methods_physics(N):
    s = slide("Methods · evidence and physics", "Physics is the referee between sources")
    blocks = [("Provenance, not merging", "Every value keeps its source and line. Conflicting values all come back; the agent "
                                          "has to decide which is right."),
              ("Checks judged by rig physics", "A flat bit depth in slips is normal; a flat hole depth while drilling is a frozen "
                                               "sensor. Also clock offsets, depth jumps, survey typos, sensor drift, units."),
              ("Friction only when it is knowable", "Soft-string torque and drag with the float-valve fill state, from my PETE 7103 "
                                                    "work. Reported only if precise, the trips agree, and it is physical (0 to 0.6).")]
    for i, (h, b) in enumerate(blocks):
        y = 1.85 + i * 1.62
        bar(s, 0.7, y + 0.05, 0.06, 1.3, GOLD)
        text(s, 0.95, y, 4.6, 1.5, [[(h, True, WHITE)], [(b, False, TEXT)]], BODY, 14, space=4)
    panel(s, 5.85, 1.75, 6.85, 5.1)
    picture(s, "physics_check.png", 6.0, 1.9, w=6.55)
    text(s, 6.05, 5.75, 6.5, 1.0, [[("Physics data check ", True, GOLD), ("(added after the test, from my PETE 7103 error "
                                    "control): no friction makes a pick-up lighter than the frictionless string. The 27 Dec "
                                    "logging trip does, so it is a record or sensor problem, not friction.", False)]], BODY, 12)
    fade(s)
    notes(s, "Three ideas. Provenance: the agent sees every candidate value with its source. Checks: each sensor is judged "
             "against what the rig was doing. Friction: the identifiability test has three parts; on the FORGE logging trips "
             "a fit gives a narrow interval but each trip gives negative friction, so a narrow CI alone is not enough. The "
             "figure: on 13 December every reading sits between zero friction and friction 0.8; on 27 December five pick-ups "
             "weigh 24 to 36 klb less than the string would with no friction at all.")


def methods_gates(N):
    s = slide("Methods · six gates", "Every answer has to survive an attempt to refute it")
    gates = [("Evidence ids", "cited tool calls must exist"), ("Number verifier", "every number must appear in a tool result"),
             ("Hypothesis ledger", "two rival explanations, tested, before concluding"),
             ("Identifiability", "no friction factor unless the tool says it is determined"),
             ("Skeptic critic", "a second model call tries to refute the answer (1 round)"),
             ("Injection guard", "instructions found in documents are marked as data")]
    for i, (name, what) in enumerate(gates):
        y = 1.85 + i * 0.82
        shape(s, MSO_SHAPE.OVAL, 0.7, y + 0.05, 0.5, 0.5, rgb("2A1D45"), GOLD, 1.25)
        text(s, 0.7, y + 0.12, 0.5, 0.35, str(i + 1), HEAD, 17, True, GOLD, align=PP_ALIGN.CENTER)
        text(s, 1.4, y + 0.02, 4.6, 0.7, [[(name, True, WHITE)], [(what, False, TEXT)]], BODY, 13, space=1)
    panel(s, 6.3, 1.75, 6.4, 5.1, line=GOLD, weight=1.25)
    text(s, 6.55, 1.9, 6.0, 0.3, "A REAL RUN · FRICTION, DEEP TRIPS 13-20 DEC 2020", HEAD, 12, True, GOLD, spacing=150)
    steps = [("c1", "friction_identifiability", "identifiable", None),
             ("c2", "friction_by_trip", "5 trips: 0.317, 0.332, 0.348, 0.355, 0.375", None),
             ("c3", "check_hookload_physics", "0 of 335 readings outside the envelope", None),
             ("", "submit_answer", "REJECTED: \"52\" appears in no tool result; no rival hypotheses", "bad"),
             ("", "add_hypothesis", "H1 friction ≈ 0.35 · H2 sensor or string errors, refuted by c3", None),
             ("", "submit_answer", "REJECTED: \"95\" appears in no tool result", "bad"),
             ("", "submit_answer", "ANSWER 0.35 (key 0.35 ± 0.05), confidence 0.7", "ok")]
    for i, (cid, tool, out, kind) in enumerate(steps):
        y = 2.35 + i * 0.6
        dot = {"bad": BAD, "ok": OK}.get(kind, MUTED)
        shape(s, MSO_SHAPE.OVAL, 6.6, y + 0.08, 0.16, 0.16, dot, None)
        if i < len(steps) - 1:
            bar(s, 6.665, y + 0.26, 0.03, 0.42, EDGE)
        out_color = {"bad": BAD_TXT, "ok": OK_TXT}.get(kind, TEXT)
        text(s, 6.95, y, 5.6, 0.6, [[(f"{cid}  " if cid else "", True, MUTED), (tool, True, WHITE)],
                                     [(out, kind is not None, out_color)]], BODY, 12, space=0)
    fade(s)
    notes(s, "The gates are the falsification part. Right: a real run from 22 November. The agent checked identifiability, "
             "fitted friction trip by trip, and ran the physics check. Its first answer was rejected twice over: it quoted a "
             "number no tool had produced, and it had not tested a rival explanation. It then wrote two hypotheses and refuted "
             "'sensor error' with the physics check. A second number problem was caught, and the third answer passed: 0.35, "
             "matching the key. In this run the critic's reply was cut off; that flaw is on the challenges slide.")


def methods_eval(N):
    sl = N["slices"]
    s = slide("Methods · evaluation", "Faults planted in real data, so every answer is known")
    steps = ["Take a real FORGE window", "Plant one fault (10 kinds)", "Oracle: the fault is findable, the control passes",
             "Question + answer key"]
    for i, st in enumerate(steps):
        x = 0.7 + i * 3.05
        panel(s, x, 1.85, 2.75, 0.95, line=FX, weight=1.25)
        text(s, x + 0.15, 1.95, 2.45, 0.75, [[(f"{i + 1}  ", True, FX), (st, True, WHITE)]], BODY, 14, anchor=MSO_ANCHOR.MIDDLE)
        if i:
            arrow(s, x - 0.27, 2.32, x - 0.03, 2.32, FX)
    tiles = [(sl["injected"], "planted faults", "frozen sensor, clock, depth, drift, mud weight, survey, units, planted text, audits", GOLD),
             (sl["control"], "clean controls", "same checks on clean data: they catch false alarms", FX),
             (sl["identify"], "answer or abstain", "friction windows: 3 determined, 4 not", FX)]
    for i, (n, lab, sub, color) in enumerate(tiles):
        stat(s, 0.7 + i * 2.55, 3.1, 2.4, 2.35, str(n), lab, None, color)
        text(s, 1.0 + i * 2.55, 4.6, 2.0, 0.8, sub, BODY, 10, color=MUTED)
    panel(s, 8.5, 3.1, 4.2, 3.75)
    text(s, 8.75, 3.25, 3.8, 0.3, "HOW IT WAS KEPT HONEST", HEAD, 12, True, GOLD, spacing=150)
    text(s, 8.75, 3.65, 3.8, 3.2, [
        [("Development set ", True, WHITE), (f"{N['n_dev_forge'] + N['n_dev_lsu']} tasks used while building ({N['n_dev_lsu']} private LSU).", False)],
        [("Held-out test set ", True, WHITE), (f"{N['n_test']} tasks from other days, hashed and frozen with 4 written predictions before any run.", False)],
        [("Baselines ", True, WHITE), ("ReAct (same model and tools; no ledger, identifiability or critic gate) and a fixed pipeline (no LLM).", False)],
        [("Statistics ", True, WHITE), (f"3 runs each, {N['runs']} per LLM system; paired by task, bootstrap CI, sign-flip p.", False)]],
        BODY, 12, space=8)
    text(s, 0.7, 5.7, 7.5, 1.1, "A planted fault is kept only if its reference check finds it, and a control only if the same check "
         "passes it on clean data. The agents never see the key, the injection or the reference check.", BODY, 13, color=TEXT)
    fade(s)
    notes(s, "Real data has no answer key, so I plant faults where the answer is known. The development set was used to fix "
             "bugs, so its scores flatter the system. The headline results come only from the held-out test set, built from "
             "other days and frozen with predictions before the first run.")


def methods_challenges(N):
    c, wf, fam, dev = N["critic"], N["without_flawed"], N["fam"], N["dev_acc"]
    s = slide("Methods · challenges", "What broke, and what I did about it")
    rows = [
        (f"Skeptic lost to ReAct while building ({dev['skeptic']:.2f} vs {dev['react']:.2f})",
         "Failure analysis: most losses were plumbing (cut-off arguments, runs with no answer), tool bugs, one wrong key. "
         "Fixed for all systems.", "SOLVED", OK),
        ("I had tuned on the development set", "Built a held-out set from other days; froze it with a hash and 4 predictions "
                                               "before any run.", "SOLVED", OK),
        ("A narrow confidence interval read as 'friction is known'", "Identifiability also needs trips that agree and a physical "
                                                                     "value (logging trips gave negative friction).", "SOLVED", OK),
        ("Frozen-sensor check flagged normal flat depths", "Judge each channel by rig physics: bit depth moves only out of slips, "
                                                           "hole depth only while drilling.", "SOLVED", OK),
        (f"After the test: clock check said 'aligned' with no drilling; {c['cut_reject']} critic rejections were cut off and "
         f"counted as accept", f"Both fixed and disclosed. Without the 2 clock tasks all systems score {wf['skeptic']:.3f}; "
                               "the critic fix is not yet scored.", "FIXED LATE", PART),
        ("Open audits: find anything wrong in a whole day", f"Not solved: Skeptic {fam.loc['audit', 'skeptic']:.2f}, ReAct "
                                                            f"{fam.loc['audit', 'react']:.2f}, fixed pipeline {fam.loc['audit', 'fixed']:.2f}.",
         "OPEN", BAD)]
    text(s, 0.95, 1.72, 4.5, 0.3, "PROBLEM", HEAD, 12, True, GOLD, spacing=150)
    text(s, 5.75, 1.72, 5.0, 0.3, "WHAT I DID", HEAD, 12, True, GOLD, spacing=150)
    for i, (prob, fix, status, color) in enumerate(rows):
        y = 2.05 + i * 0.8
        panel(s, 0.7, y, 12.0, 0.72)
        text(s, 0.95, y + 0.06, 4.6, 0.6, prob, BODY, 12, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 5.75, y + 0.06, 5.35, 0.6, fix, BODY, 12, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
        chip(s, 11.3, y + 0.21, 1.25, status, color, 11)
    fade(s)
    notes(s, f"The first test of Skeptic lost. Instead of tuning until it won, I found the causes, fixed them for every system, "
             f"and built a fresh held-out set. Two flaws turned up after the test run. The clock check returned a placeholder "
             f"and 'aligned' on days without drilling. The critic's JSON reply was sometimes cut off by its token limit "
             f"({c['cut']} of {c['all']} reviews) and the harness counted those as accept, including {c['cut_reject']} that were "
             f"rejections. Both are fixed and disclosed; neither fix has been scored.")


def results_main(N):
    t, p, sure = N["t"], N["pair"], N["sure"]
    s = slide(f"Results · held-out test, {N['n_test']} tasks x 3 runs", "Same accuracy. Half the confident errors.")
    cw = p["confident_wrong"]
    stat(s, 0.7, 1.85, 3.85, 2.1, f"{t.loc['skeptic', 'accuracy']:.2f} vs {t.loc['react', 'accuracy']:.2f}",
         "Accuracy, Skeptic vs ReAct", f"p = {p['correct']['p']:.2f}: a tie", MUTED)
    stat(s, 4.75, 1.85, 3.85, 2.1, f"{pct(t.loc['skeptic', 'confident_wrong'])} vs {pct(t.loc['react', 'confident_wrong'])}",
         "Wrong while ≥ 70% sure", f"p = {cw['p']:.3f} · 95% CI {cw['ci95_low']:+.2f} to {cw['ci95_high']:+.2f}")
    stat(s, 8.8, 1.85, 3.85, 2.1, f"{sure['skeptic'][0]:.0%} vs {sure['react'][0]:.0%}", "Right when ≥ 70% sure",
         f"{sure['skeptic'][1]} vs {sure['react'][1]} confident answers")
    panel(s, 0.7, 4.2, 8.0, 2.65)
    picture(s, "headline.png", 0.95, 4.35, w=7.5)
    panel(s, 8.95, 4.2, 3.7, 2.65)
    text(s, 9.2, 4.35, 3.3, 0.3, "THE PRICE", HEAD, 12, True, GOLD, spacing=150)
    text(s, 9.2, 4.75, 3.3, 2.0, [
        [("Abstains more: ", True, WHITE), (f"{pct(t.loc['skeptic', 'abstain'])} vs {pct(t.loc['react', 'abstain'])}", False)],
        [("Costs more: ", True, WHITE), (f"${t.loc['skeptic', 'usd_per_task']:.3f} vs ${t.loc['react', 'usd_per_task']:.3f} per task "
                                         f"({t.loc['skeptic', 'usd_per_task'] / t.loc['react', 'usd_per_task']:.1f}x)", False)],
        [("Works harder: ", True, WHITE), (f"{t.loc['skeptic', 'calls_per_task']:.0f} vs {t.loc['react', 'calls_per_task']:.0f} "
                                           "tool calls per task", False)]], BODY, 13, space=10)
    fade(s)
    notes(s, f"Accuracy is a tie: {t.loc['skeptic', 'accuracy']:.3f} against {t.loc['react', 'accuracy']:.3f}, and the fixed "
             f"pipeline scores {t.loc['fixed', 'accuracy']:.3f}. The difference is in confident errors: "
             f"{pct(t.loc['skeptic', 'confident_wrong'])} against {pct(t.loc['react', 'confident_wrong'])}, p = {cw['p']:.3f}. "
             f"When Skeptic is at least 70% sure it is right {sure['skeptic'][0]:.0%} of the time. The price is more abstentions "
             "and about twice the cost. Confident-wrong was defined before the run but was not one of the four predictions, so "
             "it is a strong secondary finding.")


def results_more(N):
    t, p = N["t"], N["pair"]
    s = slide("Results · what stands out", "Confidence you can act on, and an instructive failure")
    panel(s, 0.7, 1.85, 6.9, 5.0)
    picture(s, "risk_coverage.png", 0.85, 2.0, w=6.6)
    text(s, 0.95, 6.15, 6.5, 0.6, [[("Lower is better. ", True, WHITE), (f"Area under the curve {t.loc['skeptic', 'aurc']:.3f} vs "
                                    f"{t.loc['react', 'aurc']:.3f} (95% CI of the difference −0.19 to −0.04). Skeptic's curve "
                                    "stops early because it abstains on the rest.", False)]], BODY, 11)
    panel(s, 7.85, 1.85, 4.85, 3.05)
    text(s, 8.1, 2.0, 4.4, 0.3, "PREDICTIONS WRITTEN BEFORE THE RUN", HEAD, 12, True, GOLD, spacing=150)
    preds = [("Fewer false alarms than ReAct", f"{pct(t.loc['skeptic', 'false_alarm'])} vs {pct(t.loc['react', 'false_alarm'])}, "
                                               "not significant", "HELD", OK),
             ("Better on answer-or-abstain", f"{t.loc['skeptic', 'acc_identify']:.2f} vs {t.loc['react', 'acc_identify']:.2f}", "FAILED", BAD),
             ("Overall accuracy: no prediction", f"{p['correct']['diff']:+.3f}, p = {p['correct']['p']:.2f}", "TIE", PART),
             ("About twice the cost", f"{t.loc['skeptic', 'usd_per_task'] / t.loc['react', 'usd_per_task']:.1f}x", "HELD", OK)]
    for i, (pr, val, status, color) in enumerate(preds):
        y = 2.45 + i * 0.6
        text(s, 8.1, y, 3.2, 0.55, [[(pr, True, WHITE)], [(val, False, MUTED)]], BODY, 12, space=0)
        chip(s, 11.45, y + 0.1, 1.05, status, color, 11)
    panel(s, 7.85, 5.1, 4.85, 1.75, line=GOLD, weight=1.25)
    text(s, 8.1, 5.22, 4.4, 1.55, [[("Why prediction 2 failed: ", True, GOLD), ("after building, the friction tool itself "
                                    "tested precision, agreement and physical range, and ReAct simply followed its verdict. "
                                    "Putting the judgement into the tool helped every agent.", False)]], BODY, 13)
    fade(s)
    notes(s, "Risk-coverage: if each system answers only the questions it is most sure about, how many of those answers are "
             "wrong? Skeptic's curve is lower everywhere. Of the four written predictions two held, one had no direction, and "
             "one failed. The failure is instructive: once the judgement lives in the tool, the extra gate in the loop adds "
             "nothing.")


def conclusion(N):
    t = N["t"]
    s = slide("Conclusion", "Tools carry accuracy. Refutation makes confidence trustworthy.")
    cols = [("KEY TAKEAWAYS", GOLD, [
                ("Accuracy comes from the tools:", f" Skeptic, ReAct and a script with no LLM all score "
                                                   f"{t.accuracy.min():.2f}-{t.accuracy.max():.2f}."),
                ("Trying to refute the answer", f" halves confident errors ({pct(t.loc['skeptic', 'confident_wrong'])} vs "
                                                f"{pct(t.loc['react', 'confident_wrong'])}) at the price of more \"I can't tell\"."),
                ("Physics catches what text cannot:", " a trip lighter than a frictionless string is a record problem, not friction.")]),
            ("LIMITATIONS", BAD_TXT, [
                ("One model, one public test well:", " Claude Haiku 5.5 on FORGE 16A; LSU results stay private."),
                ("Secondary finding:", " confident-wrong was defined before the run, not one of the 4 predictions."),
                ("Two flaws found after the run", " (clock placeholder, cut-off critic replies), both disclosed."),
                ("Open audits stay hard:", f" Skeptic {N['fam'].loc['audit', 'skeptic']:.2f}.")]),
            ("FUTURE WORK", FX, [
                ("Score the post-test fixes", " (physics data check, critic) on a new held-out set."),
                ("More wells and models;", " calibrate confidence on validation data."),
                ("Standpipe-pressure questions", " from the original proposal."),
                ("Cheaper runs:", f" {t.loc['skeptic', 'calls_per_task']:.0f} tool calls per task vs ReAct's "
                                  f"{t.loc['react', 'calls_per_task']:.0f}.")])]
    for i, (name, color, items) in enumerate(cols):
        x = 0.7 + i * 4.05
        panel(s, x, 1.85, 3.85, 4.35)
        bar(s, x + 0.2, 1.85, 3.45, 0.06, color)
        text(s, x + 0.25, 2.05, 3.4, 0.3, name, HEAD, 13, True, color, spacing=200)
        text(s, x + 0.25, 2.5, 3.4, 3.6, [[(a, True, WHITE), (b, False)] for a, b in items], BODY, 14, space=12)
    text(s, 0.7, 6.42, 12, 0.4, [[("Code  ", False, MUTED), (REPO, True, GOLD, True), ("      Live demo  ", False, MUTED),
                                  ("streamlit run app.py", True, WHITE), ("      Data  ", False, MUTED),
                                  ("Utah FORGE 16A(78)-32, DOI 10.15121/1776602, CC BY 4.0", False, TEXT)]], BODY, 12,
         link="https://" + REPO)
    fade(s)
    notes(s, "The claim is narrow and supported: the falsification loop does not make the agent more accurate, but it makes "
             "its confidence something you can rely on. The limits are stated, including the two flaws found after the run. "
             "Next is a new held-out set to score the changes made since.")


if __name__ == "__main__":
    N = numbers()
    title(N)
    for f in (bluf, diagram, methods_physics, methods_gates, methods_eval, methods_challenges, results_main, results_more, conclusion):
        f(N)
    for i, s in enumerate(prs.slides):
        if i:
            text(s, 11.7, 7.08, 0.95, 0.25, f"{i + 1} / {len(prs.slides)}", HEAD, 10, color=MUTED, align=PP_ALIGN.RIGHT)
    fade(prs.slides[0])
    prs.save(OUT)
    print("wrote", OUT, len(prs.slides), "slides")
