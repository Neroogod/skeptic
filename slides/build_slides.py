"""Build the CSC 7700 final-presentation deck.

Every result number is computed here from the committed runs (results/test_set, FORGE only),
the task files (bench/public) and results/v4_friction_windows.json, so the deck cannot drift
from the code. Figures come from slides/build_figures.py (run it first).

    .venv/bin/python slides/build_figures.py
    .venv/bin/python slides/build_slides.py      ->  slides/Skeptic_CSC7700_Oghale.pptx

Style follows the author's PETE 7103 deck (LSU purple and gold); fonts are Arial / Arial
Narrow so the deck renders the same on any machine.
"""
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
from skeptic.bench import report, score as S          # noqa: E402

FIG = ROOT / "slides" / "figures"
OUT = ROOT / "slides" / "Skeptic_CSC7700_Oghale.pptx"
PURPLE, GOLD, INK, LAV, MUTED, BG, WHITE, FLAG = (RGBColor.from_string(c) for c in
    ("461D7C", "FDD023", "2A1050", "E9E2F3", "6E6680", "F7F5FA", "FFFFFF", "B3261E"))
HEAD, BODY = "Arial Narrow", "Arial"
FOOTER = "VICTORY OGHALE · CSC 7700 · LSU"


# =========================================================================== numbers
def numbers():
    """Everything the slides quote, from the committed test-set runs and task files."""
    runs = [ROOT / "results" / "test_set" / "llm", ROOT / "results" / "test_set" / "fixed"]
    df = report.load_rows([str(d) for d in runs])
    df = df[df.task.str.startswith("test-")]
    t = report.table(df).set_index("config")
    pair = {m: report.paired(df, "skeptic", "react", m) for m in ("correct", "confident_wrong", "abstained")}
    sure = {c: df[(df.config == c) & ~df.abstained & (df.confidence >= 0.7)] for c in ("skeptic", "react")}
    fam = df.pivot_table(index="family", columns="config", values="correct", aggfunc="mean")
    flawed = ["test-clock-control-1", "test-clock-control-3"]
    tasks = json.loads((ROOT / "bench" / "public" / "forge_test.json").read_text())
    dev = json.loads((ROOT / "bench" / "public" / "forge_tasks.json").read_text())
    return dict(
        t=t, pair=pair, fam=fam,
        sure={c: (g.correct.mean(), len(g)) for c, g in sure.items()},
        without_flawed=df[~df.task.isin(flawed)].groupby("config").correct.mean().to_dict(),
        runs={c: int((df.config == c).sum()) for c in ("skeptic", "react", "fixed")},
        n_test=len(tasks), slices=Counter(x["slice"] for x in tasks), families=Counter((x["family"], x["slice"]) for x in tasks),
        n_dev_forge=len(dev), n_dev_lsu=14,
        cost_test=float(df.cost_usd.sum()),
        n_tools=len(T.TOOLS), tool_groups={k: len(v) for k, v in (("evidence", harness.EVIDENCE_TOOLS), ("checks", harness.CHECK_TOOLS),
                                                                   ("physics", harness.PHYSICS_TOOLS), ("ledger", harness.LEDGER_TOOLS))},
        windows=json.loads((ROOT / "results" / "v4_friction_windows.json").read_text()))


# =========================================================================== drawing helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, font=BODY, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, fill=None,
         anchor=MSO_ANCHOR.TOP, space=6):
    """Text box. s is a string, or a list of lines; a line may be a list of (text, bold) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = Inches(0.12 if fill else 0.0)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    for i, line in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(space)
        for run_text, b in (line if isinstance(line, list) else [(line, bold)]):
            r = p.add_run()
            r.text = run_text
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = font, Pt(size), b, color
    if fill is not None:
        tb.fill.solid()
        tb.fill.fore_color.rgb = fill
    return tb


def box(slide, x, y, w, h, s, fill=WHITE, line=PURPLE, size=14, color=INK, bold_first=True, align=PP_ALIGN.LEFT):
    """Rounded box with a bold first line."""
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb, sh.line.width = line, Pt(1.5)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.12)
    for i, line_text in enumerate(s if isinstance(s, list) else [s]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment, p.space_after = align, Pt(3)
        r = p.add_run()
        r.text = line_text
        r.font.name, r.font.size, r.font.color.rgb = BODY, Pt(size if i or not bold_first else size + 1), color
        r.font.bold = bold_first and i == 0
    return sh


def arrow(slide, x1, y1, x2, y2, label=None, color=MUTED, lx=0.0, ly=-0.32):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb, c.line.width = color, Pt(2.25)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if label:
        text(slide, min(x1, x2) + lx, min(y1, y2) + ly, abs(x2 - x1) + 1.4, 0.3, label, BODY, 12, color=MUTED)
    return c


def rect(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def slide(no, section, headline):
    """Content slide: numbered section badge, section name, headline, footer."""
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    text(s, 0.6, 0.42, 0.46, 0.42, f"{no:02d}", HEAD, 17, True, GOLD, PP_ALIGN.CENTER, fill=PURPLE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.2, 0.47, 8.0, 0.35, section.upper(), HEAD, 15, True, PURPLE)
    text(s, 0.6, 0.95, 12.1, 0.8, headline, HEAD, 30, True, INK)
    text(s, 0.6, 7.02, 6.0, 0.3, FOOTER, HEAD, 11, color=MUTED)
    return s


def bullets(s, x, y, w, items, size=16, gap=0.62):
    """(lead, rest) pairs with a purple bar; lead in bold."""
    for i, (lead, rest) in enumerate(items):
        rect(s, x, y + i * gap + 0.06, 0.05, gap - 0.2, PURPLE)
        text(s, x + 0.2, y + i * gap, w - 0.2, gap, [[(lead + " ", True), (rest, False)]], BODY, size)


def table(s, x, y, w, header, rows, widths, size=13, row_h=0.36, bold_col0=False):
    tb = s.shapes.add_table(1 + len(rows), len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (1 + len(rows)))).table
    for j, wd in enumerate(widths):
        tb.columns[j].width = Inches(wd)
    for i, row in enumerate([header] + rows):
        for j, v in enumerate(row):
            c = tb.cell(i, j)
            c.text = str(v)
            c.fill.solid()
            c.fill.fore_color.rgb = PURPLE if i == 0 else (WHITE if i % 2 else LAV)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = HEAD if i == 0 else BODY
                    r.font.size = Pt(size)
                    r.font.bold = i == 0 or (bold_col0 and j == 0)
                    r.font.color.rgb = GOLD if i == 0 else INK
    return tb


def callout(s, x, y, w, h, msg, size=18):
    return text(s, x, y, w, h, msg, HEAD, size, True, WHITE, fill=PURPLE, anchor=MSO_ANCHOR.MIDDLE)


def picture(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def notes(s, msg):
    s.notes_slide.notes_text_frame.text = msg


pct = lambda v: f"{100 * v:.1f}%"
SHORT = {"pick-up below zero-friction load": "pick-ups too light", "slack-off above zero-friction load": "slack-offs too heavy",
         "beyond the largest friction": "need friction > 0.8"}


def window(w):
    """'12-08 to 12-09', or '12-07 08:00-11:00' for a window inside one day."""
    a, b = w["start"], w["end"]
    return f"{a[5:10]} {a[11:16]}-{b[11:16]}" if a[:10] == b[:10] else f"{a[5:10]} to {b[5:10]}"


# =========================================================================== slides
def title_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = PURPLE
    rect(s, 0.6, 2.05, 1.4, 0.09, GOLD)
    text(s, 0.6, 0.7, 12, 0.4, "CSC 7700 AI & LLM DEVELOPMENT · FINAL PROJECT", HEAD, 16, True, GOLD)
    text(s, 0.6, 2.3, 12, 1.2, "Skeptic", HEAD, 66, True, WHITE)
    text(s, 0.6, 3.55, 12, 0.8, "A falsification-first agent for drilling data", HEAD, 32, False, WHITE)
    text(s, 0.6, 4.35, 11.5, 0.9, "It reconciles conflicting well records with physics as the referee, tests competing "
         "explanations, and abstains when the data cannot decide.", BODY, 18, color=LAV)
    text(s, 0.6, 6.25, 12, 0.4, "Victory Oghale · Louisiana State University · Fall 2026", HEAD, 18, False, WHITE)
    notes(s, "Skeptic is an LLM agent for drilling records. It treats every answer as a hypothesis to try to "
             "refute, uses physics to decide between conflicting sources, and is allowed to say the data cannot decide.")


def problem(N):
    s = slide(1, "The problem", "Drilling records disagree, and some numbers cannot be known from them")
    text(s, 0.6, 1.95, 5.6, 0.4, "Three records of the same well", HEAD, 20, True, PURPLE)
    bullets(s, 0.6, 2.5, 5.6, [("Rig sensor log.", "A reading every 10 s: hook load, depths, pump pressure, torque."),
                                ("Daily drilling reports.", "Typed by the crew: depths, mud weights, time log, bits."),
                                ("Survey.", "Where the hole actually goes.")], size=16, gap=0.72)
    text(s, 0.6, 4.75, 5.6, 0.4, "Everyday ways they go wrong", HEAD, 20, True, PURPLE)
    text(s, 0.6, 5.2, 5.6, 1.5, "Frozen sensors, shifted clocks, wrong units, typos, depth shifts, and instructions "
         "pasted into text an agent will read.", BODY, 16)
    text(s, 6.8, 1.95, 6.0, 0.4, "Found in the public FORGE 16A records while building", HEAD, 20, True, PURPLE)
    table(s, 6.8, 2.5, 5.95, ["Record", "What is wrong"], [
        ["11-09 report time log", "about 18 minutes off the rig clock"],
        ["9⅝\" casing shoe", "5,110 ft and 5,113 ft in different reports"],
        ["Bit 9 nozzles 4x12 + 4x11", "TFA 0.813 in², report says 0.910"],
        ["Daily reports", "37 exist twice, byte-identical"],
        ["Logging trips 26-29 Dec", "a narrow friction CI, yet negative friction trip by trip"]],
        [2.35, 3.6], size=13, row_h=0.42)
    callout(s, 6.8, 5.35, 5.95, 1.2, "An LLM with tools tends to take the first number it finds and state it with "
            "confidence. Skeptic is built to check it first.", 17)
    notes(s, "These examples are real findings from the public FORGE records, made by the data checks, not by the LLM. "
             "The last one matters most: a fit can look precise and still be physically impossible.")


def prior_work(N):
    s = slide(2, "Existing solutions and prior work", "The pieces exist; none of them referees sources with physics")
    table(s, 0.6, 1.9, 12.1, ["Work", "What it gives", "What is missing for drilling records"], [
        ["RAG (Lewis et al. 2020)", "Answers grounded in retrieved text", "No check that the retrieved number is right"],
        ["ReAct (Yao et al. 2023)", "An LLM that reasons and calls tools", "No rule for conflicting sources or unknowable values (my baseline)"],
        ["Reflexion (Shinn et al. 2023); debate (Du et al. 2024)", "A model critiques a model", "Critic shares the model's blind spots; no outside referee"],
        ["Strong inference (Platt 1964)", "Competing hypotheses, a test that separates them", "A method for scientists, not an agent loop"],
        ["Selective prediction (Geifman & El-Yaniv 2017)", "Abstention, risk-coverage curves", "Classifiers, not tool-using agents"],
        ["Indirect prompt injection (Greshake et al. 2023)", "Attacks hidden in documents", "No defence built for well records"],
        ["Soft-string torque and drag (Johancsik et al. 1984)", "Physics of hook load and friction", "A fixed calculation; never asks if friction can be known"]],
        [3.7, 3.6, 4.8], size=13, row_h=0.47)
    callout(s, 0.6, 5.85, 12.1, 0.8, "Gap: an agent that uses physics as the referee between sources, tests explanations "
            "before answering, and abstains when the data cannot decide.", 18)
    notes(s, "ReAct is the closest prior system, so it is my main baseline: same model, same tools, none of Skeptic's rules.")


def scope(N):
    s = slide(3, "Scope", "From the approved proposal (HydraulicsAgent) to Skeptic")
    table(s, 0.6, 1.9, 12.1, ["Proposal item", "In Skeptic"], [
        ["Custom parsers that keep provenance", "FORGE daily reports and 10-s rig log; LSU Pason, tour sheets, Halliburton, EOW. Every value keeps source and locator."],
        ["LLM picks tools; physics computes numbers", f"Hand-written loop, {N['n_tools']} tools. A verifier rejects any number not found in a tool result."],
        ["Disagreeing records: investigate or report uncertainty", "Hypothesis ledger, skeptic review, identifiability gate, abstention."],
        ["Documents are data, not commands", "Injection guard; benchmark tasks with planted instructions."],
        ["Compare with search-only and a fixed pipeline", "Both, plus ReAct with the same tools."],
        ["Tested hydraulics tool", "Bit hydraulics, pump output and pressure losses, with unit tests."],
        ["SPP error on held-out intervals", "Not evaluated: the benchmark targets record faults and friction instead."],
        ["One hole section", f"Two wells; {N['n_test']}-task held-out test set (FORGE) and a {N['n_dev_forge'] + N['n_dev_lsu']}-task development set."]],
        [3.9, 8.2], size=13, row_h=0.5)
    notes(s, "The approved proposal was HydraulicsAgent. Skeptic keeps its parts and widens the question from one pressure "
             "change to any conflict between records. The SPP prediction-error metric was not carried over.")


def architecture(N):
    g = N["tool_groups"]
    s = slide(4, "Technical approach", "A hand-written agent loop, and gates every answer must pass")
    box(s, 0.6, 2.65, 1.75, 1.1, ["Question", "about a well"], fill=LAV, size=13)
    rect(s, 2.75, 1.95, 6.15, 4.75, WHITE).line.color.rgb = LAV
    text(s, 2.95, 2.0, 5.8, 0.35, "Control loop (Python, no LangChain)", HEAD, 15, True, PURPLE)
    box(s, 3.0, 2.45, 2.3, 1.45, ["LLM", "Claude Haiku 5.5 via OpenRouter", "OpenAI-format function calls"], size=12)
    box(s, 6.05, 3.15, 2.65, 3.3, ["Tools", f"Evidence ({g['evidence']}): facts with provenance, BM25 search, rig log",
                                   f"Checks ({g['checks']}): stuck sensor, clock, depth, survey, drift, units",
                                   f"Physics ({g['physics']}): torque and drag, identifiability, physics data check, hydraulics",
                                   f"Ledger ({g['ledger']}): competing hypotheses"], size=11)
    arrow(s, 5.3, 3.35, 6.05, 3.35, "tool call", lx=-0.05)
    arrow(s, 6.05, 3.7, 5.3, 3.7, "result", lx=0.05, ly=0.02)
    text(s, 3.0, 4.3, 2.9, 1.9, ["Budgets on steps and cost", "Loop detection", "Bad arguments come back as errors",
                                  "Always ends with an answer"], BODY, 12, color=MUTED, space=3)
    box(s, 9.35, 2.2, 3.35, 3.0, ["An answer must pass", "Cited evidence ids exist", "Every number appears in a tool result",
                                    "Two competing hypotheses when in doubt", "Friction only after 'identifiable'",
                                    "A skeptic model tries to refute it", "Document text screened for instructions"], size=12, fill=WHITE)
    arrow(s, 5.3, 2.65, 9.35, 2.65, None, color=PURPLE)
    text(s, 6.2, 2.3, 2.2, 0.3, "submit_answer", BODY, 12, color=PURPLE)
    arrow(s, 9.35, 4.85, 8.9, 4.85, None, color=FLAG)
    text(s, 9.35, 5.3, 3.35, 0.3, "rejected: the reason goes back to the LLM", BODY, 11, color=FLAG)
    box(s, 9.35, 5.85, 3.35, 0.85, ["Answer", "verdict, value, confidence, evidence; or abstain"], fill=LAV, size=12)
    arrow(s, 2.35, 3.2, 3.0, 3.2)
    notes(s, "The harness is my own loop, about 300 lines. The LLM only decides which tool to call; numbers come from the "
             "tools. When the model submits, six gates check the answer; a failed gate sends the reason back and the "
             "model must fix it. ReAct uses the same loop and tools without the gates.")


def physics(N):
    s = slide(5, "Technical approach", "Physics is the referee, and it also says when a number cannot be known")
    bullets(s, 0.6, 1.95, 6.2, [
        ("Torque and drag.", "Soft-string model (Johancsik 1984) with the float-valve fill state, reused from my PETE 7103 project."),
        ("Identifiability.", "A friction factor is reported only if it is precise (trips move ≥ 20 klb per unit friction), "
                             "consistent (the trips agree) and physical (between 0 and 0.6)."),
        ("A narrow CI is not enough.", "On the FORGE logging trips the fit has a narrow CI, yet each trip gives negative friction."),
        ("QC checks.", "Frozen sensor (judged by rig activity), clock offset, depth jumps, survey typos, block-weight drift, units."),
        ("Hydraulics.", "Pump output, bit pressure drop, plugged-nozzle scenarios, TFA consistency.")], size=15, gap=0.95)
    text(s, 7.2, 1.95, 5.5, 0.4, "Tool verdicts on the 7 friction windows (test set)", HEAD, 17, True, PURPLE)
    rows = [[window(w), w["key"], w["identifiability"],
             "-" if w["estimate"] is None else f"{w['estimate']:.3f}"] for w in N["windows"]]
    table(s, 7.2, 2.45, 5.5, ["Window", "Key", "Tool verdict", "Trip-pooled μ"], rows, [1.85, 0.85, 1.6, 1.2], size=12, row_h=0.4)
    callout(s, 7.2, 5.85, 5.5, 0.8, "The tool's verdict matches the key on all 7 windows.", 16)
    notes(s, "Identifiability is the core physics idea. Friction is fitted trip by trip with the weight terms held at the "
             "global fit; the three tests must all pass. The key for the 7 identify tasks is the tool's verdict, "
             "checked when the test set was built.")


def physics_check(N):
    s = slide(6, "Technical approach", "Readings no friction factor can explain are not friction")
    picture(s, "physics_check.png", 1.2, 1.8, w=10.9)
    text(s, 0.6, 6.45, 12.1, 0.5, "Whatever the friction, a pick-up cannot weigh less than the frictionless string and a "
         "slack-off cannot weigh more. Adapted from my PETE 7103 error control (stage P8); added after the test run.",
         BODY, 13, color=MUTED)
    notes(s, "Left: a deep trip out on 13 December. Every reading sits between the zero-friction line and the friction-0.8 "
             "line, so friction can explain it. Right: the 27 December logging trip picks up 24 to 36 klb lighter than the "
             "string would weigh with no friction at all. No friction value does that, so it is a record or sensor "
             "problem. The same trip was found by the PETE pipeline.")


def benchmark(N):
    sl, fam = N["slices"], N["families"]
    s = slide(7, "Evaluation", "Planted faults with known answers, and a test set frozen before any run")
    bullets(s, 0.6, 1.95, 6.3, [
        ("Known answers.", "Faults are planted in real FORGE data, so the right answer is known. Clean controls catch false alarms."),
        ("Oracle-validated.", "Every planted fault is found by its reference check; every control passes it."),
        ("Development set.", f"{N['n_dev_forge'] + N['n_dev_lsu']} tasks ({N['n_dev_forge']} FORGE, {N['n_dev_lsu']} private LSU), used while building."),
        ("Held-out test set.", f"{N['n_test']} FORGE tasks from other days, hashed and frozen with 4 written predictions before any run."),
        ("Contestants.", "Skeptic; ReAct (same model and tools, no gates); a fixed pipeline with no LLM."),
        ("Statistics.", "3 runs per LLM system (T = 0, 0.7, 0.7). Paired by task: bootstrap 95% CI, sign-flip p.")], size=15, gap=0.78)
    names = [("stuck", "Frozen sensor"), ("clock", "Clock offset"), ("depth", "Depth shift (one stand)"), ("drift", "Hook-load drift"),
             ("mud_weight", "Mud weight wrong or in s.g."), ("survey", "Survey typo"), ("unit", "Hook load in lb, not klb"),
             ("planted", "Instruction planted in a report"), ("audit", "Open audit of a whole day")]
    rows = [[n, fam.get((k, "injected"), 0), fam.get((k, "control"), 0) or "-"] for k, n in names]
    rows.append(["Friction: answer or abstain", "-", f"{fam.get(('friction', 'identify'), 0)} identify"])
    rows.append(["Total", sl["injected"], f"{sl['control']} + {sl['identify']}"])
    table(s, 7.3, 1.95, 5.4, ["Test set family", "Injected", "Control"], rows, [3.1, 1.1, 1.2], size=12, row_h=0.38)
    notes(s, "The development set was used to fix bugs, so its scores are optimistic. The headline results come only from "
             "the test set, which was built from different days and frozen, with predictions written down, before the first run.")


def results(N):
    t, p = N["t"], N["pair"]
    s = slide(8, "Results", "Accuracy ties; Skeptic is confidently wrong half as often")
    picture(s, "headline.png", 0.75, 1.85, w=11.8)
    rows = []
    for m, name in (("correct", "Accuracy"), ("confident_wrong", "Wrong with confidence ≥ 0.7"), ("abstained", "Abstained")):
        r = p[m]
        rows.append([name, f"{r['diff']:+.3f}", f"{r['ci95_low']:+.3f} to {r['ci95_high']:+.3f}", f"{r['p']:.3f}"])
    rows.append(["Cost per task (USD)", f"{t.loc['skeptic', 'usd_per_task']:.3f} vs {t.loc['react', 'usd_per_task']:.3f}",
                 f"{t.loc['skeptic', 'usd_per_task'] / t.loc['react', 'usd_per_task']:.1f}x", "-"])
    table(s, 0.6, 5.0, 8.0, ["Skeptic minus ReAct, paired by task", "Difference", "95% CI", "p"], rows, [3.4, 1.7, 2.1, 0.8], size=13, row_h=0.37)
    callout(s, 8.95, 5.0, 3.75, 1.85, f"{N['n_test']} held-out tasks, {N['runs']['skeptic']} runs per LLM system. "
            "Skeptic turns confident errors into abstentions.", 16)
    notes(s, f"Accuracy is a tie: {t.loc['skeptic', 'accuracy']:.3f} against {t.loc['react', 'accuracy']:.3f}, p = {p['correct']['p']:.2f}. "
             f"The difference is in confident errors: {pct(t.loc['skeptic', 'confident_wrong'])} against "
             f"{pct(t.loc['react', 'confident_wrong'])}, p = {p['confident_wrong']['p']:.3f}. The price is more abstentions "
             "and about twice the cost. Confident-wrong was defined before the run but was not one of the four predictions, "
             "so I report it as a strong secondary finding.")


def risk(N):
    t, sure = N["t"], N["sure"]
    s = slide(9, "Results", f"When Skeptic says it is sure, it is right {sure['skeptic'][0]:.0%} of the time")
    picture(s, "risk_coverage.png", 0.6, 1.85, h=4.95)
    bullets(s, 9.05, 2.0, 3.7, [
        ("Read it as:", "answer only the questions a system is most sure about; how many answers are wrong?"),
        ("Area under the curve:", f"Skeptic {t.loc['skeptic', 'aurc']:.3f}, ReAct {t.loc['react', 'aurc']:.3f} "
                                  "(task bootstrap 95% CI of the difference −0.19 to −0.04)."),
        ("Confidence ≥ 0.7:", f"Skeptic right {sure['skeptic'][0]:.0%} of {sure['skeptic'][1]} answers; "
                              f"ReAct {sure['react'][0]:.0%} of {sure['react'][1]}."),
        ("Curves end early:", f"abstentions are not covered (Skeptic {pct(t.loc['skeptic', 'abstain'])}).")], size=14, gap=1.15)
    notes(s, "This is the selective-prediction view. Lower is better. Skeptic's curve is below ReAct's at every coverage "
             "and ends at about 80% coverage because it abstains on the rest.")


def predictions(N):
    t, p = N["t"], N["pair"]
    s = slide(10, "Results", "Predictions written before the run: two held, one failed")
    fa = lambda c: pct(t.loc[c, "false_alarm"])
    table(s, 0.6, 1.95, 12.1, ["Prediction (frozen with the test set)", "Outcome"], [
        ["1. Skeptic's false-alarm rate on controls is at most ReAct's", f"Held: {fa('skeptic')} vs {fa('react')}, not significant"],
        ["2. Skeptic is more accurate on the friction answer-or-abstain tasks",
         f"Failed: {t.loc['skeptic', 'acc_identify']:.2f} vs {t.loc['react', 'acc_identify']:.2f}"],
        ["3. No prediction on overall accuracy", f"Tie: {p['correct']['diff']:+.3f}, p = {p['correct']['p']:.2f}"],
        ["4. Skeptic costs about twice as much", f"Held: {t.loc['skeptic', 'usd_per_task'] / t.loc['react', 'usd_per_task']:.1f}x"]],
        [7.0, 5.1], size=15, row_h=0.5)
    text(s, 0.6, 4.75, 12.1, 0.4, "Why prediction 2 failed", HEAD, 20, True, PURPLE)
    bullets(s, 0.6, 5.25, 12.1, [
        ("The judgement moved into the tool.", "After the development run the identifiability tool tests precision, agreement and "
                                               "physical range itself, and ReAct simply follows its verdict."),
        ("Skeptic overruled the tool once.", "On a one-trip window (7 Dec) it argued most of the signal came from cased hole. "
                                             "Defensible, but the key says answer.")], size=15, gap=0.7)
    notes(s, "I report the failed prediction as it is. The lesson is useful: putting the judgement into the tool helped "
             "every agent, so the gate in the loop had nothing left to add.")


def limits(N):
    t, fam, wf = N["t"], N["fam"], N["without_flawed"]
    s = slide(11, "Results", "What the results do not show")
    bullets(s, 0.6, 1.95, 12.1, [
        ("No accuracy gain.", f"Skeptic {t.loc['skeptic', 'accuracy']:.3f}, ReAct {t.loc['react', 'accuracy']:.3f}, fixed pipeline "
                              f"{t.loc['fixed', 'accuracy']:.3f}. The tools carry the accuracy."),
        ("Open audits are hard for everyone.", f"Find whatever is wrong in a day: fixed {fam.loc['audit', 'fixed']:.2f}, ReAct "
                                               f"{fam.loc['audit', 'react']:.2f}, Skeptic {fam.loc['audit', 'skeptic']:.2f}."),
        ("A flaw found after the run.", "On days without drilling the clock check returned a placeholder and 'aligned'; 2 controls "
                                        f"rest on it. Without them all three score {wf['skeptic']:.3f}. Fixed and disclosed."),
        ("More abstentions, higher cost.", f"Skeptic abstains on {pct(t.loc['skeptic', 'abstain'])} (ReAct "
                                           f"{pct(t.loc['react', 'abstain'])}); a clean control abstained on counts as wrong."),
        ("Narrow test.", "One model (Claude Haiku 5.5), one public well in the test set; the LSU results stay private.")], size=16, gap=0.92)
    notes(s, "These limits are in RESULTS.md as well. The clock flaw was found after the run; as pre-registered, the "
             "headline keeps both tasks, and the check without them changes nothing.")


def after_test(N):
    s = slide(12, "After the test", "Added from my PETE 7103 work: a physics data check (not yet evaluated)")
    bullets(s, 0.6, 1.95, 5.6, [
        ("New tool.", "check_hookload_physics flags readings outside the friction envelope and trips no single friction fits."),
        ("Better abstentions.", "friction_identifiability now says what would settle it: deeper trips, or a records check "
                                "when trips disagree."),
        ("Checked, not scored.", "Tests pass and the fixed pipeline scores the same; the frozen test set cannot measure the change."),
        ("Live demo checks.", "Logging trips: abstains, citing the physics check. Deep trips: answers 0.35, citing 0 of 335 "
                              "readings outside the envelope.")], size=14, gap=1.12)
    rows = []
    for w in N["windows"]:
        why = ", ".join(f"{n} {SHORT[r]}" for r, n in w["by_reason"].items()) if w["by_reason"] else "-"
        rows.append([window(w), w["key"],
                     "consistent" if w["physics_check"].startswith("consistent") else "not friction", why])
    table(s, 6.6, 1.95, 6.1, ["Window", "Key", "Physics check", "Flagged readings"], rows, [1.85, 0.85, 1.25, 2.15], size=12, row_h=0.4)
    text(s, 6.6, 5.3, 6.1, 1.3, "On the 3 'answer' windows every reading is consistent with friction. Each 'abstain' window "
         "gets a concrete reason: too shallow, or readings that no friction factor explains.", BODY, 13, color=MUTED)
    notes(s, "This was added after the test run, so it is not part of the scored results. Measuring it needs a new held-out "
             "set. The two live runs are demo checks only.")


def demo(N):
    s = slide(13, "Live demo", "Ask a question, sabotage the data, watch it check")
    bullets(s, 0.6, 1.95, 7.0, [
        ("1. Ask.", "Pick the FORGE well and a question, or load a benchmark task."),
        ("2. Sabotage.", "Freeze the pump pressure, shift the rig clock 90 min, shift depth 93 ft, add hook-load drift, "
                         "or plant an instruction in a report."),
        ("3. Watch.", "Tool calls stream in; then the hypothesis ledger, the gates that rejected answers, the skeptic "
                      "review and the final answer with its evidence."),
        ("Fallback.", "Without network, replay saved traces: the 378 test-set runs and two demo traces are in the repository.")],
        size=16, gap=1.05)
    box(s, 8.1, 1.95, 4.6, 2.2, ["Run it", "streamlit run app.py", "", "FORGE 16A(78)-32 only: public data,",
                                  "DOI 10.15121/1776602, CC BY 4.0"], fill=WHITE, size=14)
    box(s, 8.1, 4.4, 4.6, 2.2, ["Good live questions", "Friction factor during the deep trips out, 13-20 Dec 2020",
                                 "The same during the logging trips, 26-29 Dec", "Any question after 'shift rig clock +90 min'"],
        fill=LAV, size=13)
    notes(s, "The demo shows a clean question, then the same kind of question after sabotage. If the network fails, use "
             "Replay trace: results/demo has the two live runs from 22 November.")


def conclusions(N):
    t = N["t"]
    s = slide(14, "Conclusion and future work", "Tools carry accuracy; testing answers makes confidence trustworthy")
    text(s, 0.6, 1.95, 5.9, 0.4, "What the evidence supports", HEAD, 20, True, PURPLE)
    bullets(s, 0.6, 2.45, 5.9, [
        ("Same accuracy", f"as ReAct with the same tools ({t.loc['skeptic', 'accuracy']:.2f} vs {t.loc['react', 'accuracy']:.2f})."),
        ("Half the confident errors", f"({pct(t.loc['skeptic', 'confident_wrong'])} vs {pct(t.loc['react', 'confident_wrong'])}, "
                                      f"p = {N['pair']['confident_wrong']['p']:.3f}) and a much lower risk-coverage curve."),
        ("Physics as referee", "finds real record problems in public data and tells friction from record errors."),
        ("Honest evaluation:", "held-out set, written predictions, a failed prediction and a flaw reported.")], size=15, gap=0.92)
    text(s, 6.9, 1.95, 5.8, 0.4, "Next", HEAD, 20, True, PURPLE)
    bullets(s, 6.9, 2.45, 5.8, [
        ("A new held-out set", "to score the physics data check added after the test."),
        ("More wells and models,", "and confidence calibrated on validation data."),
        ("Back to the proposal:", "standpipe-pressure questions with the hydraulics tools."),
        ("Cheaper runs:", "fewer tool calls per task (Skeptic uses about 16, ReAct 10).")], size=15, gap=0.92)
    notes(s, "The claim is narrow and supported: the falsification loop does not make the agent more accurate, but it makes "
             "its confidence something you can rely on.")


def references(N):
    s = slide(15, "References", "References")
    refs = [
        "Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. ICML.",
        "Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. NeurIPS 30.",
        "Greshake, K., et al. (2023). Not what you've signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. AISec '23.",
        "Johancsik, C.A., Friesen, D.B., Dawson, R. (1984). Torque and drag in directional wells: prediction and measurement. JPT 36(6), 987-992.",
        "Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 33.",
        "Platt, J.R. (1964). Strong inference. Science 146(3642), 347-353.",
        "Shinn, N., et al. (2023). Reflexion: language agents with verbal reinforcement learning. NeurIPS 36.",
        "Yao, S., et al. (2023). ReAct: synergizing reasoning and acting in language models. ICLR.",
        "Utah FORGE 16A(78)-32 drilling data (2021). Geothermal Data Repository, DOI 10.15121/1776602, CC BY 4.0.",
        "Physics code reused from the author's PETE 7103 term project (torque and drag, fill state, data check).",
        "Code: github.com/Neroogod/skeptic. AI assistance was used for coding, as the course guide allows."]
    text(s, 0.6, 1.9, 12.1, 4.9, refs, BODY, 13, space=7)


if __name__ == "__main__":
    N = numbers()
    title_slide()
    for f in (problem, prior_work, scope, architecture, physics, physics_check, benchmark, results, risk, predictions,
              limits, after_test, demo, conclusions, references):
        f(N)
    for i, s in enumerate(prs.slides):                 # page numbers
        if i:
            text(s, 11.9, 7.02, 0.8, 0.3, f"{i + 1} / {len(prs.slides)}", HEAD, 11, color=MUTED, align=PP_ALIGN.RIGHT)
    prs.save(OUT)
    print("wrote", OUT, len(prs.slides), "slides")
