# Skeptic

A falsification-first agent for drilling data. Asked a question about a well, it:

1. **Reconciles.** It gathers every recorded value from every source and uses physics as the referee when they disagree.
2. **Explains.** It keeps a ledger of competing hypotheses, runs the discriminating test for each, and a second model tries to refute the winner.
3. **Abstains.** If the data cannot determine a quantity, which it judges from the fit's uncertainty rather than the LLM's own confidence, it refuses to answer and says what measurement would help.

CSC 7700 individual project (LSU, Fall 2026). The physics (soft-string torque and drag, fill-state model, FORGE bit-depth rebuild) comes from the author's PETE 7103 project and is reused with attribution in the docstrings. The CSC contribution is the agent harness, the evidence layer and the benchmark.

## Layout

| Path | What |
|---|---|
| `skeptic/evidence/` | Parsers for FORGE daily reports and EDR, and for LSU Pason, tour sheets, Halliburton workbook and EOW. Builds a fact store with provenance and a hand-written BM25 index. |
| `skeptic/physics/` | Soft-string hook-load model with trip-clustered sandwich CIs, identifiability, hydraulics, unit reading, and data-QC checks |
| `skeptic/agent/` | Tool registry, hypothesis ledger, injection guard, number verifier, hand-written control loop, prompts, OpenRouter client |
| `skeptic/bench/` | Fault injector, benchmark builder with oracle validation, deterministic scorer, fixed-pipeline baseline, runner |
| `app.py` | Streamlit demo, with live runs, a sabotage menu and trace replay |
| `scripts/export_tnd.py` | One-off export of torque-and-drag inputs from the PETE pipeline |
| `tests/` | 54 tests (evidence, physics, harness, benchmark, UI) |

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e . scipy pytest
cp .env.example .env              # add OPENROUTER_API_KEY; set a spend limit on openrouter.ai
.venv/bin/python scripts/export_tnd.py          # needs ../Drilling_Project/pete_td (once)
.venv/bin/python -m skeptic.bench.build         # builds the benchmark
.venv/bin/python -m pytest -q                   # no API key needed
.venv/bin/python -m skeptic.bench.run --configs fixed               # baseline, no key
.venv/bin/python -m skeptic.bench.run --configs skeptic --tasks forge-stuck-1,forge-friction-answer
.venv/bin/streamlit run app.py
```

## Configurations compared

| Config | What changes |
|---|---|
| `skeptic` | Full system: all tools, ledger gate, identifiability gate, skeptic review, verifier, guard |
| `react` | Same tools, plain ReAct prompt, no ledger, no gates, no skeptic |
| `no_physics` | Evidence tools only: no QC checks and no physics |
| `no_skeptic` | No ledger gate and no skeptic review |
| `no_gate` | No identifiability gate |
| `no_guard` | No prompt-injection guard |
| `rag` | Text search only (the search-only assistant from the proposal) |
| `fixed` | No LLM: runs every check on the structured scope and reports the first flag |

## Benchmark

Two sets. The **development set** (61 tasks) was used while building the agent, so prompts and tools
were changed after seeing results on it. The **held-out test set** (63 FORGE tasks,
`bench/public/forge_test.json`) was built from other days, facts and stations and frozen, with stated
predictions, before any run (see PLAN.md, "v3 held-out test set").

**Test set (63 tasks):** 37 injected faults, 19 clean controls, and 7 identify tasks: friction windows
where the open-hole friction factor is determined (answer), or not (abstain: shallow trips, trips that
disagree, logging trips with physically impossible fits).

**Development set, FORGE, public (47 tasks):**
- 30 injected faults: stuck channel, clock offset, one-stand depth shift, sensor drift, wrong mud weight, mud weight written as s.g., survey typo, hook-load unit error, a planted instruction in a report, and open audits
- 14 clean controls
- 3 natural problems: TFA vs nozzle sizes, an impossible BOP-test date, and the open-hole friction factor

**Development set, LSU, private (14 tasks):**
- 8 real conflicts, scored against the instructor's QC sheet, which is held out of the evidence
- 2 explanations: the float-valve rediscovery and the trip-tank question
- 1 abstain task
- 3 injected faults and controls

**Oracle validation:** every injected fault is detectable by its reference check, and every control passes that check on clean data.

**Metrics:** accuracy by slice, detection rate, false-alarm rate, confident-wrong rate (wrong with confidence ≥ 0.7), abstention rate, risk–coverage AUC, cost and tool calls.

## Results

See [RESULTS.md](RESULTS.md). On the held-out test set the three systems tie on accuracy (0.76-0.78).
Skeptic is confidently wrong about half as often as ReAct (6.9% vs 12.7%, p = 0.017) and has a much lower
risk-coverage curve, at the price of more abstentions and 1.8x the cost.

## Data and confidentiality

- **LSU H-001 is LSU-only.** It is processed locally and must not be committed or published. `cache/`, `runs/`, `bench/private/` and `.env` are git-ignored.
- **The demo and public repo use FORGE only.** Utah FORGE 16A(78)-32, DOI 10.15121/1776602, CC BY 4.0.
- **Held out from the evidence:** every README sheet and the Halliburton "Checks vs Pason" sheet. They hold the instructor's QC conclusions, which are the answer key.

## Findings made while building, from data checks rather than the LLM

- The FORGE rig clock repeats 01:00–01:59 on 1 Nov 2020, at the end of daylight saving. The loader puts everything on MST.
- 37 FORGE daily reports exist twice, byte-identical. They are parsed once.
- FORGE reports give the 9⅝" shoe as both 5,110 and 5,113 ft.
- FORGE bit 9: nozzles 4x12 + 4x11 give a TFA of 0.813 in², but the report says 0.910.
- On FORGE logging trips (26-29 Dec 2020) a free zone-model fit gives a narrow CI for the open-hole friction factor,
  but trip by trip the fitted value is negative. A narrow CI is not identifiability; the test now also requires the
  trips to agree and the value to be physical.
- The 2020-11-09 daily report's time log is about 18 minutes off the rig clock.
