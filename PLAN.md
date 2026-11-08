# Skeptic: build plan

Engineering plan for the CSC 7700 project build. This is not proposal text.

## Goal

Skeptic is an agent that answers questions about a well only when the records support the answer. It does three things:

1. **Reconcile.** It finds every candidate value for a quantity across sources, then uses physics or cross-checks to decide which one is right.
2. **Explain.** It keeps a list of hypotheses. Each one gets a test that tells it apart from the others, and a skeptic pass that tries to refute the leading one.
3. **Abstain.** If the measurement cannot pin down the parameter, it refuses to answer, says why, and names the measurement that would fix it.

## Layout

```
skeptic/
  skeptic/
    config.py            paths and env vars (data never inside the repo)
    evidence/
      store.py           Fact + FactStore: every value carries unit, source and locator
      forge.py           FORGE 16A daily-report parser, 10-s EDR loader, bit-depth rebuild
      lsu.py             LSU H-001: Pason CSV + memos, tour sheets, Halliburton, supplement, EOW
      text.py            text chunks + BM25 index (hand-written)
      workspace.py       one task's view of a well (evidence + EDR + text), copy-on-write
    physics/
      units.py           unit detection and conversion
      hydraulics.py      pump output, bit pressure drop, Bingham-plastic pressure losses
      tnd.py             soft-string torque/drag (adapted from PETE 7103 tdcore.py)
      qc.py              data-quality checks: stuck channel, unit scale, depth, survey, drift, clock offset
      identify.py        sensitivity / Fisher-information identifiability
    agent/
      llm.py             OpenRouter client + ScriptedLLM for tests
      tools.py           tool registry: schema, validation, errors returned as text
      ledger.py          hypothesis ledger (structured state)
      guard.py           prompt-injection screen for text coming out of documents
      verify.py          every number in the answer must trace to a tool output
      harness.py         the control loop: budgets, loop detection, skeptic pass
      prompts.py
    bench/
      tasks.py           task schema
      inject.py          fault injector (answer key known by construction)
      build.py           builds the benchmark (FORGE public; LSU private)
      score.py           deterministic scorer
      baselines.py       fixed pipeline, search-only RAG, no-physics agent
      run.py             runs configs × tasks × seeds, writes runs/*.jsonl
  app.py                 Streamlit UI
  tests/
```

## Phases and gates

Each phase ends with a gate. The next phase does not start until the gate passes.

| # | Phase | Gate (automated unless noted) |
|---|---|---|
| 0 | Env, config, .gitignore, data paths | `pytest` runs; LSU paths are git-ignored |
| 1 | Evidence layer: FORGE parser and EDR, LSU parsers, fact store, BM25 | Parser golden tests: values hand-checked from source files (FORGE report 11/16/2020, LSU shoes/DP ID/bit 3); every fact has a source and locator; the answer-key sheets are excluded |
| 2 | Physics and QC tools | Closed-form tests: vertical-well hook load = buoyed weight; loop solver = fast solver; textbook bit ΔP and pump output; every QC check fires on a synthetic fault and stays silent on clean data |
| 3 | Fault injector and benchmark builder | Oracle test: for every injected task, the reference check detects it, and the matching clean control passes |
| 4 | Harness: LLM client, tools, ledger, guard, verifier, loop | Scripted-LLM tests: tool errors returned as text, budget stop, loop detection, injection screen, verifier rejects invented numbers, skeptic pass can veto |
| 5 | Scorer and baselines | Scorer unit tests; the fixed-pipeline baseline runs on the whole benchmark end to end with no API key |
| 6 | Live LLM runs (needs an API key) | Smoke test on 3 tasks, then the full benchmark: full system, 3 ablations, baselines, 3 seeds |
| 7 | UI and demo | Manual: demo script runs on FORGE in under 5 minutes |
| 8 | Self-review | Re-read every module against the TA code-review standard; full test suite green |

## Benchmark

| Slice | Well | Answer key |
|---|---|---|
| Real conflicts | LSU (private) | Instructor's "Checks vs Pason" sheet and PETE results. Held out, so never indexed as evidence. |
| Injected faults | FORGE (public), LSU | Known by construction: unit error, clock offset, stuck channel, depth shift, wrong value, survey error, sensor drift, planted instruction |
| Clean controls | both | Same question, no fault. Measures false alarms. |
| Answer or abstain | both | PETE identifiability results |
| Explanation | both | PETE float-valve finding; EOW narratives |

**Held out from evidence:** the Halliburton README and the "Checks vs Pason" sheet, and the README sheets in the Pason package, tour sheets and supplement. These contain the instructor's QC conclusions, so including them would turn the tasks into retrieval.

## Confidentiality

- LSU data stays local: `cache/`, `runs/`, `bench/private/` and `.env` are git-ignored.
- The public repo and the live demo use FORGE only.
- LSU results are reported as aggregates.

## Needs from Victory

1. OpenRouter API key in `.env`, with a spend limit set on OpenRouter.
2. Instructor's OK on the pivot. The note to the instructor must be written by Victory, because of the GenAI ban on proposals.
3. GitHub repo name, and whether it is public (FORGE only) or private.
4. Later: about 20 ChatGPT/Claude web-upload runs for the "existing interface" baseline.

## Status (2026-11-08)

| Phase | State |
|---|---|
| 0 Env | done (Python 3.12 venv, .gitignore guards LSU data) |
| 1 Evidence | done: FORGE 4.8k facts / 476 chunks, LSU 1.3k facts / 2k chunks; golden tests pass |
| 2 Physics + QC | done: reproduces the PETE fits and FORGE per-trip mu (within 0.010) |
| 3 Benchmark | done: development set 61 tasks; held-out test set 63 tasks (below) |
| 4 Harness | done: scripted-LLM tests for every gate |
| 5 Scorer + baselines | done |
| 6 Live LLM runs | v1, v2 (development set) and v3 (held-out test set, 3 seeds) done; see RESULTS.md |
| 7 UI | done (Streamlit, sabotage menu, replay); renders headlessly in tests |
| 8 Self-review | done twice (2026-11-07, 2026-11-08), see below |

## v2 failure analysis (2026-11-08, development set)

Skeptic lost 8 tasks that ReAct won and won 4 that ReAct lost. From the traces:

| Cause | Tasks | Fix (applies to every configuration) |
|---|---|---|
| Run ended with no answer (model stopped calling tools) | 1 Skeptic, 2 ReAct | submit_answer is forced; if nothing passes the gates the run abstains, and that abstention is scored wrong |
| fault_found without a fault_type | 2 | gate added on 2026-11-07 |
| Answer key ignored the report's natural 18-min offset on 2020-11-09 | 1 | clock keys accept 15 min around the injected offset or the injected plus the natural offset |
| Stuck check flagged bit depth in slips / hole depth off bottom | 1 + 1 ReAct false alarm | depth channels judged by physics (drilling / out of slips), flat runs with a physical cause reported as explained |
| Reviewer judged drift on the whole well, not the asked week | 1 | one-line scope rule in the reviewer prompt |
| Verdict 'answer' for a correct clock offset | 1 | none (model error) |
| LSU explanation (float valve) abstained | 1 | none |

The identifiability gate only checked that the tool was *called*; it now requires the tool to have said
"identifiable". The identifiability test itself now requires precision, agreement between trips and a
physical value, because the free zone fit gave narrow CIs on logging trips whose trip-by-trip mu is negative.

The reviewer, measured on v2 traces: 7 rejections, 1 turned a wrong answer right, 1 turned a right answer
into an abstention, 5 changed nothing.

A 10-task check after the fixes (Skeptic only, $0.30) passed 6 of the 8 earlier losses. It re-ran tasks
chosen *because* they had failed, so it verifies the fixes and is not a result.

## v3 held-out test set: pre-registration (frozen 2026-11-08, before any LLM run on it)

- **Tasks:** `bench/public/forge_test.json`, 63 tasks, SHA-256
  `d97f431ba3302e90a19d8e7bc61b508bf4eeeb01e78944b7acf9642954e397c2`. Built by
  `python -m skeptic.bench.build test` (spec `TEST` in `build.py`): 13 drilling days whose windows do
  not overlap any development window, seed 11, other offsets and shift sizes, no mud check / pick-up
  report used by the development set, no survey station the development set injected.
  37 injected, 19 controls, 7 identify (3 windows where the open-hole friction factor is determined,
  4 where the right answer is to abstain). Every injected fault is oracle-validated; every control
  passes its oracle.
- **Code:** frozen at the commit that adds this section. No prompt, tool, gate or key change after
  the first test run; a later change needs a new test set.
- **Runs:** `skeptic` and `react` with Claude Haiku 5.5, 3 seeds (seed 0 at temperature 0, seeds 1-2
  at 0.7); `fixed` (no LLM).
- **Primary analysis:** accuracy, Skeptic minus ReAct, paired by task (mean over seeds), bootstrap 95%
  CI over tasks and sign-flip permutation p (`skeptic.bench.report`).
- **Predictions made before the run:**
  1. Skeptic's false-alarm rate on controls is at most ReAct's.
  2. Skeptic is more accurate than ReAct on the identify slice.
  3. No prediction on overall accuracy; it is reported either way.
  4. Skeptic costs about twice as much per task as ReAct.
- Every result is reported, including any that go against these predictions.

### v3 outcome (2026-11-08)

Run at commit `2a2e1e6`, $5.36. Accuracy: Skeptic 0.757, ReAct 0.778, fixed 0.762 (Skeptic - ReAct -0.021,
95% CI -0.10 to +0.05, p = 0.70). Prediction 1 held (false alarms 8.8% vs 14.0%, n.s.), prediction 2 failed
(identify 0.86 vs 1.00), prediction 4 held (1.8x cost). Secondary: confident-wrong 6.9% vs 12.7% (p = 0.017),
risk-coverage AUC 0.060 vs 0.168. Found after the run: 2 clock controls rest on a sentinel output of the clock
check (fixed afterwards; without them all three systems score 0.770). Full tables in RESULTS.md.
