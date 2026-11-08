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

## Status (2026-11-07)

| Phase | State |
|---|---|
| 0 Env | done (Python 3.12 venv, .gitignore guards LSU data) |
| 1 Evidence | done: FORGE 4.8k facts / 476 chunks, LSU 1.3k facts / 2k chunks; golden tests pass |
| 2 Physics + QC | done: reproduces the PETE fits and FORGE per-trip mu (within 0.010) |
| 3 Benchmark | done: 61 tasks, every fault oracle-validated, controls clean |
| 4 Harness | done: scripted-LLM tests for every gate |
| 5 Scorer + baselines | done: fixed pipeline scores 0.79 overall (injected 0.81, real 0.60, identify 0.50, explain 0.00) |
| 6 Live LLM runs | **blocked: needs OPENROUTER_API_KEY** |
| 7 UI | done (Streamlit, sabotage menu, replay); renders headlessly in tests |
| 8 Self-review | first pass done (fixed: identifiability gate bypass, verifier tolerance, QC false positives) |
