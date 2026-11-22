# Code tour

How one question moves through the code, then where the physics and the benchmark live.
Line numbers are for commit `HEAD` of 2026-11-22.

## 1. One question, start to finish

| Step | What happens | Where |
|---|---|---|
| Load the well | Parsed evidence (facts, text chunks, rig log) is loaded from cache or built from the raw files | `skeptic/evidence/workspace.py:53` `load` |
| Start the loop | System prompt + well description + question; the tool list depends on the configuration | `skeptic/agent/harness.py:167` `run`, configurations at `harness.py:51` |
| Ask the model | One chat call with the tool schemas; a cut-off tool call comes back marked `_unparseable` | `skeptic/agent/llm.py:13` `OpenRouterLLM.chat` |
| Run a tool | Arguments checked against the schema, the function runs, errors come back as text, output capped at 4,000 characters | `skeptic/agent/tools.py:84` `call`, `tools.py:58` `validate` |
| Screen the result | Text that reads like an instruction to an AI is marked as data | `skeptic/agent/guard.py:19` `screen` |
| Keep hypotheses | `add_hypothesis` / `update_hypothesis` write to the ledger, citing call ids | `skeptic/agent/ledger.py:29`, `:69`, `:78` |
| Submit | `submit_answer` hands the answer to the gates | `skeptic/agent/ledger.py:117` |
| Gates | Evidence ids exist, numbers trace to tool results, two hypotheses tested, identifiability, then the critic | `skeptic/agent/harness.py:79` `check_answer`; numbers: `skeptic/agent/verify.py:33` |
| Critic | A second model call tries to refute; a cut-off reply keeps its visible decision | `harness.py:129` `review`, `harness.py:150` `salvage_review`, prompt at `prompts.py:48` |
| Finish | Accepted answer, or an abstention if nothing passed within budget | end of `harness.run` |

The loop itself is plain Python: `for step in range(max_steps)`, one model call per step, every tool call
appended to `ctx.calls` with an id (`c1`, `c2`, ...) that later claims must cite.

## 2. Evidence layer

- `skeptic/evidence/store.py:49` `Fact`: one value with unit, source and locator (file and line); `:63` `FactStore.get`
  returns every candidate, conflicts included.
- `skeptic/evidence/text.py:39` `TextIndex`: hand-written BM25 over report chunks.
- `skeptic/evidence/forge.py:97` `parse_report`: one FORGE daily report into facts and chunks, one section per table.
- `skeptic/evidence/lsu.py`: the same for the private LSU files (never committed).

## 3. Physics

- `skeptic/physics/tnd.py:50` `softstring`: soft-string tension, Picard iteration over 20-ft elements.
- `tnd.py:209` `identifiability`: precise, consistent across trips, physical. `tnd.py:263` `track_open_friction`
  fits friction trip by trip.
- `tnd.py:378` `physics_check`: readings outside the zero-friction / friction-0.8 envelope (added after the test).
- `skeptic/physics/qc.py:34` `stuck_channel` (judged by rig activity), `qc.py:100` `clock_offset_activity`, and the
  other data checks.

## 4. Benchmark

| Step | Where |
|---|---|
| Plant a fault in a copy of the well | `skeptic/bench/inject.py:17` `apply` |
| Oracle: keep a task only if its reference check finds the fault | `skeptic/bench/build.py:57` `oracle`, `build.py:125` `forge_tasks` |
| No-LLM baseline | `skeptic/bench/baselines.py:24` `fixed_pipeline` |
| Run tasks x configurations x seeds | `skeptic/bench/run.py:47` `run_one` |
| Score against the key | `skeptic/bench/score.py:21` `score` |
| Paired statistics | `skeptic/bench/report.py:68` `paired` |

## 5. Questions a reviewer is likely to ask

- **Why no LangChain?** The loop is about 70 lines (`harness.run`); writing it by hand keeps every gate visible
  and testable (`tests/test_harness.py` drives it with a scripted model, no API key).
- **Where does a number in an answer come from?** It must appear in a tool result in the same run, or
  `verify.unverified_numbers` rejects the answer.
- **How does ReAct differ?** Same loop and tools and a short prompt. The ledger gate, identifiability gate and critic
  are off; the evidence-id and number checks and the injection guard stay on (`harness.py:51`, `harness.py:79`).
- **How do you know a planted fault is findable?** `build.oracle` runs the reference check on the corrupted copy;
  a task is kept only if it fires, and a control only if it stays quiet on clean data.
