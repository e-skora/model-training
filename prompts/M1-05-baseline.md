# Prompt M1-05: Baseline scores and pick the base model

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md` and `milestones/M1-contradiction-judge/SPEC.md`. This task is M1 Step 5: score untrained models on the frozen test set (the "before" score) and pick the base model we will fine-tune. Do not start any other step.

Create and switch to a branch named `m1-05-baseline`.

**Rules for this task**
- `data/m1/test.jsonl` is frozen. Before using it, check its SHA-256 matches the one in `STATE.md`. Never edit it.
- Use GPU type `L4` and set a timeout on every Modal function. If the whole step would cost more than $1, stop and ask Elias first.
- Check every claim about Unsloth, Modal or a model's settings against current docs, and write the URLs in your report.

**Do this**
1. **One shared prompt.** Write `src/judge/prompt.py` with one function that turns a pair (person, memory A with date, memory B with date) into chat messages for the model. The system message explains the task, the three reading rules from the SPEC, the labels and types, and says to answer with JSON only, in the SPEC's format. Training in Step 7 will reuse this exact function, so keep it in one place.
2. **Scoring code.** Write `src/judge/score.py`. It reads the test pairs and a file of model answers and computes:
   - contradiction precision and recall (contradiction is the "positive" label)
   - overall label accuracy
   - type accuracy on the non-contradictions the model got right
   - how many answers were not valid JSON or used an unknown label (these count as wrong)
   Write tests for it in `tests/test_score.py` using small made-up answers (no GPU, no network, temporary folders only), add `pytest` as a dev dependency, and make the tests pass.
3. **Baseline run on Modal.** Write `scripts/run_baseline.py`. It loads each candidate model on an L4 with Unsloth (4-bit), answers all 200 test pairs with greedy decoding (temperature 0, so answers repeat the same every time), and saves the raw answers to `results/m1/baseline/<model-short-name>.jsonl`. The candidates are:
   - `unsloth/Qwen3-4B-Instruct-2507` (Apache 2.0)
   - `unsloth/Qwen3-8B` (Apache 2.0). It is a "thinking" model, so turn thinking off as Qwen's docs describe.
   - `unsloth/Phi-4-mini-instruct` (MIT)
   Reuse the image from `scripts/hello_gpu.py`. If the pinned Unsloth (2025.7.8) or transformers can't load a candidate, check PyPI and Unsloth's docs for current versions, update the pins in a shared image module, rerun `scripts/hello_gpu.py` to prove it still works, and write a lesson about it. If one candidate still can't load, skip it and say why.
4. **Results.** Score each model's answers and write `results/m1/baseline/summary.md`: a plain-language table with one row per model and these columns: precision, recall, accuracy, type accuracy, invalid answers, minutes, cost. Under the table, show 3 example mistakes from the best model, with a one-line note on each.
5. **Pick the base model.** Choose the candidate with the best contradiction precision and recall combined that fits training on an L4. If two are close, choose the smaller one. Add a dated entry to `DECISIONS.md` saying which model and why (it is a Default; Elias can override). Log every GPU run in `runs/RUNLOG.md`.
6. **Close.** Update `STATE.md` (Step 5 done; next action: Codex reviews the scoring code, then Step 6). Close the Unsloth-pin open item in `STATE.md` with what you found. Add new terms to `GLOSSARY.md` (for example: greedy decoding, 4-bit, system message, chat template) and any lesson or mistake to `LESSONS.md`. Commit in small steps, push the branch, and do not merge.

**Report back** in plain language: the summary table, which model you picked and why, total minutes and cost, whether the pins had to change, and the doc URLs you checked.
