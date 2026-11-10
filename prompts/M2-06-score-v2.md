# M2-06: score v2 once on the two frozen test sets

**Where:** Claude Code (the desktop Code panel), **new session**, folder `~/code/model-training`, branch picker **`main`**, worktree box **unchecked**. Make the branch `m2-06-test-score` from `main` at `22e4c96` or later.

Copy everything below the line into Claude Code.

---

Read `STATE.md` first, then `milestones/M2-rl-judge/SPEC.md` ("Done means"), `DECISIONS.md` (D-024, D-041, D-056, D-057, D-060) and `LESSONS.md`.

**Goal:** M2 Step 6. Score v2 once on `data/m2/test-hard.jsonl` and once on `data/m1/test.jsonl`, in all three versions (as written, swapped, no dates). Then say plainly whether v2 meets D-041.

**v2** is the step-250 adapter chosen in Step 5, in the private repo `eskora/m2-contradiction-judge-qwen3-4b-grpo-lora` at commit `0964d031073408707397ce7438fc7df7b8a772c0` (record: `results/m2/grpo/`). Load it at that exact commit, on the same pinned base as v1.

**The rule is fixed before any score (D-060).** Write it into code and commit it before any GPU run.

1. **Compare v2's adapter with v1's adapter.** v1's saved answers are the baseline: `results/m2/baseline/trained-v1/` (hard set) and `results/m1/trained-v1/qwen3-4b-trained-v1.*.jsonl` (M1 set). Do not run v1 again.
2. **Item 1 (hard set):** worst-case recall (the lowest of the three versions) must be at least v1's worst-case recall + 0.03. Worst-case precision must be at least v1's worst-case precision - 0.01. Use exact fractions, like `choose_checkpoint`.
3. **Item 2 (M1 set):** in each version, v2's precision and recall must each be at least v1's - 0.02 in that same version (SPEC "Done means" item 2).
4. **Item 3:** 0 invalid answers in every version of both sets. Also report answers that break v1's format (`judge.reward.read_answer`).
5. Write `scripts/compare_v2.py`. It reads saved answer files only, applies 1 to 4, and prints PASS or MISS for each item and overall. Add tests built from made-up answers (no test file opened). Commit and push it before the GPU run.

**How to run:**
- Score v2 the way v1 was scored: `scripts/eval_trained.py`, greedy, same prompt and settings. Teach it to take the v2 adapter (repo plus commit) without changing how it scores v1. If you change anything under `src/`, the reward fingerprint changes; record that.
- First a smoke run on 8 validation pairs into a scratch folder outside `results/`.
- Then the two real runs, into `results/m2/v2/hard/` and `results/m2/v2/m1/`. Each test set is answered **once**. If a run fails before it saves answers, fixing it and running again is fine. Once answers are saved, never run that set again.
- Expected cost: well under $1 for each run (v1's hard-set run was 7.2 L4 minutes). If a run looks likely to go over $1, stop and ask Elias first (D-019).

**Never:**
- Score any other checkpoint (step 300, step 314 or any other) on either test set.
- Train again, or change the reward or the choice, because of a test score.
- Open either frozen test file outside the scoring code, and never run `shasum` on them (LESSONS 2026-11-04).

**If v2 misses item 1, 2 or 3,** that is a real result (D-041). Write it plainly. Do not try to rescue it. Step 7 (GGUF for the Mac) waits for Elias in that case.

**Report:**
- `results/m2/v2/summary.md`, answer first. A table with v1's and v2's precision, recall, `unclear` and invalid counts in each version of both sets, with exact counts. PASS or MISS for each D-041 item. Per-trap results on the hard set (`judge.score.score_by_trap`), v1 against v2.
- A `runs/RUNLOG.md` line for each GPU run, with the billed cost once Modal reports it.
- Update `STATE.md`, then commit and push on `m2-06-test-score`. Do not merge: Claude in Cowork rescores v2 on its own and merges.
- Then tell Elias, in plain words, whether v2 beats v1, and that his next step is to tell Claude in Cowork.
