# M3-02: the M3 baseline (v1 and v2 on the two new sets)

**Where:** Claude Code (the desktop Code panel), **new session**, folder `~/code/model-training`, branch picker **`main`**, worktree box **unchecked**. Make the branch `m3-02-baseline` from `main`.

Copy everything below the line into Claude Code.

---

Read `STATE.md` first, then `docs/m2-lessons-and-fixes.md`, `milestones/M3-rl-judge-attempt-2/SPEC.md` (Steps 2 and 6 to 8), `DECISIONS.md` (D-041, D-060, D-063), `LESSONS.md` and `results/m2/baseline/summary.md`.

**Goal:** M3 Step 2. Score v1's adapter and v2's adapter once each on the frozen `data/m3/test-hard-2.jsonl` (600 pairs) and the frozen `data/m3/valid-hard-2.jsonl` (200 pairs), in all three versions. This is the baseline the M3 bar is measured from (v1) and the starting point training builds on (v2). Lessons applied: A2, A4, A6, F2.

**How:**
- Use `scripts/eval_trained.py` exactly as in M2 (`--test`, `--model v1` or `--model v2`), so both adapters are scored the way M2 scored them (A4). Change no scoring code. If the script cannot take one of these files as it is, make the smallest change and add a test.
- First one smoke run (8 validation pairs) into a scratch folder outside `results/`.
- Then four real runs, each answered once: `results/m3/baseline/v1/test`, `results/m3/baseline/v1/valid`, `results/m3/baseline/v2/test`, `results/m3/baseline/v2/valid`. If a run fails before saving answers, fix it and run again. Once answers are saved, never run that set with that model again.
- Expected cost: about $0.60 in all, each run under $1 (v1's 900-answer hard-set run in M2 was 7.2 L4 minutes, $0.11). If a run looks likely to go over $1, stop and ask Elias first (D-019).
- Log the time spent waiting for a GPU apart from the work (F2).

**Report** `results/m3/baseline/summary.md`, answer first:
- For each model, set and version: caught / said / real contradictions, precision, recall, `unclear`, invalid, and answers that break v1's format (`judge.reward.read_answer`). Print counts, not reduced fractions (A6).
- Per trap on test set 2 (`judge.score.score_by_trap`), v1 against v2.
- The pair-by-pair count on test set 2 in each version: pairs v1 misses that v2 gets, and the reverse.
- **What the M3 bar will be,** from v1's numbers: worst-case recall at least v1's + 0.03 of 300, worst-case precision at least v1's - 0.01. Give both as pair counts. This is a record only; the rule itself is coded in Step 6.
- Where v1 and v2 differ from their M2 hard-set results, and whether test set 2 looks harder or easier than M2's set (per-trap shares), as facts only.

**Never:** train, change the reward, or pick anything by these numbers except the record above. Never open `data/m1/test.jsonl` or `data/m2/test-hard.jsonl`, and never run `shasum` on frozen files.

**Finish:** `runs/RUNLOG.md` lines with billed costs once Modal reports them, update `STATE.md`, commit and push `m3-02-baseline`. Do not merge: Claude in Cowork rescores on its own and merges. Then tell Elias in plain words what the baseline shows and that his next step is to tell Claude in Cowork.
