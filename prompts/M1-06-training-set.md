# Prompt M1-06: Training set, Aeverum-fit changes, new baseline

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md`, `milestones/M1-contradiction-judge/SPEC.md`, `DECISIONS.md` (especially D-021 to D-027) and `LESSONS.md` (especially the test-set lessons). This task is M1 Step 6. Do not start training (Step 7).

Create and switch to a branch named `m1-06-training-set`. Commit after each part.

## Part A: code changes for Aeverum (D-025, D-026, D-027)
1. **Prompt** (`src/judge/prompt.py`): allow `unclear` as a third label for pairs the text alone cannot settle. Say that the order of the two memories means nothing, and that dates may be missing, in which case updates are judged from the wording. Let the function build a pair with or without dates. Add a function that returns the SHA-256 of the prompt template (D-027).
2. **Scorer** (`src/judge/score.py`): accept `unclear` as a valid label. It is not a "yes", so it never enters precision, it lowers recall when the right label is contradiction, and it gets its own count. Add scoring of two derived versions of the frozen test set, built in memory without touching the file (D-026): **swapped** (memories in the other order) and **no dates**. Report all three versions side by side. Add tests for all of this.
3. **Training formatter** (`src/judge/format.py`): turns pairs into chat training examples (the prompt's messages plus the correct JSON answer). Using a fixed seed (write it in the code), it shows the later memory first in about half the examples and leaves dates out of about a third. Add tests.

## Part B: write the data
4. **Training set:** `data/m1/train.jsonl`, about 1,500 pairs, in the same format as `data/m1/test.jsonl`. Keep the earlier memory as `a`; the formatter does the shuffling. Label `unclear` uses `"type": null`. The mix:
   - about 45% `contradiction`
   - about 50% `non_contradiction`, split evenly across `update`, `compatible_detail` and `merely_related`
   - about 5% `unclear` (pairs that could clash or not depending on facts the text doesn't give, such as unclear scope or time; never the "silent change" pairs, which stay `contradiction` under reading rule 1)
5. **Validation set:** `data/m1/valid.jsonl`, 100 more pairs, same mix. Training uses them only to watch for overfitting, which is when a model memorizes its practice examples instead of learning the rule. Never train on them.
6. **Same quality bar as the test set, from the start** (see LESSONS.md):
   - The hard cases by name: change words in every label; at least 10% of updates said indirectly; contradictions and non-contradictions that contain change words.
   - Dates spread about half same-day per label (except updates, which are always dated later).
   - No sentence template used more than 3 times in 100 pairs.
   - Everyday topics only, many cultures, every pair about one person named in both memories.
   - Write in batches of about 50 and reread each batch against the reading rules. You may split the batches across subagents, each given the rules, the hard-case list and a separate range of names and topics.
7. **Walls:**
   - No person's name from `data/m1/test.jsonl` may appear in train or valid.
   - No pair in train or valid may be a near copy of a test pair (more than 60% shared words between the two pairs' text).
   - No pair may be repeated across train and valid.
   - Never Mara or Lindqvist, and nothing from Aeverum.
   - Extend `scripts/check_dataset.py` to check all of these (`--against data/m1/test.jsonl`), then run it until it passes on both files.
8. **Spot-check sheet:** `data/m1/spotcheck-train.md`, 20 pairs from train chosen with seed 11, including at least 2 `unclear` pairs and at least 5 hard cases. Use the same `Elias: OK` format as before.

## Part C: new baseline with the new prompt
9. The prompt changed, so the Step 5 numbers are no longer a fair "before" score. Rerun the baseline for `unsloth/Qwen3-4B-Instruct-2507` and `unsloth/Qwen3-8B` (thinking off) on all three test versions (as is, swapped, no dates). Save the results under `results/m1/baseline-v2/` with a `summary.md` that has the same columns plus `unclear` and the prompt's SHA-256. Keep `results/m1/baseline/` as it is. Use an L4 and log the runs in `runs/RUNLOG.md`. If this would cost more than $1, stop and ask.

## Close
10. Update `data/m1/README.md`, `STATE.md` (Step 6 written, waiting for Elias's spot-check of `data/m1/spotcheck-train.md`), `GLOSSARY.md` (validation set, overfitting, and any other new terms) and `LESSONS.md`. Run `uv run pytest` until it passes. Push the branch. Do not merge.

**Report back** in plain language: the label counts for train and valid, what the checker printed, the new baseline table across all three test versions, total GPU minutes and cost, and the spot-check file path.
