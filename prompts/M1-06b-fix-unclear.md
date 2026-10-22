# Prompt M1-06b: Fix the `unclear` pairs after the blind review

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m1-06-training-set`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training` on branch `m1-06-training-set`. Read `AGENTS.md` first and follow it. Then read `STATE.md`, the SPEC's reading rules (`milestones/M1-contradiction-judge/SPEC.md`), which now have rules 4 and 5, `DECISIONS.md` D-029, and the review `reviews/M1-06-blind-review.md`. This task fixes the training and validation data. **No GPU runs.** Do not start Step 7.

1. **Recheck every `unclear` pair** in `data/m1/train.jsonl` and `data/m1/valid.jsonl` against rules 4 and 5. Keep a pair `unclear` only if a truly missing fact decides it and neither reading is more natural. Relabel all the others and rewrite their reasons:
   - Start with the 34 the review lists. The reviewer's answers are a strong hint but not final; apply the rules yourself.
   - Inventing an escape ("maybe two dogs") does not make a pair unclear.
2. **Apply rule 5 across the whole file,** not just the sample. Find every pair about the same activity on different days, times or places. With no exclusive word ("only", "always", "every day", "never"), it is `non_contradiction` / `merely_related` (or `compatible_detail` when one adds detail to the other). With an exclusive word, decide by the words.
3. **Top `unclear` back up to about 5%** with new pairs that pass the review's standard, such as the examples it agreed were truly undecidable. Keep the train and valid totals and the label mix close to before (about 45% contradiction, 50% non-contradiction, 5% unclear). New pairs follow every wall in the checker.
4. **Vary the `unclear` reasons.** No more than 3 in any 100 may start the same way. Today 73 of 74 start "It depends on whether..." or "Whether they clash depends on...".
5. Run `scripts/check_dataset.py` on both files with `--against data/m1/test.jsonl` until it passes. Add a check for the reason-opening rule in item 4.
6. **Close.** Write `reviews/M1-06-fixes.md`: how many pairs were relabelled and to what, how many new `unclear` pairs were added, and the new label counts. Update `data/m1/README.md`, `STATE.md` (next action: Claude in Cowork re-checks the `unclear` pairs, then merges) and `LESSONS.md`. Run `uv run pytest`. Commit, push the branch, do not merge.

**Report back** in plain language: the relabel counts, the new label counts for train and valid, and what the checker printed.
