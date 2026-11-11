# M3-01: write hard test set 2 and hard validation set 2

**Where:** Claude Code (the desktop Code panel), **new session**, folder `~/code/model-training`, branch picker **`main`**, worktree box **unchecked**. Make the branch `m3-01-test-set-2` from `main`. Paste this only after M2-08 has finished and Claude in Cowork has merged it.

Copy everything below the line into Claude Code.

---

Read `STATE.md` first, then `docs/m2-lessons-and-fixes.md`, `milestones/M3-rl-judge-attempt-2/SPEC.md` (Step 1), `DECISIONS.md` (D-038, D-039, D-042, D-063), `LESSONS.md`, `data/m2/README.md` and `reviews/blind-brief.md`.

**Goal:** M3 Step 1. Write `data/m3/test-hard-2.jsonl` (600 pairs) and `data/m3/valid-hard-2.jsonl` (200 pairs). No GPU, no Modal, no model runs. Lessons applied: A2, A3, A5, C1, C4.

**Counts:** the test set has the same 11 traps as `data/m2/test-hard.jsonl`, each at twice M2's count: 300 contradictions, 100 each of update, compatible_detail and merely_related, 0 unclear. Same-day and change-word shares per trap as in M2's table. The validation set has the same mix at a third of that (100 contradictions, 100 others), 0 unclear. Every pair has a different person, and no person or name from any earlier file (all of `data/m1/` and `data/m2/`).

**1. Situation plan first (C1).** Before writing any pair, write `data/m3/situation-plan.jsonl`: one line per pair (800 lines), giving its set, trap, label and type, the person's name and culture, the topic, and in one short phrase what the clash or the change is. No two lines may share a topic and clash idea. Check the plan for repeated ideas against itself and against every pair in `data/m1/` and `data/m2/` before writing. Commit it.

**2. Write the pairs from the plan.** You may use helper agents, each with its own slice of the plan and the reading rules. No helper sees another's slice or any model's answers. Reread every batch against reading rules 1 to 5 before the next. Dates follow the time words in the text (C4).

**3. A cross-pair check (C1).** Add it to `scripts/check_dataset.py`: compare every pair with every other pair across all files given (the new files and every earlier file), and flag pairs whose texts are too alike. Choose the measure and the limit by trying it on M2's training pairs, where repeated ideas were found by hand (LESSONS 2026-10-27). It should flag those and few others. Write down the measure, the limit and that test in `data/m3/README.md`. It fails above the limit. Add tests.

**4. Run every check.** The existing checker (traps, names, dates, labels, `source`, `label_by`) and the new cross-pair check, on both files. Both must PASS. Fix and rerun until they do.

**5. Shuffle and ids:** the same no-more-than-3-in-a-row rule as M2, with seed 20261111. Ids `h2-001` to `h2-600` and `v2-001` to `v2-200`.

**Never:** open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl` except through the checker's name and repeat walls, and never run `shasum` on them. Never run a model on the new files.

**Finish:** write `data/m3/README.md` (counts, traps, how it was written, the cross-pair check), give each file's SHA-256 (not frozen yet), run the tests, update `STATE.md` and `GLOSSARY.md`, then commit and push `m3-01-test-set-2`. Do not merge. Claude in Cowork runs the blind review with control pairs (C3) and freezes the sets. Then tell Elias in plain words what is ready and that his next step is to tell Claude in Cowork.
