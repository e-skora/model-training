# Prompt M1-04b: Make the test set harder

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `m1-04-test-set`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training` on branch `m1-04-test-set`. Read `AGENTS.md` first and follow it, then `STATE.md` and `milestones/M1-contradiction-judge/SPEC.md`. This task revises M1 Step 4. Do not start any other step. No GPU runs.

An independent checker read all 200 pairs in `data/m1/test.jsonl`. It found **no wrong labels**, but the set is too easy and too repetitive. A model could score well with shortcuts instead of understanding:
- Every `update` has a change word ("now", "moved", "switched", "since", "stopped", "later", "replaced"), and no other pair has one. A rule like "change word means update" would get almost everything right.
- Dates follow patterns: 28 of 33 `merely_related` pairs share one date, and all updates have gaps of months.
- Templates repeat with only a noun or colour swapped: "only X is ...", "favourite X is A / B", "has no X at all / has X", "never did X / did X". `compatible_detail` is always "vague fact + more specific fact", and `merely_related` is almost always "A's thing 1 / A's thing 2".
- t-068: memory b is dated earlier than a. Fix the dates so a is earlier.
- t-140, t-174, t-179: the two memories are about different people. Every pair must be about one person, named in both memories. Rewrite them.

**Do this:**
1. Rewrite about 60 pairs. Keep the totals exactly: 100 `contradiction`; 34 `update`, 33 `compatible_detail`, 33 `merely_related`. Keep each kept pair's `id`; rewritten pairs keep their slot's `id`.
2. Add these hard cases:
   - At least 15 `contradiction` pairs whose memories contain a change word but are not updates. Examples: both written the same day ("Ana now cycles to work" / "Ana now takes the bus to work"); or the change word is about something else ("Since moving, Omar keeps his bike in the hallway" / "Omar keeps his bike in the shed", same moment).
   - At least 15 `compatible_detail` or `merely_related` pairs that contain a change word but do not clash ("Lena switched to oat milk" / "Lena drinks her coffee black in the afternoons" can both be true).
   - At least 8 `update` pairs whose change is said indirectly, without the common words above ("These days...", "After the renovation...", "As of June...", "Her new flat is...").
3. Spread the dates: about half of each label and type on the same date, half on different dates. The exception is `update`, where b is always dated later.
4. Use no sentence template more than 3 times across the whole file.
5. Make `compatible_detail` and `merely_related` varied: not only "vague + specific" or "thing 1 / thing 2".
6. **Checker.** Extend `scripts/check_dataset.py` so it also fails when: an `update` has b dated on or before a; the person's first name is missing from either memory. Have it print how many pairs per label contain a change word from the list above. Run it until it passes.
7. **Spot-check sheet.** Rewrite `data/m1/spotcheck-test.md` with 20 new pairs picked at random with seed 7 (write the seed in the file). At least 8 contradictions, at least 8 non-contradictions, and at least 5 of the new hard cases. Keep the same `Elias: OK` format and instructions at the top.
8. **Close.** Update `data/m1/README.md` and `STATE.md` (Step 4 revised; next action is Elias's spot-check). Add to `LESSONS.md`: "2026-10-21: Mistake: the first test set followed obvious templates and keyword patterns; an independent check caught it before freezing. Lesson: ask for hard cases by name, check for shortcut patterns, and have a second agent review before freezing." Commit in small steps, push the branch, do not merge.

**Report back** in plain language: what changed, the label counts, the change-word counts per label, what the checker printed, and the spot-check file path.
