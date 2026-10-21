# Prompt M1-04: Write the test set

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then `STATE.md` and `milestones/M1-contradiction-judge/SPEC.md` (the "What the model does", "Reading rules" and "Data rules" sections are the contract for this task). This task is M1 Step 4: write the test set. Do not start any other step. No GPU runs are needed.

Create and switch to a branch named `m1-04-test-set`.

1. **Write 200 test pairs** to `data/m1/test.jsonl`, one JSON object per line:
   ```json
   {"id": "t-001", "person": "Jonas Okafor", "a": {"text": "Jonas's favourite season is autumn.", "written_at": "2026-03-04"}, "b": {"text": "Jonas's favourite season is spring.", "written_at": "2026-03-04"}, "label": "contradiction", "type": null, "reason": "Both state his favourite season at the same moment and name different seasons."}
   ```
   - Exactly 100 `contradiction` pairs and 100 `non_contradiction` pairs: 34 `update`, 33 `compatible_detail`, 33 `merely_related`.
   - Every pair is about one fictional person, and both memories name that person. Use at least 60 different people with names from many cultures. Never use the name Mara or Lindqvist.
   - Everyday topics only: home, work, food, hobbies, travel, pets, routines, belongings, tastes, places. No health, money, religion, politics or other sensitive topics.
   - Make it hard. Most pairs should look alike on the surface:
     - Contradictions where the difference is small (a different day, colour, number or place).
     - Same-moment contradictions where the two memories were written on different dates but neither describes a change. These are still `contradiction`, per reading rule 1.
     - Updates that use a change word ("later", "now", "switched", "moved", "since then") with a later `written_at`.
     - Compatible details and merely related pairs that share many words with each other.
   - Shuffle the order so labels do not come in runs.
   - Write in batches of about 25 and reread each batch against the reading rules before writing the next.
2. **Checker script.** Write `scripts/check_dataset.py`. Given a JSONL file, it prints counts per label and type, and fails with a clear message if: a line is not valid JSON or is missing a field; `type` is set on a contradiction or missing on a non-contradiction; an `id` or an (a, b) text pair repeats; any text contains "Mara" or "Lindqvist". Run it on `data/m1/test.jsonl` until it passes.
3. **Spot-check sheet.** Write `data/m1/spotcheck-test.md`: 20 pairs chosen at random with a fixed seed (write the seed in the file), at least 8 contradictions and 8 non-contradictions. For each, show the two memories with dates, the label, the type and the reason in plain text, then a line `Elias: OK` that Elias changes to `Elias: WRONG, <why>` if he disagrees. Put a short note at the top explaining what to do.
4. **How it was made.** Write `data/m1/README.md`: what the files are, that all people and memories are made up by Claude Code, the label counts, and that the test set is never used for training.
5. **Close.** Update `STATE.md`: Step 4 written, waiting for Elias's spot-check; the next action is Elias's. Add new terms to `GLOSSARY.md` and any lesson or mistake to `LESSONS.md`. Commit in small steps and push the branch. Do not merge; Claude in Cowork checks, freezes and merges.

**Report back** in plain language: the label counts, how many different people, what the checker printed, and exactly what Elias has to do with the spot-check sheet (the file path).
