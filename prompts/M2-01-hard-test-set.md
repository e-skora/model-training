# Prompt M2-01: Write the hard test set

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `milestones/M2-rl-judge/SPEC.md`, whose section "The hard test set" is the contract for this task;
- `milestones/M1-contradiction-judge/SPEC.md`, for "What the model does", "Reading rules" 1 to 5 and "Data rules";
- `DECISIONS.md` D-036 to D-042;
- `LESSONS.md`, above all the 2026-10-21 lesson on shortcut patterns and the 2026-10-22 lesson "when dates can vanish, a contradiction must not look like an update".

This task is M2 Step 1: write the hard test set. Do not start any other step. No GPU or Modal runs.

Create and switch to a branch named `m2-01-hard-test`.

1. **Write 300 pairs** to `data/m2/test-hard.jsonl`, one JSON object per line.
   - **Format:** the M1 format plus one field, `hard`, naming the pair's trap:
     ```json
     {"id": "h-001", "person": "...", "a": {"text": "...", "written_at": "2026-02-11"}, "b": {"text": "...", "written_at": "2026-02-11"}, "label": "contradiction", "type": null, "hard": "explain_away", "reason": "..."}
     ```
   - **Contradictions, 150:**
     - `explain_away` (40): a reader could invent a way both are true ("maybe she has two"), but reading rule 4 makes it a contradiction.
     - `paraphrase` (30): the clash is said in different words, with little word overlap.
     - `small_difference` (30): the two memories differ by one number, day, colour or place.
     - `change_word_contradiction` (25): a change word appears, but both describe the same moment, or the same past change told two different ways.
     - `far_apart_no_change` (25): written months apart, with no change described, about a current fact (reading rule 1).
   - **Updates, 50:**
     - `indirect_update` (25): the change is said without the usual change words ("later", "now", "moved", "switched").
     - `update_with_overlap` (25): wording close to a contradiction, plus a clear change.
   - **Compatible details, 50:**
     - `looks_like_clash` (30): units, counts, part and whole, nicknames, or general and specific.
     - `near_identical` (20): almost the same sentence, and both are true.
   - **Merely related, 50:**
     - `rule5_bait` (25): different days, times or places with no exclusive word (reading rule 5).
     - `near_identical` (25): almost the same sentence, but a different fact.
   - **Every pair can be decided from its text alone** (D-038). An update must show its change in its words. A contradiction must never depend on the dates. Check each pair twice: once with dates, once as if the dates were removed. Both readings must give the same label.
   - **Dates:** about half of each trap has both memories written the same day. An update always has memory b dated later. No pair has b dated before a.
   - **People:** one fictional person per pair, named in both memories. 300 different people, with names from many cultures. No name that appears in any file in `data/m1/`. Never Mara or Lindqvist.
   - **Topics:** everyday only, as in M1.
   - **Order:** shuffle the lines with a fixed seed (write the seed in the README), so no label comes more than 3 times in a row.
   - **Write to be hard by trap, never by guessing what v1 gets wrong** (D-039). Do not run any model on these pairs.
   - **How to write them:** in batches of about 25, rereading each batch against reading rules 1 to 5 before writing the next. You may split the writing across helper agents if they all work from one shared brief, saved as `data/m2/test-writing-brief.md`, as in M1 Step 6. Reread everything they write yourself.
2. **Checker.** Extend `scripts/check_dataset.py` so that:
   - when a file has `hard` fields, every pair has one, it is one of the tags above, and it fits its label and type;
   - it prints counts per trap, and same-day counts and change-word counts per trap;
   - the name wall covers every file passed with `--against` or `--distinct-from`, not only the frozen test set (check how it works today first).

   M1's files must still pass exactly as before. Add tests for the new checks in `tests/`, with temporary files only (AGENTS.md rule 10). Then run the checker until it passes:
   ```
   uv run python scripts/check_dataset.py data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl
   ```
   Run the whole test suite with `uv run pytest`.
3. **Loader and scorer.** Make sure `src/judge` reads the new file with its `hard` field, and that scoring can report precision, recall and `unclear` per trap. Do not change how M1 files are read or scored; the M1 saved answers must rescore to exactly the same numbers (prove this with a test or a rescore). Do not freeze the file: Claude in Cowork freezes it after the blind review.
4. **README.** Write `data/m2/README.md`. Cover:
   - what the file is;
   - that all people and memories are made up by Claude Code;
   - the counts per label, type and trap;
   - the shuffle seed;
   - that it is never used for training.
5. **Close.**
   - Update `STATE.md`: M2 Step 1 written, waiting for the blind review by Claude in Cowork.
   - Add new terms to `GLOSSARY.md`, and any lesson or mistake to `LESSONS.md`.
   - Commit in small steps and push the branch.
   - Do not merge.

**Report back** in plain language:
- the counts per label, type and trap;
- how many different people;
- what the checker printed;
- the number of tests that pass;
- the file's SHA-256;
- anything that is Elias's to do (expected: nothing; Claude in Cowork reviews next).
