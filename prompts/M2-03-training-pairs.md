# Prompt M2-03: Write the hard training and validation pairs

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `milestones/M2-rl-judge/SPEC.md` (section "Training data" is the contract);
- `milestones/M1-contradiction-judge/SPEC.md` (reading rules 1 to 5);
- `DECISIONS.md` D-028 to D-030 and D-036 to D-044;
- `LESSONS.md`, above all: "AI-written 'unclear' examples drift toward dodging", "the name wall catches ordinary capitalised words too", and "splitting data writing across helper agents works with one shared brief and a checker";
- `data/m1/train-writing-brief.md`;
- `data/m2/README.md`.

This task is M2 Step 3, part A: write the hard training and validation pairs. Do not start any other step. No GPU or Modal runs.

Create and switch to a branch named `m2-03-training-pairs`.

**Never open `data/m2/test-hard.jsonl` to read its pairs, and never copy or rework them.** It is the frozen exam. The checker compares against it by itself. Its per-trap results in `results/m2/baseline/` must not steer the mix either: the trap counts below are fixed in advance, and Step 3 part B lets v1's own answers pick which pairs RL trains on.

1. **Write a brief first,** `data/m2/train-writing-brief.md`. It covers:
   - the format, labels, reading rules 1 to 5, and D-044;
   - the trap list with definitions and one made-up example each, written fresh (not from any test file);
   - the date rules: about half the pairs written on the same day; an update always has b later; b is never dated before a;
   - the rule that every pair can be decided from its text alone, except true `unclear` pairs;
   - the rules for `unclear`, from D-029 and D-030;
   - the name rules: fresh people, and check every capitalised word against the name wall;
   - everyday topics only;
   - no templates;
   - what each writer hands back.
2. **Write the pairs** to `data/m2/train-hard.jsonl` (1,000 pairs) and `data/m2/valid-hard.jsonl` (100 pairs).
   - **Format:** the hard test format plus two fields (D-043): `"source": "synthetic"` and `"label_by": "claude-code"`. Ids are `tr2-0001` and `va2-001`.
   - **Training counts:**
     - contradiction 470: explain_away 125, paraphrase 95, small_difference 95, change_word_contradiction 77, far_apart_no_change 78;
     - update 160: indirect_update 80, update_with_overlap 80;
     - compatible_detail 170: looks_like_clash 100, near_identical 70;
     - merely_related 170: rule5_bait 85, near_identical 85;
     - unclear 30: tag `true_unclear`.
   - **Validation counts:** about one tenth of each: contradiction 47 (13, 9, 9, 8, 8), update 16 (8, 8), compatible_detail 17 (10, 7), merely_related 17 (9, 8), unclear 3.
   - **`unclear` pairs:**
     - Each one must be one where a truly missing fact decides the label and neither reading is more natural (rule 4).
     - Have a second helper agent judge every `unclear` pair blind (D-030). Keep only those it also calls `unclear`, and write new ones until the count is met.
     - No `unclear` reason may start with the same three words more than 3 times in any 100.
   - **People:** fresh in every pair. No name, or other capitalised word, that the name wall rejects against all M1 files and the hard test set.
   - **How to write them:**
     - Helper agents from the one brief, in batches of about 25. Shuffle with a fixed seed, and write the seed in the README.
     - Reread every batch yourself against reading rules 1 to 5 before accepting it. Fix or replace weak pairs, and keep a short log in `data/m2/README.md` of how many you fixed and why.
3. **Checker.** Extend `scripts/check_dataset.py` so that:
   - `true_unclear` is a valid tag only with `label` `unclear`;
   - files with `source` or `label_by` have both on every line, with allowed values `synthetic` and `claude-code`;
   - the hard test set can be passed with `--against`, so the name wall and the near-copy check cover it.

   Add tests, with temporary files only. Run:
   ```
   uv run python scripts/check_dataset.py data/m2/train-hard.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/valid-hard.jsonl
   uv run python scripts/check_dataset.py data/m2/valid-hard.jsonl --against data/m2/test-hard.jsonl --against data/m1/test.jsonl --distinct-from data/m1/train.jsonl --distinct-from data/m1/valid.jsonl --distinct-from data/m2/train-hard.jsonl
   ```
   If `--against` takes only one file today, make it take several. Both must pass. `uv run pytest` must pass.
4. **README.** Add a section to `data/m2/README.md`. Cover:
   - the counts per label, type and trap for both files;
   - the shuffle seed;
   - the helper-agent and reread log;
   - the `unclear` double-check result;
   - SHA-256 of both files;
   - that neither is ever used to score the final model.
5. **Close.**
   - Update `STATE.md`: M2 Step 3 part A written, waiting for the blind review by Claude in Cowork.
   - Add any lesson or mistake to `LESSONS.md`, and new terms to `GLOSSARY.md`.
   - Commit in small steps and push the branch.
   - Do not merge.

**Report back** in plain language:
- the counts;
- how many pairs you fixed or replaced while rereading;
- the `unclear` double-check numbers;
- what the checker printed for both files;
- the number of tests that pass;
- both SHA-256 values;
- anything that is Elias's to do (expected: nothing).
