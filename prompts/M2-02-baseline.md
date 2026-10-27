# Prompt M2-02: Score v1 and the untrained models on the hard test set (the M2 baseline)

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked
**Before you paste:** open the Ollama app on the Mac, so the judge `m1-judge:v1` can answer.

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it. Then read:
- `STATE.md`;
- `milestones/M2-rl-judge/SPEC.md`;
- `DECISIONS.md` D-036 to D-042;
- `LESSONS.md`;
- `reviews/M2-01-blind-review.md`;
- `docs/run-judge-on-mac.md`.

This task is M2 Step 2: score v1 and the untrained models on the frozen hard test set. Do not start any other step.

Create and switch to a branch named `m2-02-baseline`.

**The hard test set.** `data/m2/test-hard.jsonl` is frozen (SHA-256 in `STATE.md`). Never edit it, never train on it, and load it only through `judge.frozen.load_frozen`. Each model below answers it **once**, in all three versions: as is, swapped, and no dates.

1. **Let the scripts take a test file.** `scripts/run_baseline.py` and `scripts/eval_trained.py` have the M1 test file written in. Add a way to pass another frozen file:
   - The default must stay `data/m1/test.jsonl`, so every M1 command works exactly as before.
   - The frozen check must apply to whichever file is passed.
   - Output must never write over an existing file.
   - Add tests, with temporary files only, and keep every existing test passing.

   `scripts/run_ollama.py` already takes a file and `--final`.
2. **Smoke runs first**, on 8 validation pairs, written to a scratch folder outside `results/`, to prove each path works:
   - v1 through Ollama;
   - v1 as the adapter on Modal;
   - the untrained models on Modal.
3. **The one run on the hard set**, each into its own folder under `results/m2/baseline/`:
   - **v1 on the Mac:** `m1-judge:v1` through Ollama with `scripts/run_ollama.py ... --final`. This is the file Aeverum would use.
   - **v1 adapter on Modal:** the epoch-2 adapter (D-031) on its 4-bit base, with `scripts/eval_trained.py`. This is the starting point of RL, so v2's adapter will be compared with it.
   - **The untrained Qwen3-8B and Qwen3-4B-Instruct-2507 on Modal,** with `scripts/run_baseline.py`. These show how hard the set is.
4. **Score everything** with the scorer, per version and per trap: precision, recall, accuracy, type accuracy, `unclear`, invalid answers.
5. **Write `results/m2/baseline/summary.md`**, in plain language:
   - one table of the 4 models by the 3 versions;
   - one table of v1 (Mac) per trap;
   - v1's M1 test-set numbers next to its hard-set numbers;
   - the 5 pairs v1 gets wrong most often, in all three versions, by trap, quoting the pair and v1's reason;
   - **"Room for M2":** v1's worst-case recall and precision on the hard set, and whether the D-041 bar (+0.03 worst-case recall) is reachable. Say plainly if v1 is already near the top.
6. **Cost:** Modal runs on L4s. Expect well under $1 in total; if anything would go over $1 in one run, stop and ask (D-019). Add one line per GPU run to `runs/RUNLOG.md`.
7. **Close:**
   - Update `STATE.md`: M2 Step 2 done, waiting for the check by Claude in Cowork, with the headline numbers.
   - Add any lesson or mistake to `LESSONS.md`, and new terms to `GLOSSARY.md`.
   - Commit in small steps and push the branch.
   - Do not merge.

**Report back** in plain language:
- the main table;
- v1's weakest traps;
- the "Room for M2" verdict;
- GPU minutes and cost;
- the number of tests that pass;
- anything that is Elias's to do (expected: nothing).
