# Prompt M2-04b review: Codex checks the fixes to its M2-04 review

**Where:** Codex (ChatGPT desktop app), **new session**, working directly in the folder `~/code/model-training` on your Mac.
**Branch:** `m2-04-pool-and-reward` (the folder is already on it; do not switch).

Copy everything below the line into Codex.

---

You are reviewing code in `~/code/model-training`, which is on branch `m2-04-pool-and-reward`. Read `AGENTS.md` first. **This is a read-only review:**
- Do not switch branches.
- Do not edit, commit or delete anything.
- Do not start any GPU or Modal job, and do not use the network.
- Never open `data/m2/test-hard.jsonl` or `data/m1/test.jsonl`.
- The only file you write is `reviews/M2-04b-codex-review.md`.

Context: your review `reviews/M2-04-codex-review.md` said CHANGES REQUIRED, with three must-fix findings: (1) the reward fingerprint missed the answer reader's helpers and constants; (2) pool selection did not check the recorded prompt fingerprint; (3) a repeated sample ID could hide a missing candidate. Claude Code fixed them in the commits after `a28db37`. What changed and which test covers each: `reviews/M2-04-fixes.md`. The offline recheck of the saved pool: `results/m2/pool/reward-recheck.md`.

Review `git diff a28db37..HEAD` and check:
1. **Finding 1:** does `reward_sha256()` now change for every change that alters what earns points? Retry your three reproductions (remove `update` from the allowed types, read repeated keys with a plain dictionary, let `NaN` through) and try others: the fence pattern, `LABELS`, a new helper called from `parse_answer` or `format_problem`, `completion_text`. Is there any way left to change a reward without changing the fingerprint? Can `rule_parts` loop forever, miss a name, or crash on something the reward code uses?
2. **Finding 2:** does `choose_pool` stop on a changed prompt, show seed, keep seed or answers per pair, before it writes anything?
3. **Finding 3:** does `check_sampled_ids` stop on repeated, missing and unexpected IDs, for full and smoke runs, before anything is written?
4. **The recheck:** run the script in `results/m2/pool/reward-recheck.md` and confirm its numbers. Is it right that the saved files keep the old fingerprint `a369e730...` and Step 5 records the new one?
5. **Nothing else changed:** no reward value, saved sample, pool row, always-wrong row or summary differs from `a28db37`.
6. **Rerun:** run `uv run pytest` and report the count.

Write `reviews/M2-04b-codex-review.md` in plain language a non-engineer can follow:
- a verdict line at the top: `ACCEPT` or `CHANGES REQUIRED`
- then each finding, with the file and line, what is wrong, why it matters, and the fix
- mark each finding `must fix` or `nice to have`
- if you find nothing, say so
