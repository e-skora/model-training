ACCEPT

Reviewed by Codex on 2026-11-01 in `/Users/frank/code/model-training`.
Branch: `m2-04-pool-and-reward`.
Commit: `0713ec2d74899d070c6bb4fa0d9acccf0ff381ec`.
Review range: `61616b64a3f39707004d016a28e66c89b3082db7..0713ec2d74899d070c6bb4fa0d9acccf0ff381ec`.

The three earlier must-fix findings and the skipped-file issue are fixed. I found no remaining must-fix issue within D-053, Elias's rule that the check must catch ordinary mistakes, rather than code written to fool it. One deliberate bypass is a known limit. One documentation correction is nice to have. Neither blocks acceptance of these fixes. This review does not approve a Step 5 training run.

A fingerprint is a SHA-256 code that identifies the files being checked. An allow-list says what may pass, and stops everything else. All changes made for these checks were in temporary folders. This report is the only repository file written. I did not switch branches, edit implementation files, commit, push, use the network, start a GPU or Modal job, or open either frozen test file.

## Findings

### 1. Known limit: an allowed function can deliberately be used to recover `sys`

**File and line:** `src/judge/reward.py:123-125`. Related exclusions: `reviews/M2-04-fixes.md:145-153`.

The check allows reading `sys.exit`. Python also lets code reach the module that owns that built-in function through its `__self__` attribute. This deliberately indirect spelling passes:

```python
import sys
from pathlib import Path
runtime = sys.exit.__self__
runtime.path.append(str(Path(__file__).resolve().parents[2] / "plugins"))
```

I put that same code in two temporary copies of `judge/__init__.py`, made the copied scorer import `read_fields` from `policy`, and put the helper in `plugins/policy.py`. One helper rejects repeated JSON fields; the other accepts them. Both fresh processes returned fingerprint:

`ac6a5864e2bccb514eb32749a008c0e1ee2ea34d436bf705712d6236397a58ce`

On the same repeated-label answer, both the direct reward and the training wrapper's reward changed from **-2.0 to +1.0**. No source was edited during either process, and neither reward function was patched in memory.

**Why this is a known limit under D-053:** reaching the import search list through the owner of the exit function is a deliberately indirect substitute for `sys.path`. The normal versions, including `runtime = sys`, now stop. This experiment was written specifically to evade the allowed spellings; it is not an ordinary refactor or helper edit.

**Action:** no implementation fix is required for acceptance under D-053. Claude in Cowork should add this concrete example alongside the deliberate-code exclusion in the fixes note. Broadening the checker to defend against deliberate introspection would change the agreed scope. Introspection means asking a running Python object about its own internals.

### 2. Nice to have: the frozen-file warning names too few test functions

**File and line:** `reviews/M2-04-fixes.md:102`, `STATE.md:50`, and `prompts/M2-04h-codex-review.md:39` (check 15). Relevant code: `tests/test_prepublish_check.py:26-33`, `66-85`; `scripts/prepublish_check.py:306`, `359-371`.

The notes name four test functions that read the real frozen M1 file. Three more functions also cause that read through shared test setup, called fixtures:

- `test_wrong_score_in_a_table_is_caught` uses `scores`.
- `test_unknown_row_in_a_score_table_is_caught` uses `scores`.
- `test_untraced_number_in_prose_is_caught` uses `facts`.

**Why it matters:** excluding only the four named functions would still read the protected file. I inspected the shared setup and excluded all affected tests before running them. The guard reported no attempted protected reads.

**Action:** list all six affected publishing-check functions, plus the seven cases of the saved-answer scoring test, when documenting the exclusions. This is a documentation correction, not a failure of the reward fingerprint or a reason to block Step 5.

## Earlier findings, retried

| Earlier finding | Verified result |
|---|---|
| `runtime = sys; runtime.path.append(...)`, helper in `plugins/policy.py` | Stops at `judge/__init__.py`, line 3, before returning a fingerprint. |
| `__path__.append(...)` in `judge/__init__.py`, scorer imports `judge.policy` | Stops at `judge/__init__.py`, line 2. |
| `strict/src/judge` and `loose/src/judge` link to `external/src/judge`, with different neighboring helpers | Both stop at their linked `src/judge` folder. The fingerprint itself raises the error. |
| `match Worker(): case re:` followed by `re.compile` | Executes in the temporary module, then the fingerprint stops at the binding's file and line. |
| `import re; from judge.other import *`, exporting another `re` | Executes in the temporary modules, then stops on the other binding. A separate test also rejects a star import whose exports do not currently replace `re`. |
| Regular file named `__pycache__` | Moves the fingerprint at the source root, inside `judge`, and inside another source folder. |
| Changed contents inside a `__pycache__` folder | Do not move the fingerprint, as documented. |

The independently retried alias, package-path, linked-package and pattern/star examples used fresh Python processes. The repository tests also cover links at `src` and `reward.py`, links inside `src`, and links named `__pycache__` or ending in `.pyc`; all pass. A separate probe confirmed that a link at the repository folder itself is allowed and preserves the current fingerprint. That folder is outside the three checked source components, as requested.

## Allow-lists and claims

- **`sys`:** one plain import in the file's outer body permits the four documented attribute reads. Other attributes, aliasing, passing the module, different bindings, direct attribute assignment and deletion stop. Star imports prevent the allowed reads from passing. This is a check of written code, not a restriction on everything an allowed object could later expose; finding 1 demonstrates that distinction.
- **`re`:** the full code-tree check catches the requested assignment, loop, context-manager, exception, walrus, pattern, scope-declaration, argument, definition, import and deletion forms. I also tried nested sequence/mapping/class patterns, an alternative pattern, asynchronous loops/context managers, and type-parameter forms: 24 additional checks across `re`, `sys` and `__path__` all stopped.
- **Nonbinding fields:** excluding an attribute's name, a call's keyword, a class pattern's keyword names, and a `from` import's module name from the binding check is correct. The actual bound names remain in child nodes and are checked. I found no binding missed by those exclusions in the current syntax.
- **`__path__`:** the tested plain reads, writes, attribute access, import, argument and keyword forms stop. Strings are deliberately excluded, consistent with the documented limits.
- **Honest code:** today's `score.py`, `frozen.py` and `format.py` pass. D-054 intentionally rejects harmless local names such as an argument called `re`, imports inside functions, and unrelated uses of the name `sys`. These choices can require small future edits, but they are explicit, give a location, and do not obstruct today's work. I found no additional must-fix false rejection.
- **Documentation:** the current glossary entry, latest STATE entry, fourth-review fixes section and `reward_sha256()` description agree on the 11 files, allowed forms, stops and exclusions. Later text explicitly replaces the historical fingerprints and older boundary descriptions. Apart from finding 2, I found no material mismatch. The deliberate-code exclusion already covers finding 1, but its concrete reproduction should be carried forward.

The exclusions remain real: Python and outside packages, outside data/settings, runtime changes, outside wrappers, `.pyc` files, unlisted loaders such as `pkgutil`, and deliberately evasive code. I found no new ordinary-edit bypass outside those exclusions. No claim about a provider's current service or behavior was needed or checked online.

## Test evidence

**Current code:** the offline `uv run` test execution collected 523 cases and ran **498 passed, 25 deselected**, in 17.18 seconds. All **341 reward tests** ran and passed. Bytecode and pytest cache writes were disabled; temporary reports and test folders were outside the repository. A temporary guard rejected protected-file reads, repository writes, live credential/config reads, and network connections/binds. It recorded zero denials. The inspected reward probes use `-I -B` and an empty environment, load temporary source copies, and do not access live services or the real source files for writing.

The 25 excluded cases were:

- Six functions in `tests/test_prepublish_check.py`: `test_real_cards_pass`, `test_wrong_score_in_a_table_is_caught`, `test_unknown_row_in_a_score_table_is_caught`, `test_wrong_count_is_caught`, `test_untraced_number_in_prose_is_caught`, and `test_every_fingerprint_in_the_cards_is_known`.
- Seven parameter cases of `tests/test_score.py::test_m1_saved_answers_rescore_to_the_same_numbers`.
- Twelve cases in `tests/test_run_ollama.py` that use the `setup` fixture, which starts a local socket server. These were excluded to preserve the no-network boundary.

I did not run the unrestricted suite and do not claim to have independently reproduced all 523 passes.

**Old code:** I ran today's entire reward test file against `61616b6` in a temporary source tree: **142 failed, 199 passed**. Relaxing only the expected error-line number gave **113 failed, 228 passed**. The difference is exactly the documented 29: 25 new binding-before-compile cases and four changed older expectations. Of the remaining 113 failures, 110 returned successfully when they should have stopped, and three failed to move the fingerprint for a regular `__pycache__` file. This confirms the note's split between behavior failures and line-number failures.

**Deliberately broken copies:** all eight new cases that also pass on the old code failed against a suitable broken current copy. Removing `sys` permissions or the `re.compile` exception broke all eight because today's copied code itself was rejected. Counting attribute/call keywords as bindings, counting class-pattern keywords as bindings, counting the `from` module as a binding, or rejecting every star import each broke exactly its intended allow-list case, with the other seven passing. Counting string contents as names broke the strings/comments test. Removing the cache-folder exclusion broke the non-`.pyc` cache test, with the other three still passing. These checks support the fixes note's claims without treating tests that pass on old code as regression failures.

The four changed expected lines are correct: the stricter rule now points to the binding rather than the later `re.compile`. The `PROBE` change from real-path comparison to absolute-path comparison is also correct: linked test layouts now reach the fingerprint's own rejection instead of being rejected by the helper first. No changed test writes to the real `src/judge`, starts a live job, or uses the network.

Temporary scripts, logs, source copies and test reports for this review are in `/private/tmp/m2-04h-review-0713ec2-mdhraudm`. They are local review evidence, not changes proposed for the repository.

## Saved-pool recheck and unchanged files

I extracted and ran the Python script printed in `results/m2/pool/reward-recheck.md`, unchanged, from the repository folder. It completed with these results:

| Check | Verified result |
|---|---|
| Recorded reward fingerprint and old method on today's rules | Both `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c` |
| Current reward fingerprint | `769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7` |
| Recorded prompt matches today's | Yes |
| Show seed, keep seed, answers per pair | `2026102704`, `2026102705`, `8`; match today's constants |
| Candidate-file fingerprints match | Yes |
| Sample rows, different IDs, candidates | `1,018`, `1,018`, `1,018` |
| Answers checked | `8,144` |
| Stored reward lists, counts and outcomes | `323`, zero differences |
| Pool SHA-256 | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` |

I independently compared these files' bytes with `61616b6`; all are unchanged:

| File | Current SHA-256 |
|---|---|
| `results/m2/pool/v1-samples.jsonl` | `04739ad04cfb061aabdd1023f6d639784db033fdc5b8642482e63d28c0fcd88d` |
| `results/m2/pool/v1-samples.meta.json` | `fbfd0db356c63faaf183de2b849e487dd390c35e971986886f204cc376ae2dc4` |
| `data/m2/rl-pool.jsonl` | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` |
| `data/m2/rl-pool-always-wrong.jsonl` | `a923b0391dcc0982777fbc77f735828a9daa01791d38794c9284e38808ea27da` |
| `results/m2/pool/summary.md` | `674d69932f4bc58b89817451896f082035a01f473b936f835f743e301390e071` |

Only `src/judge/reward.py` changed under `src/`. Comparing parsed code, with line locations ignored, confirmed that `REWARD`, `read_answer`, `reward`, `completion_text` and `GRPOReward` are unchanged. The current fingerprint covers exactly the documented 11 files. The review-range whitespace check also passed.

## Done and next action

1. **Codex: finished.** Review and permitted offline checks are complete. Only `/Users/frank/code/model-training/reviews/M2-04h-codex-review.md` was added. No implementation repair, control-file update, commit or push was made because the review prompt prohibits them.
2. **Elias: next.** In Claude Desktop, continue the Cowork planning chat in **Fine Tuning and RL training**. Give it this report's absolute path, the SHA-256 supplied with the final review reply, and verdict **ACCEPT**, for branch `m2-04-pool-and-reward` at `0713ec2`. No new decision is needed to close this review.
3. **Claude in Cowork:** verify the report and its hash, carry the known limit and test-exclusion correction into the project records, and complete the existing check/merge process. Then prepare Step 5's training-code review under the existing plan. Acceptance of this fingerprint review grants no new GPU spending, provider, publishing or deployment authority. Preserve the saved samples, pool and frozen-file boundaries. This report has not been sent to another app, and no receiver pickup is claimed.
