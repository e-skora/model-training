CHANGES REQUIRED

Reviewed by Codex on 2026-11-01 in `/Users/frank/code/model-training`.
Branch: `m2-04-pool-and-reward`.
Commit: `61616b64a3f39707004d016a28e66c89b3082db7`.
Review range: `dd96f9c..61616b6`.

The three previous must-fix findings are fixed for the requested examples. The saved answers, reward rules and pool remain correct. Three gaps remain in the new stops. Two allow a source file to change the points without changing the fingerprint, which is the SHA-256 code used to identify the reward files. The third breaks Elias's rule that only Python's own `re.compile` may pass. There is also one small mismatch in the skipped-file rule.

All reproductions used temporary folders and fresh Python processes. No implementation file, saved result, branch or commit was changed. No GPU, Modal job or network connection was started. Neither frozen test file was opened. This report is the only repository file written.

## Findings

### 1. Must fix, P2: an ordinary alias bypasses the `sys.path` stop

**File:** `src/judge/reward.py:104`, `111-112`. Related claim: `reviews/M2-04-fixes.md:67`.

The check recognizes `sys` and names introduced by `import sys as name`. It does not recognize a name assigned to `sys` afterward. In two temporary copies, `judge/__init__.py` contains the same code:

```python
import sys
from pathlib import Path
runtime = sys
runtime.path.append(str(Path(__file__).resolve().parents[2] / "plugins"))
```

`sys.path` is Python's list of folders to search for imports. This code changes that list through the name `runtime`, but the fingerprint succeeds. The copied scorer uses `from policy import read_fields`. The helper lives at `plugins/policy.py`, outside `src/`. Stop 2 sees the directory name `plugins`, but the import names `policy`, so it does not stop this either.

The two helper files differ only in whether they reject repeated JSON fields. One raises `ValueError` on repeated fields and then returns `dict(items)`; the other simply returns `dict(items)`. On the same repeated-label answer, the raw and wrapped points change from **-2.0 to +1.0**. Both copies return:

`c10cc6564c1a707d0f403b4ea262b3b0701fbed01473d77e7df219e8a51db3b5`

No code name was hidden, no reward function was patched in memory, and no source file changed during execution. This directly violates the promised stop on changes to `sys.path`.

**Fix:** recognize simple assignments that give `sys` another name, or reject uses the check cannot safely identify. Add a fresh-process test using this exact alias and an outside helper. Require a clear file-and-line error before returning a fingerprint.

**Related boundary:** ordinary package search-folder changes are also unchecked. With `__path__.append(...)` in `judge/__init__.py` and `from judge.policy import read_fields` in the scorer, the same outside-helper experiment gives -2.0 and +1.0 with identical fingerprint `86586ae174d90d1a17e9bf158cefe9d7e351ef4fdae3ad68bc0365a575322ebc`. `__path__` is a package's own list of import folders. Address this when stating the outside-code boundary: stop unsupported package search-folder changes, or explicitly declare that limitation. The current notes do not identify it.

### 2. Must fix, P2: resolving the source path first hides a linked `judge` folder

**File:** `src/judge/reward.py:37-38`, `135-144`. Related test: `tests/test_reward.py:551-558`.

`Path(__file__).resolve()` follows links before the source folder is chosen. If `src/judge` itself is a link, the walk starts at its destination. It never sees that link. If the destination's parent is also named `src`, the folder-name check passes too.

I created this layout in temporary folders:

```text
external/src/judge/       copied package, scorer imports policy
strict/src/judge         link to external/src/judge
strict/src/policy.py     rejects repeated fields
loose/src/judge          link to external/src/judge
loose/src/policy.py      allows repeated fields
```

Each fresh process imports `judge` through its own `strict/src` or `loose/src` search folder. Python finds that folder's `policy.py`. The fingerprint instead reads `external/src` and silently omits the helper. Raw and wrapped points change from **-2.0 to +1.0**, while both fingerprints are:

`e1c6c30d605141681d244701b518e4a07948a4b82b225a66ee2319e38a43110c`

**Why it matters:** D-051 specifically promises that a link under `src/` stops. A linked package is a normal folder layout, and here it makes the fingerprint read the wrong source tree.

**Fix:** check the original package path and its components for links before resolving them. Reject a linked `judge` directory or reward file with a clear error. Test a destination whose parent is also named `src`.

The existing test helper checks that the real imported package path starts inside the temporary source folder. That assertion rejects this layout before the fingerprint runs. I used a fresh-process probe without that test-helper assertion to check the production behavior. The new regression test must reach the fingerprint's own check.

### 3. Must fix, P2: two ordinary bindings let another object's `compile` pass

**File:** `src/judge/reward.py:85-103`, `109-110`. Related claim: `reviews/M2-04-fixes.md:69`.

The exception for `re.compile` tries to ensure `re` still names Python's text-pattern module. It misses names bound by pattern matching and names supplied by `from ... import *`.

This covered file imports and executes successfully, and its fingerprint succeeds:

```python
import re

class Worker:
    def compile(self, text):
        return "not a regex"

match Worker():
    case re:
        RESULT = re.compile("x")
```

Here `case re` gives the name `re` to the `Worker` object. Its `compile` method runs instead of Python's `re.compile`. The code reader stores this binding differently from an ordinary assignment, so `gives_re_another_meaning()` misses it.

I also confirmed the same behavior with `import re; from judge.other import *`, where `other.py` exports `re = Worker()`. The star import replaces `re`, but the checker still allows `re.compile`.

**Why it matters:** the exception is broader than D-050 permits. These examples do not themselves evade hashing of files under `src/`; they prove the promised stop is missing.

**Fix:** account for pattern-bound names and treat star imports as an unknown binding when deciding whether to allow `re.compile`. Require an unaliased `import re`. Add executing tests for both examples and require a file-and-line error.

### 4. Nice to have, P3: a regular file named `__pycache__` is also skipped

**File:** `src/judge/reward.py:140-142`. Related claim: `reviews/M2-04-fixes.md:88`.

The rule says to skip `__pycache__` folders and `.pyc` files. The implementation skips any path containing the component `__pycache__`, even when that component is a regular file. Adding an ordinary file at `src/__pycache__` leaves the fingerprint unchanged. It is neither a cache folder nor a `.pyc` file.

**Fix:** skip a `__pycache__` entry itself only when it is a directory, while continuing to skip that directory's contents. Add a test for a regular file with that name. This is a completeness mismatch; today's 11-file source tree has no such file.

## Checks that hold

- **Previous finding 1:** all three requested `runpy` forms now stop with a file-and-line error: `run_module`, `run_path`, and `from runpy import run_module`.
- **Previous finding 2:** both imports of a neighboring folder without `__init__.py` now change the fingerprint when its helper changes. The tests connect the helper to the answer reader and prove the points change too.
- **Previous finding 3:** the preloaded nested `judge.sub.policy` case now changes both points and fingerprint.
- **Previous nice-to-have:** the added note correctly separates the earlier 44 regression failures from the two behavior/API checks.
- The direct loader forms, direct `sys.path` forms, and all 18 outside-import cases in the new tests pass. Today's source is not wrongly blocked by either stop. Ordinary links found by the walk and a package parent not named `src` are rejected.
- For ordinary regular files, paths are sorted and use `/`, and each file's path and byte length separate it from the next file. Moving the source tree does not change its fingerprint. Different file boundaries and paths are tested. I found no ambiguity in this encoding; the failures above omit files rather than break SHA-256.
- The current covered list contains exactly the documented 11 source files. The newer STATE, glossary, fixes note and function description consistently describe the folder boundary and the stated exclusions. Later entries explicitly replace the historical fingerprints. Their guarantees about the stops and skipped files need the corrections above. The documented exclusions for outside packages, outside data/settings, runtime reward patches, outside wrappers, `.pyc` files and unlisted loaders remain real limits.
- Removing tests tied only to import tracing is reasonable under the new folder boundary. That does not cover the new search-path and linked-package cases above.

## Saved-pool recheck and unchanged files

I ran the Python script printed in `results/m2/pool/reward-recheck.md`, unchanged, from the repository folder. It reports:

| Check | Verified result |
|---|---|
| Recorded fingerprint and old method on today's reward rules | Both `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c` |
| Current fingerprint | `1ffce9909b4cef446921fd0ecd0185eebef4b881b1d3c1fb39648edad42c2381` |
| Recorded prompt matches today's | Yes |
| Show seed, keep seed, answers per pair | `2026102704`, `2026102705`, `8` |
| Candidate-file fingerprints match | Yes |
| Sample rows, unique IDs, candidates | `1,018`, `1,018`, `1,018` |
| Answers checked | `8,144` |
| Stored reward lists, counts and outcomes | `323`, zero differences |
| Pool SHA-256 | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` |

I compared bytes with `dd96f9c`: `v1-samples.jsonl`, its metadata, `rl-pool.jsonl`, `rl-pool-always-wrong.jsonl` and the pool summary are all unchanged. The only changed file under `src/` is `judge/reward.py`. Its reward table, `read_answer`, `reward`, `completion_text` and `GRPOReward` have unchanged executable code.

## Test evidence and limits

- **Current code:** offline `uv run pytest` with a temporary guard, no bytecode/cache writes, and a temporary test-results file: **352 passed, 25 deselected**, in 12.38 seconds. All **195 reward tests** are included and pass.
- The 25 exclusions preserve this review's limits: 7 tests rescore the real frozen M1 test set, 6 publishing-check tests read that set, and 12 Ollama tests start a local socket server. The guard rejects protected-file reads, repository writes, live credential-config reads and socket connections/binds. **I did not rerun all 377 tests and do not claim that total passed in this review.**
- **Old code:** today's complete reward test file against `0d39055` in a temporary source tree gives **96 failed, 99 passed**, matching the note. Relaxing only the requirement for a line number gives **74 failed, 121 passed**. The 22 newly passing cases are exactly the loader tests described in the note.
- **Deliberately broken temporary copies:** removing the cache skip makes all 4 cache tests fail; including absolute source paths makes the location-stability test fail; searching raw text for a forbidden path operation makes the strings/comments test fail. For the last experiment, all 9 reward-value checks still pass, so the failure is specific to the false rejection. These checks support the note's claims about the three test groups.
- The new fingerprint tests edit temporary source copies and use fresh Python processes with no inherited environment. They do not edit the real `src/judge` files or use the network. The one in-process reward-table patch is restored by the test framework and expressly checks the documented disk-only limitation.

## Done and next action

1. **Codex: done.** Review and offline checks are finished. Only `/Users/frank/code/model-training/reviews/M2-04f-codex-review.md` was added. No commit or push was made because this review explicitly forbids both.
2. **Elias: next.** In Claude Desktop, open the Cowork project **Fine Tuning and RL training** and continue the planning chat. Give it this report's absolute path and the verdict **CHANGES REQUIRED**, tied to branch `m2-04-pool-and-reward` at `61616b6`.
3. **Claude in Cowork:** route the three must-fix items to Claude Code's desktop Code panel in `/Users/frank/code/model-training`, then request another Codex review. Preserve the saved samples and pool; no GPU run, network use, frozen-test access or reward-value change is needed to fix these findings. Step 5 remains unapproved. This report has not been sent to another app, and no receiver pickup is claimed.
