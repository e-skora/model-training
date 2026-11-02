CHANGES REQUIRED

Reviewed by Codex on 2026-10-30 in `/Users/frank/code/model-training`.
Branch: `m2-04-pool-and-reward`.
Commit: `0d3905589107011a195a95b1b23bd5142c79e353`.
Review range: `4b2d9d69ac67fa711b882a675583f33c6f6f3216..0d3905589107011a195a95b1b23bd5142c79e353`.

The two earlier findings are fixed for the requested examples. Whole-file fingerprints now see default values, captured values, imported helpers and the training wrapper. The saved reward values and pool remain correct. However, three ordinary ways of reaching more code still escape the promised checks. A fingerprint is an identifying code calculated from file contents with SHA-256. In each new reproduction, different source files produce different points but the same fingerprint.

## Findings

### 1. Must fix, P2: `runpy` loads project code without being followed or stopped

**File:** `src/judge/reward.py:39`, `93-96`, `99-113`.

The stop list catches `importlib`, `exec` and the other named loaders, but misses `runpy`, another Python code loader. In a temporary copy, I changed the answer reader to use:

```python
import runpy
read_fields = runpy.run_module("judge.policy")["read_fields"]
```

The JSON reader then calls `read_fields`. Two copies of `judge/policy.py` differ only in whether repeated fields are rejected. A repeated-label answer earns **-2.0** in one fresh process and **+1.0** in the other. Both return fingerprint:

`005758cd2c9f6f5128aa1d1a19f3d6f8ef7f79c111633a4589bf8d88bc0c9656`

The fingerprint succeeds, lists only the usual five files and leaves out `judge/policy.py`. I also reproduced this with `runpy.run_path`. No name was hidden or assembled, and no loaded function was replaced in memory. These are Python source-code dependencies, rather than the documented exclusion for data or settings.

**Why it matters:** the documented rule says plain code loaders that cannot be followed stop with an error. This one silently accepts an incomplete reward identity.

**Fix:** reject plain `runpy` imports and uses, including `from runpy import run_module` and `run_path`, or explicitly follow their supported source targets. Add fresh-process tests that change the loaded helper and require a changed fingerprint or a clear error. Keep the stated coverage aligned with the supported loaders.

### 2. Must fix, P2: neighboring project packages without `__init__.py` bypass the stop

**File:** `src/judge/reward.py:76-82`, `93-96`, `119-121`.

The check for project code beside `judge` asks whether `module_file(top)` finds a file. It misses a neighboring directory that Python can import without an `__init__.py`. This is called a namespace package.

In a temporary source folder I created `policy/reader.py` beside `judge`, with no `policy/__init__.py`. The copied scorer uses:

```python
from policy.reader import read_fields
```

Changing only `policy/reader.py` from rejecting repeated fields to allowing them changes both raw and wrapped rewards from **-2.0 to +1.0**. Both fresh processes successfully return:

`be07b32d901a1b9f2dd320022902ce8ae1e334b37aa336a1d727f153add7feee`

The helper is absent from the covered files. This is local project code, not an outside installed package.

**Why it matters:** an ordinary import bypasses the explicit promise to stop on project code beside `judge`.

**Fix:** also detect local namespace-package directories and dotted source targets. Reject these imports under the chosen boundary, or cover their source files. Test both `from policy.reader import read_fields` and `import policy.reader` with a neighboring directory lacking `__init__.py`.

### 3. Must fix, P2: the check for an already loaded helper stops after one package level

**File:** `src/judge/reward.py:103-106`. Related test: `tests/test_reward.py:286-293`.

The special check recognizes `judge.policy`, but for `judge.sub.policy.read_fields` it only adds `judge.sub`. It never follows the full dotted module path.

I reproduced this in fresh temporary processes. Each process first imports `judge.sub.policy` normally. The scorer then uses `import judge.prompt` and calls `judge.sub.policy.read_fields(items)`. Changing only the source of `judge/sub/policy.py` changes both raw and wrapped rewards from **-2.0 to +1.0**, while both fingerprints remain:

`295978ee21a4968c9c8bbc541abf694c75bbd7c4c192d8d0b0cf3779b63a2f6c`

The covered list includes `judge/sub/__init__.py`, but omits `judge/sub/policy.py`. The process uses normal imports, with no runtime patching or edits during execution. The missing helper is called inside the covered scorer; it is not an outside wrapper changing the returned rewards.

**Why it matters:** this is the same already-loaded-module case the new code and test expressly support, extended into a subfolder. Its result can depend on what the caller imported first.

**Fix:** follow complete dotted module references, including supported aliases, or stop clearly on references that cannot be resolved. Add an execution test that preloads the nested helper and proves both points and fingerprint change. The existing test checks only that the fingerprint moves; it never connects its new `read_fields` function to the JSON reader, so it does not test actual use of that helper.

### 4. Nice to have, P3: the old-code failure explanation counts two different kinds of failure together

**File:** `reviews/M2-04-fixes.md:34`.

I ran today's complete reward test file against a temporary copy with the old `4b2d9d6` reward module. The result is indeed **46 failed, 79 passed**. However, **44** fail because the fingerprint misses a source change or fails to reject unsupported code. The other two fail for different reasons:

- `test_fingerprint_reads_files_not_memory` deliberately expects the new disk-only behavior, which the old implementation does not have.
- `test_fingerprint_covers_these_files_today` calls the new `fingerprinted_files` function, which does not exist in the old implementation.

**Why it matters:** the note says all 46 fail for the behavioral reasons it lists. The total is right, but that explanation is too strong.

**Fix:** report 44 regression checks plus the two behavior/API checks separately. The 14 rewritten older cases pass on both implementations, as claimed. The folder-stability test also passes on both. I separately made a scratch implementation include absolute folder paths in its hash and confirmed that this stability test fails there.

## Checks that hold

- The `STRICT` default and captured `allow` examples now change both points and fingerprint. All six explicit module-import forms pass, including aliases, relative imports and a subfolder.
- Doubling `GRPOReward.__call__` changes its returned points from +1.0 to +2.0 and changes the fingerprint, while the raw reward stays +1.0.
- Imports inside functions, transitive imports and a relative import two levels up are covered in the tested forms. A separate two-module import cycle terminates successfully. The visited-file check prevents repeated traversal.
- The current fingerprint is stable across copied folders and a different working folder. Paths inside the hash are relative and the final mapping is sorted. This verifies the tested local cases, not every computer or Python environment.
- `judge/__init__.py` correctly belongs in the list because it runs when the package is imported.
- The current tests confirm clear stops for the listed `importlib`, `builtins`, `__import__`, `exec`, `eval`, bare `import judge`, star-import, missing-module, escaping-relative-import and ordinary neighboring-file cases. No listed stop blocks today's reward code. The three gaps above remain.
- The newer STATE entry and fix-note correction clearly replace the old name-walk claims. The exclusions for Python, outside packages, data, settings, runtime changes and outside wrappers are substantially clearer. The statements that unsupported plain loaders and neighboring code always stop need the fixes above. The disk-only limitation is explicitly acknowledged.
- The new tests edit temporary copies rather than the real source files. Their child processes use isolated Python with an empty environment and disabled bytecode writes. The memory-limit test temporarily changes the live test process's reward dictionary and pytest restores it; it does not edit a source file. These tests do not contact a live service.

## Offline saved-pool recheck

I executed the unchanged Python block directly from `results/m2/pool/reward-recheck.md`. An audit guard blocked protected-file access and network operations. The recheck triggered no blocked attempts.

| Check | Verified result |
|---|---|
| Recorded sampling fingerprint and old method on today's code | Both `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c` |
| Current fingerprint | `4d64ca4c54a353e3cb60522cfd49ed5dd38a14ade5ec55f8e87a98ee15f8a42a` |
| Recorded prompt equals current prompt | True |
| Show seed, keep seed, answers per pair | 2026102704, 2026102705, 8 |
| Candidate-file fingerprints match | True |
| Sample rows, distinct IDs, candidates | 1,018 each; exact-ID check passes |
| Answers rescored | 8,144 |
| Stored reward lists, answer counts and outcomes | 323 checked, zero differences |
| Pool SHA-256 | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` |

The current covered list is exactly `judge/__init__.py`, `judge/frozen.py`, `judge/prompt.py`, `judge/reward.py` and `judge/score.py`.

I compared the bytes of all five named saved artifacts against `4b2d9d6`: samples, sample metadata, pool, always-wrong rows and summary are identical. The reward table, answer-kind list, raw reward, answer reader and completion reader also have unchanged Python definitions. The diff contains only the eight expected source, test and documentation files. Neither frozen test file was opened or freshly hashed.

## Test results and limits

The complete suite collects **307 tests**. Running every test would violate this review's protected-data and no-network limits. I ran `uv run pytest` offline with dependency synchronization, bytecode and repository pytest-cache writes disabled, plus a temporary audit plugin.

- Reward and pool tests: **144 passed in 3.76 seconds**.
- Final broader permitted run: **280 passed, 27 deselected in 5.00 seconds**, with no blocked access attempts.
- Exclusions: seven saved-answer rescores that read the real frozen test file; six publication checks that read real test/validation data; two real-validation checks; twelve fake-Ollama tests that bind local sockets.
- An initial guarded run missed the six publication exclusions and reported **280 passed, 21 deselected, 2 failed and 4 setup errors**. Those failures were the guard refusing protected-file access before the files opened. After adding those exclusions, the final permitted run passed. These are review restrictions, not six demonstrated product bugs.
- Old-code comparison and the deliberately broken absolute-path check were run only in temporary copies. Their expected failures are described above.

No GPU or Modal job ran, no network was used, no branch was switched, and no implementation or saved-result file was edited. Only this report was written in the checkout. I did not update STATE or commit/push because the specific review prompt prohibits those actions.

## Done and next action

Done: reviewed the requested revision, retried the prior examples, reproduced three new gaps, checked the saved artifacts and ran the permitted tests. The review is finished; implementation acceptance remains blocked by findings 1 to 3. No new GPU sampling is needed for these fixes.

Elias's next steps:

1. In Claude Desktop, continue the Cowork project **Fine Tuning and RL training** for `/Users/frank/code/model-training`.
2. Give it this report: `/Users/frank/code/model-training/reviews/M2-04d-codex-review.md`, verdict **CHANGES REQUIRED** at commit `0d3905589107011a195a95b1b23bd5142c79e353`.
3. Ask Claude in Cowork to route the three source-tracking fixes and their tests to Claude Code's desktop Code panel in that same folder and branch, then request another Codex review. Keep reward values and saved artifacts unchanged; no GPU, network or protected-test access is needed for this correction pass. Step 5 training is not approved by this review.
