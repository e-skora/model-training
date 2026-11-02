CHANGES REQUIRED

Reviewed by Codex on 2026-10-29. Folder: `/Users/frank/code/model-training`. Branch: `m2-04-pool-and-reward`. Commit: `becefcd9e707fef8339123e4760a49a6ee20b3b8`. Review range: `a28db37..becefcd`.

The prompt and sample-ID checks are fixed. The original fingerprint reproductions are fixed too, but the wider fingerprint guarantee still has two gaps. A fingerprint is an identifying code calculated with SHA-256; it must change when the rules it identifies change. The two findings below concern that safeguard. The current reward values and saved pool remain correct.

## Findings

### 1. Must fix, P2: a new helper can still change points without changing the fingerprint

**File:** `src/judge/reward.py:95-107`, especially `100-107`. Also `describe` at `72-78`.

The new code follows names inside function bodies and records their source text. It does not record default argument values or settings captured outside a helper's body. It also skips every module, including a module containing the project's own helper rules. These are silent omissions, rather than a clear error saying a rule cannot be fingerprinted.

**Reproduced:** replace the repeated-field reader in memory with this ordinary helper, loaded from source held in memory:

```python
STRICT = True

def read_fields(items, strict=STRICT):
    if strict and len(items) != len(dict(items)):
        raise ValueError("repeated field")
    return dict(items)
```

Then load a second version with only `STRICT = False` changed. An answer with two `label` fields changes from **-2.0 to +1.0**, but both versions give the same fingerprint:

`9021ec3c7b06e413fb21dec4c1372dd71e3898185c9d2d2b9025c4824e66deb7`

The function body sees `strict` as an argument. The walker never visits the `STRICT` value used to set its default. I also reproduced unchanged fingerprints while changing points with:

- A helper that captures an `allow` flag from its enclosing function. This is called a closure: a function keeps a value from the place where it was created.
- A reader calling `policy.read_fields(items)`, where `policy` is another module. Changing that module's reader from rejecting repeated fields to allowing them changes points, while line 106 skips the module entirely.

**Why it matters:** the fix and its documentation promise that a newly added helper is covered automatically. These normal helper forms break that promise and can let different reader rules share one recorded reward identity. These were controlled probes of the requested future-helper cases, not evidence that today's saved pool used such a helper.

**Fix:** make the supported boundary explicit. Either include the relevant project source files and rule values, or extend the walker to include default values, captured values and project-module dependencies. If a construct cannot be covered reliably, stop with an error instead of silently skipping it. Standard-library and outside-package versions can remain a separate environment record. Add regression tests for the default-setting example, a captured setting and a project helper reached through a module. Update the broad claims in `GLOSSARY.md`, `STATE.md` and `reviews/M2-04-fixes.md` to match what is actually checked.

### 2. Must fix, P2: the actual training reward wrapper is outside the fingerprint

**File:** `src/judge/reward.py:120`. The omitted scoring code is `GRPOReward.__call__` at `148-163`.

The fingerprint starts from `reward` and `completion_text`. It never includes `GRPOReward.__call__`, which chooses how the returned training rewards are built from those functions.

**Reproduced:** keep the table, raw-answer reward and completion reader unchanged, but replace the wrapper's call method in memory with one that doubles its returned values. A correct contradiction changes from **+1.0 to +2.0** through `GRPOReward`. The fingerprint remains the current value, `92a043f1...c3a53`.

**Why it matters:** Step 5 calls this wrapper, so a change to its reward calculation or pairing of answers and right labels is a real training-rule change. Covering the text reader alone does not identify the complete reward used during training.

**Fix:** include the wrapper's scoring method and its relevant dependencies in the fingerprint, or include the whole reward source file under a clearly defined source-file policy. Add a regression test where the raw reward and text reader stay unchanged but the wrapper's returned points change. This does not call for changing today's reward values.

## Checks of the three original findings

| Original finding | Result at this commit |
|---|---|
| Reader helpers and constants omitted from the fingerprint | The original cases now work. Removing `update`, allowing repeated keys, allowing `NaN`, changing the fence pattern, and changing accepted labels each changed both points and the fingerprint. `completion_text` changes are covered. The two wider gaps above remain. |
| Pool selection accepts another prompt or settings | Fixed. A different prompt fingerprint, show seed, keep seed or answers-per-pair value stops selection before outputs are created. |
| Repeated IDs can hide missing candidates | Fixed. Full runs must contain every candidate exactly once. Smoke runs, which are small trial runs, must contain exactly the intended first 16 candidates. |

I separately tried repeated, missing and unexpected IDs in both full and smoke runs, plus each of the four changed metadata values. All **10** bad-input cases stopped before an output directory was written. The new tests also check this boundary.

The walker terminates for directly or mutually recursive functions: its `seen` set prevents visiting the same function again. It also examines nested code, such as list comprehensions. It runs successfully on all dependencies used by the current reward code, including the built-in `dict` replacement from the original reproduction. An unsupported plain object raises `TypeError` under `describe`, as designed. These checks do not resolve the silent omissions in finding 1.

## Documented offline recheck

I executed the Python block in `results/m2/pool/reward-recheck.md` directly from that document, without creating `recheck.py`. An in-memory guard prohibited access to both real test files, both validation files, and network sockets. No prohibited access occurred.

Its printed results match the document:

- Saved reward fingerprint: `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c`.
- Old fingerprint method on today's code: the same `a369e730...b0c70c`.
- New fingerprint method: `92a043f18ac483f60aac7036375b3d98abfde4e6c82191d8303af7f70a8c3a53`.
- Saved prompt matches today's prompt. Show seed, keep seed and answers per pair are **2026102704, 2026102705, 8**.
- Candidate-file fingerprints match. There are **1,018 sample rows, 1,018 different IDs and 1,018 candidates**.
- **8,144 answers** were rescored. All **323** stored reward lists, answer counts and outcomes match: 314 pool rows plus 9 always-wrong rows.
- Pool SHA-256 remains `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`.

Keeping the old fingerprint in the original saved metadata is correct: it preserves the record of how the original run was identified. The separate recheck connects that record to the revised method without rewriting history. After the remaining fingerprint fixes, record the final fingerprint and repeat this offline check before Step 5. A further fingerprint change alone does not require another GPU run.

The stricter `choose_pool` deliberately rejects the old saved metadata because its reward fingerprint differs. The documented standalone recheck is the explicit bridge for these existing files; it does not weaken that normal selection check.

## Scope and unchanged files

I reviewed every change in `git diff a28db37..HEAD`, including the fix notes, recheck, tests and control-file updates. Git confirms no difference between the two commits in:

- The raw sample file and its metadata.
- The pool and always-wrong files.
- The pool summary.
- The shared scorer, pool rules and generation code.
- Either frozen test file.

The reward table and the original raw-answer reward function did not change. The old-method fingerprint and the rescoring also agree with that conclusion. I did not open either frozen test file or calculate a fresh hash of its bytes.

## Tests

The directly relevant tests pass:

```text
PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/m2-04b-uv-cache \
uv run --offline --no-sync pytest -p no:cacheprovider tests/test_reward.py tests/test_pool.py
90 passed in 0.54s
```

The broader permitted run reports **228 passed, 25 deselected in 2.57s**, out of 253 collected tests. It used pytest through `uv run --offline --no-sync python`, an in-memory collection filter, no bytecode or pytest-cache writes, and guards against protected-file reads and network socket use.

The 25 excluded cases are:

- The same 13 frozen-data cases identified in `reviews/M2-04-codex-review.md`: seven saved-M1-answer checks and six prepublication checks.
- Twelve `tests/test_run_ollama.py` cases using its `setup` fixture, which starts a local HTTP server. They were excluded before execution to honor this review's no-network rule. The previous review also found that this sandbox blocks their socket binding.

I therefore do not claim a fresh full **253-test pass**. The fix note reports that result from Claude Code's Mac run; it was not reproduced here. `git diff --check a28db37..HEAD` passes. The additional fingerprint and bad-input probes above were separate checks, not added to the repository's test count.

## Done and next action

1. **Codex: finished.** Wrote `/Users/frank/code/model-training/reviews/M2-04b-codex-review.md`. The review left source files, datasets, saved results and control files unchanged. No branch switch, commit, push, GPU job, Modal job or network use occurred. Only this report was written in the checkout; test fixtures used temporary folders.
2. **Elias: next.** In Claude Desktop, continue the Cowork project **Fine Tuning and RL training** and provide this report's absolute path. Ask it to route the two remaining fingerprint fixes to Claude Code's desktop Code panel in `/Users/frank/code/model-training`, branch `m2-04-pool-and-reward`.
3. **Claude Code, then Codex:** add the missing coverage or clear unsupported-case stops, add the regression tests, repeat the offline recheck with the final fingerprint, and obtain another review before Step 5. Preserve the saved samples and pool. These fixes need no GPU run, frozen-test access or reward-value change. Cowork retains merge responsibility. This report was not sent to another app or chat.
