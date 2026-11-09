# M2-05d: fix after Codex's narrow review

Codex's narrow review `reviews/M2-05d-codex-review.md` (at `cfd104d`, 2026-11-09, scope D-058) said CHANGES REQUIRED with one must-fix finding. The padding and billing fixes passed. Claude Code fixed the finding on branch `m2-05-grpo` on 2026-11-09. No GPU or Modal job ran. No file under `src/`, no saved pool file and no saved trial result changed.

**Tests:** 650 pass (649 before). `tests/test_train_grpo.py` has 127 cases.

## The finding (must fix): a writer right after the repo was made could become the upload's starting point

Another writer could add a file after `create_repo` and before the upload's first look at the new repo. The script then took that writer's commit as the repo's first commit and as the upload's parent, and could overwrite the file.

**Fix.** In `upload_checked`, right after making the repo, the script looks at the newest commit and fingerprints every file there. If it holds anything but the `.gitattributes` Hugging Face adds, it stops before saving that commit in its note. It sends nothing. The same commit is then the one the pre-send check and `parent_commit` use. Because the note never names that commit, a retry cannot resume into the repo either.

**Test:** `test_a_writer_right_after_the_repo_is_made_is_never_overwritten`. The fake Hugging Face adds another writer's `adapter_config.json` right after the repo is made. The upload stops, sends nothing, leaves the file as it was, and saves no first commit; a retry with that note stops too.

**Proof the test tests the real thing:** with the new check turned off, the test fails. The script was then restored byte for byte (SHA-256 checked).

**Known limit.** A writer who makes the repo's first commit look empty, for example by changing only `.gitattributes`, is not stopped here. That writer's later files are still protected by the pre-send check and `parent_commit`, and the script never uploads `.gitattributes`, so nothing of theirs is overwritten.
