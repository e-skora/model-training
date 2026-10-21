# AGENTS.md: rules for every coding agent in this repo

This repo is where Elias learns to fine-tune and train open AI models, hands-on. Claude Code (desktop Code panel) builds. Codex reviews. Claude in the Cowork project "Fine Tuning and RL training" plans, checks and merges.

## Read first, in this order
1. `STATE.md` (where things stand, the next action)
2. The active milestone's `SPEC.md` (named in STATE.md)
3. `DECISIONS.md` (what is already decided; do not re-propose what it rules out)
4. `LESSONS.md` (mistakes not to repeat)

## Rules
1. **Plain language.** Every report to Elias, and every doc, is written so a child could follow it. Define each technical term the first time, and add new terms to `GLOSSARY.md`.
2. **Say who does what.** Every report ends by listing what is done and what, if anything, is Elias's to do, step by step, with the exact folder and app.
3. **No em dashes** in public files: README, model cards, dataset cards, posts.
4. **No secrets in git, ever.** No tokens, keys, `.env` files or config values. Tokens live in Modal secrets or the local Hugging Face login, never in the repo.
5. **No real personal data.** All training and test data is made up, about fictional people.
6. **Aeverum is off limits.** Never copy text, data or code from `~/code/Aeverum-Auditer`, `~/code/Aeverum` or `~/GPT-Projects`. Above all, never use Aeverum's corpus, benchmark or holdout pairs (the Mara Lindqvist data) for training or testing here. Aeverum scores our model on its own benchmark later, at its step 12.
7. **Check before you claim.** Any statement about how Modal, Unsloth, Hugging Face or another outside tool currently works (commands, prices, settings, screens) is checked against that tool's current docs first, with the doc URL written in the report. If it cannot be checked, say so.
8. **Keep GPU cost small.** Default GPU is an L4. Every Modal function sets a timeout. Every GPU run gets one line in `runs/RUNLOG.md`: date, what ran, GPU, minutes, estimated cost.
9. **Git.** `main` always works. Each task happens on its own short branch named after what it does, with small commits. Never mix unrelated changes.
10. **Tests never touch live things.** Tests use temporary folders and fake data. They never write to the real Hugging Face account, never start paid GPU jobs, and never read your real home folder.
11. **Freeze the test set.** Once a milestone's test set is written and checked, it is never edited and never used for training. Its file fingerprint (SHA-256) is recorded in `STATE.md`.
12. **Close every task.** Before finishing: update `STATE.md`, add any lesson or mistake to `LESSONS.md`, add new terms to `GLOSSARY.md`, commit, push the branch.
