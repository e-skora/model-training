# Prompt M1-03: Tool setup and first GPU test

**Where:** Claude Code (desktop Code panel), **new session**.
**Folder:** `~/code/model-training`
**Branch picker:** `main`
**Worktree box:** unchecked

Copy everything below the line into Claude Code.

---

You are working in `~/code/model-training`. Read `AGENTS.md` first and follow it, then read `STATE.md` and `milestones/M1-contradiction-judge/SPEC.md`. This task is M1 Step 3: install the tools and prove that a GPU job runs on Modal. Do not start any other step.

Create and switch to a branch named `m1-03-tool-setup`.

1. **Python project.** Use `uv` (already installed) to make this repo a Python project (Python 3.12) and add `modal` as a dependency. Commit `pyproject.toml` and `uv.lock`. Do not commit `.venv/`.
2. **Modal login.** A Modal login file already exists at `~/.modal.toml` (profile `e-skora`). Check that the login works with a read-only Modal command run through `uv run`. If it does not work, stop and tell Elias exactly what to do (Modal's docs say the login command is `modal setup`; check https://modal.com/docs/guide first).
3. **Hugging Face login.** Check `hf auth whoami` shows user `eskora`. Never print the token.
4. **Hugging Face token for Modal.** Later steps upload models from Modal, so Modal needs the Hugging Face token. Look up the current command in Modal's docs for creating a secret, then create a Modal secret named `huggingface` with the key `HF_TOKEN`, reading the value from `~/.cache/huggingface/token` without printing it or writing it to any file in the repo. Write the doc URL you used in your report.
5. **First GPU job.** Write `scripts/hello_gpu.py`: a Modal app with one function on an `L4` GPU with `timeout=300`. Build its container image the way Modal's Unsloth example does (https://modal.com/docs/examples/unsloth_finetune), so the image has Unsloth installed. The function prints the GPU name and its memory (VRAM), imports `unsloth` and prints its version, and checks that the `huggingface` secret is visible (print only "HF token present: yes/no"). Run it once with `uv run modal run scripts/hello_gpu.py`. If anything would cost more than $1, stop and ask.
6. **Log it.** Add one line to `runs/RUNLOG.md`: date, "M1-03 hello GPU", GPU, minutes, estimated cost (check the L4 price at https://modal.com/pricing).
7. **Short how-to.** Write `docs/how-to-run.md`: in plain language, how to run a Modal job from this repo and where to see its cost in Modal's dashboard.
8. **Close.** Update `STATE.md` (Step 3 done, next action is Step 4), add new terms to `GLOSSARY.md` (for example: container image, Modal secret, uv), add any lesson or mistake to `LESSONS.md`. Commit in small steps and push the branch to `origin`. Do not merge to `main`; Claude in Cowork checks and merges.

**Report back** in plain language: what worked, the GPU name and memory printed, the Unsloth version, the run's minutes and cost, the doc URLs you checked, and anything that is Elias's to do.
