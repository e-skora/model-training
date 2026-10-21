# How to run a Modal job from this repo

Modal rents us a GPU (a chip that does training math fast) and bills by the second. Each job is a Python file in `scripts/`.

## Run a job
1. Open Terminal and go to the repo: `cd ~/code/model-training`
2. Run the file through uv (the tool that holds this project's Python and packages):
   `uv run modal run scripts/hello_gpu.py`
3. Modal builds the container image (the ready-made software package the job runs in) if it changed, starts a GPU machine, runs the job, prints its output, and shuts the machine down.
4. The first lines and the last line print a link, "View run at https://modal.com/apps/...". Open it to see the run's logs and how long it ran.

The first run of a new image takes longer because Modal builds it. Later runs reuse it.

## Where to see the cost
- **Total spend this month:** in Modal's website, open Settings, then "Usage & Billing" (https://modal.com/settings/usage). Modal's free plan includes $30 of compute each month.
- **One run:** open the run's link (step 4) to see how long it ran, then multiply. An L4 GPU costs $0.000222 per second, about $0.80 an hour (https://modal.com/pricing, checked 2026-10-21). Our free plan does not show cost per run; per-app cost reports are only on Modal's paid Team plan (https://modal.com/docs/guide/billing).
- Every GPU run gets one line in `runs/RUNLOG.md` with its minutes and estimated cost.

## Safety rules
- Every Modal function sets a `timeout`, so a stuck job stops itself.
- Default GPU is `L4`. Ask Elias before any run expected to cost more than $1.
- The Hugging Face token lives in the Modal secret `huggingface` (key `HF_TOKEN`), never in this repo. Code reads it with `os.environ["HF_TOKEN"]` after adding `secrets=[modal.Secret.from_name("huggingface")]` to the function.
