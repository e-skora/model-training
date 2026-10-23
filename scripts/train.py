"""M1 Step 7: fine-tune Qwen3-4B-Instruct-2507 on the training set with QLoRA (D-011), on one L4.

QLoRA: the base model is loaded in 4-bit and frozen, and small LoRA adapters
are trained on top. The LoRA settings start from Unsloth's notebook for this
model (r=32, alpha=32, all seven projection layers), and the training
settings from Unsloth's LoRA guide (learning rate 2e-4, effective batch 16,
1 to 3 epochs, warmup 5%, weight decay 0.01). Only the answer (the
assistant's JSON) is trained on; `judge.train_data.check_masks` proves it
for every row before training starts. The validation loss is measured during
training. The test set is never read here.

Run from the repo folder, one stage at a time:
    uv run modal run scripts/train.py --stage trial    # 20 steps: speed, memory, loading check
    uv run modal run scripts/train.py --stage train    # full run, saves an adapter after each epoch
    uv run modal run scripts/train.py --stage choose   # answer the validation set with each epoch's adapter, pick one
    uv run modal run scripts/train.py --stage upload   # upload the chosen adapter to the private Hugging Face repo
Records go to results/m1/trained-v1/ (train.meta.json, valid/). Adapters stay
on the Modal volume `m1-train` until uploaded; they never go into git.
"""

import math

import modal

from judge.modal_image import image

app = modal.App("m1-train")
volume = modal.Volume.from_name("m1-train", create_if_missing=True)
RUNS = "/runs"  # where the volume is mounted on the GPU machine

# Asked for "unsloth/Qwen3-4B-Instruct-2507" in 4-bit, Unsloth loads this ready-made 4-bit copy
# instead and ignores the revision (its log says so). So the baseline's answers came from this
# repo, whose only upload is this commit (2025-08-06). Naming it directly makes the pin real.
BASE_REPO = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
BASE_REVISION = "7744afa8566e264af1a92a806d8d9aae00cc7c78"
HF_REPO = "eskora/m1-contradiction-judge-qwen3-4b-lora"  # private until Step 9
TRAIN_PATH = "data/m1/train.jsonl"
VALID_PATH = "data/m1/valid.jsonl"
OUT = "results/m1/trained-v1"

# Qwen's chat format (ChatML): where a user turn and an assistant turn start.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

SEED = 3407  # Unsloth's notebook seed, for LoRA start values, shuffling and dropout
LORA = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEED,
    "use_rslora": False,
    "loftq_config": None,
}
TRAIN = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # 4 x 4 = 16 examples per step
    "per_device_eval_batch_size": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "max_length": 1024,  # every row is checked to fit, so nothing is cut off
    "packing": False,
    "eval_steps": 12,  # about 8 validation-loss points per epoch
    "logging_steps": 2,
}
WARMUP_SHARE = 0.05
BATCH_SIZE = 16  # for answering, the same as the baseline


def planned_steps(examples, max_steps=0):
    """How many optimizer steps the run takes: max_steps if set, else the epochs times steps per epoch."""
    if max_steps:
        return max_steps
    per_step = TRAIN["per_device_train_batch_size"] * TRAIN["gradient_accumulation_steps"]
    return TRAIN["num_train_epochs"] * math.ceil(examples / per_step)


def warmup_steps(total):
    return max(1, round(WARMUP_SHARE * total))


def valid_items(pairs):
    """One item per validation pair and version (as is, swapped, no dates); the id says which version."""
    from judge.prompt import build_messages
    from judge.score import VERSIONS, derive

    return [{"id": f"{version}/{shown['id']}", "messages": build_messages(shown)}
            for version in VERSIONS for shown in derive(pairs, version)]


@app.function(image=image, gpu="L4", timeout=3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def train(run, train_pairs, valid_pairs, max_steps=0, eval_steps=None):
    import time

    start = time.time()
    import unsloth  # import first, as Unsloth asks
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    from judge.load import MAX_SEQ_LENGTH
    from judge.train_data import build_rows, check_masks, loss_curves, plain

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_REPO, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, revision=BASE_REVISION
    )
    model = FastLanguageModel.get_peft_model(model, **LORA)

    train_rows = build_rows(train_pairs, tokenizer)
    valid_rows = build_rows(valid_pairs, tokenizer)
    total = planned_steps(len(train_rows), max_steps)
    settings = {k: v for k, v in TRAIN.items() if k != "eval_steps"}
    config = SFTConfig(
        output_dir=f"{RUNS}/{run}/trainer",
        dataset_text_field="text",
        max_steps=max_steps or -1,
        warmup_steps=warmup_steps(total),
        eval_strategy="steps",
        eval_steps=eval_steps or TRAIN["eval_steps"],
        save_strategy="no",
        report_to="none",
        seed=SEED,
        **settings,
    )

    saved = []

    class SaveEachEpoch(TrainerCallback):
        """Save the adapter at the end of every epoch, and measure the validation loss there too."""

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            control.should_evaluate = True
            whole = round(state.epoch)
            name = f"epoch-{whole}" if abs(state.epoch - whole) < 0.01 and whole > 0 else f"step-{state.global_step}"
            model.save_pretrained(f"{RUNS}/{run}/{name}")
            tokenizer.save_pretrained(f"{RUNS}/{run}/{name}")
            saved.append({"name": name, "step": state.global_step, "epoch": round(state.epoch, 3)})
            return control

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets.Dataset.from_list([{"text": r["text"]} for r in train_rows]),
        eval_dataset=datasets.Dataset.from_list([{"text": r["text"]} for r in valid_rows]),
        args=config,
        callbacks=[SaveEachEpoch()],
    )
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    masks = {
        "train": check_masks(train_rows, trainer.train_dataset, tokenizer),
        "valid": check_masks(valid_rows, trainer.eval_dataset, tokenizer),
    }
    for split, summary in masks.items():
        if summary["longest_row_tokens"] > TRAIN["max_length"]:
            raise ValueError(f"{split}: a row has {summary['longest_row_tokens']} tokens, over max_length")
    print(f"masks checked: {masks}")

    # The validation loss before any training, as the first point of the curve.
    before = trainer.evaluate()
    loaded = time.time()
    result = trainer.train()
    trained = time.time()
    if not saved or saved[-1]["step"] != trainer.state.global_step:  # a run that stopped mid-epoch (the trial)
        name = f"step-{trainer.state.global_step}"
        model.save_pretrained(f"{RUNS}/{run}/{name}")
        tokenizer.save_pretrained(f"{RUNS}/{run}/{name}")
        saved.append({"name": name, "step": trainer.state.global_step, "epoch": round(trainer.state.epoch, 3)})
    volume.commit()

    return plain({
        "run": run,
        "loaded_model": model.config._name_or_path,
        "planned_steps": total,
        "global_step": trainer.state.global_step,
        "warmup_steps": config.warmup_steps,
        "eval_steps": config.eval_steps,
        "valid_loss_before": round(float(before["eval_loss"]), 4),
        "train_metrics": result.metrics,
        "curves": loss_curves(trainer.state.log_history),
        "saved": saved,
        "masks": masks,
        "rows": {"train": len(train_rows), "valid": len(valid_rows),
                 "train_swapped": sum(r["swapped"] for r in train_rows),
                 "train_no_dates": sum(not r["dates"] for r in train_rows),
                 "valid_swapped": sum(r["swapped"] for r in valid_rows),
                 "valid_no_dates": sum(not r["dates"] for r in valid_rows)},
        "load_seconds": round(loaded - start, 1),
        "train_seconds": round(trained - loaded, 1),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "versions": {"unsloth": unsloth.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__, "torch": torch.__version__,
                     "datasets": datasets.__version__},
    })


@app.function(image=image, gpu="L4", timeout=1800, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def answer(run, checkpoint, items):
    import time

    start = time.time()
    import torch

    from judge.generate import answer_items
    from judge.load import load_judge

    model, tokenizer = load_judge(BASE_REPO, BASE_REVISION, adapter=f"{RUNS}/{run}/{checkpoint}")
    loaded = time.time()
    with torch.no_grad():
        results, settings = answer_items(model, tokenizer, items, {}, BATCH_SIZE)
    return {
        "checkpoint": checkpoint,
        "loaded_model": model.base_model.config._name_or_path,
        "results": results,
        "load_seconds": round(loaded - start, 1),
        "answer_seconds": round(time.time() - loaded, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        **settings,
    }


@app.function(image=image, timeout=900, secrets=[modal.Secret.from_name("huggingface")], volumes={RUNS: volume})
def upload(run, checkpoint, meta_text):
    """Upload one adapter folder plus train.meta.json to the private repo. The token comes from HF_TOKEN."""
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(HF_REPO).private:
        raise SystemExit(f"STOP: {HF_REPO} exists and is public; nothing uploaded")
    with tempfile.TemporaryDirectory() as folder:
        shutil.copytree(f"{RUNS}/{run}/{checkpoint}", folder, dirs_exist_ok=True)
        (Path(folder) / "train.meta.json").write_text(meta_text, encoding="utf-8")
        files = sorted(p.name for p in Path(folder).iterdir())
        commit = api.upload_folder(folder_path=folder, repo_id=HF_REPO, repo_type="model",
                                   commit_message=f"M1 Step 7: LoRA adapter {run}/{checkpoint}")
    info = api.model_info(HF_REPO)
    return {"repo": HF_REPO, "commit": info.sha, "commit_url": commit.commit_url, "private": info.private,
            "files": files}


def run_settings():
    """The settings train.meta.json records about how a run was made, as today's script makes them."""
    from judge.format import SEED as FORMAT_SEED
    from judge.train_data import plain

    return plain({
        "base_model": BASE_REPO,
        "base_revision": BASE_REVISION,
        "seed": SEED,
        "format_seed": FORMAT_SEED,
        "lora": LORA,
        "training": TRAIN,
        "warmup_share": WARMUP_SHARE,
        "instruction_part": INSTRUCTION_PART,
        "response_part": RESPONSE_PART,
    })


def check_record(meta, valid_sha256):
    """Stop unless the validation file, the prompt and the run settings are the ones train.meta.json records.

    `valid_sha256` is the fingerprint of the validation bytes about to be used.
    Called before any remote work, so a changed file or a changed script
    constant can never pick or upload a model under an old record.
    """
    from judge.prompt import prompt_sha256

    problems = []
    if meta.get("valid_sha256") != valid_sha256:
        problems.append(f"{VALID_PATH} SHA-256 is {valid_sha256}, the record says {meta.get('valid_sha256')}")
    if meta.get("prompt_sha256") != prompt_sha256():
        problems.append(f"prompt SHA-256 is {prompt_sha256()}, the record says {meta.get('prompt_sha256')}")
    for key, value in run_settings().items():
        if meta.get(key) != value:
            problems.append(f"{key} is {value!r} in the script, {meta.get(key)!r} in the record")
    if problems:
        raise SystemExit("STOP: train.meta.json does not match what this run would use; nothing ran.\n  "
                         + "\n  ".join(problems))


def modal_calls():
    """The GPU and upload jobs, run on Modal. Tests pass fakes with the same names."""
    from types import SimpleNamespace

    return SimpleNamespace(train=train.remote, answer=answer.remote,
                           answer_many=lambda args: list(answer.starmap(args)), upload=upload.remote)


def run_stage(stage, run, calls, out=OUT, train_path=TRAIN_PATH, valid_path=VALID_PATH):
    import json
    import time
    from pathlib import Path

    from judge.prompt import prompt_sha256
    from judge.score import VERSIONS, score_versions
    from judge.train_data import CHOICE_RULE, choose, per_epoch, read_pairs

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "train.meta.json"
    # The fingerprints come from the same bytes that are sent to Modal, read once, before dispatch.
    valid_pairs, valid_sha256 = read_pairs(valid_path)

    if stage in ("trial", "train"):
        train_pairs, train_sha256 = read_pairs(train_path)
        trial = stage == "trial"
        name = "trial" if trial else run
        start = time.time()
        result = calls.train(name, train_pairs, valid_pairs, max_steps=20 if trial else 0,
                             eval_steps=10 if trial else None)
        wall = round(time.time() - start, 1)
        meta = {
            "stage": stage,
            "train_sha256": train_sha256,
            "valid_sha256": valid_sha256,
            "prompt_sha256": prompt_sha256(),
            **run_settings(),
            "wall_seconds": wall,
            **result,
        }
        if not trial:
            meta["per_epoch"] = per_epoch(result["curves"], TRAIN["num_train_epochs"])
        path = out / ("trial.meta.json" if trial else "train.meta.json")
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {result['global_step']} steps in {result['train_seconds']} s "
              f"({result['train_seconds'] / result['global_step']:.2f} s a step), load {result['load_seconds']} s, "
              f"peak VRAM {result['peak_vram_gib']} GiB, wall {wall / 60:.1f} min; saved {result['saved']}")
        if trial:
            # Prove a saved adapter loads and answers, on 8 validation items.
            checkpoint = result["saved"][-1]["name"]
            check = calls.answer(name, checkpoint, valid_items(valid_pairs)[:8])
            print(f"trial answers from {checkpoint}: load {check['load_seconds']} s, answer {check['answer_seconds']} s")
            for line in check["results"]:
                print(f"  {line['id']}: {line['raw']}")
        return

    if stage not in ("choose", "upload"):
        raise SystemExit(f"unknown stage {stage!r}; choose trial, train, choose or upload")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_record(meta, valid_sha256)

    if stage == "choose":
        checkpoints = [s["name"] for s in meta["saved"]]
        items = valid_items(valid_pairs)
        start = time.time()
        outcomes = calls.answer_many([(meta["run"], c, items) for c in checkpoints])
        wall = round(time.time() - start, 1)
        valid_dir = out / "valid"
        valid_dir.mkdir(exist_ok=True)
        scores, runs = {}, {}
        for outcome in outcomes:
            name = outcome["checkpoint"]
            answers = {version: [] for version in VERSIONS}
            for line in outcome.pop("results"):
                version, pair_id = line["id"].split("/", 1)
                answers[version].append({"id": pair_id, "raw": line["raw"]})
            for version, lines in answers.items():
                with open(valid_dir / f"{name}.{version}.jsonl", "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            scores[name] = score_versions(valid_pairs, answers)
            runs[name] = outcome
        chosen = choose(scores)
        meta["choice"] = {"rule": CHOICE_RULE, "chosen": chosen, "valid_scores": scores, "answer_runs": runs,
                          "wall_seconds": wall}
        # An upload is kept only when it is provably this run's copy of the new choice.
        old = meta.pop("upload", None)
        if old and old.get("run") == meta["run"] and old.get("checkpoint") == chosen:
            meta["upload"] = old
        elif old:
            print(f"dropped the old upload record ({old.get('run')}/{old.get('checkpoint')}): it is not "
                  f"{meta['run']}/{chosen}; run --stage upload before scoring")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for name, by_version in scores.items():
            cells = ", ".join(f"{v} P {m['precision']:.2f} R {m['recall']:.2f} inv {m['invalid']} unc {m['unclear']}"
                              for v, m in by_version.items())
            print(f"{name}: {cells} (load {runs[name]['load_seconds']} s, answer {runs[name]['answer_seconds']} s)")
        print(f"chosen on the validation set: {chosen}; wall {wall / 60:.1f} min")
        return

    chosen = meta["choice"]["chosen"]
    result = calls.upload(meta["run"], chosen, json.dumps(meta, indent=2) + "\n")
    meta["upload"] = {"checkpoint": chosen, "run": meta["run"], **result}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"uploaded {chosen} to https://huggingface.co/{result['repo']} (private: {result['private']}), "
          f"commit {result['commit']}, files {result['files']}")


@app.local_entrypoint()
def main(stage: str = "trial", run: str = "v1"):
    run_stage(stage, run, modal_calls())
