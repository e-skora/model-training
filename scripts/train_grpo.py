"""M2 Step 5: train judge v1 further with GRPO (reinforcement learning), on one L4.

GRPO: for each pair the model writes 8 answers, `judge.reward.GRPOReward`
scores them, and training pushes the model toward the answers that scored
above the group's average.

- **Start point (D-031, D-037):** the 4-bit base files and v1's epoch-2 LoRA
  adapter, each at its pinned commit. The adapter is loaded as trainable and
  keeps training; no new adapter is added. A check proves its weights are
  exactly v1's file before training starts.
- **Data:** every row of `data/m2/rl-pool.jsonl` (314 pairs), each shown in the
  version saved in that row (`swapped`, `dates`), with `judge.prompt.build_messages`.
  `format_pairs` is not run again.
- **Reward:** `judge.reward.GRPOReward()`, unchanged. Its fingerprint is taken
  here on the Mac (D-051) and must be the recorded one.
- **Sampling matches the pool:** 8 answers a pair, temperature 1.0, top_p 1.0,
  top_k off, no min_p, 256 new tokens at most, the same chat template options.
  The GPU job records the settings `generate` really used and stops at the
  first answer if they differ, or if a prompt is not one of the pool's.
- **Settings:** in GRPO below, each with its source in results/m2/grpo/settings.md.
- `data/m2/valid-hard.jsonl` and `data/m1/valid.jsonl` only choose a checkpoint
  (D-056); they never reach training. The frozen test files are never opened.

Run from the repo folder, one stage at a time:
    uv run modal run scripts/train_grpo.py --stage trial    # 20 steps, checkpoints at 10 and 20, then 8 answers
    uv run modal run scripts/train_grpo.py --stage train    # the full run (needs Elias's yes, D-055)
    uv run modal run scripts/train_grpo.py --stage choose   # step 0 and each checkpoint answer both validation sets (D-056)
    uv run modal run scripts/train_grpo.py --stage upload   # the chosen adapter to a private Hugging Face repo
    uv run python scripts/train_grpo.py bill trial          # add the billed cost from `modal billing report` (no GPU)
Records go to results/m2/grpo/ (trial.meta.json, train.meta.json, valid/). Adapters stay on
the Modal volume `m2-grpo` until uploaded; they never go into git.
"""

import math
import sys
from fractions import Fraction

import modal

from judge.modal_image import image

app = modal.App("m2-grpo")
volume = modal.Volume.from_name("m2-grpo", create_if_missing=True)
RUNS = "/runs"  # where the volume is mounted on the GPU machine

# v1 exactly as D-031 pinned it (the same pins scripts/sample_pool.py checks).
BASE_REPO = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
BASE_REVISION = "7744afa8566e264af1a92a806d8d9aae00cc7c78"
V1_ADAPTER_REPO = "eskora/m1-contradiction-judge-qwen3-4b-lora"
V1_ADAPTER_REVISION = "1d9ebbf66e1a2abcad9932932bb7d810a38c59b9"
V1_CHECKPOINT = "epoch-2"
HF_REPO = "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora"  # private until M2 Step 8

POOL_PATH = "data/m2/rl-pool.jsonl"
POOL_SHA256 = "2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77"
REWARD_SHA256 = "769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7"
VALID_FILES = {"valid_hard": "data/m2/valid-hard.jsonl", "m1_valid": "data/m1/valid.jsonl"}
TRAIN_META = "results/m1/trained-v1/train.meta.json"  # v1's record: its upload must be the pins above
OUT = "results/m2/grpo"

SEED = 3407  # Unsloth's seed, as in M1: shuffling and sampling
SAMPLES = 8  # answers per pair, as the pool was sampled (judge.pool.SAMPLES)

# Every GRPO setting in one place. Sources: results/m2/grpo/settings.md.
GRPO = {
    # Sampling: the same as the pool (judge.generate.SAMPLING), checked by a test.
    "num_generations": SAMPLES,
    "max_completion_length": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,  # 0 = off
    "min_p": None,  # off; Unsloth's notebook uses 0.1 with vLLM, not copied
    "repetition_penalty": 1.0,
    "use_vllm": False,
    # Batch: 8 answers per micro-batch (one pair's group), 2 micro-batches per step = 2 pairs, 16 answers.
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    # Optimizer and schedule.
    "learning_rate": 5e-6,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "weight_decay": 0.001,
    "max_grad_norm": 1.0,
    # The GRPO loss.
    "beta": 0.0,  # no pull back toward a reference model, so none is loaded
    "loss_type": "dapo",
    "epsilon": 0.2,
    # Unsloth sets 0.28 by itself when this is left empty with "dapo" (the trial ran with it); set here so it shows.
    "epsilon_high": 0.28,
    "num_iterations": 1,
    "scale_rewards": "group",
    "mask_truncated_completions": False,  # a cut-off answer is invalid and must still cost -2.0
    # Bookkeeping.
    "num_train_epochs": 2,
    "logging_steps": 1,
    "shuffle_dataset": True,
    "log_completions": False,
    "remove_unused_columns": False,
    "save_strategy": "no",  # the adapter is saved by SaveCheckpoints below, every SAVE_EVERY steps
    "report_to": "none",
}
WARMUP_SHARE = 0.1  # Unsloth's notebook (warmup_ratio 0.1)
GRADIENT_CHECKPOINTING = "unsloth"
SAVE_EVERY = 50
TRIAL = {"max_steps": 20, "save_every": 10, "check_items": 8}
LORA_EXPECTED = {"r": 32, "lora_alpha": 32, "lora_dropout": 0.0,
                 "target_modules": sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])}

# The reward-hacking watch: warn when, over the last WATCH["window"] logged steps, ...
WATCH = {"window": 20, "invalid": 0.02, "one_label": 0.85, "unclear": 0.15}
LABEL_KINDS = ("contradiction", "non_contradiction", "unclear")
ANSWER_KINDS = (*LABEL_KINDS, "invalid")
# The metrics kept from each logged step.
LOG_KEYS = ("reward", "reward_std", "frac_reward_zero_std", "completions/mean_length",
            "completions/clipped_ratio", "rewards/judge_reward/mean", "loss", "grad_norm",
            "learning_rate", "entropy", *(f"answers/{k}" for k in ANSWER_KINDS))

# Logits processors that change sampling. With the pool's settings, generate must use none of them.
SAMPLING_PROCESSORS = ("TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper", "MinPLogitsWarper",
                       "TypicalLogitsWarper", "TopHLogitsWarper", "EpsilonLogitsWarper", "EtaLogitsWarper",
                       "RepetitionPenaltyLogitsProcessor")
BATCH_SIZE = 16  # greedy answering, as scripts/train.py
L4_PER_SECOND = 0.000222  # https://modal.com/pricing (as in the M2 SPEC)

CHOICE_RULE = (
    "D-056: candidates are step 0 (v1) and every checkpoint. Out: any invalid answer in either validation file; "
    "worst-case precision on valid-hard more than 0.02 below step 0's; worst-case precision or recall on M1 valid "
    "more than 0.03 below step 0's. Pick the highest worst-case recall on valid-hard, then the higher worst-case "
    "precision there, then the earlier step. Worst case is the lowest of the three versions."
)


# ---------- Mac side: checks, data, rules (no GPU libraries; tests run these) ----------

def planned_steps(pairs, max_steps=0):
    """Optimizer steps: max_steps if set, else epochs times steps per epoch (pairs per step = batch x accumulation / 8)."""
    if max_steps:
        return max_steps
    per_step = GRPO["per_device_train_batch_size"] * GRPO["gradient_accumulation_steps"] // GRPO["num_generations"]
    return GRPO["num_train_epochs"] * math.ceil(pairs / per_step)


def warmup_steps(total):
    return max(1, round(WARMUP_SHARE * total))


def save_points(total, every):
    """The steps at which an adapter is saved: every `every` steps, and at the end."""
    points = list(range(every, total + 1, every))
    if not points or points[-1] != total:
        points.append(total)
    return points


def sampling_settings():
    """The sampling the GRPO config asks for, in the keys of judge.generate.SAMPLING (plus answers per pair)."""
    return {
        "max_new_tokens": GRPO["max_completion_length"],
        "do_sample": True,  # TRL's generation config always samples
        "num_beams": 1,
        "temperature": GRPO["temperature"],
        "top_p": GRPO["top_p"],
        "top_k": GRPO["top_k"],
        "min_p": GRPO["min_p"],
        "repetition_penalty": GRPO["repetition_penalty"],
        "num_return_sequences": GRPO["num_generations"],
    }


def check_not_frozen(paths, state_path="STATE.md"):
    """Stop if any file this script would read is a frozen test file."""
    from pathlib import Path

    from judge.frozen import frozen_entries, frozen_key

    frozen = frozen_entries(Path(state_path))
    hits = [p for p in paths if frozen_key(p, state_path) in frozen]
    if hits:
        raise SystemExit(f"STOP: {', '.join(hits)} is a frozen test file; nothing ran")


def check_reward():
    """The reward fingerprint, taken here on the Mac (D-051). Stops unless it is the recorded one."""
    from judge.reward import reward_sha256

    found = reward_sha256()
    if found != REWARD_SHA256:
        raise SystemExit(f"STOP: reward fingerprint is {found}, not the recorded {REWARD_SHA256}; nothing ran")
    return found


def read_pool(path=None, expected=None):
    """The pool rows and their SHA-256, from one read. Stops unless the fingerprint is the recorded one.

    The defaults are read when called, not when this file loads, so the check always uses today's constants.
    """
    from judge.train_data import read_pairs

    path = path or POOL_PATH
    expected = expected or POOL_SHA256
    rows, sha = read_pairs(path)
    if sha != expected:
        raise SystemExit(f"STOP: {path} SHA-256 is {sha}, not the recorded {expected}; nothing ran")
    return rows, sha


def dataset_rows(pool):
    """One GRPO row per pool row: the prompt in the row's saved version, the right label and type, and the id.

    The prompt is built with judge.format.to_example from the row's own `swapped`
    and `dates`, so it is the prompt the pool sampled (judge.pool.show) without
    running format_pairs again.
    """
    from judge.format import to_example
    from judge.pool import version_name

    rows = []
    for row in pool:
        if version_name(row["swapped"], row["dates"]) != row["version"]:
            raise SystemExit(f"STOP: pool row {row['id']} says {row['version']} but swapped={row['swapped']}, "
                             f"dates={row['dates']}")
        example = to_example(row, swap=row["swapped"], dates=row["dates"])
        rows.append({"prompt": example["messages"][:-1], "gold_label": row["label"], "gold_type": row["type"],
                     "id": row["id"]})
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("STOP: two pool rows share an id")
    return rows


def check_apart(pool, valid):
    """Stop if a validation pair shares an id or a memory text with a pool pair (validation never trains)."""
    pool_ids = {p["id"] for p in pool}
    pool_texts = {p[w]["text"] for p in pool for w in ("a", "b")}
    for name, pairs in valid.items():
        ids = pool_ids & {p["id"] for p in pairs}
        texts = pool_texts & {p[w]["text"] for p in pairs for w in ("a", "b")}
        if ids or texts:
            raise SystemExit(f"STOP: {name} overlaps the pool ({len(ids)} ids, {len(texts)} memory texts)")


def check_v1(train_meta):
    """Stop unless v1's training record names the pinned base and the pinned epoch-2 upload."""
    upload = train_meta.get("upload") or {}
    if (train_meta.get("base_model"), train_meta.get("base_revision")) != (BASE_REPO, BASE_REVISION):
        raise SystemExit("STOP: v1's record names a different base model or commit")
    if ((upload.get("repo"), upload.get("commit"), upload.get("checkpoint")) != (V1_ADAPTER_REPO, V1_ADAPTER_REVISION,
                                                                                 V1_CHECKPOINT)
            or (train_meta.get("choice") or {}).get("chosen") != V1_CHECKPOINT):
        raise SystemExit("STOP: v1's record does not name the pinned epoch-2 adapter upload")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def watch_warnings(steps, watch=WATCH):
    """Reward-hacking warnings over the last `window` logged steps (none until that many steps exist).

    `steps` is the step log: dicts with "step" and the "answers/<kind>" shares.
    Each step has the same number of answers, so the window's share is the mean of its steps' shares.
    """
    window = [s for s in steps if "answers/invalid" in s][-watch["window"]:]
    if len(window) < watch["window"]:
        return []
    where = f"steps {window[0]['step']}-{window[-1]['step']}"
    found = []
    invalid = mean(s.get("answers/invalid") for s in window)
    if invalid > watch["invalid"]:
        found.append(f"WARNING ({where}): invalid answers are {invalid:.1%}, over {watch['invalid']:.0%}")
    for kind in LABEL_KINDS:
        share = mean(s.get(f"answers/{kind}") for s in window)
        if share is not None and share > watch["one_label"]:
            found.append(f"WARNING ({where}): {kind} is {share:.1%} of answers, over {watch['one_label']:.0%}")
    unclear = mean(s.get("answers/unclear") for s in window)
    if unclear is not None and unclear > watch["unclear"]:
        found.append(f"WARNING ({where}): unclear is {unclear:.1%} of answers, over {watch['unclear']:.0%}")
    return found


def valid_items(files):
    """One item per validation pair and version, for every validation file: id "<file>/<version>/<pair id>"."""
    from judge.prompt import build_messages
    from judge.score import VERSIONS, derive

    return [{"id": f"{name}/{version}/{shown['id']}", "messages": build_messages(shown)}
            for name, pairs in files.items() for version in VERSIONS for shown in derive(pairs, version)]


def counts(pairs, answers):
    """Exact counts for one version: caught, said (contradiction answers), real contradictions, invalid."""
    from judge.score import score

    _, rows = score(pairs, answers)
    return {
        "caught": sum(r["label"] == "contradiction" and r["gold_label"] == "contradiction" for r in rows),
        "said": sum(r["label"] == "contradiction" for r in rows),
        "real": sum(r["gold_label"] == "contradiction" for r in rows),
        "invalid": sum(bool(r["problem"]) for r in rows),
    }


def precision(c):
    """Exact precision; a candidate that never says contradiction gets 0 (the strict reading)."""
    return Fraction(c["caught"], c["said"]) if c["said"] else Fraction(0)


def recall(c):
    return Fraction(c["caught"], c["real"]) if c["real"] else Fraction(0)


def worst(by_version, rate):
    return min(rate(c) for c in by_version.values())


def choose_checkpoint(candidates):
    """D-056, coded before any training result. Returns (chosen name or None, {name: [reasons it is out]}).

    `candidates` maps a name to {"step": n, "valid_hard": {version: counts}, "m1_valid": {version: counts}},
    where counts come from `counts`. "step-0" is v1. All comparisons use exact fractions.
    """
    base = candidates["step-0"]
    p0 = worst(base["valid_hard"], precision)
    m1_p0, m1_r0 = worst(base["m1_valid"], precision), worst(base["m1_valid"], recall)
    out = {}
    for name, c in candidates.items():
        reasons = []
        invalid = sum(v["invalid"] for f in ("valid_hard", "m1_valid") for v in c[f].values())
        if invalid:
            reasons.append(f"{invalid} invalid answers")
        if worst(c["valid_hard"], precision) < p0 - Fraction(2, 100):
            reasons.append("valid-hard worst-case precision more than 0.02 below step 0")
        if worst(c["m1_valid"], precision) < m1_p0 - Fraction(3, 100):
            reasons.append("M1 valid worst-case precision more than 0.03 below step 0")
        if worst(c["m1_valid"], recall) < m1_r0 - Fraction(3, 100):
            reasons.append("M1 valid worst-case recall more than 0.03 below step 0")
        if reasons:
            out[name] = reasons
    left = [name for name in candidates if name not in out]
    if not left:
        return None, out
    chosen = max(left, key=lambda n: (worst(candidates[n]["valid_hard"], recall),
                                      worst(candidates[n]["valid_hard"], precision),
                                      -candidates[n]["step"]))
    return chosen, out


def check_free(paths):
    taken = [str(p) for p in paths if p.exists()]
    if taken:
        raise SystemExit(f"STOP: these files already exist and would be replaced: {', '.join(taken)}. Nothing ran.")


def run_settings():
    """What a record says about how the run was made, as today's script makes it."""
    from judge.train_data import plain

    return plain({
        "base_model": BASE_REPO,
        "base_revision": BASE_REVISION,
        "adapter_repo": V1_ADAPTER_REPO,
        "adapter_revision": V1_ADAPTER_REVISION,
        "adapter_checkpoint": V1_CHECKPOINT,
        "seed": SEED,
        "grpo": GRPO,
        "warmup_share": WARMUP_SHARE,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "save_every": SAVE_EVERY,
        "watch": WATCH,
        "sampling_asked": sampling_settings(),
    })


def check_record(meta, pool_sha, valid_shas=None):
    """Stop unless the pool, prompt, reward and settings (and, once chosen, the validation files) match the record."""
    from judge.prompt import prompt_sha256

    problems = []
    if meta.get("pool_sha256") != pool_sha:
        problems.append(f"pool SHA-256 is {pool_sha}, the record says {meta.get('pool_sha256')}")
    if meta.get("prompt_sha256") != prompt_sha256():
        problems.append("the prompt changed since training")
    if meta.get("reward_sha256") != REWARD_SHA256:
        problems.append("the reward fingerprint differs from the record")
    for key, value in run_settings().items():
        if meta.get(key) != value:
            problems.append(f"{key} is {value!r} in the script, {meta.get(key)!r} in the record")
    if valid_shas is not None and meta.get("choice") and meta["choice"].get("valid_sha256") != valid_shas:
        problems.append("a validation file changed since the choice")
    if problems:
        raise SystemExit("STOP: the record does not match what this run would use; nothing ran.\n  "
                         + "\n  ".join(problems))


# ---------- GPU side ----------

@app.function(image=image, gpu="L4", timeout=3 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def train(run, rows, max_steps, save_every):
    """Load v1 as trainable, train with GRPO, save adapters. Returns a plain record."""
    import os
    import time

    start = time.time()
    import unsloth  # import first, as Unsloth asks
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from huggingface_hub import hf_hub_download
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from transformers.generation.utils import GenerationMixin
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from judge.generate import TEMPLATE_BASE
    from judge.load import MAX_SEQ_LENGTH
    from judge.reward import GRPOReward
    from judge.train_data import plain

    # 1. The start point: the pinned base, then v1's adapter at its pinned commit, trainable,
    #    then Unsloth's own patch (the three steps Unsloth's loader runs for an adapter, with the base pinned).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_REPO, revision=BASE_REVISION, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
        fast_inference=False)
    model = PeftModel.from_pretrained(model, V1_ADAPTER_REPO, revision=V1_ADAPTER_REVISION, is_trainable=True)
    model = FastLanguageModel.patch_peft_model(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)

    lora = model.peft_config["default"]
    lora_found = {"r": lora.r, "lora_alpha": lora.lora_alpha, "lora_dropout": float(lora.lora_dropout),
                  "target_modules": sorted(lora.target_modules)}
    if lora_found != LORA_EXPECTED:
        raise RuntimeError(f"v1's adapter settings are {lora_found}, not {LORA_EXPECTED}")
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise RuntimeError("the trainable weights are not exactly the LoRA adapter's")
    saved_v1 = load_file(hf_hub_download(V1_ADAPTER_REPO, "adapter_model.safetensors", revision=V1_ADAPTER_REVISION))
    loaded = get_peft_model_state_dict(model)
    if set(saved_v1) != set(loaded):
        raise RuntimeError("the loaded adapter's weight names differ from v1's file")
    largest_diff = max(float((loaded[k].float().cpu() - saved_v1[k].float()).abs().max()) for k in saved_v1)
    if largest_diff != 0.0:
        raise RuntimeError(f"the loaded adapter differs from v1's file (largest difference {largest_diff})")
    adapter_check = {"lora": lora_found, "trainable_tensors": len(trainable),
                     "trainable_parameters": sum(n for _, n in trainable),
                     "weights_equal_v1_file": True, "tensors_compared": len(saved_v1)}

    # 2. The prompts the pool sampled, as token ids built the way judge.generate builds them.
    expected_prompts = {}
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"], **TEMPLATE_BASE)
        expected_prompts[tuple(tokenizer(text, add_special_tokens=False)["input_ids"])] = row["id"]

    # 3. Record what generate really used, and stop at the first answer if it is not the pool's sampling.
    asked = sampling_settings()
    seen = {"generate_calls": 0, "configs": {}, "processors": {}, "prompts_checked": 0, "ids": set()}
    original_prepare = GenerationMixin._prepare_generation_config
    original_processors = GenerationMixin._get_logits_processor

    def recording_prepare(self, generation_config, **kwargs):
        config, model_kwargs = original_prepare(self, generation_config, **kwargs)
        used = {k: getattr(config, k, None) for k in ("do_sample", "num_beams", "temperature", "top_p", "top_k",
                                                    "min_p", "typical_p", "top_h", "repetition_penalty",
                                                    "max_new_tokens", "num_return_sequences")}
        key = repr(sorted(used.items()))
        seen["configs"][key] = seen["configs"].get(key, 0) + 1
        seen["generate_calls"] += 1
        differs = {k: (used[k], asked[k]) for k in ("do_sample", "num_beams", "temperature", "top_p", "max_new_tokens")
                   if used[k] != asked[k]}
        if used["top_k"] not in (0, None) or used["min_p"] is not None or used["repetition_penalty"] not in (1.0, None):
            differs["top_k/min_p/repetition_penalty"] = (used["top_k"], used["min_p"], used["repetition_penalty"])
        if differs:
            raise RuntimeError(f"generate used other sampling than the pool: {differs}")
        ids, mask = kwargs.get("input_ids"), kwargs.get("attention_mask")
        if ids is None or mask is None:
            raise RuntimeError("generate was called without input_ids and attention_mask; prompts cannot be checked")
        for row_ids, row_mask in zip(ids.tolist(), mask.tolist()):
            prompt = tuple(t for t, m in zip(row_ids, row_mask) if m)
            if prompt not in expected_prompts:
                raise RuntimeError("generate got a prompt that is not one of the pool's prompts")
            seen["prompts_checked"] += 1
            seen["ids"].add(expected_prompts[prompt])
        return config, model_kwargs

    def recording_processors(self, generation_config, *args, **kwargs):
        processors = original_processors(self, generation_config, *args, **kwargs)
        names = tuple(type(p).__name__ for p in processors)
        seen["processors"][repr(names)] = seen["processors"].get(repr(names), 0) + 1
        changing = [n for n in names if n in SAMPLING_PROCESSORS]
        if changing:
            raise RuntimeError(f"generate changed the sampling with {changing}")
        return processors

    GenerationMixin._prepare_generation_config = recording_prepare
    GenerationMixin._get_logits_processor = recording_processors

    # 4. The trainer.
    total = planned_steps(len(rows), max_steps)
    points = save_points(total, save_every)
    settings = {k: v for k, v in GRPO.items() if k != "num_train_epochs"}
    config = GRPOConfig(output_dir=f"{RUNS}/{run}/trainer", max_steps=total, warmup_steps=warmup_steps(total),
                        seed=SEED, **settings)
    reward = GRPOReward()
    steps, warnings_seen, saved = [], [], []
    os.makedirs(f"{RUNS}/{run}", exist_ok=True)
    log_file = open(f"{RUNS}/{run}/run.log", "w", encoding="utf-8")
    clock = {"started": time.time()}

    def write(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    class Watch(TrainerCallback):
        """Keep each step's metrics and write a warning line when the reward-hacking watch fires."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "reward" not in logs:
                return
            entry = {"step": state.global_step, "seconds": round(time.time() - clock["started"], 1)}
            entry.update({k: float(logs[k]) for k in LOG_KEYS if k in logs})
            steps.append(entry)
            shares = " ".join(f"{k}={entry.get('answers/' + k, float('nan')):.2f}" for k in ANSWER_KINDS)
            write(f"step {entry['step']}: reward {entry.get('reward', float('nan')):.3f} "
                  f"zero_std {entry.get('frac_reward_zero_std', float('nan')):.2f} "
                  f"length {entry.get('completions/mean_length', float('nan')):.1f} {shares}")
            for line in watch_warnings(steps):
                warnings_seen.append(line)
                write(line)

    class SaveCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step in points and not any(s["step"] == state.global_step for s in saved):
                name = f"step-{state.global_step}"
                model.save_pretrained(f"{RUNS}/{run}/{name}")
                tokenizer.save_pretrained(f"{RUNS}/{run}/{name}")
                saved.append({"name": name, "step": state.global_step,
                              "seconds": round(time.time() - clock["started"], 1)})
                volume.commit()
            return control

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=datasets.Dataset.from_list(rows),
        callbacks=[Watch(), SaveCheckpoints()],
    )
    resolved = {k: getattr(trainer.args, k, None) for k in (
        *settings, "max_steps", "warmup_steps", "seed", "bf16", "fp16", "generation_batch_size",
        "steps_per_generation", "gradient_checkpointing", "chat_template_kwargs", "generation_kwargs", "delta",
        "importance_sampling_level", "vllm_importance_sampling_correction", "top_entropy_quantile",
        "router_aux_loss_coef", "disable_dropout")}
    decode_path = ("parse_response" if getattr(trainer._tokenizer, "response_template", None) is not None
                   or getattr(trainer._tokenizer, "response_schema", None) is not None
                   else "batch_decode(skip_special_tokens=True)")
    write(f"loaded in {time.time() - start:.0f} s; {total} steps, saving at {points}; settings {resolved}")
    clock["started"] = time.time()
    result = trainer.train()
    trained = time.time()
    GenerationMixin._prepare_generation_config = original_prepare
    GenerationMixin._get_logits_processor = original_processors
    log_file.close()
    volume.commit()

    return plain({
        "run": run,
        "loaded_model": model.base_model.config._name_or_path,
        "planned_steps": total,
        "global_step": trainer.state.global_step,
        "save_points": points,
        "saved": saved,
        "adapter_check": adapter_check,
        "resolved_config": resolved,
        "completion_decode": decode_path,
        "sampling_used": {"generate_calls": seen["generate_calls"],
                          "configs": [{"settings": k, "calls": n} for k, n in seen["configs"].items()],
                          "logits_processors": [{"processors": k, "calls": n} for k, n in seen["processors"].items()],
                          "prompts_checked": seen["prompts_checked"],
                          "distinct_pool_prompts_seen": len(seen["ids"])},
        "steps": steps,
        "warnings": warnings_seen,
        "reward_answer_counts": reward.total_counts,
        "train_metrics": result.metrics,
        "load_seconds": round(clock["started"] - start, 1),
        "train_seconds": round(trained - clock["started"], 1),
        "seconds_per_step": round((trained - clock["started"]) / max(1, trainer.state.global_step), 2),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "versions": {"unsloth": unsloth.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
                     "peft": peft.__version__, "torch": torch.__version__, "datasets": datasets.__version__},
    })


@app.function(image=image, gpu="L4", timeout=3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def answer(run, checkpoint, items):
    """Greedy answers from one candidate: "step-0" is v1 from Hugging Face, any other name a saved checkpoint."""
    import time

    start = time.time()
    import torch

    from judge.generate import answer_items
    from judge.load import load_judge

    if checkpoint == "step-0":
        model, tokenizer = load_judge(BASE_REPO, BASE_REVISION, adapter=V1_ADAPTER_REPO,
                                      adapter_revision=V1_ADAPTER_REVISION)
    else:
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
                                   commit_message=f"M2 Step 5: GRPO LoRA adapter {run}/{checkpoint}")
    info = api.model_info(HF_REPO)
    return {"repo": HF_REPO, "commit": info.sha, "commit_url": commit.commit_url, "private": info.private,
            "files": files}


def modal_calls():
    """The GPU and upload jobs on Modal. Tests pass fakes with the same names."""
    from types import SimpleNamespace

    return SimpleNamespace(train=train.remote, answer=answer.remote,
                           answer_many=lambda args: list(answer.starmap(args)), upload=upload.remote,
                           app_id=lambda: app.app_id)


# ---------- Stages ----------

def read_valid():
    from judge.train_data import read_pairs

    files, shas = {}, {}
    for name, path in VALID_FILES.items():
        files[name], shas[path] = read_pairs(path)
    return files, shas


def run_stage(stage, run, calls, out=OUT, state_path="STATE.md", train_meta_path=TRAIN_META):
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    from judge.prompt import prompt_sha256
    from judge.score import VERSIONS, format_problem, parse_answer, score_versions

    if stage not in ("trial", "train", "choose", "upload"):
        raise SystemExit(f"unknown stage {stage!r}; choose trial, train, choose or upload")
    check_not_frozen([POOL_PATH, *VALID_FILES.values()], state_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "train.meta.json"

    if stage in ("trial", "train"):
        trial = stage == "trial"
        path = out / ("trial.meta.json" if trial else "train.meta.json")
        check_free([path])
        check_v1(json.loads(Path(train_meta_path).read_text(encoding="utf-8")))
        reward_sha = check_reward()
        pool, pool_sha = read_pool()
        valid, valid_shas = read_valid()
        check_apart(pool, valid)
        rows = dataset_rows(pool)
        name = "trial" if trial else run
        max_steps = TRIAL["max_steps"] if trial else 0
        save_every = TRIAL["save_every"] if trial else SAVE_EVERY
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start = time.time()
        result = calls.train(name, rows, max_steps, save_every)
        wall = round(time.time() - start, 1)
        meta = {
            "stage": stage,
            "started_at": started_at,
            "pool_path": POOL_PATH,
            "pool_sha256": pool_sha,
            "pool_rows": len(rows),
            "valid_kept_apart": valid_shas,
            "prompt_sha256": prompt_sha256(),
            "reward_sha256": reward_sha,
            **run_settings(),
            "modal_app_id": calls.app_id(),
            "wall_seconds": wall,
            "cost_estimate_inside": round((result["load_seconds"] + result["train_seconds"]) * L4_PER_SECOND, 2),
            "cost_estimate_wall": round(wall * L4_PER_SECOND, 2),
            **result,
        }
        if trial:
            # Prove the last checkpoint loads and answers in v1's format, on 8 valid-hard items.
            checkpoint = result["saved"][-1]["name"]
            items = valid_items({"valid_hard": valid["valid_hard"]})[:TRIAL["check_items"]]
            check = calls.answer(name, checkpoint, items)
            lines = []
            for line in check.pop("results"):
                parsed, problem, _ = parse_answer(line["raw"])
                lines.append({**line, "problem": problem,
                              "format_problem": format_problem(parsed) if parsed else None})
            meta["load_check"] = {"checkpoint": checkpoint, "answers": lines, **check}
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {result['global_step']} steps in {result['train_seconds']} s ({result['seconds_per_step']} s a "
              f"step), load {result['load_seconds']} s, peak VRAM {result['peak_vram_gib']} GiB, wall {wall / 60:.1f} "
              f"min; saved {[s['name'] for s in result['saved']]}; warnings {len(result['warnings'])}")
        if trial:
            for line in meta["load_check"]["answers"]:
                print(f"  {line['id']}: {line['raw']}  problem={line['problem']} format={line['format_problem']}")
        return meta

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pool, pool_sha = read_pool()
    valid, valid_shas = read_valid()
    check_record(meta, pool_sha, valid_shas if stage == "upload" else None)

    if stage == "choose":
        if meta.get("choice"):
            raise SystemExit("STOP: train.meta.json already holds a choice; nothing ran")
        valid_dir = out / "valid"
        names = ["step-0", *[s["name"] for s in meta["saved"]]]
        check_free([valid_dir / f"{n}.{f}.{v}.jsonl" for n in names for f in VALID_FILES for v in VERSIONS])
        items = valid_items(valid)
        start = time.time()
        outcomes = calls.answer_many([(meta["run"], n, items) for n in names])
        wall = round(time.time() - start, 1)
        valid_dir.mkdir(exist_ok=True)
        steps = {"step-0": 0, **{s["name"]: s["step"] for s in meta["saved"]}}
        candidates, scores, runs = {}, {}, {}
        for outcome in outcomes:
            n = outcome["checkpoint"]
            answers = {f: {v: [] for v in VERSIONS} for f in VALID_FILES}
            for line in outcome.pop("results"):
                f, version, pair_id = line["id"].split("/", 2)
                answers[f][version].append({"id": pair_id, "raw": line["raw"]})
            for f, by_version in answers.items():
                for version, lines in by_version.items():
                    with open(valid_dir / f"{n}.{f}.{version}.jsonl", "x", encoding="utf-8") as handle:
                        for line in lines:
                            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            candidates[n] = {"step": steps[n],
                             **{f: {v: counts(valid[f], answers[f][v]) for v in VERSIONS} for f in VALID_FILES}}
            scores[n] = {f: score_versions(valid[f], answers[f]) for f in VALID_FILES}
            runs[n] = outcome
        chosen, out_reasons = choose_checkpoint(candidates)
        meta["choice"] = {"rule": CHOICE_RULE, "chosen": chosen, "out": out_reasons, "counts": candidates,
                          "valid_scores": scores, "answer_runs": runs, "valid_sha256": valid_shas,
                          "wall_seconds": wall, "modal_app_id": calls.app_id()}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for n in names:
            cells = ", ".join(f"{f} {v} P {m['precision'] or 0:.2f} R {m['recall'] or 0:.2f} inv {m['invalid']}"
                              for f in VALID_FILES for v, m in scores[n][f].items())
            print(f"{n}: {cells}{'  OUT: ' + '; '.join(out_reasons[n]) if n in out_reasons else ''}")
        print(f"chosen on validation (D-056): {chosen}"
              + ("  (step 0: RL did not help on validation; the hard test set is not run)" if chosen == "step-0" else ""))
        return meta

    chosen = (meta.get("choice") or {}).get("chosen")
    if not chosen or chosen == "step-0":
        raise SystemExit(f"STOP: nothing to upload (chosen: {chosen}); nothing ran")
    if meta.get("upload"):
        raise SystemExit("STOP: train.meta.json already records an upload; nothing ran")
    result = calls.upload(meta["run"], chosen, json.dumps(meta, indent=2) + "\n")
    meta["upload"] = {"checkpoint": chosen, "run": meta["run"], **result}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"uploaded {chosen} to https://huggingface.co/{result['repo']} (private: {result['private']}), "
          f"commit {result['commit']}")
    return meta


def billed(report, app_ids):
    """The billed cost per app id, added up from `modal billing report --json` rows."""
    from decimal import Decimal

    totals = {}
    for row in report:
        if row.get("object_id") in app_ids:
            totals[row["object_id"]] = totals.get(row["object_id"], Decimal(0)) + Decimal(str(row["cost"]))
    return {k: float(v) for k, v in totals.items()}


def bill(name, out=OUT, run_report=None):
    """Add the billed cost of a run's Modal apps to its record, from `modal billing report` (no GPU).

    Modal reports full hours only, so run this after the hour the run ended in has passed.
    """
    import json
    import subprocess
    from pathlib import Path

    path = Path(out) / f"{name}.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    ids = [i for i in (meta.get("modal_app_id"), (meta.get("choice") or {}).get("modal_app_id")) if i]
    command = [sys.executable, "-m", "modal", "billing", "report", "--start", meta["started_at"][:10],
               "-r", "h", "--tz", "local", "--show-resources", "--json"]
    text = run_report(command) if run_report else subprocess.run(command, check=True, capture_output=True,
                                                                  text=True).stdout
    found = billed(json.loads(text), ids)
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"STOP: no billing rows yet for {missing}; try again after the hour has passed")
    meta["billed_usd"] = found
    meta["billed_total_usd"] = round(sum(found.values()), 4)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: billed {found} (total ${meta['billed_total_usd']})")


@app.local_entrypoint()
def main(stage: str = "trial", run: str = "v2"):
    run_stage(stage, run, modal_calls())


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "bill":
    bill(sys.argv[2])
