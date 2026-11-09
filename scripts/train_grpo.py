"""M2 Step 5: train judge v1 further with GRPO (reinforcement learning), on one L4.

GRPO: for each pair the model writes 8 answers, `judge.reward.GRPOReward`
scores them, and training pushes the model toward the answers that scored
above the group's average.

- **Start point (D-031, D-037):** the 4-bit base files and v1's epoch-2 LoRA
  adapter, each at its pinned commit. The adapter is loaded as trainable and
  keeps training; no new adapter is added. A check proves its trainable weights
  are exactly v1's file, by name and by value, before training starts.
- **Data:** every row of `data/m2/rl-pool.jsonl` (314 pairs), each shown in the
  version saved in that row (`swapped`, `dates`), with `judge.prompt.build_messages`.
  `format_pairs` is not run again.
- **Reward:** `judge.reward.GRPOReward()`, unchanged. Its fingerprint is taken
  here on the Mac (D-051), before every stage, and must be the recorded one.
- **Sampling matches the pool:** 8 answers a pair, temperature 1.0, top_p 1.0,
  top_k off, no min_p, 256 new tokens at most, the same chat template options.
  The GPU job watches every `generate` call and stops at the first one that
  differs: other settings, any word filter, any stop rule but the 256-token
  limit and the pool's two end tokens, prompt tokens that are not exactly TRL's
  prompts, more than one answer per prompt copy, a group that is not 8 copies
  of one pool prompt, or counts that do not add up.
- **Every run gets a fresh folder** on the volume, and every saved adapter is
  fingerprinted; choose and upload check those fingerprints before they use it.
- **Settings:** in GRPO below, each with its source in results/m2/grpo/settings.md.
- `data/m2/valid-hard.jsonl` and `data/m1/valid.jsonl` only choose a checkpoint
  (D-056); they never reach training. The frozen test files are never opened.

Run from the repo folder, one stage at a time:
    uv run modal run scripts/train_grpo.py --stage trial    # 20 steps, checkpoints at 10 and 20, then 8 answers
    uv run modal run scripts/train_grpo.py --stage train    # the full run (needs Elias's yes, D-055)
    uv run modal run scripts/train_grpo.py --stage choose   # step 0 and each checkpoint answer both validation sets (D-056)
    uv run modal run scripts/train_grpo.py --stage upload   # the chosen adapter to a new private Hugging Face repo
    uv run python scripts/train_grpo.py bill train          # add the billed cost from `modal billing report` (no GPU)
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
V1_FILES = ("adapter_config.json", "adapter_model.safetensors")  # what step 0 loads from v1's repo
HF_REPO = "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora"  # private until M2 Step 8

POOL_PATH = "data/m2/rl-pool.jsonl"
POOL_SHA256 = "2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77"
POOL_ROWS = 314
REWARD_SHA256 = "769f58fe9fd3cccd062725e25111bc86dc5c8d8f449c95795a3c1ff165d025e7"
VALID_FILES = {"valid_hard": "data/m2/valid-hard.jsonl", "m1_valid": "data/m1/valid.jsonl"}
TRAIN_META = "results/m1/trained-v1/train.meta.json"  # v1's record: its upload must be the pins above
OUT = "results/m2/grpo"

SEED = 3407  # Unsloth's seed, as in M1: shuffling and sampling
SAMPLES = 8  # answers per pair, as the pool was sampled (judge.pool.SAMPLES)

# Every GRPO setting in one place. Sources: results/m2/grpo/settings.md.
GRPO = {
    # Sampling: the same as the pool (judge.generate.SAMPLING), checked by a test and in the run.
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
# Every logged step must hold these; a missing one stops the run.
REQUIRED_STEP_KEYS = ("reward", "frac_reward_zero_std", "completions/mean_length",
                      *(f"answers/{k}" for k in ANSWER_KINDS))
# The metrics kept from each logged step.
LOG_KEYS = (*REQUIRED_STEP_KEYS, "reward_std", "completions/clipped_ratio", "rewards/judge_reward/mean", "loss",
            "grad_norm", "learning_rate", "entropy")

# Allow-lists for what `generate` may add (D-052's rule: say what may pass, stop everything else).
ALLOWED_PROCESSORS = ()  # no word filter at all: the pool sampled from the model's plain chances
ALLOWED_STOPPING = ("MaxLengthCriteria", "EosTokenCriteria")  # the 256-token limit and the end token
# The end tokens the pool's generate stopped on: judge.generate passes none of its own, so they are the pinned
# base's generation_config.json ([151645, 151643]: <|im_end|>, <|endoftext|>). TRL 1.15.0 passes the model's list
# plus the tokenizer's end token (<|im_end|>, already in it), grpo_trainer.py:398-405.
POOL_EOS_IDS = (151643, 151645)
BATCH_SIZE = 16  # greedy answering, as scripts/train.py
L4_PER_SECOND = 0.000222  # https://modal.com/pricing (as in the M2 SPEC)

CHOICE_RULE = (
    "D-056, with D-057's Style A gate: candidates are step 0 (v1) and every checkpoint. Out: any answer in either "
    "validation file that the scorer counts invalid or that breaks v1's answer format (judge.reward.read_answer); "
    "worst-case precision on valid-hard more than 0.02 below step 0's; worst-case precision or recall on M1 valid "
    "more than 0.03 below step 0's. Pick the highest worst-case recall on valid-hard, then the higher worst-case "
    "precision there, then the earlier step. Worst case is the lowest of the three versions."
)


# ---------- Mac side: checks, data, rules (no GPU libraries; tests run these) ----------

def pairs_per_step():
    return GRPO["per_device_train_batch_size"] * GRPO["gradient_accumulation_steps"] // GRPO["num_generations"]


def planned_steps(pairs, max_steps=0):
    """Optimizer steps: max_steps if set, else epochs times steps per epoch."""
    if max_steps:
        return max_steps
    return GRPO["num_train_epochs"] * math.ceil(pairs / pairs_per_step())


def warmup_steps(total):
    return max(1, round(WARMUP_SHARE * total))


def save_points(total, every):
    """The steps at which an adapter is saved: every `every` steps, and at the end."""
    points = list(range(every, total + 1, every))
    if not points or points[-1] != total:
        points.append(total)
    return points


def full_run_checkpoints():
    """The checkpoint names a finished full run must have saved."""
    return [f"step-{p}" for p in save_points(planned_steps(POOL_ROWS), SAVE_EVERY)]


def sampling_settings():
    """The sampling the GRPO config asks for, in the keys of judge.generate.SAMPLING.

    `num_return_sequences` here is the answers per pair, as the pool asked for them. TRL gets them
    by putting each prompt in the batch 8 times with one answer each; the run checks exactly that.
    """
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


def new_run_id(stage):
    """A fresh run name, never used before: the stage, the UTC time and a random tag."""
    import secrets
    from datetime import datetime, timezone

    return f"{stage}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def files_sha256(folder):
    """{file name: SHA-256} for every regular file directly in `folder`, in name order."""
    import hashlib
    from pathlib import Path

    found = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{path} is not a plain file")
        found[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def write_new_json(path, data):
    """Write a record that must not exist yet, all at once: a half-written file never takes its name."""
    import json
    import os
    from pathlib import Path

    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(temp, path)  # fails if the file exists
    except FileExistsError:
        raise SystemExit(f"STOP: {path} already exists and would be replaced. Nothing was written.") from None
    finally:
        temp.unlink()


def replace_json(path, data):
    """Replace a record all at once, so a crash leaves the old record or the new one, never half of one."""
    import json
    import os
    from pathlib import Path

    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def check_not_frozen(paths, state_path="STATE.md"):
    """Stop if any file this script would read is a frozen test file."""
    from pathlib import Path

    from judge.frozen import frozen_entries, frozen_key

    frozen = frozen_entries(Path(state_path))
    hits = [p for p in paths if frozen_key(p, state_path) in frozen]
    if hits:
        raise SystemExit(f"STOP: {', '.join(hits)} is a frozen test file; nothing ran")


def check_reward():
    """The reward fingerprint, recomputed here on the Mac (D-051). Stops unless it is the recorded one."""
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
    values = list(values)
    return sum(values) / len(values) if values else None


class GenerationCheckError(RuntimeError):
    """The run's answers were not written the way the pool's were, or the run's counts do not add up."""


def check_step_log(logs):
    """Stop the run if a logged training step lacks a metric the watch needs. Returns the kept metrics."""
    missing = [k for k in REQUIRED_STEP_KEYS if k not in logs]
    if missing:
        raise GenerationCheckError(f"a logged step lacks {missing}; the reward-hacking watch cannot read it")
    return {k: float(logs[k]) for k in LOG_KEYS if k in logs}


def watch_warnings(steps, watch=WATCH):
    """Reward-hacking warnings over the last `window` logged steps (none until that many steps exist).

    `steps` is the step log: dicts with "step" and the "answers/<kind>" shares (check_step_log makes
    sure every step has them). Each step has the same number of answers, so the window's share is
    the mean of its steps' shares.
    """
    window = steps[-watch["window"]:]
    if len(window) < watch["window"]:
        return []
    where = f"steps {window[0]['step']}-{window[-1]['step']}"
    found = []
    invalid = mean(s["answers/invalid"] for s in window)
    if invalid > watch["invalid"]:
        found.append(f"WARNING ({where}): invalid answers are {invalid:.1%}, over {watch['invalid']:.0%}")
    for kind in LABEL_KINDS:
        share = mean(s[f"answers/{kind}"] for s in window)
        if share > watch["one_label"]:
            found.append(f"WARNING ({where}): {kind} is {share:.1%} of answers, over {watch['one_label']:.0%}")
    unclear = mean(s["answers/unclear"] for s in window)
    if unclear > watch["unclear"]:
        found.append(f"WARNING ({where}): unclear is {unclear:.1%} of answers, over {watch['unclear']:.0%}")
    return found


def token_rows(x):
    """A tensor (or a plain list) of token rows as lists of ints."""
    return x.tolist() if hasattr(x, "tolist") else [list(row) for row in x]


def unpadded_rows(input_ids, attention_mask):
    """The tokens the model really attends to, row by row, and the padded width.

    Each row's mask must be one unbroken run of 1s (padding on one side only), so the kept
    tokens are exactly the prompt, in order."""
    ids, mask = token_rows(input_ids), token_rows(attention_mask)
    if len(ids) != len(mask) or any(len(i) != len(m) for i, m in zip(ids, mask)):
        raise GenerationCheckError("generate's input_ids and attention_mask differ in shape")
    rows = []
    for row_ids, row_mask in zip(ids, mask):
        kept = [k for k, m in enumerate(row_mask) if m == 1]
        if any(m not in (0, 1) for m in row_mask) or not kept or kept != list(range(kept[0], kept[-1] + 1)):
            raise GenerationCheckError("generate's attention mask is not one unbroken run per row")
        rows.append(tuple(row_ids[k] for k in kept))
    return rows, len(ids[0]) if ids else 0


def stop_rule(criterion):
    """One stop rule as plain data: its class name, and its length limit or end tokens if it has them."""
    rule = {"name": type(criterion).__name__}
    if hasattr(criterion, "max_length"):
        rule["max_length"] = int(criterion.max_length)
    if hasattr(criterion, "eos_token_id"):
        ends = criterion.eos_token_id
        ends = ends.tolist() if hasattr(ends, "tolist") else ends
        rule["eos_token_id"] = sorted(ends if isinstance(ends, (list, tuple)) else [ends])
    return rule


class GenerationWatch:
    """Checks every answer-writing call of a GRPO run against the pool's sampling (no GPU libraries here).

    The GPU job feeds it from five places (`wrap_generation`, `wrap_trainer_turn`): `generate`
    itself with the exact tokens and mask the model gets (`entry`), the settings `generate` ends up
    with (`config`), the word filters and stop rules it adds (`processors`, `stopping`), and TRL's
    own generation step, which sees the prompts and answers (`trainer_call`). Each check stops at
    once. `check_counts` makes sure no call went past a check unseen.
    """

    def __init__(self, asked, expected_prompts, group_size):
        self.asked = asked
        self.expected = expected_prompts  # {tuple of prompt token ids: pool id}
        self.group = group_size
        self.calls = dict.fromkeys(("entry", "config", "processors", "stopping", "trainer"), 0)
        self.configs, self.processor_sets, self.stopping_sets = {}, {}, {}
        self.groups = {}  # pool id -> how many groups of 8 answers it got
        self.answers = 0
        self.pending = None  # the unpadded prompt rows generate got, until TRL's step returns its answers
        self.width = None  # their padded width, which the length limit counts from

    @staticmethod
    def _count(table, key):
        table[key] = table.get(key, 0) + 1

    def entry(self, input_ids, attention_mask):
        """`generate`'s own input: the token rows and their mask, as the model gets them."""
        self.calls["entry"] += 1
        if self.pending is not None:
            raise GenerationCheckError("generate ran again before TRL's step returned the last answers")
        if input_ids is None or attention_mask is None:
            raise GenerationCheckError("generate got no input_ids or no attention_mask; its prompts cannot be checked")
        self.pending, self.width = unpadded_rows(input_ids, attention_mask)

    def config(self, used):
        """`used`: the final settings of one generate call (keys of judge.generate.SAMPLING and a few more)."""
        self.calls["config"] += 1
        self._count(self.configs, repr(sorted(used.items())))
        differs = {k: (used.get(k), self.asked[k]) for k in ("do_sample", "num_beams", "temperature", "top_p",
                                                             "max_new_tokens") if used.get(k) != self.asked[k]}
        if used.get("top_k") not in (0, None):
            differs["top_k"] = used.get("top_k")
        if used.get("min_p") is not None:
            differs["min_p"] = used.get("min_p")
        if used.get("repetition_penalty") not in (1.0, None):
            differs["repetition_penalty"] = used.get("repetition_penalty")
        if used.get("num_return_sequences") != 1:
            differs["num_return_sequences"] = (used.get("num_return_sequences"), 1)
        if differs:
            raise GenerationCheckError(f"generate used other sampling than the pool: {differs}")

    def processors(self, names):
        self.calls["processors"] += 1
        self._count(self.processor_sets, repr(tuple(names)))
        extra = [n for n in names if n not in ALLOWED_PROCESSORS]
        if extra:
            raise GenerationCheckError(f"generate changed the word chances with {extra}")

    def stopping(self, rules):
        """`rules`: the stop rules generate added, as `stop_rule` gives them."""
        self.calls["stopping"] += 1
        names = [r["name"] for r in rules]
        self._count(self.stopping_sets, repr([(r["name"], r.get("eos_token_id")) for r in rules]))
        if sorted(names) != sorted(ALLOWED_STOPPING):
            raise GenerationCheckError(f"generate used other stop rules than the pool: {names}")
        limit = next(r for r in rules if r["name"] == "MaxLengthCriteria").get("max_length")
        if self.width is None or limit != self.width + self.asked["max_new_tokens"]:
            raise GenerationCheckError(f"generate stops at {limit} tokens for prompts {self.width} wide, not after "
                                       f"{self.asked['max_new_tokens']} new tokens")
        ends = next(r for r in rules if r["name"] == "EosTokenCriteria").get("eos_token_id")
        if sorted(ends or []) != sorted(POOL_EOS_IDS):
            raise GenerationCheckError(f"generate stops on the end tokens {ends}, the pool on {list(POOL_EOS_IDS)}")

    def trainer_call(self, prompt_ids, completion_ids):
        """TRL's prompts (unpadded token ids, one row per answer) and the answers it got back."""
        self.calls["trainer"] += 1
        given, self.pending = self.pending, None
        if given is None:
            raise GenerationCheckError("TRL's step returned answers without a checked generate call")
        if given != [tuple(p) for p in prompt_ids]:
            raise GenerationCheckError("generate got other tokens than TRL's prompts (changed, cut, reordered or "
                                       "a different number of rows)")
        if len(completion_ids) != len(prompt_ids):
            raise GenerationCheckError(f"{len(prompt_ids)} prompt copies got {len(completion_ids)} answers")
        if any(len(c) > self.asked["max_new_tokens"] for c in completion_ids):
            raise GenerationCheckError(f"an answer is longer than {self.asked['max_new_tokens']} tokens")
        if not prompt_ids or len(prompt_ids) % self.group:
            raise GenerationCheckError(f"{len(prompt_ids)} prompt copies is not whole groups of {self.group}")
        for start in range(0, len(prompt_ids), self.group):
            block = [tuple(p) for p in prompt_ids[start:start + self.group]]
            if len(set(block)) != 1:
                raise GenerationCheckError(f"a group of {self.group} answers mixes different prompts")
            if block[0] not in self.expected:
                raise GenerationCheckError("generate got a prompt that is not one of the pool's prompts")
            self._count(self.groups, self.expected[block[0]])
        self.answers += len(completion_ids)

    def check_counts(self, steps_done, rewarded):
        """After each step: one checked generation per step, and as many rewarded answers as written ones."""
        if any(n != steps_done for n in self.calls.values()):
            raise GenerationCheckError(f"after {steps_done} steps the checks saw {self.calls}; a call went unseen")
        if self.answers != steps_done * pairs_per_step() * self.group or rewarded != self.answers:
            raise GenerationCheckError(f"after {steps_done} steps: {self.answers} answers written, {rewarded} rewarded")

    def summary(self):
        return {"calls": dict(self.calls), "answers": self.answers,
                "configs": [{"settings": k, "calls": n} for k, n in self.configs.items()],
                "logits_processors": [{"processors": k, "calls": n} for k, n in self.processor_sets.items()],
                "stopping_criteria": [{"criteria": k, "calls": n} for k, n in self.stopping_sets.items()],
                "groups_per_pool_pair": dict(sorted(self.groups.items())),
                "distinct_pool_prompts_seen": len(self.groups)}


def wrap_generation(mixin, watch):
    """Put the watch into `generate` and the three steps it sets itself up with, on `mixin`
    (transformers' GenerationMixin on the GPU, a fake in the tests). Returns a function that takes it out."""
    import functools

    names = ("generate", "_prepare_generation_config", "_get_logits_processor", "_get_stopping_criteria")
    originals = {name: getattr(mixin, name) for name in names}

    @functools.wraps(originals["generate"])
    def checked_generate(self, *args, **kwargs):
        ids = kwargs["input_ids"] if "input_ids" in kwargs else (args[0] if args else kwargs.get("inputs"))
        watch.entry(ids, kwargs.get("attention_mask"))
        return originals["generate"](self, *args, **kwargs)

    def checked_prepare(self, generation_config, **kwargs):
        config, model_kwargs = originals["_prepare_generation_config"](self, generation_config, **kwargs)
        watch.config({k: getattr(config, k, None) for k in (
            "do_sample", "num_beams", "temperature", "top_p", "top_k", "min_p", "typical_p", "top_h",
            "repetition_penalty", "max_new_tokens", "num_return_sequences")})
        return config, model_kwargs

    def checked_processors(self, generation_config, *args, **kwargs):
        processors = originals["_get_logits_processor"](self, generation_config, *args, **kwargs)
        watch.processors([type(p).__name__ for p in processors])
        return processors

    def checked_stopping(self, generation_config, *args, **kwargs):
        criteria = originals["_get_stopping_criteria"](self, generation_config, *args, **kwargs)
        watch.stopping([stop_rule(c) for c in criteria])
        return criteria

    for name, function in zip(names, (checked_generate, checked_prepare, checked_processors, checked_stopping)):
        setattr(mixin, name, function)

    def restore():
        for name, function in originals.items():
            setattr(mixin, name, function)

    return restore


def wrap_trainer_turn(trainer, watch):
    """Put the watch after TRL's generation step, which holds the prompts it asked for and the answers."""
    original = trainer._generate_single_turn

    def checked_turn(prompt_ids, *args, **kwargs):
        completion_ids, logprobs = original(prompt_ids, *args, **kwargs)
        watch.trainer_call(prompt_ids, completion_ids)
        return completion_ids, logprobs

    trainer._generate_single_turn = checked_turn


def coverage_problems(groups, pool_ids, steps, each=None):
    """What is wrong with how often each pool pair got a group: `each` times exactly (a full run), else at most once."""
    problems = []
    unknown = set(groups) - set(pool_ids)
    if unknown:
        problems.append(f"{len(unknown)} groups for pairs not in the pool")
    if sum(groups.values()) != steps * pairs_per_step():
        problems.append(f"{sum(groups.values())} groups in {steps} steps, not {steps * pairs_per_step()}")
    if each is not None:
        off = [i for i in pool_ids if groups.get(i, 0) != each]
        if off:
            problems.append(f"{len(off)} pool pairs did not get exactly {each} groups (for example {off[0]})")
    elif any(n > 1 for n in groups.values()):
        problems.append("a pair got more than one group in a run shorter than one epoch")
    return problems


def valid_items(files):
    """One item per validation pair and version, for every validation file: id "<file>/<version>/<pair id>"."""
    from judge.prompt import build_messages
    from judge.score import VERSIONS, derive

    return [{"id": f"{name}/{version}/{shown['id']}", "messages": build_messages(shown)}
            for name, pairs in files.items() for version in VERSIONS for shown in derive(pairs, version)]


def counts(pairs, answers):
    """Exact counts for one version: caught, said (contradiction answers), real contradictions, invalid,
    and style_a_breaks: answers the training reward's reader rejects (invalid, or a missing or wrong
    type or reason), so not in v1's exact format (D-040, D-057)."""
    from judge.reward import read_answer
    from judge.score import score

    _, rows = score(pairs, answers)
    return {
        "caught": sum(r["label"] == "contradiction" and r["gold_label"] == "contradiction" for r in rows),
        "said": sum(r["label"] == "contradiction" for r in rows),
        "real": sum(r["gold_label"] == "contradiction" for r in rows),
        "invalid": sum(bool(r["problem"]) for r in rows),
        "style_a_breaks": sum(read_answer(r["raw"])[0] is None for r in rows),
    }


def precision(c):
    """Exact precision; a candidate that never says contradiction gets 0 (the strict reading)."""
    return Fraction(c["caught"], c["said"]) if c["said"] else Fraction(0)


def recall(c):
    return Fraction(c["caught"], c["real"]) if c["real"] else Fraction(0)


def worst(by_version, rate):
    return min(rate(c) for c in by_version.values())


def choose_checkpoint(candidates):
    """D-056 with D-057's Style A gate, coded before any training result.

    Returns (chosen name or None, {name: [reasons it is out]}). `candidates` maps a name to
    {"step": n, "valid_hard": {version: counts}, "m1_valid": {version: counts}}, where counts come
    from `counts`. "step-0" is v1. All comparisons use exact fractions.
    """
    base = candidates["step-0"]
    p0 = worst(base["valid_hard"], precision)
    m1_p0, m1_r0 = worst(base["m1_valid"], precision), worst(base["m1_valid"], recall)
    out = {}
    for name, c in candidates.items():
        reasons = []
        every = [v for f in ("valid_hard", "m1_valid") for v in c[f].values()]
        invalid = sum(v["invalid"] for v in every)
        breaks = sum(v["style_a_breaks"] for v in every) - invalid
        if invalid:
            reasons.append(f"{invalid} invalid answers")
        if breaks:
            reasons.append(f"{breaks} answers not in v1's exact format (type or reason)")
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
        "allowed_processors": list(ALLOWED_PROCESSORS),
        "allowed_stopping": list(ALLOWED_STOPPING),
    })


def check_finished_run(meta):
    """Stop unless the record is a finished full training run whose own checks passed."""
    problems = []
    total = planned_steps(POOL_ROWS)
    if meta.get("stage") != "train":
        problems.append(f"the record is a {meta.get('stage')!r} run, not a full training run")
    if meta.get("pool_rows") != POOL_ROWS:
        problems.append(f"the record trained on {meta.get('pool_rows')} rows, not {POOL_ROWS}")
    if not (meta.get("planned_steps") == meta.get("global_step") == total):
        problems.append(f"the run planned {meta.get('planned_steps')} and did {meta.get('global_step')} steps, "
                        f"not {total}")
    saved = meta.get("saved") or []
    if [s.get("name") for s in saved] != full_run_checkpoints():
        problems.append(f"saved checkpoints are {[s.get('name') for s in saved]}, not {full_run_checkpoints()}")
    if any(not s.get("files") or "adapter_model.safetensors" not in s["files"] for s in saved):
        problems.append("a saved checkpoint has no file fingerprints")
    if not set(((meta.get("adapter_check") or {}).get("v1_files") or {})) >= set(V1_FILES):
        problems.append("the record has no fingerprints of v1's files for step 0")
    if len(meta.get("steps") or []) != total:
        problems.append(f"the step log has {len(meta.get('steps') or [])} steps, not {total}")
    if ((meta.get("sampling_used") or {}).get("calls") or {}).get("trainer") != total:
        problems.append("the record does not show one checked generation per step")
    checks = meta.get("run_checks") or {}
    if checks.get("passed") is not True:
        problems.append(f"the run's own checks did not pass: {checks.get('problems')}")
    if problems:
        raise SystemExit("STOP: train.meta.json is not a finished, checked full run; nothing ran.\n  "
                         + "\n  ".join(problems))


def check_record(meta, pool_sha, valid_shas):
    """Stop unless the pool, validation files, prompt, reward (recomputed now) and settings match the record."""
    from judge.prompt import prompt_sha256

    problems = []
    try:
        check_reward()
    except SystemExit as stop:
        problems.append(str(stop))
    if meta.get("pool_sha256") != pool_sha:
        problems.append(f"pool SHA-256 is {pool_sha}, the record says {meta.get('pool_sha256')}")
    if meta.get("valid_kept_apart") != valid_shas:
        problems.append(f"the validation files are {valid_shas}, the record says {meta.get('valid_kept_apart')}")
    if meta.get("prompt_sha256") != prompt_sha256():
        problems.append("the prompt changed since training")
    if meta.get("reward_sha256") != REWARD_SHA256:
        problems.append("the record's reward fingerprint differs from the recorded one")
    for key, value in run_settings().items():
        if meta.get(key) != value:
            problems.append(f"{key} is {value!r} in the script, {meta.get(key)!r} in the record")
    if meta.get("choice") and meta["choice"].get("valid_sha256") != valid_shas:
        problems.append("a validation file changed since the choice")
    if problems:
        raise SystemExit("STOP: the record does not match what this run would use; nothing ran.\n  "
                         + "\n  ".join(problems))


def check_outcomes(outcomes, expected_files, item_ids):
    """Stop unless every planned candidate answered exactly once, every item exactly once, from the
    fingerprinted files. Called before anything is scored or written."""
    names = [o.get("checkpoint") for o in outcomes]
    if sorted(names) != sorted(expected_files):
        raise SystemExit(f"STOP: answers came back for {names}, not for {list(expected_files)}; nothing written")
    for outcome in outcomes:
        ids = [line.get("id") for line in outcome.get("results") or []]
        if len(ids) != len(item_ids) or set(ids) != set(item_ids):
            raise SystemExit(f"STOP: {outcome['checkpoint']} answered {len(ids)} items ({len(set(ids))} distinct), "
                             f"not each of the {len(item_ids)} once; nothing written")
        if outcome.get("files") != expected_files[outcome["checkpoint"]]:
            raise SystemExit(f"STOP: {outcome['checkpoint']} answered from files that differ from its saved "
                             "fingerprints; nothing written")


def split_answers(results):
    """One candidate's answers by validation file and version: {file: {version: [{"id", "raw"}]}}."""
    from judge.score import VERSIONS

    answers = {f: {v: [] for v in VERSIONS} for f in VALID_FILES}
    for line in results:
        f, version, pair_id = line["id"].split("/", 2)
        answers[f][version].append({"id": pair_id, "raw": line["raw"]})
    return answers


def score_choice(answers_by, valid, steps):
    """Counts, scores and the D-056 choice from every candidate's answers."""
    from judge.score import VERSIONS, score_versions

    candidates, scores = {}, {}
    for n, answers in answers_by.items():
        candidates[n] = {"step": steps[n],
                         **{f: {v: counts(valid[f], answers[f][v]) for v in VERSIONS} for f in VALID_FILES}}
        scores[n] = {f: score_versions(valid[f], answers[f]) for f in VALID_FILES}
    chosen, out = choose_checkpoint(candidates)
    return {"rule": CHOICE_RULE, "chosen": chosen, "out": out, "counts": candidates, "valid_scores": scores}


def answer_file(name, f, version):
    return f"{name}.{f}.{version}.jsonl"


def items_sha256(item_ids):
    import hashlib

    return hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()


def publish_choice(out, valid_dir, answers_by, choice, run, expected_files, item_ids):
    """Write every answer file and the whole choice (choice.json) into a scratch folder, then give it the name
    `valid/` in one move. So `valid/` either does not exist or holds a finished choice that a retry can check."""
    import json
    import os
    import tempfile
    from pathlib import Path

    scratch = Path(tempfile.mkdtemp(prefix=".valid-", dir=out))
    shas = {}
    for n, answers in answers_by.items():
        for f, by_version in answers.items():
            for version, lines in by_version.items():
                name = answer_file(n, f, version)
                with open(scratch / name, "x", encoding="utf-8") as handle:
                    for line in lines:
                        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                shas[name] = file_sha256(scratch / name)
    write_new_json(scratch / "choice.json", {"run": run, "expected_files": expected_files,
                                             "reward_sha256": REWARD_SHA256, "items_sha256": items_sha256(item_ids),
                                             "answer_files": shas, "choice": choice})
    os.rename(scratch, valid_dir)


def saved_choice(valid_dir, run, expected_files, valid, valid_shas, item_ids, steps):
    """The choice an earlier choose saved in `valid/`, checked again from its answer files, or a STOP.

    It must be this run's, for the same checkpoint fingerprints, validation files, items and reward; every
    answer file must keep its bytes; and the answers must still give the saved counts, scores and choice."""
    import json

    from judge.score import VERSIONS

    keep = "it is never overwritten. Move it aside to answer again (that pays for the answers again)."
    path = valid_dir / "choice.json"
    if not path.is_file():
        raise SystemExit(f"STOP: {valid_dir} exists without choice.json; {keep}")
    note = json.loads(path.read_text(encoding="utf-8"))
    choice = note.get("choice") or {}
    wanted_files = sorted(answer_file(n, f, v) for n in expected_files for f in VALID_FILES for v in VERSIONS)
    problems = []
    if note.get("run") != run:
        problems.append(f"it is from run {note.get('run')}, not {run}")
    if note.get("expected_files") != expected_files:
        problems.append("its checkpoint fingerprints differ from the training record's")
    if choice.get("valid_sha256") != valid_shas:
        problems.append("the validation files changed since it was saved")
    if note.get("reward_sha256") != REWARD_SHA256:
        problems.append("it was saved with another reward")
    if note.get("items_sha256") != items_sha256(item_ids):
        problems.append("it answered other items")
    if sorted(note.get("answer_files") or {}) != wanted_files:
        problems.append("it does not list one answer file per candidate, file and version")
    if sorted(p.name for p in valid_dir.iterdir()) != sorted([*wanted_files, "choice.json"]):
        problems.append("the folder holds other files than the ones it lists")
    if not problems:
        problems += [f"{name} changed since it was saved" for name, sha in note["answer_files"].items()
                     if file_sha256(valid_dir / name) != sha]
    if problems:
        raise SystemExit(f"STOP: {valid_dir} does not hold a finished choice of this run; {keep}\n  "
                         + "\n  ".join(problems))
    answers_by = {}
    for n in expected_files:
        answers_by[n] = {f: {v: [json.loads(line) for line in
                                 (valid_dir / answer_file(n, f, v)).read_text(encoding="utf-8").splitlines()]
                             for v in VERSIONS} for f in VALID_FILES}
        ids = [f"{f}/{v}/{a['id']}" for f in VALID_FILES for v in VERSIONS for a in answers_by[n][f][v]]
        if sorted(ids) != sorted(item_ids):
            raise SystemExit(f"STOP: {valid_dir} does not hold each item once for {n}; {keep}")
    again = json.loads(json.dumps(score_choice(answers_by, valid, steps)))
    if again != {k: choice.get(k) for k in again}:
        raise SystemExit(f"STOP: the saved answers in {valid_dir} no longer give the saved choice; {keep}")
    return {**choice, "recovered_from": "valid/choice.json"}


# ---------- GPU side ----------

@app.function(image=image, gpu="L4", timeout=3 * 3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def train(run, rows, max_steps, save_every, each_pair_groups):
    """Load v1 as trainable, train with GRPO, save fingerprinted adapters. Returns a plain record."""
    import hashlib
    import os
    import time

    start = time.time()
    volume.reload()
    run_dir = f"{RUNS}/{run}"
    os.makedirs(run_dir, exist_ok=False)  # a fresh folder: a run never writes where another one did
    volume.commit()

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

    # 1. The answer-writing checks go in before the model loads, so Unsloth's own copy of generate is the checked one.
    watch = GenerationWatch(sampling_settings(), {}, GRPO["num_generations"])
    restore_generation = wrap_generation(GenerationMixin, watch)

    # 2. The start point: the pinned base, then v1's adapter at its pinned commit, trainable,
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
    v1_paths = {name: hf_hub_download(V1_ADAPTER_REPO, name, revision=V1_ADAPTER_REVISION) for name in V1_FILES}
    saved_v1 = load_file(v1_paths["adapter_model.safetensors"])
    trainable = {n.replace(".default.", "."): p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(saved_v1):
        raise RuntimeError(f"the trainable weights are not exactly v1's: {len(set(trainable) - set(saved_v1))} extra, "
                           f"{len(set(saved_v1) - set(trainable))} missing")
    loaded = get_peft_model_state_dict(model)
    if set(saved_v1) != set(loaded):
        raise RuntimeError("the loaded adapter's weight names differ from v1's file")
    largest_diff = max(float((loaded[k].float().cpu() - saved_v1[k].float()).abs().max()) for k in saved_v1)
    if largest_diff != 0.0:
        raise RuntimeError(f"the loaded adapter differs from v1's file (largest difference {largest_diff})")
    v1_files = {}
    for name, path in v1_paths.items():
        with open(path, "rb") as handle:
            v1_files[name] = hashlib.sha256(handle.read()).hexdigest()
    adapter_check = {"lora": lora_found, "trainable_tensors": len(trainable),
                     "trainable_parameters": sum(trainable.values()),
                     "trainable_names_equal_v1_file": True, "weights_equal_v1_file": True,
                     "tensors_compared": len(saved_v1), "v1_files": v1_files}

    # 3. The prompts the pool sampled, as token ids built the way judge.generate builds them.
    for row in rows:
        text = tokenizer.apply_chat_template(row["prompt"], **TEMPLATE_BASE)
        watch.expected[tuple(tokenizer(text, add_special_tokens=False)["input_ids"])] = row["id"]

    # 4. The trainer.
    total = planned_steps(len(rows), max_steps)
    points = save_points(total, save_every)
    settings = {k: v for k, v in GRPO.items() if k != "num_train_epochs"}
    config = GRPOConfig(output_dir=f"{run_dir}/trainer", max_steps=total, warmup_steps=warmup_steps(total),
                        seed=SEED, **settings)
    reward = GRPOReward()
    steps, warnings_seen, saved = [], [], []
    log_file = open(f"{run_dir}/run.log", "x", encoding="utf-8")
    clock = {"started": time.time()}

    def write(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    class Watch(TrainerCallback):
        """Check each step, keep its metrics, and write a warning line when the reward-hacking watch fires."""

        def on_step_end(self, args, state, control, **kwargs):
            watch.check_counts(state.global_step, sum(reward.total_counts.values()))
            if state.global_step > 1 and (not steps or steps[-1]["step"] != state.global_step - 1):
                raise GenerationCheckError(f"step {state.global_step - 1} was never logged")
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not any(k in logs for k in REQUIRED_STEP_KEYS):
                return  # the trainer's end-of-run summary, which holds no step metrics
            entry = {"step": state.global_step, "seconds": round(time.time() - clock["started"], 1),
                     **check_step_log(logs)}
            steps.append(entry)
            shares = " ".join(f"{k}={entry['answers/' + k]:.2f}" for k in ANSWER_KINDS)
            write(f"step {entry['step']}: reward {entry['reward']:.3f} zero_std {entry['frac_reward_zero_std']:.2f} "
                  f"length {entry['completions/mean_length']:.1f} {shares}")
            for line in watch_warnings(steps):
                warnings_seen.append(line)
                write(line)

    class SaveCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step in points and not any(s["step"] == state.global_step for s in saved):
                name = f"step-{state.global_step}"
                os.makedirs(f"{run_dir}/{name}", exist_ok=False)
                model.save_pretrained(f"{run_dir}/{name}")
                tokenizer.save_pretrained(f"{run_dir}/{name}")
                saved.append({"name": name, "step": state.global_step, "files": files_sha256(f"{run_dir}/{name}"),
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
    wrap_trainer_turn(trainer, watch)
    resolved = {k: getattr(trainer.args, k, None) for k in (
        *settings, "max_steps", "warmup_steps", "seed", "bf16", "fp16", "generation_batch_size",
        "steps_per_generation", "gradient_checkpointing", "chat_template_kwargs", "generation_kwargs", "delta",
        "importance_sampling_level", "vllm_importance_sampling_correction", "top_entropy_quantile",
        "router_aux_loss_coef", "disable_dropout")}
    decode_path = ("parse_response" if getattr(trainer._tokenizer, "response_template", None) is not None
                   or getattr(trainer._tokenizer, "response_schema", None) is not None
                   else "batch_decode(skip_special_tokens=True)")
    write(f"run {run}: loaded in {time.time() - start:.0f} s; {total} steps, saving at {points}; settings {resolved}")
    clock["started"] = time.time()
    result = trainer.train()
    trained = time.time()
    restore_generation()
    log_file.close()

    # 5. After the run: does everything add up?
    problems = []
    try:
        watch.check_counts(trainer.state.global_step, sum(reward.total_counts.values()))
    except GenerationCheckError as error:
        problems.append(str(error))
    if trainer.state.global_step != total:
        problems.append(f"the run stopped at step {trainer.state.global_step}, not {total}")
    if len(steps) != trainer.state.global_step:
        problems.append(f"{len(steps)} steps logged, not {trainer.state.global_step}")
    if [s["step"] for s in saved] != points:
        problems.append(f"saved at {[s['step'] for s in saved]}, not {points}")
    problems += coverage_problems(watch.groups, [r["id"] for r in rows], trainer.state.global_step, each_pair_groups)
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
        "sampling_used": watch.summary(),
        "run_checks": {"passed": not problems, "problems": problems, "each_pair_groups": each_pair_groups},
        "steps": steps,
        "warnings": warnings_seen,
        "reward_answer_counts": reward.total_counts,
        "train_metrics": result.metrics,
        "load_seconds": round(clock["started"] - start, 1),
        "train_seconds": round(trained - clock["started"], 1),
        "first_step_seconds": steps[0]["seconds"] if steps else None,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "versions": {"unsloth": unsloth.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
                     "peft": peft.__version__, "torch": torch.__version__, "datasets": datasets.__version__},
    })


def hf_file_sha256(name):
    """SHA-256 of one of v1's files, at its pinned commit (downloaded to the cache on the GPU machine)."""
    import hashlib

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(V1_ADAPTER_REPO, name, revision=V1_ADAPTER_REVISION), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@app.function(image=image, gpu="L4", timeout=3600, secrets=[modal.Secret.from_name("huggingface")],
              volumes={RUNS: volume})
def answer(run, checkpoint, items, expected_files):
    """Greedy answers from one candidate, after checking its files against their recorded fingerprints.

    "step-0" is v1 from Hugging Face at its pinned commit; any other name is a saved checkpoint of `run`.
    """
    import time

    start = time.time()
    import torch

    from judge.generate import answer_items
    from judge.load import load_judge

    if checkpoint == "step-0":
        files = {n: hf_file_sha256(n) for n in V1_FILES}
    else:
        volume.reload()
        files = files_sha256(f"{RUNS}/{run}/{checkpoint}")
    if files != expected_files:
        raise RuntimeError(f"{run}/{checkpoint}: files differ from the recorded fingerprints; nothing answered")
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
        "files": files,
        "loaded_model": model.base_model.config._name_or_path,
        "results": results,
        "load_seconds": round(loaded - start, 1),
        "answer_seconds": round(time.time() - loaded, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        **settings,
    }


UPLOAD_ALSO_IN_REPO = (".gitattributes",)  # Hugging Face puts this file into every new repo by itself


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_wanted(meta_text, expected_files):
    """The exact files the repo must hold, by SHA-256: the checked adapter folder and the run and choice record."""
    import hashlib

    return {**expected_files, "train.meta.json": hashlib.sha256(meta_text.encode("utf-8")).hexdigest()}


def repo_files(api, fetch, revision):
    """{file: SHA-256} of every file in HF_REPO at one commit, but the one Hugging Face adds by itself.
    `fetch(name, revision)` downloads one file at that commit and returns its local path."""
    names = api.list_repo_files(HF_REPO, revision=revision, repo_type="model")
    return {n: file_sha256(fetch(n, revision)) for n in sorted(names) if n not in UPLOAD_ALSO_IN_REPO}


def upload_checked(api, fetch, folder_in, run, checkpoint, meta_text, expected_files, note, save_note, copy=None):
    """What the upload job does, with Hugging Face (`api`, `fetch`) and its note on the volume passed in.

    `note` is what an earlier try of this run's upload saved (or None); `save_note` saves a new one.
    - No repo yet: save a note that this try reserves the repo, make it private, add its first commit to
      the note, then upload a checked copy.
    - The repo holds exactly the wanted files (adapter folder and record) at its newest commit: an earlier
      try finished and only the local record is missing. Recorded as recovered, at that commit.
    - The repo is empty, and the note shows this upload made it and nothing came after: an earlier try
      stopped after making it. The upload resumes into it.
    - Anything else stops, and nothing is uploaded.
    After an upload, the commit it returned must hold exactly the wanted files.
    """
    import shutil
    import tempfile
    from pathlib import Path

    if files_sha256(folder_in) != expected_files:
        raise SystemExit(f"STOP: {run}/{checkpoint} files differ from the recorded fingerprints; nothing uploaded")
    wanted = upload_wanted(meta_text, expected_files)
    mine = {"repo": HF_REPO, "run": run, "checkpoint": checkpoint, "files": wanted}
    resumed = False
    if api.repo_exists(HF_REPO):
        info = api.model_info(HF_REPO)
        if not info.private:
            raise SystemExit(f"STOP: {HF_REPO} exists and is public; nothing uploaded")
        held = repo_files(api, fetch, info.sha)
        if held == wanted:
            return {"repo": HF_REPO, "commit": info.sha, "private": True, "files": sorted(wanted),
                    "recovered": True, "resumed": False}
        if held or note != {**mine, "first_commit": info.sha}:
            unproven = not held and note == {**mine, "first_commit": None}
            raise SystemExit(
                f"STOP: {HF_REPO} already exists; nothing uploaded. "
                + ("It is empty and this run's note shows it was being made, but its first commit was never noted, "
                   "so this script cannot prove it made it. Delete the empty repo on Hugging Face, then retry."
                   if unproven else
                   "It does not hold exactly this adapter and record at its newest commit, and it is not an empty "
                   "repo this upload made."))
        resumed = True
    else:
        save_note({**mine, "first_commit": None})
        api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=False)
        save_note({**mine, "first_commit": api.model_info(HF_REPO).sha})
    with tempfile.TemporaryDirectory() as folder:
        (copy or shutil.copytree)(folder_in, folder, dirs_exist_ok=True)
        (Path(folder) / "train.meta.json").write_text(meta_text, encoding="utf-8")
        if files_sha256(folder) != wanted:
            raise SystemExit("STOP: the copy made for the upload differs from the checked files; nothing uploaded")
        commit = api.upload_folder(folder_path=folder, repo_id=HF_REPO, repo_type="model",
                                   commit_message=f"M2 Step 5: GRPO LoRA adapter {run}/{checkpoint}")
    if repo_files(api, fetch, commit.oid) != wanted:
        raise SystemExit(f"STOP: commit {commit.oid} of {HF_REPO} does not hold exactly the files sent")
    if not api.model_info(HF_REPO).private:
        raise SystemExit(f"STOP: {HF_REPO} is public after the upload; make it private now")
    return {"repo": HF_REPO, "commit": commit.oid, "commit_url": commit.commit_url, "private": True,
            "files": sorted(wanted), "recovered": False, "resumed": resumed}


@app.function(image=image, timeout=900, secrets=[modal.Secret.from_name("huggingface")], volumes={RUNS: volume})
def upload(run, checkpoint, meta_text, expected_files):
    """Upload one checked adapter folder plus train.meta.json to a new private repo (`upload_checked`).
    Token from HF_TOKEN. The upload's note lives next to the run on the volume: <run>/hf-upload.json."""
    import json
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download

    volume.reload()
    note_path = Path(f"{RUNS}/{run}/hf-upload.json")
    note = json.loads(note_path.read_text(encoding="utf-8")) if note_path.exists() else None

    def save_note(data):
        replace_json(note_path, data)
        volume.commit()

    with tempfile.TemporaryDirectory() as cache:
        def fetch(name, revision):
            return hf_hub_download(HF_REPO, name, revision=revision, repo_type="model", cache_dir=cache)

        return upload_checked(HfApi(), fetch, f"{RUNS}/{run}/{checkpoint}", run, checkpoint, meta_text,
                              expected_files, note, save_note)


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


def run_stage(stage, calls, out=OUT, state_path="STATE.md", train_meta_path=TRAIN_META):
    import json
    import time
    from datetime import datetime
    from pathlib import Path

    from judge.prompt import prompt_sha256
    from judge.score import format_problem, parse_answer

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
        if len(rows) != POOL_ROWS:
            raise SystemExit(f"STOP: the pool has {len(rows)} rows, not {POOL_ROWS}; nothing ran")
        run = new_run_id(stage)
        max_steps = TRIAL["max_steps"] if trial else 0
        save_every = TRIAL["save_every"] if trial else SAVE_EVERY
        each = None if trial else GRPO["num_train_epochs"]
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start = time.time()
        result = calls.train(run, rows, max_steps, save_every, each)
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
            "save_every_used": save_every,
            "modal_app_id": calls.app_id(),
            "wall_seconds": wall,
            **result,
        }
        gpu = {"train": round(result["load_seconds"] + result["train_seconds"], 1)}
        if trial:
            # Prove the last checkpoint loads and answers in v1's format, on 8 valid-hard items.
            last = result["saved"][-1]
            items = valid_items({"valid_hard": valid["valid_hard"]})[:TRIAL["check_items"]]
            check = calls.answer(run, last["name"], items, last["files"])
            lines = []
            for line in check.pop("results"):
                parsed, problem, _ = parse_answer(line["raw"])
                lines.append({**line, "problem": problem,
                              "format_problem": format_problem(parsed) if parsed else None})
            meta["load_check"] = {"checkpoint": last["name"], "answers": lines, **check}
            gpu["load_check"] = round(check["load_seconds"] + check["answer_seconds"], 1)
        meta["gpu_seconds"] = gpu
        meta["cost_estimate_gpu_only"] = round(sum(gpu.values()) * L4_PER_SECOND, 2)
        write_new_json(path, meta)
        print(f"{run}: {result['global_step']} steps in {result['train_seconds']} s, load {result['load_seconds']} s, "
              f"peak VRAM {result['peak_vram_gib']} GiB, wall {wall / 60:.1f} min; saved "
              f"{[s['name'] for s in result['saved']]}; warnings {len(result['warnings'])}; run checks "
              f"{'passed' if result['run_checks']['passed'] else 'FAILED: ' + str(result['run_checks']['problems'])}")
        return meta

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_finished_run(meta)
    pool, pool_sha = read_pool()
    valid, valid_shas = read_valid()
    check_record(meta, pool_sha, valid_shas)
    expected_files = {"step-0": {n: meta["adapter_check"]["v1_files"][n] for n in V1_FILES},
                      **{s["name"]: s["files"] for s in meta["saved"]}}

    if stage == "choose":
        if meta.get("choice"):
            raise SystemExit("STOP: train.meta.json already holds a choice; nothing ran")
        valid_dir = out / "valid"
        items = valid_items(valid)
        item_ids = [i["id"] for i in items]
        names = list(expected_files)
        steps = {"step-0": 0, **{s["name"]: s["step"] for s in meta["saved"]}}
        if valid_dir.exists():
            # An earlier choose saved every answer and its choice, then stopped before the record was written.
            choice = saved_choice(valid_dir, meta["run"], expected_files, valid, valid_shas, item_ids, steps)
            print("valid/ holds a finished, checked choice from an earlier try; no candidate answered again")
        else:
            start = time.time()
            outcomes = calls.answer_many([(meta["run"], n, items, expected_files[n]) for n in names])
            wall = round(time.time() - start, 1)
            check_outcomes(outcomes, expected_files, item_ids)
            answers_by = {o["checkpoint"]: split_answers(o.pop("results")) for o in outcomes}
            runs = {o["checkpoint"]: o for o in outcomes}
            choice = {**score_choice(answers_by, valid, steps), "answer_runs": runs, "valid_sha256": valid_shas,
                      "gpu_seconds": round(sum(r["load_seconds"] + r["answer_seconds"] for r in runs.values()), 1),
                      "wall_seconds": wall, "modal_app_id": calls.app_id()}
            publish_choice(out, valid_dir, answers_by, choice, meta["run"], expected_files, item_ids)
        meta["choice"] = choice
        replace_json(meta_path, meta)
        chosen, out_reasons, scores = choice["chosen"], choice["out"], choice["valid_scores"]
        for n in names:
            cells = ", ".join(f"{f} {v} P {m['precision'] or 0:.2f} R {m['recall'] or 0:.2f} inv {m['invalid']}"
                              for f in VALID_FILES for v, m in scores[n][f].items())
            print(f"{n}: {cells}{'  OUT: ' + '; '.join(out_reasons[n]) if n in out_reasons else ''}")
        print(f"chosen on validation (D-056): {chosen}"
              + ("  (step 0: RL did not help on validation; the hard test set is not run)" if chosen == "step-0" else ""))
        return meta

    choice = meta.get("choice") or {}
    chosen = choice.get("chosen")
    if not chosen or chosen == "step-0" or chosen not in expected_files:
        raise SystemExit(f"STOP: nothing to upload (chosen: {chosen}); nothing ran")
    if set(choice.get("counts") or {}) != set(expected_files) or choose_checkpoint(choice["counts"])[0] != chosen:
        raise SystemExit("STOP: the recorded counts no longer pick the recorded choice by D-056; nothing ran")
    if meta.get("upload"):
        raise SystemExit("STOP: train.meta.json already records an upload; nothing ran")
    result = calls.upload(meta["run"], chosen, json.dumps(meta, indent=2) + "\n", expected_files[chosen])
    meta["upload"] = {"checkpoint": chosen, "run": meta["run"], "modal_app_id": calls.app_id(), **result}
    replace_json(meta_path, meta)
    print(f"uploaded {chosen} to https://huggingface.co/{result['repo']} (private: {result['private']}), "
          f"commit {result['commit']}{' (recovered an earlier finished upload)' if result.get('recovered') else ''}"
          f"{' (resumed into the empty repo an earlier try made)' if result.get('resumed') else ''}")
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
    """Add the billed cost of a run's Modal apps (train or trial, choose, upload) to its record (no GPU).

    Modal reports full hours only, so run this after the hour the last stage ended in has passed.
    """
    import json
    import subprocess
    from pathlib import Path

    path = Path(out) / f"{name}.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    stages = {"train": meta.get("modal_app_id"), "choose": (meta.get("choice") or {}).get("modal_app_id"),
              "upload": (meta.get("upload") or {}).get("modal_app_id")}
    stages = {k: v for k, v in stages.items() if v}
    command = [sys.executable, "-m", "modal", "billing", "report", "--start", meta["started_at"][:10],
               "-r", "h", "--tz", "local", "--show-resources", "--json"]
    text = run_report(command) if run_report else subprocess.run(command, check=True, capture_output=True,
                                                                  text=True).stdout
    found = billed(json.loads(text), list(stages.values()))
    missing = [i for i in stages.values() if i not in found]
    if missing:
        raise SystemExit(f"STOP: no billing rows yet for {missing}; try again after the hour has passed")
    meta["billed_usd"] = {stage: found[app_id] for stage, app_id in stages.items()}
    meta["billed_total_usd"] = round(sum(meta["billed_usd"].values()), 4)
    replace_json(path, meta)
    print(f"{name}: billed {meta['billed_usd']} (total ${meta['billed_total_usd']})")


@app.local_entrypoint()
def main(stage: str = "trial"):
    run_stage(stage, modal_calls())


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "bill":
    bill(sys.argv[2])
