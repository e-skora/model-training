"""Tests for scripts/train_grpo.py (M2 Step 5): data, fingerprints, sampling, the watch and the D-056 rule.

No GPU, no network, no Modal job: every remote call is a fake that records
what it was asked. Results are written only to a temporary folder; the real
repo is only read. tests/conftest.py points Modal at an empty temporary config
before this file imports it. The frozen test files are never opened, and one
test proves the script does not open them either.
"""

import builtins
import importlib.util
import io
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import judge.reward
from judge.format import answer_text
from judge.generate import SAMPLING
from judge.pool import SAMPLES, candidates, show
from judge.score import VERSIONS
from judge.train_data import read_pairs

REPO = Path(__file__).resolve().parents[1]
FROZEN = {(REPO / "data/m1/test.jsonl").resolve(), (REPO / "data/m2/test-hard.jsonl").resolve()}


def load_script():
    spec = importlib.util.spec_from_file_location("train_grpo_script", REPO / "scripts" / "train_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # builds Modal objects lazily; no network, no GPU
    return module


grpo = load_script()
V1_FILES = {"adapter_config.json": "c" * 64, "adapter_model.safetensors": "d" * 64}


def files_of(name):
    return {"adapter_config.json": f"{name}-config", "adapter_model.safetensors": f"{name}-weights"}


# ---------- fakes ----------

class Calls:
    """Fake Modal jobs. `answers` maps a checkpoint to a function (pair, version) -> raw answer; default: right.

    `train` returns a record shaped like the real one, finished and checked unless `train_changes` says otherwise.
    """

    def __init__(self, answers=None, train_changes=None, answer_files=None, drop_items=0, upload_result=None):
        self.log = []
        self.answers = answers or {}
        self.train_changes = train_changes or {}
        self.answer_files = answer_files or {}
        self.drop_items = drop_items
        self.upload_result = upload_result

    def app_id(self):
        return "ap-test"

    def train(self, run, rows, max_steps, save_every, each):
        self.log.append(("train", run, [r["id"] for r in rows], max_steps, save_every, each))
        total = grpo.planned_steps(len(rows), max_steps)
        points = grpo.save_points(total, save_every)
        result = {
            "run": run, "planned_steps": total, "global_step": total, "save_points": points,
            "saved": [{"name": f"step-{p}", "step": p, "files": files_of(f"step-{p}")} for p in points],
            "adapter_check": {"v1_files": dict(V1_FILES)},
            "sampling_used": {"calls": dict.fromkeys(("entry", "config", "processors", "stopping", "trainer"), total)},
            "run_checks": {"passed": True, "problems": [], "each_pair_groups": each},
            "steps": [{"step": i + 1} for i in range(total)], "warnings": [],
            "train_seconds": 40.0, "load_seconds": 60.0, "peak_vram_gib": 12.0,
        }
        result.update(self.train_changes)
        return result

    def answer(self, run, checkpoint, items, expected_files):
        self.log.append(("answer", run, checkpoint, [i["id"] for i in items]))
        valid, _ = grpo.read_valid()
        by_file = {f: {p["id"]: p for p in pairs} for f, pairs in valid.items()}
        results = []
        for item in items[self.drop_items:]:
            name, version, pair_id = item["id"].split("/", 2)
            pair = by_file[name][pair_id]
            rule = self.answers.get(checkpoint)
            results.append({"id": item["id"], "raw": rule(pair, version) if rule else answer_text(pair)})
        return {"checkpoint": checkpoint, "load_seconds": 1, "answer_seconds": 1, "results": results,
                "files": self.answer_files.get(checkpoint, expected_files)}

    def answer_many(self, args):
        self.log.append(("answer_many", [a[1] for a in args]))
        return [self.answer(*a) for a in args]

    def upload(self, run, checkpoint, meta_text, expected_files):
        self.log.append(("upload", run, checkpoint, expected_files))
        return self.upload_result or {"repo": "test/repo", "commit": "c1", "private": True, "files": [],
                                      "recovered": False}


@pytest.fixture
def at_repo(monkeypatch):
    """Run from the repo folder (the script reads its data by repo path); outputs go to tmp_path only."""
    monkeypatch.chdir(REPO)


def trial(tmp_path, calls=None):
    calls = calls or Calls()
    return grpo.run_stage("trial", calls, out=tmp_path), calls


def finished_run(tmp_path, **changes):
    return grpo.run_stage("train", Calls(train_changes=changes), out=tmp_path)


def label_only(pair, version):
    return json.dumps({"label": pair["label"]})


# ---------- Modal's settings stay out of the tests (finding 5) ----------

def test_modal_reads_an_empty_temporary_config_not_the_real_one():
    import modal.config

    assert Path(modal.config.user_config_path).resolve() == Path(os.environ["MODAL_CONFIG_PATH"]).resolve()
    assert Path(modal.config.user_config_path).resolve() != Path("~/.modal.toml").expanduser().resolve()
    assert Path(modal.config.user_config_path).read_text() == ""
    assert not [k for k in os.environ if k.startswith(("HF_TOKEN", "MODAL_TOKEN"))]


def test_collecting_the_tests_never_reads_a_home_modal_config(tmp_path):
    # A home folder whose .modal.toml cannot be read: if collection tried to read it, Modal's import would fail.
    home = tmp_path / "home"
    home.mkdir()
    locked = home / ".modal.toml"
    locked.write_text("[broken\n")
    locked.chmod(0)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODAL_", "HF_"))}
    env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
    try:
        done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                               "tests/test_train_grpo.py", "tests/test_run_records.py"],
                              cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    finally:
        locked.chmod(0o600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "error" not in done.stdout.lower()


# ---------- the data ----------

def test_dataset_rows_are_the_pool_rows_in_their_saved_versions():
    pool, sha = read_pairs(REPO / grpo.POOL_PATH)
    assert sha == grpo.POOL_SHA256
    rows = grpo.dataset_rows(pool)
    assert [r["id"] for r in rows] == [p["id"] for p in pool]
    assert len(rows) == grpo.POOL_ROWS == 314
    assert all(set(r) == {"prompt", "gold_label", "gold_type", "id"} for r in rows)
    assert [(r["gold_label"], r["gold_type"]) for r in rows] == [(p["label"], p["type"]) for p in pool]
    # The prompt the pool really sampled: judge.pool.show over all candidates, as scripts/sample_pool.py did.
    hard, _ = read_pairs(REPO / "data/m2/train-hard.jsonl")
    m1, _ = read_pairs(REPO / "data/m1/train.jsonl")
    shown = {s["id"]: s for s in show(candidates(hard, m1))}
    for row, pair in zip(rows, pool):
        assert row["prompt"] == shown[pair["id"]]["messages"]
        assert (shown[pair["id"]]["swapped"], shown[pair["id"]]["dates"]) == (pair["swapped"], pair["dates"])
        assert shown[pair["id"]]["version"] == pair["version"]


def test_a_row_whose_version_disagrees_with_its_flags_stops():
    pool, _ = read_pairs(REPO / grpo.POOL_PATH)
    bad = [dict(pool[0], version="no_dates" if pool[0]["version"] != "no_dates" else "as_is")]
    with pytest.raises(SystemExit, match="says"):
        grpo.dataset_rows(bad)


def test_validation_pairs_overlapping_the_pool_stop(at_repo):
    pool, _ = read_pairs(REPO / grpo.POOL_PATH)
    valid, _ = grpo.read_valid()
    grpo.check_apart(pool, valid)  # the real files are apart
    with pytest.raises(SystemExit, match="overlaps the pool"):
        grpo.check_apart(pool, {"valid_hard": [pool[3]]})
    first = valid["valid_hard"][0]
    copied = dict(first, a=dict(first["a"], text=pool[5]["b"]["text"]))
    with pytest.raises(SystemExit, match="1 memory texts"):
        grpo.check_apart(pool, {"valid_hard": [copied]})


def test_training_gets_only_pool_rows_never_validation_rows(at_repo, tmp_path):
    meta, calls = trial(tmp_path)
    pool, _ = read_pairs(REPO / grpo.POOL_PATH)
    trained = [entry for entry in calls.log if entry[0] == "train"]
    assert len(trained) == 1
    assert trained[0][2] == [p["id"] for p in pool]
    valid, _ = grpo.read_valid()
    valid_ids = {p["id"] for pairs in valid.values() for p in pairs}
    assert not valid_ids & set(trained[0][2])
    # Validation is used only to prove the trial checkpoint loads: 8 valid-hard items, answered, not trained on.
    answered = [entry for entry in calls.log if entry[0] == "answer"]
    assert len(answered) == 1 and answered[0][2] == "step-20" and len(answered[0][3]) == 8
    assert all(i.startswith("valid_hard/as_is/") for i in answered[0][3])
    assert meta["load_check"]["answers"][0]["problem"] is None


# ---------- run folders are fresh, records are whole (finding 1) ----------

def test_every_run_gets_a_fresh_name(at_repo, tmp_path):
    first = trial(tmp_path / "a")[0]["run"]
    second = trial(tmp_path / "b")[0]["run"]
    full = finished_run(tmp_path / "c")["run"]
    assert first.startswith("trial-") and second.startswith("trial-") and full.startswith("train-")
    assert len({first, second, full}) == 3


def test_files_sha256_fingerprints_each_plain_file(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text("{}")
    found = grpo.files_sha256(tmp_path)
    assert list(found) == ["adapter_config.json", "adapter_model.safetensors"]
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights!")
    assert grpo.files_sha256(tmp_path)["adapter_model.safetensors"] != found["adapter_model.safetensors"]
    (tmp_path / "link").symlink_to(tmp_path / "adapter_config.json")
    with pytest.raises(RuntimeError, match="not a plain file"):
        grpo.files_sha256(tmp_path)


def test_records_are_written_whole_and_a_new_one_never_replaces_an_old_one(tmp_path):
    path = tmp_path / "trial.meta.json"
    grpo.write_new_json(path, {"a": 1})
    with pytest.raises(SystemExit, match="already exists"):
        grpo.write_new_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
    grpo.replace_json(path, {"a": 3})
    assert json.loads(path.read_text()) == {"a": 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trial.meta.json"]  # no scratch file left behind


def test_a_saved_trial_record_is_never_replaced(at_repo, tmp_path):
    trial(tmp_path)
    calls = Calls()
    with pytest.raises(SystemExit, match="already exist"):
        trial(tmp_path, calls)
    assert calls.log == []


# ---------- fingerprints stop before anything runs ----------

def test_a_wrong_pool_fingerprint_stops_before_anything_runs(at_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(grpo, "POOL_SHA256", "0" * 64)
    calls = Calls()
    with pytest.raises(SystemExit, match="rl-pool.jsonl SHA-256"):
        trial(tmp_path, calls)
    assert calls.log == []
    assert not (tmp_path / "trial.meta.json").exists()


def test_a_changed_pool_file_stops(tmp_path):
    pool = REPO.joinpath(grpo.POOL_PATH).read_bytes()
    changed = tmp_path / "rl-pool.jsonl"
    changed.write_bytes(pool.replace(b'"contradiction"', b'"non_contradiction"', 1))
    with pytest.raises(SystemExit, match="SHA-256"):
        grpo.read_pool(changed)


def test_a_wrong_reward_fingerprint_stops_before_anything_runs(at_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(judge.reward, "reward_sha256", lambda: "f" * 64)
    calls = Calls()
    with pytest.raises(SystemExit, match="reward fingerprint"):
        trial(tmp_path, calls)
    assert calls.log == []


def test_the_reward_fingerprint_today_is_the_recorded_one():
    assert grpo.check_reward() == grpo.REWARD_SHA256


def test_a_wrong_v1_record_stops_before_anything_runs(at_repo, tmp_path):
    record = json.loads((REPO / grpo.TRAIN_META).read_text(encoding="utf-8"))
    record["upload"]["commit"] = "0" * 40
    fake = tmp_path / "train.meta.json"
    fake.write_text(json.dumps(record), encoding="utf-8")
    calls = Calls()
    with pytest.raises(SystemExit, match="epoch-2 adapter"):
        grpo.run_stage("trial", calls, out=tmp_path / "out", train_meta_path=fake)
    assert calls.log == []


def test_the_real_v1_record_names_the_pins():
    grpo.check_v1(json.loads((REPO / grpo.TRAIN_META).read_text(encoding="utf-8")))


# ---------- sampling (finding 3) ----------

def test_sampling_matches_the_pool():
    asked = grpo.sampling_settings()
    for key, value in SAMPLING.items():
        assert asked[key] == value, key  # top_k 0 means off in both TRL and transformers
    assert asked["num_return_sequences"] == SAMPLES == 8
    assert asked["min_p"] is None and asked["repetition_penalty"] == 1.0
    saved = json.loads((REPO / "results/m2/pool/v1-samples.meta.json").read_text(encoding="utf-8"))
    assert saved["generation"] == {k: asked[k] for k in saved["generation"]}
    assert saved["chat_template_options"] == {"tokenize": False, "add_generation_prompt": True}
    assert grpo.GRPO["use_vllm"] is False


def test_every_setting_unsloth_would_fill_in_is_set_here():
    # Unsloth 2026.10.7 fills these in by itself when left empty (results/m2/grpo/settings.md, Part A item 2).
    for key, value in {"loss_type": "dapo", "beta": 0.0, "mask_truncated_completions": False, "epsilon_high": 0.28,
                       "learning_rate": 5e-6, "weight_decay": 0.001, "per_device_train_batch_size": 8,
                       "gradient_accumulation_steps": 2, "optim": "adamw_8bit", "logging_steps": 1}.items():
        assert grpo.GRPO[key] == value, key


GOOD_CONFIG = {"do_sample": True, "num_beams": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": None,
               "typical_p": 1.0, "top_h": None, "repetition_penalty": 1.0, "max_new_tokens": 256,
               "num_return_sequences": 1}  # what the trial's 20 generate calls really used
PROMPT_A, PROMPT_B, PROMPT_C = (1, 2, 3), (4, 5, 6), (1, 2, 3, 4, 5)  # C is longer, so it pads A


POOL_ENDS = [151643, 151645]  # <|endoftext|> and <|im_end|>, the pinned base's generation_config.json


def new_watch():
    return grpo.GenerationWatch(grpo.sampling_settings(), {PROMPT_A: "tr-a", PROMPT_B: "tr-b", PROMPT_C: "tr-c"}, 8)


def left_padded(rows, pad=0):
    """Rows of different lengths as TRL hands them to generate: padded on the left, with their mask."""
    width = max(len(r) for r in rows)
    return ([[pad] * (width - len(r)) + list(r) for r in rows],
            [[0] * (width - len(r)) + [1] * len(r) for r in rows])


def right_padded(rows, pad=0):
    width = max(len(r) for r in rows)
    return ([list(r) + [pad] * (width - len(r)) for r in rows],
            [[1] * len(r) + [0] * (width - len(r)) for r in rows])


def good_stops(width):
    return [{"name": "MaxLengthCriteria", "max_length": width + 256},
            {"name": "EosTokenCriteria", "eos_token_id": list(POOL_ENDS)}]


def one_step(watch, prompts=(PROMPT_A, PROMPT_B), skip=()):
    """Feed the watch one generation the way the GPU job does; `skip` leaves out recording points."""
    rows = [list(p) for p in prompts for _ in range(8)]
    ids, mask = left_padded(rows)
    if "entry" not in skip:
        watch.entry(ids, mask)
    if "config" not in skip:
        watch.config(GOOD_CONFIG)
    if "processors" not in skip:
        watch.processors([])
    if "stopping" not in skip:
        watch.stopping(good_stops(len(ids[0])))
    if "trainer" not in skip:
        watch.trainer_call(rows, [[9]] * len(rows))


def test_the_watch_passes_the_pools_sampling_and_counts_groups():
    watch = new_watch()
    one_step(watch)
    watch.check_counts(1, rewarded=16)
    one_step(watch, prompts=(PROMPT_B, PROMPT_A))
    watch.check_counts(2, rewarded=32)
    assert watch.summary()["groups_per_pool_pair"] == {"tr-a": 2, "tr-b": 2}


@pytest.mark.parametrize("change", [
    {"temperature": 0.7}, {"top_p": 0.8}, {"top_k": 20}, {"min_p": 0.1}, {"repetition_penalty": 1.1},
    {"max_new_tokens": 512}, {"do_sample": False}, {"num_return_sequences": 2}, {"num_beams": 2},
])
def test_other_sampling_stops_at_once(change):
    with pytest.raises(grpo.GenerationCheckError, match="other sampling"):
        new_watch().config({**GOOD_CONFIG, **change})


def test_any_word_filter_stops():
    with pytest.raises(grpo.GenerationCheckError, match="word chances"):
        new_watch().processors(["NoBadWordsLogitsProcessor"])


def entered(rows):
    """A watch that has seen generate get `rows`, as the GPU job feeds it."""
    watch = new_watch()
    watch.entry(*left_padded(rows))
    return watch


@pytest.mark.parametrize("rules, message", [
    ([{"name": "MaxLengthCriteria", "max_length": 259}, {"name": "StopStringCriteria"}], "stop rules"),
    ([{"name": "EosTokenCriteria", "eos_token_id": POOL_ENDS}], "stop rules"),
    ([*good_stops(3), {"name": "MaxTimeCriteria"}], "stop rules"),
    ([{"name": "MaxLengthCriteria", "max_length": 3 + 512}, good_stops(3)[1]], "stops at 515 tokens"),
    ([good_stops(3)[0], {"name": "EosTokenCriteria", "eos_token_id": [151645]}], "end tokens"),
    ([good_stops(3)[0], {"name": "EosTokenCriteria", "eos_token_id": [151645, 151643, 0]}], "end tokens"),
])
def test_other_stop_rules_stop(rules, message):
    with pytest.raises(grpo.GenerationCheckError, match=message):
        entered([list(PROMPT_A)] * 8).stopping(rules)


def test_a_group_must_be_eight_copies_of_one_pool_prompt():
    mixed = [list(PROMPT_A)] * 7 + [list(PROMPT_B)] + [list(PROMPT_B)] * 8
    with pytest.raises(grpo.GenerationCheckError, match="mixes different prompts"):
        entered(mixed).trainer_call(mixed, [[9]] * 16)
    with pytest.raises(grpo.GenerationCheckError, match="not one of the pool's"):
        entered([[7, 7]] * 8).trainer_call([[7, 7]] * 8, [[9]] * 8)
    with pytest.raises(grpo.GenerationCheckError, match="whole groups"):
        entered([list(PROMPT_A)] * 12).trainer_call([list(PROMPT_A)] * 12, [[9]] * 12)
    with pytest.raises(grpo.GenerationCheckError, match="got 32 answers"):
        entered([list(PROMPT_A)] * 16).trainer_call([list(PROMPT_A)] * 16, [[9]] * 32)
    with pytest.raises(grpo.GenerationCheckError, match="longer than 256"):
        entered([list(PROMPT_A)] * 8).trainer_call([list(PROMPT_A)] * 8, [[9] * 257] * 8)


@pytest.mark.parametrize("given, message", [
    ([list(PROMPT_A)[1:]] * 8, "other tokens than TRL's prompts"),  # shortened on the way to generate
    ([list(PROMPT_A)] * 7, "other tokens than TRL's prompts"),  # a row dropped
    ([list(PROMPT_A) + [0]] * 8, "other tokens than TRL's prompts"),  # a token added
])
def test_generate_must_get_exactly_trls_prompts(given, message):
    with pytest.raises(grpo.GenerationCheckError, match=message):
        entered(given).trainer_call([list(PROMPT_A)] * 8, [[9]] * 8)


def test_the_mask_must_be_one_unbroken_run_per_row():
    rows = [list(PROMPT_A)] * 8
    ids, mask = left_padded(rows)
    holed = [[1, 0, 1]] * 8
    with pytest.raises(grpo.GenerationCheckError, match="unbroken"):
        new_watch().entry(ids, holed)
    with pytest.raises(grpo.GenerationCheckError, match="no input_ids or no attention_mask"):
        new_watch().entry(ids, None)
    with pytest.raises(grpo.GenerationCheckError, match="padded on the right"):
        new_watch().entry(*right_padded([list(PROMPT_A), list(PROMPT_C)]))
    watch = new_watch()
    watch.entry(ids, mask)
    with pytest.raises(grpo.GenerationCheckError, match="ran again"):
        watch.entry(ids, mask)


@pytest.mark.parametrize("skip, message", [
    ("entry", "prompts None wide"),
    ("config", "went unseen"), ("processors", "went unseen"), ("stopping", "went unseen"), ("trainer", "went unseen"),
])
def test_a_call_that_skips_any_check_stops_at_the_step_end(skip, message):
    watch = new_watch()
    with pytest.raises(grpo.GenerationCheckError, match=message):
        one_step(watch, skip={skip})
        watch.check_counts(1, rewarded=16)


# The real wrappers (wrap_generation, wrap_trainer_turn) around fakes shaped like transformers and TRL 1.15.0.

class MaxLengthCriteria:
    def __init__(self, max_length):
        self.max_length = max_length


class EosTokenCriteria:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def fake_mixin(stops=None, config=None):
    """A fresh stand-in for GenerationMixin: generate sets itself up the way transformers does, then adds one token."""

    class Mixin:
        def generate(self, inputs=None, generation_config=None, **kwargs):
            self.width = len(kwargs["input_ids"][0])
            settings, _ = self._prepare_generation_config(generation_config, **kwargs)
            self._get_logits_processor(settings)
            self._get_stopping_criteria(settings, None)
            return [row + [9] for row in kwargs["input_ids"]]

        def _prepare_generation_config(self, generation_config, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(**(config or GOOD_CONFIG)), kwargs

        def _get_logits_processor(self, generation_config, *args, **kwargs):
            return []

        def _get_stopping_criteria(self, generation_config, *args, **kwargs):
            return stops(self.width) if stops else [MaxLengthCriteria(self.width + 256), EosTokenCriteria([151645, 151643])]

    return Mixin


class FakeTrainer:
    """TRL 1.15.0's regular text path (grpo_trainer.py:1769-1828): left-pad, generate, cut the prompt off."""

    def __init__(self, model, change=None, calls=1, pad=left_padded):
        self.model, self.change, self.calls, self.pad = model, change, calls, pad

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations):
        rows = self.change([list(p) for p in prompt_ids]) if self.change else [list(p) for p in prompt_ids]
        ids, mask = self.pad(rows, pad=151654)
        for _ in range(self.calls):
            out = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=None)
        return [row[len(ids[0]):] for row in out], None


def run_wrapped(mixin, prompts, change=None, calls=1, pad=left_padded):
    watch = new_watch()
    restore = grpo.wrap_generation(mixin, watch)
    try:
        trainer = FakeTrainer(mixin(), change, calls, pad)
        grpo.wrap_trainer_turn(trainer, watch)
        trainer._generate_single_turn([list(p) for p in prompts for _ in range(8)], None, {}, 8)
        watch.check_counts(1, rewarded=8 * len(prompts))
    finally:
        restore()
    return watch


def test_the_real_wrappers_pass_a_correct_step_and_come_back_out():
    mixin = fake_mixin()
    before = {name: getattr(mixin, name) for name in ("generate", "_get_stopping_criteria")}
    watch = run_wrapped(mixin, (PROMPT_A, PROMPT_C))  # two prompts of different lengths, so one is padded
    assert watch.summary()["calls"] == dict.fromkeys(("entry", "config", "processors", "stopping", "trainer"), 1)
    assert {name: getattr(mixin, name) for name in before} == before


@pytest.mark.parametrize("make, message", [
    (lambda: (fake_mixin(), {"change": lambda rows: [r[1:] for r in rows]}), "other tokens than TRL's prompts"),
    (lambda: (fake_mixin(), {"change": lambda rows: rows[::-1]}), "other tokens than TRL's prompts"),
    (lambda: (fake_mixin(), {"calls": 2}), "ran again"),
    (lambda: (fake_mixin(), {"pad": right_padded}), "padded on the right"),
    (lambda: (fake_mixin(stops=lambda w: [MaxLengthCriteria(w + 256), EosTokenCriteria([151645])]), {}), "end tokens"),
    (lambda: (fake_mixin(stops=lambda w: [MaxLengthCriteria(w + 256), EosTokenCriteria(7)]), {}), "end tokens"),
    (lambda: (fake_mixin(stops=lambda w: [MaxLengthCriteria(w + 256)]), {}), "stop rules"),
    (lambda: (fake_mixin(stops=lambda w: [MaxLengthCriteria(w + 100), EosTokenCriteria(POOL_ENDS)]), {}), "stops at"),
    (lambda: (fake_mixin(config={**GOOD_CONFIG, "temperature": 0.7}), {}), "other sampling"),
])
def test_the_real_wrappers_stop_a_changed_step(make, message):
    mixin, options = make()
    with pytest.raises(grpo.GenerationCheckError, match=message):
        run_wrapped(mixin, (PROMPT_A, PROMPT_C), **options)


def test_stop_rule_reads_tensors_lists_and_single_ids():
    class Tensor:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    assert grpo.stop_rule(EosTokenCriteria(Tensor([151645, 151643]))) == {"name": "EosTokenCriteria",
                                                                           "eos_token_id": POOL_ENDS}
    assert grpo.stop_rule(EosTokenCriteria(Tensor(151645)))["eos_token_id"] == [151645]
    assert grpo.stop_rule(MaxLengthCriteria(300)) == {"name": "MaxLengthCriteria", "max_length": 300}


def test_rewarded_answers_must_equal_written_ones():
    watch = new_watch()
    one_step(watch)
    with pytest.raises(grpo.GenerationCheckError, match="16 answers written, 15 rewarded"):
        watch.check_counts(1, rewarded=15)


def test_coverage_in_a_full_run_is_each_pair_exactly_twice():
    ids = ["a", "b", "c", "d"]
    assert grpo.coverage_problems({"a": 2, "b": 2, "c": 2, "d": 2}, ids, steps=4, each=2) == []
    problems = grpo.coverage_problems({"a": 3, "b": 1, "c": 2, "d": 2}, ids, steps=4, each=2)
    assert any("did not get exactly 2" in p for p in problems)
    assert any("not in the pool" in p for p in grpo.coverage_problems({"z": 2}, ids, steps=1, each=None))
    assert any("more than one group" in p for p in grpo.coverage_problems({"a": 2}, ids, steps=1, each=None))
    assert any("groups in 3 steps" in p for p in grpo.coverage_problems({"a": 1, "b": 1}, ids, steps=3, each=None))


def test_a_logged_step_missing_a_watch_metric_stops():
    good = {"reward": 1, "frac_reward_zero_std": 0.5, "completions/mean_length": 40,
            **{f"answers/{k}": 0.25 for k in grpo.ANSWER_KINDS}}
    assert grpo.check_step_log(good)["reward"] == 1.0
    for key in grpo.REQUIRED_STEP_KEYS:
        with pytest.raises(grpo.GenerationCheckError, match="lacks"):
            grpo.check_step_log({k: v for k, v in good.items() if k != key})


def test_the_record_holds_what_the_trial_asked_and_used(at_repo, tmp_path):
    meta, _ = trial(tmp_path)
    assert meta["sampling_asked"] == grpo.sampling_settings()
    assert meta["sampling_used"]["calls"]["trainer"] == 20
    assert meta["pool_sha256"] == grpo.POOL_SHA256 and meta["reward_sha256"] == grpo.REWARD_SHA256
    assert (meta["base_revision"], meta["adapter_revision"]) == (grpo.BASE_REVISION, grpo.V1_ADAPTER_REVISION)
    assert meta["modal_app_id"] == "ap-test"
    assert meta["save_every_used"] == 10 and meta["save_points"] == [10, 20]
    assert set(meta["gpu_seconds"]) == {"train", "load_check"}


# ---------- steps, warmup and checkpoints ----------

def test_steps_and_checkpoints():
    assert grpo.planned_steps(314) == 314  # 2 pairs a step, 2 epochs
    assert grpo.warmup_steps(314) == 31
    assert grpo.save_points(314, grpo.SAVE_EVERY) == [50, 100, 150, 200, 250, 300, 314]
    assert grpo.full_run_checkpoints() == [f"step-{p}" for p in (50, 100, 150, 200, 250, 300, 314)]
    assert grpo.save_points(20, 10) == [10, 20]
    assert grpo.planned_steps(314, max_steps=20) == 20


# ---------- the reward-hacking watch ----------

def log(n, **shares):
    base = {"answers/contradiction": 0.5, "answers/non_contradiction": 0.4, "answers/unclear": 0.1,
            "answers/invalid": 0.0}
    return [{"step": i + 1, **base, **shares} for i in range(n)]


def test_no_warning_on_a_healthy_log():
    assert grpo.watch_warnings(log(40)) == []


def test_no_warning_before_twenty_steps():
    assert grpo.watch_warnings(log(19, **{"answers/invalid": 0.5})) == []


@pytest.mark.parametrize("shares, expected", [
    ({"answers/invalid": 0.03}, "invalid answers are 3.0%"),
    ({"answers/contradiction": 0.9, "answers/non_contradiction": 0.05, "answers/unclear": 0.05}, "contradiction is 90.0%"),
    ({"answers/contradiction": 0.05, "answers/non_contradiction": 0.9, "answers/unclear": 0.05},
     "non_contradiction is 90.0%"),
    ({"answers/contradiction": 0.4, "answers/non_contradiction": 0.4, "answers/unclear": 0.2}, "unclear is 20.0%"),
])
def test_each_warning_fires(shares, expected):
    found = grpo.watch_warnings(log(20, **shares))
    assert len(found) == 1 and expected in found[0] and "steps 1-20" in found[0]


def test_the_watch_reads_only_the_last_twenty_steps():
    steps = log(20, **{"answers/invalid": 0.5}) + [dict(s, step=s["step"] + 20) for s in log(20)]
    assert grpo.watch_warnings(steps) == []
    assert grpo.watch_warnings(steps[:-1]) != []  # the window still holds one bad step: 0.5 / 20 = 2.5%


def test_the_limits_themselves_do_not_warn():
    assert grpo.watch_warnings(log(20, **{"answers/invalid": 0.02})) == []
    assert grpo.watch_warnings(log(20, **{"answers/unclear": 0.15})) == []


# ---------- the D-056 rule, with D-057's Style A gate (finding 4) ----------

def version_counts(caught, said, real, invalid=0, breaks=None):
    return {v: {"caught": caught, "said": said, "real": real, "invalid": invalid,
                "style_a_breaks": invalid if breaks is None else breaks} for v in VERSIONS}


def cand(step, hard=(40, 42, 47), m1=(40, 42, 46), hard_invalid=0, m1_invalid=0, hard_breaks=None, **per_version):
    c = {"step": step, "valid_hard": version_counts(*hard, hard_invalid, hard_breaks),
         "m1_valid": version_counts(*m1, m1_invalid)}
    for key, value in per_version.items():  # e.g. valid_hard__no_dates=(30, 32, 47)
        f, v = key.split("__")
        c[f][v] = {"caught": value[0], "said": value[1], "real": value[2], "invalid": 0, "style_a_breaks": 0}
    return c


def test_the_highest_worst_case_recall_wins():
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(43, 45, 47)),
                                          "step-100": cand(100, hard=(42, 44, 47))})
    assert chosen == "step-50" and out == {}


def test_worst_case_is_the_lowest_version():
    # step-50 is best as written but worst without dates; step-100 is better in its worst case.
    chosen, _ = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(46, 48, 47),
                                                                        valid_hard__no_dates=(39, 41, 47)),
                                        "step-100": cand(100, hard=(42, 44, 47))})
    assert chosen == "step-100"


def test_a_recall_tie_goes_to_higher_precision_then_the_earlier_step():
    tie_precision = {"step-0": cand(0), "step-50": cand(50, hard=(43, 46, 47)), "step-100": cand(100, hard=(43, 45, 47))}
    assert grpo.choose_checkpoint(tie_precision)[0] == "step-100"
    full_tie = {"step-0": cand(0), "step-50": cand(50, hard=(43, 45, 47)), "step-100": cand(100, hard=(43, 45, 47))}
    assert grpo.choose_checkpoint(full_tie)[0] == "step-50"
    # The same tie listed the other way round: the earlier step still wins, so the record's order does not matter.
    reversed_tie = {"step-0": cand(0), "step-100": cand(100, hard=(43, 45, 47)), "step-50": cand(50, hard=(43, 45, 47))}
    assert grpo.choose_checkpoint(reversed_tie)[0] == "step-50"


def test_step_zero_wins_when_nothing_beats_it():
    chosen, _ = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(39, 41, 47)),
                                        "step-100": cand(100, hard=(40, 42, 47))})
    assert chosen == "step-0"  # tied with step-100 on everything, so the earlier step


def test_invalid_answers_disqualify():
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(47, 47, 47), m1_invalid=1)})
    assert chosen == "step-0" and out["step-50"] == ["3 invalid answers"]


def test_answers_not_in_v1s_format_disqualify():
    # Right labels everywhere, but no type or reason in 97 answers per version: the scorer calls them valid, the reward not.
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(47, 47, 47), hard_breaks=97)})
    assert chosen == "step-0"
    assert out["step-50"] == ["291 answers not in v1's exact format (type or reason)"]


def test_a_precision_drop_on_valid_hard_disqualifies_only_beyond_0_02():
    # step 0 precision 40/42 = 0.952; 45/48 = 0.9375 is within 0.02, 45/49 = 0.918 is not.
    within = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(45, 48, 47))})
    assert within[0] == "step-50"
    beyond = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(45, 49, 47))})
    assert beyond[0] == "step-0" and "valid-hard worst-case precision" in beyond[1]["step-50"][0]


def test_exactly_0_02_lower_precision_stays_in():
    # step 0 precision 48/50 = 0.96; 47/50 = 0.94 is exactly 0.02 lower, which is not "more than".
    assert Fraction(47, 50) == Fraction(48, 50) - Fraction(2, 100)
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0, hard=(48, 50, 50)), "step-50": cand(50, hard=(47, 50, 49))})
    assert out == {} and chosen == "step-0"  # in, but step 0's recall 48/50 beats 47/49


@pytest.mark.parametrize("m1, kept", [
    ((94, 100, 94), True),  # precision 0.94: exactly 0.03 below step 0's 0.97, so in
    ((94, 97, 100), True),  # recall 0.94: exactly 0.03 below step 0's 0.97, so in
    ((93, 100, 93), False),  # precision 0.93: more than 0.03 below
    ((93, 96, 100), False),  # recall 0.93: more than 0.03 below
])
def test_the_m1_limits_at_exactly_0_03(m1, kept):
    base = cand(0, m1=(97, 100, 100))  # M1 precision and recall both 0.97
    better = cand(50, hard=(41, 43, 47), m1=m1)  # better than step 0 on valid-hard
    chosen, out = grpo.choose_checkpoint({"step-0": base, "step-50": better})
    assert (chosen == "step-50") is kept and ("step-50" not in out) is kept


@pytest.mark.parametrize("m1, reason", [
    ((40, 45, 46), "M1 valid worst-case precision"),  # 40/45 = 0.889 against 0.952
    ((38, 40, 46), "M1 valid worst-case recall"),  # 38/46 = 0.826 against 0.870
])
def test_forgetting_m1_disqualifies(m1, reason):
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(46, 47, 47), m1=m1)})
    assert chosen == "step-0" and reason in out["step-50"][0]


def test_no_candidate_left_returns_none():
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0, hard_invalid=1), "step-50": cand(50, m1_invalid=1)})
    assert chosen is None and set(out) == {"step-0", "step-50"}


def test_never_saying_contradiction_counts_as_precision_zero():
    assert grpo.precision({"caught": 0, "said": 0, "real": 47, "invalid": 0}) == 0
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, valid_hard__swapped=(0, 0, 47))})
    assert chosen == "step-0" and "step-50" in out


def test_counts_come_from_the_scorer_and_the_rewards_reader():
    pairs, _ = read_pairs(REPO / "data/m2/valid-hard.jsonl")
    right = [{"id": p["id"], "raw": answer_text(p)} for p in pairs]
    real = sum(p["label"] == "contradiction" for p in pairs)
    assert grpo.counts(pairs, right) == {"caught": real, "said": real, "real": real, "invalid": 0, "style_a_breaks": 0}
    broken = [dict(a, raw="not json") for a in right]
    assert grpo.counts(pairs, broken)["invalid"] == grpo.counts(pairs, broken)["style_a_breaks"] == len(pairs)
    bare = [{"id": p["id"], "raw": label_only(p, "as_is")} for p in pairs]
    c = grpo.counts(pairs, bare)
    assert (c["invalid"], c["style_a_breaks"], c["caught"]) == (0, len(pairs), real)


# ---------- choose and upload need a finished, checked run (finding 2) ----------

def test_choose_answers_every_candidate_and_applies_the_rule(at_repo, tmp_path):
    finished_run(tmp_path)

    def alarm(pair, version):
        return json.dumps({"label": "contradiction", "type": None, "reason": "Always."})

    calls = Calls(answers={"step-300": alarm, "step-250": label_only})
    meta = grpo.run_stage("choose", calls, out=tmp_path)
    names = ["step-0", "step-50", "step-100", "step-150", "step-200", "step-250", "step-300", "step-314"]
    assert calls.log[0] == ("answer_many", names)
    assert meta["choice"]["chosen"] == "step-0"  # all right answers tie, so the earliest step
    assert "step-300" in meta["choice"]["out"]
    assert any("v1's exact format" in r for r in meta["choice"]["out"]["step-250"])
    assert len(list((tmp_path / "valid").iterdir())) == len(names) * 2 * 3 + 1  # and choice.json
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    with pytest.raises(SystemExit, match="already holds a choice"):
        grpo.run_stage("choose", Calls(), out=tmp_path)


@pytest.mark.parametrize("changes, message", [
    ({"run_checks": {"passed": False, "problems": ["x"]}}, "own checks did not pass"),
    ({"global_step": 50}, "did 50 steps"),
    ({"saved": [{"name": "step-50", "step": 50, "files": files_of("step-50")}]}, "saved checkpoints are"),
    ({"steps": []}, "step log has 0 steps"),
    ({"sampling_used": {}}, "one checked generation per step"),
    ({"adapter_check": {}}, "fingerprints of v1's files"),
])
def test_choose_refuses_an_unfinished_or_unchecked_run(at_repo, tmp_path, changes, message):
    finished_run(tmp_path, **changes)
    calls = Calls()
    with pytest.raises(SystemExit, match=message):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert calls.log == []


def test_choose_refuses_a_trial_record(at_repo, tmp_path):
    trial(tmp_path)
    (tmp_path / "trial.meta.json").rename(tmp_path / "train.meta.json")
    calls = Calls()
    with pytest.raises(SystemExit, match="'trial' run"):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert calls.log == []


def test_choose_refuses_changed_validation_files_and_a_changed_reward(at_repo, tmp_path, monkeypatch):
    finished_run(tmp_path)
    record = json.loads((tmp_path / "train.meta.json").read_text())
    record["valid_kept_apart"]["data/m2/valid-hard.jsonl"] = "0" * 64
    (tmp_path / "train.meta.json").write_text(json.dumps(record))
    calls = Calls()
    with pytest.raises(SystemExit, match="validation files"):
        grpo.run_stage("choose", calls, out=tmp_path)
    record["valid_kept_apart"] = grpo.read_valid()[1]
    (tmp_path / "train.meta.json").write_text(json.dumps(record))
    monkeypatch.setattr(judge.reward, "reward_sha256", lambda: "f" * 64)
    with pytest.raises(SystemExit, match="reward fingerprint is f"):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert calls.log == []


@pytest.mark.parametrize("calls, message", [
    (Calls(drop_items=1), "not each of the"),
    (Calls(answer_files={"step-100": files_of("other")}), "differ from its saved fingerprints"),
])
def test_choose_writes_nothing_when_answers_are_missing_or_from_other_files(at_repo, tmp_path, calls, message):
    finished_run(tmp_path)
    before = (tmp_path / "train.meta.json").read_text()
    with pytest.raises(SystemExit, match=message):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert (tmp_path / "train.meta.json").read_text() == before
    assert not (tmp_path / "valid").exists()
    grpo.run_stage("choose", Calls(), out=tmp_path)  # a clean retry works


def interrupted_choose(tmp_path, monkeypatch):
    """A choose whose every candidate answered, stopped by a failed write of train.meta.json after valid/ was named."""
    finished_run(tmp_path)
    real = grpo.replace_json

    def fails_once(path, data):
        monkeypatch.setattr(grpo, "replace_json", real)
        raise OSError("the disk said no")

    monkeypatch.setattr(grpo, "replace_json", fails_once)
    with pytest.raises(OSError):
        grpo.run_stage("choose", Calls(answers={"step-250": label_only}), out=tmp_path)
    assert (tmp_path / "valid" / "choice.json").is_file()
    assert "choice" not in json.loads((tmp_path / "train.meta.json").read_text())


def test_a_choose_stopped_after_its_answers_were_saved_finishes_without_answering_again(at_repo, tmp_path,
                                                                                        monkeypatch):
    interrupted_choose(tmp_path / "a", monkeypatch)
    calls = Calls()
    meta = grpo.run_stage("choose", calls, out=tmp_path / "a")
    assert calls.log == []  # no candidate answered again, nothing paid twice
    assert meta["choice"]["recovered_from"] == "valid/choice.json"
    finished_run(tmp_path / "b")
    clean = grpo.run_stage("choose", Calls(answers={"step-250": label_only}), out=tmp_path / "b")
    for key in ("chosen", "out", "counts", "valid_scores", "valid_sha256", "rule"):
        assert meta["choice"][key] == clean["choice"][key], key
    assert any("v1's exact format" in r for r in meta["choice"]["out"]["step-250"])


def retag(valid, **changes):
    path = valid / "choice.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


@pytest.mark.parametrize("damage, message", [
    (lambda valid: valid.joinpath("step-50.valid_hard.as_is.jsonl").write_text("{}\n"), "changed since it was saved"),
    (lambda valid: valid.joinpath("extra.txt").write_text("x"), "other files"),
    (lambda valid: valid.joinpath("choice.json").unlink(), "without choice.json"),
    (lambda valid: retag(valid, run="train-other"), "from run train-other"),
    (lambda valid: retag(valid, items_sha256="0" * 64), "answered other items"),
])
def test_saved_answers_that_do_not_check_out_are_never_used_or_overwritten(at_repo, tmp_path, monkeypatch, damage,
                                                                           message):
    interrupted_choose(tmp_path, monkeypatch)
    damage(tmp_path / "valid")
    before = sorted(p.name for p in (tmp_path / "valid").iterdir())
    calls = Calls()
    with pytest.raises(SystemExit, match=message):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert calls.log == [] and sorted(p.name for p in (tmp_path / "valid").iterdir()) == before
    assert "choice" not in json.loads((tmp_path / "train.meta.json").read_text())


def test_saved_answers_that_no_longer_give_the_saved_choice_stop(at_repo, tmp_path, monkeypatch):
    interrupted_choose(tmp_path, monkeypatch)
    path = tmp_path / "valid" / "choice.json"
    note = json.loads(path.read_text())
    note["choice"]["chosen"] = "step-314"
    path.write_text(json.dumps(note))
    with pytest.raises(SystemExit, match="no longer give the saved choice"):
        grpo.run_stage("choose", Calls(), out=tmp_path)


def test_check_outcomes_wants_each_candidate_once():
    expected = {"step-0": V1_FILES, "step-50": files_of("step-50")}
    one = {"checkpoint": "step-0", "results": [{"id": "x"}], "files": V1_FILES}
    with pytest.raises(SystemExit, match="answers came back for"):
        grpo.check_outcomes([one, one], expected, ["x"])
    twice = {"checkpoint": "step-50", "results": [{"id": "x"}, {"id": "x"}], "files": files_of("step-50")}
    with pytest.raises(SystemExit, match="answered 2 items"):
        grpo.check_outcomes([one, twice], expected, ["x"])


def test_choose_sends_each_candidates_fingerprints(at_repo, tmp_path):
    finished_run(tmp_path)
    seen = []

    class Recording(Calls):
        def answer(self, run, checkpoint, items, expected_files):
            seen.append((checkpoint, expected_files))
            return super().answer(run, checkpoint, items, expected_files)

    grpo.run_stage("choose", Recording(), out=tmp_path)
    assert dict(seen) == {"step-0": V1_FILES, **{f"step-{p}": files_of(f"step-{p}")
                                                 for p in (50, 100, 150, 200, 250, 300, 314)}}


def choose_with_winner(tmp_path, winner="step-150"):
    """A finished run and a choice that picks `winner`: it catches one more contradiction than the rest."""
    finished_run(tmp_path)
    valid, _ = grpo.read_valid()
    miss = next(p["id"] for p in valid["valid_hard"] if p["label"] == "contradiction")

    def misses_one(pair, version):
        if pair["id"] == miss:
            return json.dumps({"label": "non_contradiction", "type": "update", "reason": "Missed."})
        return answer_text(pair)

    rules = {n: misses_one for n in ["step-0", *grpo.full_run_checkpoints()] if n != winner}
    return grpo.run_stage("choose", Calls(answers=rules), out=tmp_path)


def test_upload_sends_the_chosen_fingerprints_and_records_its_own_commit(at_repo, tmp_path):
    meta = choose_with_winner(tmp_path)
    assert meta["choice"]["chosen"] == "step-150"
    calls = Calls(upload_result={"repo": "test/repo", "commit": "abc123", "private": True, "files": [],
                                 "recovered": False})
    meta = grpo.run_stage("upload", calls, out=tmp_path)
    assert calls.log[0][2:] == ("step-150", files_of("step-150"))
    assert meta["upload"]["commit"] == "abc123" and meta["upload"]["modal_app_id"] == "ap-test"
    with pytest.raises(SystemExit, match="already records an upload"):
        grpo.run_stage("upload", Calls(), out=tmp_path)


def test_upload_refuses_a_choice_the_counts_no_longer_make(at_repo, tmp_path):
    choose_with_winner(tmp_path)
    record = json.loads((tmp_path / "train.meta.json").read_text())
    record["choice"]["chosen"] = "step-100"
    (tmp_path / "train.meta.json").write_text(json.dumps(record))
    calls = Calls()
    with pytest.raises(SystemExit, match="no longer pick"):
        grpo.run_stage("upload", calls, out=tmp_path)
    assert calls.log == []


def test_upload_refuses_step_zero_and_a_missing_choice(at_repo, tmp_path):
    finished_run(tmp_path)
    calls = Calls()
    with pytest.raises(SystemExit, match="nothing to upload"):
        grpo.run_stage("upload", calls, out=tmp_path)
    grpo.run_stage("choose", Calls(), out=tmp_path)
    with pytest.raises(SystemExit, match="nothing to upload"):  # step-0 was chosen
        grpo.run_stage("upload", calls, out=tmp_path)
    assert calls.log == []


def test_a_changed_setting_stops_choose_before_any_remote_call(at_repo, tmp_path, monkeypatch):
    finished_run(tmp_path)
    monkeypatch.setitem(grpo.GRPO, "learning_rate", 1e-5)
    calls = Calls()
    with pytest.raises(SystemExit, match="grpo is"):
        grpo.run_stage("choose", calls, out=tmp_path)
    assert calls.log == []


# ---------- the upload job itself, with a fake Hugging Face (finding 2 of the re-review) ----------

class FakeHub:
    """A fake Hugging Face with one repo: a list of commits, each the whole set of files at that commit."""

    def __init__(self, tmp_path):
        self.dir = tmp_path / "hub"
        self.dir.mkdir(parents=True)
        self.commits = []
        self.private = True
        self.fail_upload = 0
        self.drop_on_upload = None
        self.public_on_create = False
        self.intrude = None  # another writer's files, landing just before this upload's commit
        self.log = []

    def add_commit(self, files):
        held = dict(self.commits[-1][1]) if self.commits else {}
        held.update(files)
        sha = f"c{len(self.commits)}"
        self.commits.append((sha, held))
        return sha

    def repo_exists(self, repo):
        self.log.append("repo_exists")
        return bool(self.commits)

    def model_info(self, repo):
        from types import SimpleNamespace

        return SimpleNamespace(sha=self.commits[-1][0], private=self.private)

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.log.append("create_repo")
        assert private and not exist_ok and not self.commits
        self.add_commit({".gitattributes": b"*.safetensors filter=lfs\n"})
        self.private = not self.public_on_create

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message, parent_commit=None):
        from types import SimpleNamespace

        self.log.append("upload_folder")
        if self.intrude:
            self.add_commit(self.intrude)
            self.intrude = None
        if parent_commit is not None and parent_commit != self.commits[-1][0]:
            raise RuntimeError("412: the repo is not at the parent commit")  # what Hugging Face does (create_commit)
        if self.fail_upload:
            self.fail_upload -= 1
            raise ConnectionError("the connection dropped")
        files = {p.name: p.read_bytes() for p in Path(folder_path).iterdir() if p.name != self.drop_on_upload}
        sha = self.add_commit(files)
        return SimpleNamespace(oid=sha, commit_url=f"https://huggingface.co/test/commit/{sha}")

    def list_repo_files(self, repo, revision, repo_type):
        return sorted(dict(self.commits)[revision])

    def fetch(self, name, revision):
        path = self.dir / f"{revision}-{name}"
        path.write_bytes(dict(self.commits)[revision][name])
        return path


@pytest.fixture
def adapter(tmp_path):
    folder = tmp_path / "run" / "step-150"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text('{"r": 32}')
    (folder / "adapter_model.safetensors").write_bytes(b"weights-150")
    (folder / "tokenizer.json").write_text("{}")
    return folder


META_TEXT = '{"run": "train-x", "choice": {"chosen": "step-150"}}\n'


def send(hub, folder, note=None, **options):
    """Run the real upload job's logic; returns (result, the note it saved last)."""
    saved = {"note": note}
    result = grpo.upload_checked(hub, hub.fetch, folder, "train-x", "step-150", META_TEXT, grpo.files_sha256(folder),
                                 note, lambda data: saved.update(note=data), **options)
    return result, saved["note"]


def held_now(hub):
    return {n: b for n, b in hub.commits[-1][1].items() if n != ".gitattributes"}


def test_a_fresh_upload_reserves_the_repo_and_records_its_own_commit(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    result, note = send(hub, adapter)
    assert (result["commit"], result["recovered"], result["resumed"]) == ("c1", False, False)
    assert note["first_commit"] == "c0" and note["run"] == "train-x" and note["checkpoint"] == "step-150"
    assert held_now(hub) == {**{p.name: p.read_bytes() for p in adapter.iterdir()},
                             "train.meta.json": META_TEXT.encode()}


def test_an_upload_that_failed_after_making_the_repo_resumes_into_it(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    hub.fail_upload = 1
    saved = {}
    with pytest.raises(ConnectionError):
        grpo.upload_checked(hub, hub.fetch, adapter, "train-x", "step-150", META_TEXT, grpo.files_sha256(adapter),
                            None, lambda data: saved.update(note=data))
    assert saved["note"]["first_commit"] == "c0" and len(hub.commits) == 1  # an empty repo, reserved by this try
    result, _ = send(hub, adapter, note=saved["note"])
    assert (result["commit"], result["recovered"], result["resumed"]) == ("c1", False, True)
    # Without this run's note, the same empty repo is refused: nothing proves this upload made it.
    other = FakeHub(tmp_path / "other")
    other.add_commit({".gitattributes": b""})
    with pytest.raises(SystemExit, match="not an empty repo this upload made"):
        send(other, adapter)
    with pytest.raises(SystemExit, match="cannot prove this repo is that one.*before deleting anything"):
        send(other, adapter, note={**saved["note"], "first_commit": None})
    assert "upload_folder" not in other.log


def test_a_finished_upload_whose_local_record_was_lost_is_recovered(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    send(hub, adapter)
    result, _ = send(hub, adapter)  # the files and the record are proven at the newest commit, note or not
    assert (result["commit"], result["recovered"]) == ("c1", True)
    assert hub.log.count("upload_folder") == 1


@pytest.mark.parametrize("change", [
    {"adapter_config.json": b'{"r": 16}'},  # the same weights with another adapter config
    {"train.meta.json": b'{"run": "train-y"}\n'},  # another run's or another choice's record
    {"README.md": b"extra"},  # a file that was never checked
])
def test_a_repo_that_differs_in_any_file_is_refused(adapter, tmp_path, change):
    hub = FakeHub(tmp_path)
    send(hub, adapter)
    hub.add_commit(change)
    with pytest.raises(SystemExit, match="does not hold exactly this adapter and record"):
        send(hub, adapter)
    hub2 = FakeHub(tmp_path / "two")
    hub2.add_commit({".gitattributes": b"", **{p.name: p.read_bytes() for p in adapter.iterdir()}, **change})
    with pytest.raises(SystemExit, match="does not hold exactly"):
        send(hub2, adapter)
    assert hub.log.count("upload_folder") == 1 and "upload_folder" not in hub2.log


def test_the_upload_refuses_a_public_repo_a_changed_copy_and_a_short_commit(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    hub.add_commit({".gitattributes": b""})
    hub.private = False
    with pytest.raises(SystemExit, match="is public"):
        send(hub, adapter)

    def bad_copy(source, target, dirs_exist_ok):
        import shutil

        shutil.copytree(source, target, dirs_exist_ok=dirs_exist_ok)
        Path(target, "adapter_model.safetensors").write_bytes(b"other weights")

    fresh = FakeHub(tmp_path / "fresh")
    with pytest.raises(SystemExit, match="copy made for the upload differs"):
        send(fresh, adapter, copy=bad_copy)
    assert "upload_folder" not in fresh.log
    short = FakeHub(tmp_path / "short")
    short.drop_on_upload = "tokenizer.json"
    with pytest.raises(SystemExit, match="does not hold exactly the files sent"):
        send(short, adapter)


def test_another_writer_between_the_check_and_the_commit_is_never_overwritten(adapter, tmp_path):
    other = {"adapter_config.json": b'{"written": "by someone else"}'}

    def copy_while_someone_writes(source, target, dirs_exist_ok):
        import shutil

        shutil.copytree(source, target, dirs_exist_ok=dirs_exist_ok)
        hub.add_commit(other)

    hub = FakeHub(tmp_path)
    with pytest.raises(SystemExit, match="changed after it was checked"):
        send(hub, adapter, copy=copy_while_someone_writes)
    assert "upload_folder" not in hub.log and hub.commits[-1][1]["adapter_config.json"] == other["adapter_config.json"]
    late = FakeHub(tmp_path / "late")
    late.intrude = other  # lands after the last look, just before the commit
    with pytest.raises(RuntimeError, match="parent commit"):
        send(late, adapter)
    assert late.commits[-1][1]["adapter_config.json"] == other["adapter_config.json"]


def test_a_repo_that_is_public_after_it_was_made_gets_nothing(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    hub.public_on_create = True
    with pytest.raises(SystemExit, match="is public; nothing uploaded"):
        send(hub, adapter)
    assert "upload_folder" not in hub.log


def bill_now(out):
    grpo.bill("train", out=out, run_report=lambda command: json.dumps([{"object_id": "ap-test", "cost": "0.5"}]))


@pytest.mark.parametrize("first_try", ["stops after making the repo", "lands, but its receipt is lost"])
def test_recording_the_bill_between_two_upload_tries_changes_nothing(at_repo, tmp_path, adapter, first_try):
    out = tmp_path / "out"
    choose_with_winner(out)
    hub = FakeHub(tmp_path / "h")
    note = {}

    class Uploading(Calls):
        def upload(self, run, checkpoint, meta_text, expected_files):
            self.log.append(meta_text)
            result = grpo.upload_checked(hub, hub.fetch, adapter, run, checkpoint, meta_text,
                                         grpo.files_sha256(adapter), note.get("n"), lambda data: note.update(n=data))
            if self.lose_receipt:
                raise OSError("the local record was not written")
            return result

    first = Uploading()
    first.lose_receipt = first_try.startswith("lands")
    hub.fail_upload = 0 if first.lose_receipt else 1
    with pytest.raises((ConnectionError, OSError)):
        grpo.run_stage("upload", first, out=out)
    bill_now(out)  # the bill is recorded between the two tries
    assert "billed_usd" in json.loads((out / "train.meta.json").read_text())
    retry = Uploading()
    retry.lose_receipt = False
    meta = grpo.run_stage("upload", retry, out=out)
    assert retry.log == first.log  # the same record text both times
    assert "billed_usd" not in json.loads(retry.log[0]) and json.loads(retry.log[0])["choice"]["chosen"] == "step-150"
    assert (meta["upload"]["recovered"], meta["upload"]["resumed"]) == (first.lose_receipt, not first.lose_receipt)
    assert hub.log.count("create_repo") == 1


def test_the_upload_stops_before_hugging_face_when_the_folder_changed(adapter, tmp_path):
    hub = FakeHub(tmp_path)
    expected = grpo.files_sha256(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differ from the recorded fingerprints"):
        grpo.upload_checked(hub, hub.fetch, adapter, "train-x", "step-150", META_TEXT, expected, None, print)
    assert hub.log == []


# ---------- the frozen test files ----------

def test_the_script_never_opens_a_frozen_test_file(at_repo, tmp_path, monkeypatch):
    opened = []
    real_open = builtins.open

    def watching_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            opened.append(Path(file).resolve())
            if Path(file).resolve() in FROZEN:
                pytest.fail(f"opened the frozen file {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watching_open)
    monkeypatch.setattr(io, "open", watching_open)
    grpo.run_stage("trial", Calls(), out=tmp_path / "trial")
    grpo.run_stage("train", Calls(), out=tmp_path / "train")
    grpo.run_stage("choose", Calls(), out=tmp_path / "train")
    assert (REPO / grpo.POOL_PATH).resolve() in opened  # the watch really saw the reads
    assert not FROZEN & set(opened)


def test_a_frozen_file_in_the_read_list_stops(tmp_path):
    state = tmp_path / "STATE.md"
    (tmp_path / "data/m2").mkdir(parents=True)
    state.write_text("## Frozen files\n- `data/m2/valid-hard.jsonl`, SHA-256 `" + "a" * 64 + "`\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen test file"):
        grpo.check_not_frozen([str(tmp_path / "data/m2/valid-hard.jsonl")], state)


# ---------- the bill ----------

def test_billed_adds_up_each_app():
    report = [{"object_id": "ap-a", "cost": "0.10"}, {"object_id": "ap-a", "cost": "0.02"},
              {"object_id": "ap-b", "cost": "1.00"}, {"object_id": "ap-c", "cost": "5"}]
    assert grpo.billed(report, ["ap-a", "ap-b"]) == {"ap-a": 0.12, "ap-b": 1.0}


def test_bill_records_each_stage_and_stops_when_rows_are_missing(tmp_path):
    record = {"modal_app_id": "ap-a", "started_at": "2026-11-04T18:00", "choice": {"modal_app_id": "ap-b"},
              "upload": {"modal_app_id": "ap-c"}}
    (tmp_path / "train.meta.json").write_text(json.dumps(record), encoding="utf-8")
    rows = [{"object_id": "ap-a", "cost": "0.2"}, {"object_id": "ap-b", "cost": "0.7"}]
    with pytest.raises(SystemExit, match="no billing rows yet for \\['ap-c'\\]"):
        grpo.bill("train", out=tmp_path, run_report=lambda command: json.dumps(rows))
    rows.append({"object_id": "ap-c", "cost": "0.01"})
    grpo.bill("train", out=tmp_path, run_report=lambda command: json.dumps(rows))
    meta = json.loads((tmp_path / "train.meta.json").read_text(encoding="utf-8"))
    assert meta["billed_usd"] == {"train": 0.2, "choose": 0.7, "upload": 0.01}
    assert meta["billed_total_usd"] == 0.91
