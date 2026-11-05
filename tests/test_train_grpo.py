"""Tests for scripts/train_grpo.py (M2 Step 5): data, fingerprints, sampling, the watch and the D-056 rule.

No GPU, no network, no Modal job: every remote call is a fake that records
what it was asked. Results are written only to a temporary folder; the real
repo is only read. The frozen test files are never opened, and one test
proves the script does not open them either.
"""

import builtins
import importlib.util
import io
import json
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


# ---------- fakes ----------

class Calls:
    """Fake Modal jobs. `answers` maps a checkpoint to a function (pair, version) -> raw answer; default: right."""

    def __init__(self, answers=None):
        self.log = []
        self.answers = answers or {}

    def app_id(self):
        return "ap-test"

    def train(self, run, rows, max_steps, save_every):
        self.log.append(("train", run, [r["id"] for r in rows], max_steps, save_every))
        total = grpo.planned_steps(len(rows), max_steps)
        points = grpo.save_points(total, save_every)
        return {"run": run, "global_step": total, "planned_steps": total, "train_seconds": 40.0, "load_seconds": 60.0,
                "seconds_per_step": 2.0, "peak_vram_gib": 12.0, "warnings": [], "steps": [],
                "saved": [{"name": f"step-{p}", "step": p} for p in points], "sampling_used": {}}

    def answer(self, run, checkpoint, items):
        self.log.append(("answer", run, checkpoint, [i["id"] for i in items]))
        valid, _ = grpo.read_valid()
        by_file = {f: {p["id"]: p for p in pairs} for f, pairs in valid.items()}
        results = []
        for item in items:
            name, version, pair_id = item["id"].split("/", 2)
            pair = by_file[name][pair_id]
            rule = self.answers.get(checkpoint)
            results.append({"id": item["id"], "raw": rule(pair, version) if rule else answer_text(pair)})
        return {"checkpoint": checkpoint, "load_seconds": 1, "answer_seconds": 1, "results": results}

    def answer_many(self, args):
        self.log.append(("answer_many", [a[1] for a in args]))
        return [self.answer(run, c, items) for run, c, items in args]

    def upload(self, run, checkpoint, meta_text):
        self.log.append(("upload", run, checkpoint))
        return {"repo": "test/repo", "commit": "c1", "private": True, "files": []}


@pytest.fixture
def at_repo(monkeypatch):
    """Run from the repo folder (the script reads its data by repo path); outputs go to tmp_path only."""
    monkeypatch.chdir(REPO)


def trial(tmp_path, calls=None):
    calls = calls or Calls()
    return grpo.run_stage("trial", "v2", calls, out=tmp_path), calls


def finished_run(tmp_path):
    return grpo.run_stage("train", "v2", Calls(), out=tmp_path)


# ---------- the data ----------

def test_dataset_rows_are_the_pool_rows_in_their_saved_versions():
    pool, sha = read_pairs(REPO / grpo.POOL_PATH)
    assert sha == grpo.POOL_SHA256
    rows = grpo.dataset_rows(pool)
    assert [r["id"] for r in rows] == [p["id"] for p in pool]
    assert len(rows) == 314
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
        grpo.run_stage("trial", "v2", calls, out=tmp_path / "out", train_meta_path=fake)
    assert calls.log == []


def test_the_real_v1_record_names_the_pins():
    grpo.check_v1(json.loads((REPO / grpo.TRAIN_META).read_text(encoding="utf-8")))


def test_a_saved_trial_record_is_never_replaced(at_repo, tmp_path):
    trial(tmp_path)
    calls = Calls()
    with pytest.raises(SystemExit, match="already exist"):
        trial(tmp_path, calls)
    assert calls.log == []


# ---------- sampling ----------

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


def test_the_record_holds_what_the_trial_asked_and_used(at_repo, tmp_path):
    meta, _ = trial(tmp_path)
    assert meta["sampling_asked"] == grpo.sampling_settings()
    assert "sampling_used" in meta
    assert meta["pool_sha256"] == grpo.POOL_SHA256 and meta["reward_sha256"] == grpo.REWARD_SHA256
    assert (meta["base_revision"], meta["adapter_revision"]) == (grpo.BASE_REVISION, grpo.V1_ADAPTER_REVISION)
    assert meta["modal_app_id"] == "ap-test"


# ---------- steps, warmup and checkpoints ----------

def test_steps_and_checkpoints():
    assert grpo.planned_steps(314) == 314  # 2 pairs a step, 2 epochs
    assert grpo.warmup_steps(314) == 31
    assert grpo.save_points(314, grpo.SAVE_EVERY) == [50, 100, 150, 200, 250, 300, 314]
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


# ---------- the D-056 rule ----------

def version_counts(caught, said, real, invalid=0):
    return {v: {"caught": caught, "said": said, "real": real, "invalid": invalid} for v in VERSIONS}


def cand(step, hard=(40, 42, 47), m1=(40, 42, 46), hard_invalid=0, m1_invalid=0, **per_version):
    c = {"step": step, "valid_hard": version_counts(*hard, hard_invalid), "m1_valid": version_counts(*m1, m1_invalid)}
    for key, value in per_version.items():  # e.g. valid_hard__no_dates=(30, 32, 47)
        f, v = key.split("__")
        c[f][v] = {"caught": value[0], "said": value[1], "real": value[2], "invalid": 0}
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


def test_step_zero_wins_when_nothing_beats_it():
    chosen, _ = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(39, 41, 47)),
                                        "step-100": cand(100, hard=(40, 42, 47))})
    assert chosen == "step-0"  # tied with step-100 on everything, so the earlier step


def test_invalid_answers_disqualify():
    chosen, out = grpo.choose_checkpoint({"step-0": cand(0), "step-50": cand(50, hard=(47, 47, 47), m1_invalid=1)})
    assert chosen == "step-0" and out["step-50"] == ["3 invalid answers"]


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


def test_counts_come_from_the_scorer():
    pairs, _ = read_pairs(REPO / "data/m2/valid-hard.jsonl")
    right = [{"id": p["id"], "raw": answer_text(p)} for p in pairs]
    real = sum(p["label"] == "contradiction" for p in pairs)
    assert grpo.counts(pairs, right) == {"caught": real, "said": real, "real": real, "invalid": 0}
    broken = [dict(a, raw="not json") for a in right]
    assert grpo.counts(pairs, broken)["invalid"] == len(pairs)


# ---------- the choose and upload stages, with fake answers ----------

def test_choose_answers_every_candidate_and_applies_the_rule(at_repo, tmp_path):
    finished_run(tmp_path)

    def alarm(pair, version):
        return json.dumps({"label": "contradiction", "type": None, "reason": "Always."})

    calls = Calls(answers={"step-300": alarm})
    meta = grpo.run_stage("choose", "v2", calls, out=tmp_path)
    names = ["step-0", "step-50", "step-100", "step-150", "step-200", "step-250", "step-300", "step-314"]
    assert calls.log[0] == ("answer_many", names)
    assert meta["choice"]["chosen"] == "step-0"  # all right answers tie, so the earliest step
    assert "step-300" in meta["choice"]["out"]
    assert len(list((tmp_path / "valid").iterdir())) == len(names) * 2 * 3
    with pytest.raises(SystemExit, match="already holds a choice"):
        grpo.run_stage("choose", "v2", Calls(), out=tmp_path)


def test_a_changed_setting_stops_choose_before_any_remote_call(at_repo, tmp_path, monkeypatch):
    finished_run(tmp_path)
    monkeypatch.setitem(grpo.GRPO, "learning_rate", 1e-5)
    calls = Calls()
    with pytest.raises(SystemExit, match="grpo is"):
        grpo.run_stage("choose", "v2", calls, out=tmp_path)
    assert calls.log == []


def test_upload_refuses_step_zero_and_a_missing_choice(at_repo, tmp_path):
    finished_run(tmp_path)
    calls = Calls()
    with pytest.raises(SystemExit, match="nothing to upload"):
        grpo.run_stage("upload", "v2", calls, out=tmp_path)
    grpo.run_stage("choose", "v2", Calls(), out=tmp_path)
    with pytest.raises(SystemExit, match="nothing to upload"):  # step-0 was chosen
        grpo.run_stage("upload", "v2", calls, out=tmp_path)
    assert calls.log == []


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
    grpo.run_stage("trial", "v2", Calls(), out=tmp_path / "trial")
    grpo.run_stage("train", "v2", Calls(), out=tmp_path / "train")
    grpo.run_stage("choose", "v2", Calls(), out=tmp_path / "train")
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


def test_bill_records_the_cost_and_stops_when_rows_are_missing(tmp_path):
    (tmp_path / "trial.meta.json").write_text(json.dumps({"modal_app_id": "ap-a", "started_at": "2026-11-04T18:00"}),
                                              encoding="utf-8")
    with pytest.raises(SystemExit, match="no billing rows yet"):
        grpo.bill("trial", out=tmp_path, run_report=lambda command: "[]")
    grpo.bill("trial", out=tmp_path, run_report=lambda command: json.dumps([{"object_id": "ap-a", "cost": "0.2"}]))
    meta = json.loads((tmp_path / "trial.meta.json").read_text(encoding="utf-8"))
    assert meta["billed_usd"] == {"ap-a": 0.2} and meta["billed_total_usd"] == 0.2
