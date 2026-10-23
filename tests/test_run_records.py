"""Tests for the safeguards around saved results in scripts/train.py and scripts/eval_trained.py.

No GPU, no network, no Modal job: every remote call is a fake that records
what it was asked. All files live in a temporary folder; the real repo is only
read, never written.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import judge.frozen
from judge.format import answer_text
from judge.prompt import prompt_sha256
from judge.score import VERSIONS
from judge.train_data import read_pairs

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # builds Modal objects lazily; no network, no GPU
    return module


train_script = load_script("train")
eval_script = load_script("eval_trained")


def pair(i):
    return {
        "id": f"v-{i:03d}",
        "person": "Test Person",
        "a": {"text": f"Test Person likes colour {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person likes colour {i + 1}.", "written_at": "2025-01-02"},
        "label": "contradiction" if i % 2 else "non_contradiction",
        "type": None if i % 2 else "merely_related",
        "reason": "They name different favourite colours.",
    }


PAIRS = [pair(i) for i in range(4)]


def write_pairs(path, pairs=PAIRS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Calls:
    """Fake Modal jobs. Answers from `good` are right; every other checkpoint calls everything a contradiction."""

    def __init__(self, good="epoch-2"):
        self.good = good
        self.log = []

    def _results(self, checkpoint, items):
        by_id = {p["id"]: p for p in PAIRS}
        alarm = json.dumps({"label": "contradiction", "type": None, "reason": "Always."})
        return [{"id": i["id"], "raw": answer_text(by_id[i["id"].split("/", 1)[1]]) if checkpoint == self.good
                 else alarm} for i in items]

    def answer_many(self, args):
        self.log.append(("answer_many", [a[1] for a in args]))
        return [{"checkpoint": c, "results": self._results(c, items), "load_seconds": 0, "answer_seconds": 0}
                for _, c, items in args]

    def upload(self, run, checkpoint, meta_text):
        self.log.append(("upload", run, checkpoint))
        return {"repo": "test/repo", "commit": f"commit-of-{checkpoint}", "private": True, "files": []}

    def train(self, run, train_pairs, valid_pairs, max_steps=0, eval_steps=None):
        self.log.append(("train", run, len(train_pairs), len(valid_pairs)))
        return {"run": run, "global_step": 3, "train_seconds": 3, "load_seconds": 1, "peak_vram_gib": 1,
                "curves": {"train": [], "valid": []}, "saved": [{"name": "epoch-1"}, {"name": "epoch-2"}]}


def make_run(tmp_path, **changes):
    """A made-up finished training run: validation file, and train.meta.json with an upload of epoch-2."""
    valid = tmp_path / "valid.jsonl"
    digest = write_pairs(valid)
    meta = {
        "stage": "train", "run": "v1", "valid_sha256": digest, "prompt_sha256": prompt_sha256(),
        **train_script.run_settings(),
        "saved": [{"name": "epoch-1"}, {"name": "epoch-2"}],
        "choice": {"chosen": "epoch-2"},
        "upload": {"checkpoint": "epoch-2", "run": "v1", "repo": "test/repo", "commit": "c2", "private": True},
    }
    meta.update(changes)
    out = tmp_path / "out"
    out.mkdir()
    (out / "train.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out, valid


def read_meta(out):
    return json.loads((out / "train.meta.json").read_text(encoding="utf-8"))


def choose(out, valid, calls):
    train_script.run_stage("choose", "v1", calls, out=out, valid_path=valid)


def no_test_set(monkeypatch):
    monkeypatch.setattr(judge.frozen, "load_frozen", lambda *a, **k: pytest.fail("the test set was read"))


# Finding 3: selection and upload check the record first.

def test_the_real_record_matches_todays_files_and_settings():
    meta = json.loads((REPO / "results/m1/trained-v1/train.meta.json").read_text(encoding="utf-8"))
    _, valid_sha256 = read_pairs(REPO / "data/m1/valid.jsonl")
    train_script.check_record(meta, valid_sha256)  # does not stop


@pytest.mark.parametrize("change, message", [
    ({"valid_sha256": "0" * 64}, "valid.jsonl SHA-256"),
    ({"prompt_sha256": "0" * 64}, "prompt SHA-256"),
    ({"base_revision": "0" * 40}, "base_revision"),
    ({"seed": 1}, "seed is 3407"),
])
def test_a_mismatch_stops_selection_and_upload_before_any_remote_call(tmp_path, change, message):
    out, valid = make_run(tmp_path, **change)
    before = (out / "train.meta.json").read_text(encoding="utf-8")
    for stage in ("choose", "upload"):
        calls = Calls()
        with pytest.raises(SystemExit, match=message):
            train_script.run_stage(stage, "v1", calls, out=out, valid_path=valid)
        assert calls.log == []
    assert (out / "train.meta.json").read_text(encoding="utf-8") == before


def test_a_changed_lora_setting_stops_selection(tmp_path):
    lora = dict(train_script.run_settings()["lora"], r=16)
    out, valid = make_run(tmp_path, lora=lora)
    calls = Calls()
    with pytest.raises(SystemExit, match="lora"):
        choose(out, valid, calls)
    assert calls.log == []


def test_training_fingerprints_come_from_the_bytes_sent(tmp_path):
    out = tmp_path / "out"
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train_sha, valid_sha = write_pairs(train, PAIRS[:3]), write_pairs(valid)

    class Changing(Calls):
        def train(self, *args, **kwargs):
            write_pairs(train, PAIRS[:1])  # the files change while the job runs
            write_pairs(valid, PAIRS[:1])
            return super().train(*args, **kwargs)

    calls = Changing()
    train_script.run_stage("train", "v1", calls, out=out, train_path=train, valid_path=valid)
    meta = read_meta(out)
    assert calls.log[0] == ("train", "v1", 3, 4)
    assert (meta["train_sha256"], meta["valid_sha256"]) == (train_sha, valid_sha)
    # The record holds every setting the later stages check.
    train_script.check_record(meta, valid_sha)


# Finding 2: a new choice drops an upload of the old one, and scoring refuses a mismatch.

def test_a_new_choice_drops_the_old_upload_and_scoring_then_stops(tmp_path, monkeypatch):
    out, valid = make_run(tmp_path)
    choose(out, valid, Calls(good="epoch-1"))
    meta = read_meta(out)
    assert meta["choice"]["chosen"] == "epoch-1"
    assert "upload" not in meta

    no_test_set(monkeypatch)
    answered = []
    with pytest.raises(SystemExit, match="no private adapter upload"):
        eval_script.evaluate(False, tmp_path / "answers", lambda *a: answered.append(a),
                             train_meta_path=out / "train.meta.json")
    assert answered == []

    # Uploading again records the new choice, and then the two match.
    calls = Calls()
    train_script.run_stage("upload", "v1", calls, out=out, valid_path=valid)
    assert calls.log == [("upload", "v1", "epoch-1")]
    assert read_meta(out)["upload"]["checkpoint"] == "epoch-1"
    eval_script.check_upload(read_meta(out))  # does not stop


def test_the_same_choice_keeps_its_upload(tmp_path):
    out, valid = make_run(tmp_path)
    choose(out, valid, Calls(good="epoch-2"))
    assert read_meta(out)["upload"]["commit"] == "c2"


def test_an_upload_that_does_not_name_its_run_is_not_kept(tmp_path):
    out, valid = make_run(tmp_path, upload={"checkpoint": "epoch-2", "repo": "test/repo", "commit": "c2",
                                             "private": True})
    choose(out, valid, Calls(good="epoch-2"))
    assert "upload" not in read_meta(out)


def test_scoring_stops_on_a_choice_and_upload_mismatch_before_reading_the_test_set(tmp_path, monkeypatch):
    out, _ = make_run(tmp_path, choice={"chosen": "epoch-1"})
    no_test_set(monkeypatch)
    answered = []
    with pytest.raises(SystemExit, match="validation set chose 'epoch-1'"):
        eval_script.evaluate(False, tmp_path / "answers", lambda *a: answered.append(a),
                             train_meta_path=out / "train.meta.json")
    assert answered == []
    assert not (tmp_path / "answers").exists()


# Finding 1: a smoke run cannot replace the final files, and no run replaces a saved file.

def fake_answer(calls):
    def answer(base, base_revision, adapter, adapter_revision, items):
        calls.append(adapter_revision)
        return {"results": [{"id": i["id"], "raw": "{}"} for i in items], "load_seconds": 0, "answer_seconds": 0,
                "peak_vram_gib": 0}
    return answer


def test_smoke_names_never_match_the_final_names(tmp_path):
    smoke = {p.name for p in eval_script.output_paths(tmp_path, smoke=True)}
    final = {p.name for p in eval_script.output_paths(tmp_path, smoke=False)}
    assert final == {f"qwen3-4b-trained-v1.{v}.jsonl" for v in VERSIONS} | {"qwen3-4b-trained-v1.meta.json"}
    assert len(smoke) == len(final) and not smoke & final


def test_smoke_run_in_the_results_folder_leaves_the_final_files_alone(tmp_path, monkeypatch):
    run, valid = make_run(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    for path in eval_script.output_paths(results, smoke=False):
        path.write_text("final\n", encoding="utf-8")
    no_test_set(monkeypatch)
    calls = []
    eval_script.evaluate(True, results, fake_answer(calls), train_meta_path=run / "train.meta.json",
                         valid_path=valid)
    assert calls == ["c2"]
    assert all(p.read_text(encoding="utf-8") == "final\n" for p in eval_script.output_paths(results, smoke=False))
    assert all(p.exists() for p in eval_script.output_paths(results, smoke=True))

    # A second smoke run into the same folder stops, before any remote call.
    with pytest.raises(SystemExit, match="already exist"):
        eval_script.evaluate(True, results, fake_answer(calls), train_meta_path=run / "train.meta.json",
                             valid_path=valid)
    assert calls == ["c2"]


@pytest.mark.parametrize("existing", range(len(VERSIONS) + 1))
def test_any_existing_output_stops_the_final_run_before_anything_starts(tmp_path, monkeypatch, existing):
    run, _ = make_run(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    taken = eval_script.output_paths(results, smoke=False)[existing]
    taken.write_text("saved\n", encoding="utf-8")
    no_test_set(monkeypatch)
    calls = []
    with pytest.raises(SystemExit, match="already exist"):
        eval_script.evaluate(False, results, fake_answer(calls), train_meta_path=run / "train.meta.json")
    assert calls == []
    assert taken.read_text(encoding="utf-8") == "saved\n"
    assert [p.name for p in results.iterdir()] == [taken.name]


def test_final_run_into_an_empty_folder_writes_every_file(tmp_path, monkeypatch):
    run, _ = make_run(tmp_path)
    repo = tmp_path / "repo"
    digest = write_pairs(repo / "data/m1/test.jsonl")
    (repo / "STATE.md").write_text(f"## Frozen files\n- `data/m1/test.jsonl`, SHA-256 `{digest}`\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    calls = []
    eval_script.evaluate(False, "answers", fake_answer(calls), train_meta_path=run / "train.meta.json")
    assert calls == ["c2"]
    meta = json.loads((repo / "answers/qwen3-4b-trained-v1.meta.json").read_text(encoding="utf-8"))
    assert (meta["data_sha256"], meta["adapter_checkpoint"], meta["pairs"]) == (digest, "epoch-2", len(PAIRS))
    assert all(p.exists() for p in eval_script.output_paths(repo / "answers", smoke=False))
