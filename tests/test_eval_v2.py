"""Tests for `scripts/eval_trained.py --model v2` (M2 Step 6, D-060). Fake GPU jobs and made-up pairs only."""

import importlib.util
import json
from pathlib import Path

import pytest

import judge.frozen
from judge.prompt import prompt_sha256
from judge.score import VERSIONS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_trained_v2", REPO / "scripts" / "eval_trained.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FILES = {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
BASE = {"base_model": "test/base", "base_revision": "base-rev"}


def grpo_meta(**changes):
    meta = {**BASE, "prompt_sha256": prompt_sha256(), "reward_sha256": "r" * 64,
            "saved": [{"name": "step-200", "files": {k: "0" * 64 for k in FILES}},
                      {"name": "step-250", "files": {**FILES, "README.md": "c" * 64}}],
            "choice": {"chosen": "step-250"},
            "upload": {"checkpoint": "step-250", "repo": ev.V2["repo"], "commit": ev.V2["commit"], "private": True}}
    meta.update(changes)
    return meta


def write(tmp_path, grpo=None, v1=None):
    g, v = tmp_path / "grpo.json", tmp_path / "v1.json"
    g.write_text(json.dumps(grpo or grpo_meta()), encoding="utf-8")
    v.write_text(json.dumps(v1 or BASE), encoding="utf-8")
    valid = tmp_path / "valid.jsonl"
    valid.write_text("".join(json.dumps({
        "id": f"v-{i}", "person": "Test Person",
        "a": {"text": f"Test Person has {i} bikes.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person has {i + 1} bikes.", "written_at": "2025-01-02"},
        "label": "contradiction", "type": None, "reason": "Made up."}) + "\n" for i in range(10)), encoding="utf-8")
    return g, v, valid


class Job:
    """A fake GPU job: records its arguments and reports the fingerprints it is told to."""

    def __init__(self, report=FILES):
        self.calls, self.report = [], report

    def __call__(self, base, base_revision, adapter, adapter_revision, items, **kw):
        self.calls.append((base, base_revision, adapter, adapter_revision, len(items), kw))
        return {"results": [{"id": i["id"], "raw": "{}"} for i in items], "load_seconds": 0, "answer_seconds": 0,
                "peak_vram_gib": 0, "adapter_files_sha256": dict(self.report)}


def no_test_set(monkeypatch):
    monkeypatch.setattr(judge.frozen, "load_frozen", lambda *a, **k: pytest.fail("the test set was read"))


def run(tmp_path, job, out, grpo=None, v1=None, smoke=True):
    g, v, valid = write(tmp_path, grpo, v1)
    return ev.evaluate(smoke, out, job, train_meta_path=g, valid_path=valid, model="v2", v1_meta_path=v)


def test_the_real_records_name_v2_on_v1s_base():
    grpo = json.loads((REPO / ev.V2["record"]).read_text(encoding="utf-8"))
    v1 = json.loads((REPO / ev.TRAIN_META).read_text(encoding="utf-8"))
    upload, files = ev.check_v2(grpo, v1)
    assert (upload["repo"], upload["commit"], upload["checkpoint"]) == (
        "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora", "0964d031073408707397ce7438fc7df7b8a772c0", "step-250")
    step_250 = next(s for s in grpo["saved"] if s["name"] == "step-250")
    assert files == {k: step_250["files"][k] for k in ("adapter_config.json", "adapter_model.safetensors")}


def test_a_v2_smoke_run_checks_the_files_and_writes_its_own_names(tmp_path, monkeypatch):
    no_test_set(monkeypatch)
    job = Job()
    run(tmp_path, job, tmp_path / "out")
    assert job.calls == [("test/base", "base-rev", ev.V2["repo"], ev.V2["commit"], 8 * len(VERSIONS),
                          {"expected_files": FILES})]
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == sorted([f"qwen3-4b-grpo-v2-smoke.{v}.jsonl" for v in VERSIONS] + ["qwen3-4b-grpo-v2-smoke.meta.json"])
    meta = json.loads((tmp_path / "out/qwen3-4b-grpo-v2-smoke.meta.json").read_text(encoding="utf-8"))
    assert (meta["adapter_checkpoint"], meta["adapter_revision"]) == ("step-250", ev.V2["commit"])
    assert meta["adapter_files_sha256"] == meta["adapter_files_expected"] == FILES
    assert meta["reward_sha256_training"] == "r" * 64 and len(meta["reward_sha256"]) == 64


@pytest.mark.parametrize("change, message", [
    ({"upload": {"checkpoint": "step-250", "repo": ev.V2["repo"], "commit": "other", "private": True}}, "not v2"),
    ({"choice": {"chosen": "step-300"}}, "validation set chose"),
    ({"base_revision": "other-rev"}, "not v1's"),
    ({"saved": [{"name": "step-250", "files": {"adapter_config.json": "a" * 64}}]}, "no fingerprints"),
])
def test_a_wrong_record_stops_before_anything_runs(tmp_path, monkeypatch, change, message):
    no_test_set(monkeypatch)
    job = Job()
    with pytest.raises(SystemExit, match=message):
        run(tmp_path, job, tmp_path / "out", grpo=grpo_meta(**change))
    assert job.calls == [] and not (tmp_path / "out").exists()


def test_v2_never_writes_into_v1s_folder(tmp_path, monkeypatch):
    no_test_set(monkeypatch)
    job = Job()
    with pytest.raises(SystemExit, match="holds v1's answers"):
        run(tmp_path, job, REPO / ev.OUT)
    assert job.calls == []


def test_wrong_fingerprints_from_the_job_save_nothing(tmp_path, monkeypatch):
    no_test_set(monkeypatch)
    job = Job(report={**FILES, "adapter_model.safetensors": "d" * 64})
    with pytest.raises(SystemExit, match="expected adapter fingerprints"):
        run(tmp_path, job, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_files_problem_names_each_wrong_file():
    assert ev.files_problem(FILES, FILES) is None
    assert "adapter_model.safetensors" in ev.files_problem(FILES, {"adapter_config.json": "a" * 64})


def test_v1_calls_the_job_exactly_as_before(tmp_path, monkeypatch):
    no_test_set(monkeypatch)
    _, _, valid = write(tmp_path)
    record = tmp_path / "v1-train.json"
    record.write_text(json.dumps({**BASE, "prompt_sha256": prompt_sha256(), "choice": {"chosen": "epoch-2"},
                                  "upload": {"checkpoint": "epoch-2", "repo": "test/v1", "commit": "c1",
                                             "private": True}}), encoding="utf-8")
    job = Job()
    ev.evaluate(True, tmp_path / "out", job, train_meta_path=record, valid_path=valid)
    assert job.calls == [("test/base", "base-rev", "test/v1", "c1", 8 * len(VERSIONS), {})]
    meta = json.loads((tmp_path / "out/qwen3-4b-trained-v1-smoke.meta.json").read_text(encoding="utf-8"))
    assert "reward_sha256" not in meta and "adapter_files_expected" not in meta
