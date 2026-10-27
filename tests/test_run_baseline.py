"""Tests for scripts/run_baseline.py: which file it reads, and that it never replaces saved answers.

Everything runs in temporary folders with a made-up STATE.md and fake GPU jobs. No Modal, no GPU.
"""

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

import judge.frozen
from judge.score import VERSIONS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_baseline_script", REPO / "scripts" / "run_baseline.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)  # builds Modal objects lazily; no network, no GPU


def pair(i, prefix="h"):
    return {
        "id": f"{prefix}-{i:03d}",
        "person": "Test Person",
        "a": {"text": f"Test Person likes colour {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person likes colour {i + 1}.", "written_at": "2025-01-02"},
        "label": "contradiction",
        "type": None,
        "reason": "They name different favourite colours.",
    }


def write(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_repo(tmp_path):
    """A made-up repo: the M1 test set and a second frozen file, both listed in STATE.md, plus a validation file."""
    repo = tmp_path / "repo"
    m1 = write(repo / "data/m1/test.jsonl", [pair(i, "t") for i in range(3)])
    hard = write(repo / "data/m2/test-hard.jsonl", [pair(i) for i in range(5)])
    write(repo / "data/m1/valid.jsonl", [pair(i, "v") for i in range(12)])
    (repo / "STATE.md").write_text(
        "## Frozen files\n"
        f"- `data/m1/test.jsonl`, 3 pairs, SHA-256 `{m1}`, frozen 2025-01-03.\n"
        f"- `data/m2/test-hard.jsonl`, 5 pairs, SHA-256 `{hard}`, frozen 2025-01-04.\n",
        encoding="utf-8")
    return repo, {"data/m1/test.jsonl": m1, "data/m2/test-hard.jsonl": hard}


class Fake:
    """A fake GPU job that records which pair ids it was asked about."""

    def __init__(self):
        self.calls = []

    def __call__(self, repo, template_kwargs, items):
        self.calls.append((repo, [i["id"] for i in items]))
        return {"results": [{"id": i["id"], "raw": "{}"} for i in items], "load_seconds": 0, "answer_seconds": 0,
                "peak_vram_gib": 0}


def run(repo, out, fake, **kwargs):
    kwargs.setdefault("state_path", repo / "STATE.md")
    kwargs.setdefault("valid_path", repo / "data/m1/valid.jsonl")
    return script.run_models(["qwen3-8b"], out, fake, **kwargs)


def test_the_default_test_file_is_still_the_m1_test_set():
    assert script.TEST_PATH == "data/m1/test.jsonl"
    assert inspect.signature(script.main.info.raw_f).parameters["test"].default == "data/m1/test.jsonl"
    assert inspect.signature(script.run_models).parameters["test_path"].default == "data/m1/test.jsonl"
    assert inspect.signature(script.main.info.raw_f).parameters["out"].default == "results/m1/baseline-v2"


def test_the_default_reads_the_m1_test_set(tmp_path, monkeypatch):
    repo, digests = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    fake = Fake()
    run(repo, "answers", fake)
    ids = {i.split("/", 1)[1] for i in fake.calls[0][1]}
    assert ids == {"t-000", "t-001", "t-002"}
    meta = json.loads((repo / "answers/qwen3-8b.meta.json").read_text(encoding="utf-8"))
    assert (meta["data"], meta["data_sha256"], meta["pairs"]) == ("data/m1/test.jsonl", digests["data/m1/test.jsonl"], 3)


def test_another_frozen_file_is_read_and_recorded(tmp_path):
    repo, digests = make_repo(tmp_path)
    fake = Fake()
    run(repo, tmp_path / "answers", fake, test_path=repo / "data/m2/test-hard.jsonl")
    assert len(fake.calls[0][1]) == 5 * len(VERSIONS)
    meta = json.loads((tmp_path / "answers/qwen3-8b.meta.json").read_text(encoding="utf-8"))
    assert (meta["data"], meta["data_sha256"]) == ("data/m2/test-hard.jsonl", digests["data/m2/test-hard.jsonl"])
    for version in VERSIONS:
        lines = (tmp_path / f"answers/qwen3-8b.{version}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["id"] for line in lines] == [f"h-{i:03d}" for i in range(5)]


def test_an_edited_test_file_stops_before_any_gpu_job(tmp_path):
    repo, _ = make_repo(tmp_path)
    hard = repo / "data/m2/test-hard.jsonl"
    hard.write_text(hard.read_text(encoding="utf-8").replace("colour 3", "colour 9"), encoding="utf-8")
    fake = Fake()
    with pytest.raises(SystemExit, match="not the frozen"):
        run(repo, tmp_path / "answers", fake, test_path=hard)
    assert fake.calls == [] and not (tmp_path / "answers").exists()


def test_a_file_not_listed_as_frozen_stops_before_any_gpu_job(tmp_path):
    repo, _ = make_repo(tmp_path)
    other = repo / "data/m2/other.jsonl"
    write(other, [pair(1)])
    fake = Fake()
    with pytest.raises(SystemExit, match="not listed under 'Frozen files'"):
        run(repo, tmp_path / "answers", fake, test_path=other)
    assert fake.calls == []


@pytest.mark.parametrize("existing", range(len(VERSIONS) + 1))
def test_any_existing_output_stops_before_the_test_set_is_read(tmp_path, monkeypatch, existing):
    repo, _ = make_repo(tmp_path)
    out = tmp_path / "answers"
    out.mkdir()
    taken = script.output_paths(out, "qwen3-8b")[existing]
    taken.write_text("saved\n", encoding="utf-8")
    monkeypatch.setattr(judge.frozen, "load_frozen", lambda *a, **k: pytest.fail("the test set was read"))
    fake = Fake()
    with pytest.raises(SystemExit, match="already exist"):
        run(repo, out, fake, test_path=repo / "data/m2/test-hard.jsonl")
    assert fake.calls == []
    assert taken.read_text(encoding="utf-8") == "saved\n"
    assert [p.name for p in out.iterdir()] == [taken.name]


def test_one_models_saved_answers_stop_the_whole_run(tmp_path):
    repo, _ = make_repo(tmp_path)
    out = tmp_path / "answers"
    out.mkdir()
    script.output_paths(out, "qwen3-8b")[-1].write_text("saved\n", encoding="utf-8")
    fake = Fake()
    with pytest.raises(SystemExit, match="already exist"):
        script.run_models(["qwen3-4b-instruct-2507", "qwen3-8b"], out, fake, state_path=repo / "STATE.md")
    assert fake.calls == []


def test_a_smoke_run_answers_validation_pairs_under_its_own_names(tmp_path, monkeypatch):
    repo, _ = make_repo(tmp_path)
    out = tmp_path / "answers"
    out.mkdir()
    for path in script.output_paths(out, "qwen3-8b"):
        path.write_text("final\n", encoding="utf-8")
    monkeypatch.setattr(judge.frozen, "load_frozen", lambda *a, **k: pytest.fail("the test set was read"))
    fake = Fake()
    run(repo, out, fake, smoke=True, test_path=repo / "data/m2/test-hard.jsonl")
    ids = sorted({i.split("/", 1)[1] for i in fake.calls[0][1]})
    assert ids == [f"v-{i:03d}" for i in range(script.SMOKE_PAIRS)]
    assert all(p.read_text(encoding="utf-8") == "final\n" for p in script.output_paths(out, "qwen3-8b"))
    assert all(p.exists() for p in script.output_paths(out, "qwen3-8b", smoke=True))
    meta = json.loads(script.output_paths(out, "qwen3-8b", smoke=True)[-1].read_text(encoding="utf-8"))
    assert meta["data"].endswith("data/m1/valid.jsonl") and meta["pairs"] == script.SMOKE_PAIRS
    # A second smoke run into the same folder stops before any GPU job.
    with pytest.raises(SystemExit, match="already exist"):
        run(repo, out, fake, smoke=True)
    assert len(fake.calls) == 1


def test_an_unknown_model_stops_before_anything(tmp_path):
    repo, _ = make_repo(tmp_path)
    fake = Fake()
    with pytest.raises(SystemExit, match="unknown model"):
        script.run_models(["no-such-model"], tmp_path / "answers", fake, state_path=repo / "STATE.md")
    assert fake.calls == [] and not (tmp_path / "answers").exists()
