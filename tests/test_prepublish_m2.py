"""Tests for the M2 parts of scripts/prepublish_check.py and scripts/publish_hf.py (M2-08).

Made-up data only, in temporary folders: no test here opens the real frozen test files,
the saved answers or Hugging Face. Each check is fed a problem and must report it.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load("prepublish_check_m2", "scripts/prepublish_check.py")
publish_hf = load("publish_hf_m2", "scripts/publish_hf.py")
from judge.frozen import FrozenFileError  # noqa: E402  (src is on the path once the scripts load)

SCORE_TABLE = """| Model | Test set | Version | Caught / said | Precision | Recall | Unclear answers | Invalid answers |
|---|---|---|---|---|---|---|---|
| v1 | hard | as is | 2 / 3 | 0.667 | 1.000 | 0 | 0 |
"""
SCORES = {("v1", "hard", "as_is"): {"caught_said": "2 / 3", "precision": "0.667", "recall": "1.000",
                                     "unclear": "0", "invalid": "0"}}


def pair(i, label, hard, type_=None):
    return {"id": f"h-{i:03d}", "person": f"Person {i}", "a": {"text": f"A{i}", "written_at": "2025-01-01"},
            "b": {"text": f"B{i}", "written_at": "2025-01-01"}, "label": label, "type": type_, "hard": hard, "reason": "r"}


def answer(i, label, type_=None):
    return {"id": f"h-{i:03d}", "raw": json.dumps({"label": label, "type": type_, "reason": "x"})}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def fake_root(tmp_path):
    """A tiny repo: two frozen test files with STATE.md fingerprints, and saved answers for v1 and v2."""
    hard = [pair(1, "contradiction", "paraphrase"), pair(2, "contradiction", "paraphrase"),
            pair(3, "non_contradiction", "rule5_bait", "merely_related")]
    m1 = [{k: v for k, v in p.items() if k != "hard"} for p in hard]
    write_jsonl(tmp_path / "data/m2/test-hard.jsonl", hard)
    write_jsonl(tmp_path / "data/m1/test.jsonl", m1)
    sha = {p: hashlib.sha256((tmp_path / p).read_bytes()).hexdigest() for p in ("data/m2/test-hard.jsonl", "data/m1/test.jsonl")}
    (tmp_path / "STATE.md").write_text("## Frozen files\n" + "".join(f"- `{p}`, SHA-256 `{s}`, frozen.\n" for p, s in sha.items()))
    v1 = [answer(1, "contradiction"), answer(2, "unclear"), answer(3, "contradiction")]
    v2 = [answer(1, "contradiction"), answer(2, "contradiction"), answer(3, "non_contradiction", "merely_related")]
    for (model, _), prefix in check.M2_SOURCES.items():
        for version in check.VERSIONS:
            write_jsonl(tmp_path / f"{prefix}.{version}.jsonl", v1 if model == "v1" else v2)
    return tmp_path


def test_three_places_rounds_half_up_from_counts():
    assert check.three_places(143, 148) == "0.966"
    assert check.three_places(1, 8) == "0.125"
    assert check.three_places(2, 3) == "0.667"


def test_score_and_trap_facts_come_from_the_saved_answers(fake_root):
    rows = check.m2_rows(fake_root)
    scores = check.m2_score_facts(rows)
    assert scores[("v1", "hard", "as_is")]["caught_said"] == "1 / 2"  # one caught, one false alarm
    assert scores[("v1", "hard", "as_is")]["unclear"] == "1"
    assert scores[("v2", "m1", "no_dates")]["precision"] == "1.000"
    traps = check.m2_trap_facts(rows)
    assert traps[("paraphrase (contradiction)", "v1")] == "1 / 1 / 1"
    assert traps[("paraphrase (contradiction)", "v2")] == "2 / 2 / 2"
    assert traps[("rule5_bait (merely_related)", "pairs")] == 1


def test_changed_frozen_test_file_stops_the_check(fake_root):
    path = fake_root / "data/m2/test-hard.jsonl"
    path.write_text(path.read_text().replace("A1", "A one"))
    with pytest.raises(FrozenFileError):
        check.m2_rows(fake_root)


def test_wrong_score_and_unknown_row_are_caught():
    assert check.check_m2_score_tables(SCORE_TABLE, SCORES) == []
    wrong = SCORE_TABLE.replace("| 0.667 |", "| 0.668 |")
    assert check.check_m2_score_tables(wrong, SCORES) == ["v1 hard as is, Precision: text says 0.668, answers say 0.667"]
    unknown = SCORE_TABLE.replace("| v1 | hard |", "| v3 | hard |")
    assert check.check_m2_score_tables(unknown, SCORES)[0].startswith("row not traced")


def test_wrong_trap_count_is_caught():
    table = "| Trap (right answer) | Pairs | v1 | v2 |\n|---|---|---|---|\n| paraphrase (contradiction) | 2 | 1 / 1 / 1 | 2 / 2 / 2 |\n"
    traps = {("paraphrase (contradiction)", "pairs"): 2, ("paraphrase (contradiction)", "v1"): "1 / 1 / 1",
             ("paraphrase (contradiction)", "v2"): "2 / 2 / 2"}
    assert check.check_m2_trap_tables(table, traps) == []
    assert check.check_m2_trap_tables(table.replace("| 2 / 2 / 2 |", "| 2 / 1 / 2 |"), traps) == \
        ["trap paraphrase (contradiction), v2: text says 2 / 1 / 2, answers say 2 / 2 / 2"]


def test_dataset_card_tables_are_checked_against_the_files():
    data = {"test-hard.jsonl": [pair(1, "contradiction", "paraphrase"), pair(2, "contradiction", "paraphrase")]}
    sha = {"test-hard.jsonl": "a" * 64}
    files = f"| File | Pairs | SHA-256 |\n|---|---|---|\n| `test-hard.jsonl` | 2 | `{'a' * 64}` |\n"
    traps = "| Trap | Right answer | Pairs |\n|---|---|---|\n| `paraphrase` | contradiction | 2 |\n"
    assert check.check_m2_dataset_tables(files + "\n" + traps, data, sha) == []
    assert check.check_m2_dataset_tables(files.replace("| 2 |", "| 3 |"), data, sha) == \
        ["test-hard.jsonl pairs: card says 3, file has 2"]
    assert check.check_m2_dataset_tables(files.replace("a" * 64, "b" * 64), data, sha) == ["test-hard.jsonl SHA-256 differs from the file"]
    assert check.check_m2_dataset_tables(traps.replace("| 2 |", "| 1 |"), data, sha) == \
        ["trap paraphrase (contradiction): card says 1, test file has 2"]


def test_untraced_number_in_m2_prose_is_caught():
    facts = {"138", "150", "139"}
    assert check.untraced_numbers("It caught 138 of 150; the bar was 139 (D-060).", facts) == []
    assert check.untraced_numbers("It caught 137 of 150.", facts) == ["137"]


def test_visibility_rules():
    assert check.visibility_problems("eskora/memory-contradiction-pairs-hard", private=True) == []
    assert check.visibility_problems("eskora/memory-contradiction-pairs-hard", private=False) == ["NOT PRIVATE"]
    assert check.visibility_problems("eskora/m2-contradiction-judge-qwen3-4b-grpo-lora", private=False) == ["NOT PRIVATE"]
    assert check.visibility_problems("eskora/m1-contradiction-judge-qwen3-4b-gguf", private=False) == []
    assert check.visibility_problems("eskora/m1-contradiction-judge-qwen3-4b-gguf", private=True) != []


def test_m2_dataset_repo_files_are_compared(tmp_path):
    for name, path in check.DATA_FILES_M2.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(f"content of {name}\n")
    frozen = {check.FROZEN_M2: hashlib.sha256(b"content of test-hard.jsonl\n").hexdigest()}
    remote = {name: tmp_path / path for name, path in check.DATA_FILES_M2.items()}
    assert check.m2_dataset_problems(list(remote), remote.__getitem__, frozen, tmp_path) == []
    (tmp_path / "remote-test").write_text("edited\n")
    (tmp_path / "remote-pool").write_text("edited\n")
    changed = {**remote, "test-hard.jsonl": tmp_path / "remote-test", "rl-pool.jsonl": tmp_path / "remote-pool"}
    problems = check.m2_dataset_problems(list(changed), changed.__getitem__, frozen, tmp_path)
    assert problems == ["test-hard.jsonl is not the frozen file (SHA-256 differs from STATE.md)",
                        "rl-pool.jsonl differs from data/m2/rl-pool.jsonl"]
    assert check.m2_dataset_problems(["test-hard.jsonl"], remote.__getitem__, frozen, tmp_path)[0] == "train-hard.jsonl missing"


def test_write_up_must_not_name_aeverum_or_hold_an_em_dash():
    assert check.public_facing(check.WRITE_UP)
    report = check.Report()
    check.check_texts(report, {check.WRITE_UP: "Built for Aeverum — here.\n"})
    failed = [line for line in report.lines if line.startswith("FAIL")]
    assert len(failed) == 2 and all(check.WRITE_UP in line for line in failed)


# ---------- publish_hf.py, M2 ----------

class FakeApi:
    def __init__(self, private=True):
        self.private = private
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def repo_info(self, repo_id, repo_type):
        self.calls.append(("repo_info", repo_id, repo_type))
        return SimpleNamespace(private=self.private)

    def create_commit(self, repo_id, operations, **kwargs):
        self.calls.append(("create_commit", repo_id, [op.path_in_repo for op in operations]))
        return SimpleNamespace(oid="c0ffee")

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs["repo_id"], kwargs["path_in_repo"]))
        return SimpleNamespace(oid="c0ffee")


def operation(path_in_repo, path_or_fileobj):
    return SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)


@pytest.fixture
def m2_root(tmp_path):
    for local in [*publish_hf.DATA_FILES_M2.values(), publish_hf.DATASET_M2_CARD, publish_hf.LORA_M2_CARD]:
        (tmp_path / local).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / local).write_text("{}\n" if local.endswith(".jsonl") else f"made-up {local}\n")
    frozen = publish_hf.DATA_FILES_M2[publish_hf.FROZEN_M2]
    sha = hashlib.sha256((tmp_path / frozen).read_bytes()).hexdigest()
    (tmp_path / "STATE.md").write_text(f"## Frozen files\n- `{frozen}`, SHA-256 `{sha}`, frozen.\n")
    return tmp_path, sha


def uploads(api):
    return [c for c in api.calls if c[0] in ("create_commit", "upload_file")]


def test_m2_dataset_goes_to_a_private_repo_with_all_files(m2_root):
    root, sha = m2_root
    api = FakeApi()
    entries = publish_hf.upload_dataset_m2(api, root, operation)
    assert api.calls[0] == ("create_repo", publish_hf.DATASET_M2, {"repo_type": "dataset", "private": True, "exist_ok": True})
    assert uploads(api) == [("create_commit", publish_hf.DATASET_M2,
                             ["test-hard.jsonl", "train-hard.jsonl", "valid-hard.jsonl", "rl-pool.jsonl", "README.md"])]
    assert entries[0]["files"]["test-hard.jsonl"]["sha256"] == sha  # the frozen value, from STATE.md


def test_frozen_fingerprint_is_not_computed_by_the_upload_script(m2_root, monkeypatch):
    root, _ = m2_root
    seen = []
    real = publish_hf.sha256
    monkeypatch.setattr(publish_hf, "sha256", lambda path: seen.append(Path(path).name) or real(path))
    publish_hf.upload_dataset_m2(FakeApi(), root, operation)
    assert "test-hard.jsonl" not in seen and "train-hard.jsonl" in seen


def test_no_m2_upload_to_a_public_repo(m2_root):
    root, _ = m2_root
    for upload in (lambda api: publish_hf.upload_dataset_m2(api, root, operation),
                   lambda api: publish_hf.upload_card_m2(api, root)):
        api = FakeApi(private=False)
        with pytest.raises(publish_hf.NotPrivate):
            upload(api)
        assert uploads(api) == []


def test_changed_hard_test_file_stops_the_m2_upload(m2_root):
    root, _ = m2_root
    (root / publish_hf.DATA_FILES_M2[publish_hf.FROZEN_M2]).write_text('{"edited": true}\n')
    api = FakeApi()
    with pytest.raises(FrozenFileError):
        publish_hf.upload_dataset_m2(api, root, operation)
    assert api.calls == []


def test_card_m2_goes_to_the_v2_repo(m2_root):
    root, _ = m2_root
    api = FakeApi()
    entries = publish_hf.upload_card_m2(api, root)
    assert uploads(api) == [("upload_file", publish_hf.LORA_M2, "README.md")]
    assert entries[0]["repo"] == publish_hf.LORA_M2
