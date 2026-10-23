"""Tests for scripts/publish_hf.py with a fake Hugging Face API.

No network and no real account: the fake records what it was asked. Every
file the script reads or writes is in a temporary folder.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_hf_script", REPO / "scripts" / "publish_hf.py")
publish_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_hf)


def operation(path_in_repo, path_or_fileobj):
    return SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)


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


@pytest.fixture
def root(tmp_path):
    for local in [*publish_hf.DATA_FILES.values(), *(card for _, _, card in publish_hf.CARDS)]:
        path = tmp_path / local
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"made-up content of {local}\n")
    test_sha = hashlib.sha256((tmp_path / "data/m1/test.jsonl").read_bytes()).hexdigest()
    (tmp_path / "STATE.md").write_text(f"## Frozen files\n- `data/m1/test.jsonl`, SHA-256 `{test_sha}`, frozen.\n")
    return tmp_path


def uploads(api):
    return [call for call in api.calls if call[0] in ("create_commit", "upload_file")]


def test_dataset_repo_is_created_private_and_gets_all_files(root):
    api = FakeApi()
    entries = publish_hf.upload_dataset(api, root, operation)
    assert api.calls[0] == ("create_repo", publish_hf.DATASET,
                            {"repo_type": "dataset", "private": True, "exist_ok": True})
    assert uploads(api) == [("create_commit", publish_hf.DATASET, ["train.jsonl", "valid.jsonl", "test.jsonl", "README.md"])]
    assert entries[0]["commit"] == "c0ffee" and entries[0]["private"] is True


def test_no_upload_to_a_public_dataset_repo(root):
    api = FakeApi(private=False)
    with pytest.raises(publish_hf.NotPrivate):
        publish_hf.upload_dataset(api, root, operation)
    assert uploads(api) == []


def test_no_card_upload_to_a_public_model_repo(root):
    api = FakeApi(private=False)
    with pytest.raises(publish_hf.NotPrivate):
        publish_hf.upload_cards(api, root)
    assert uploads(api) == []


def test_cards_go_to_both_model_repos(root):
    api = FakeApi()
    entries = publish_hf.upload_cards(api, root)
    assert uploads(api) == [("upload_file", publish_hf.GGUF, "README.md"), ("upload_file", publish_hf.LORA, "README.md")]
    publish_hf.record(entries, root)
    saved = json.loads((root / publish_hf.RECORD).read_text())
    assert [e["repo"] for e in saved] == [publish_hf.GGUF, publish_hf.LORA]


def test_changed_test_file_stops_the_dataset_upload(root):
    (root / "data/m1/test.jsonl").write_text("edited\n")
    api = FakeApi()
    with pytest.raises(SystemExit):
        publish_hf.upload_dataset(api, root, operation)
    assert api.calls == []
