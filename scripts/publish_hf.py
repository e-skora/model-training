"""Upload the M1 dataset and the three Hugging Face cards, keeping every repo private (D-007).

    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py dataset
    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py cards

`dataset` creates the private dataset repo (if it is missing) and uploads the
three data files with the dataset card. `cards` uploads the two model cards.
Nothing here makes a repo public: it creates repos as private and stops
before any upload to a repo that is not private. Making them public is a
separate step, done only after Elias says so. Each upload is added to
`hf/published.json`.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judge.frozen import frozen_entries  # noqa: E402

DATASET = "eskora/memory-contradiction-pairs"
GGUF = "eskora/m1-contradiction-judge-qwen3-4b-gguf"
LORA = "eskora/m1-contradiction-judge-qwen3-4b-lora"
# (repo, repo type, card file in this repo)
CARDS = [
    (DATASET, "dataset", "hf/memory-contradiction-pairs/README.md"),
    (GGUF, "model", "hf/m1-contradiction-judge-qwen3-4b-gguf/README.md"),
    (LORA, "model", "hf/m1-contradiction-judge-qwen3-4b-lora/README.md"),
]
# File name on Hugging Face -> file in this repo
DATA_FILES = {
    "train.jsonl": "data/m1/train.jsonl",
    "valid.jsonl": "data/m1/valid.jsonl",
    "test.jsonl": "data/m1/test.jsonl",
}
RECORD = "hf/published.json"


class NotPrivate(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_private(api, repo_id, repo_type):
    """Stop unless the repo exists and is private."""
    info = api.repo_info(repo_id, repo_type=repo_type)
    if not info.private:
        raise NotPrivate(f"{repo_type} repo {repo_id} is not private; nothing was uploaded")


def check_frozen_test(root):
    """The test file must still match its fingerprint in STATE.md before it leaves the Mac."""
    frozen = dict(frozen_entries(Path(root) / "STATE.md"))
    expected = frozen.get(DATA_FILES["test.jsonl"])
    actual = sha256(Path(root) / DATA_FILES["test.jsonl"])
    if expected != actual:
        raise SystemExit(f"STOP: {DATA_FILES['test.jsonl']} SHA-256 {actual} is not the frozen {expected}")


def upload_dataset(api, root=REPO, CommitOperationAdd=None):
    if CommitOperationAdd is None:
        from huggingface_hub import CommitOperationAdd

    check_frozen_test(root)
    repo_id, repo_type, card = CARDS[0]
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    require_private(api, repo_id, repo_type)
    files = {**DATA_FILES, "README.md": card}
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(Path(root) / local))
                  for name, local in files.items()]
    commit = api.create_commit(repo_id, operations, repo_type=repo_type,
                               commit_message="M1 dataset: train, validation and test splits, with the dataset card")
    return [entry(repo_id, repo_type, commit, files, root)]


def upload_cards(api, root=REPO):
    entries = []
    for repo_id, repo_type, card in CARDS[1:]:
        require_private(api, repo_id, repo_type)
        commit = api.upload_file(path_or_fileobj=str(Path(root) / card), path_in_repo="README.md",
                                 repo_id=repo_id, repo_type=repo_type, commit_message="Model card for M1 judge v1")
        entries.append(entry(repo_id, repo_type, commit, {"README.md": card}, root))
    return entries


def entry(repo_id, repo_type, commit, files, root):
    return {
        "when_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": repo_id,
        "repo_type": repo_type,
        "private": True,
        "commit": commit.oid,
        "files": {name: {"from": local, "sha256": sha256(Path(root) / local)} for name, local in files.items()},
    }


def record(entries, root=REPO):
    path = Path(root) / RECORD
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(old + entries, indent=2) + "\n")


def main(argv):
    if len(argv) != 2 or argv[1] not in ("dataset", "cards"):
        sys.exit("usage: python scripts/publish_hf.py dataset|cards")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        entries = upload_dataset(api) if argv[1] == "dataset" else upload_cards(api)
    except NotPrivate as error:
        sys.exit(f"STOP: {error}")
    record(entries)
    for e in entries:
        print(f"{e['repo']} ({e['repo_type']}, private): commit {e['commit']}, files {', '.join(e['files'])}")


if __name__ == "__main__":
    main(sys.argv)
