"""Upload datasets and Hugging Face cards, keeping every repo private (D-007).

    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py dataset
    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py cards
    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py dataset-m2
    uv run --with huggingface_hub==1.33.0 python scripts/publish_hf.py card-m2

`dataset` creates the private M1 dataset repo (if it is missing) and uploads the
three data files with the dataset card. `cards` uploads the two M1 model cards.
`dataset-m2` does the same for the M2 hard data (D-062): it checks the frozen hard
test file through `judge.frozen` first and records its frozen fingerprint, so this
script never fingerprints that file itself. `card-m2` uploads v2's model card.
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

from judge.frozen import FrozenFileError, frozen_entries, load_frozen  # noqa: E402

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

# M2 (D-062): the hard data and v2's card.
DATASET_M2 = "eskora/memory-contradiction-pairs-hard"
LORA_M2 = "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora"
DATASET_M2_CARD = "hf/memory-contradiction-pairs-hard/README.md"
LORA_M2_CARD = "hf/m2-contradiction-judge-qwen3-4b-grpo-lora/README.md"
DATA_FILES_M2 = {
    "test-hard.jsonl": "data/m2/test-hard.jsonl",
    "train-hard.jsonl": "data/m2/train-hard.jsonl",
    "valid-hard.jsonl": "data/m2/valid-hard.jsonl",
    "rl-pool.jsonl": "data/m2/rl-pool.jsonl",
}
FROZEN_M2 = "test-hard.jsonl"


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


def entry(repo_id, repo_type, commit, files, root, known=None):
    """One record line. `known` gives fingerprints already checked elsewhere (a frozen file),
    so that file is not read again here."""
    known = known or {}
    return {
        "when_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": repo_id,
        "repo_type": repo_type,
        "private": True,
        "commit": commit.oid,
        "files": {name: {"from": local, "sha256": known.get(name) or sha256(Path(root) / local)}
                  for name, local in files.items()},
    }


def check_frozen_m2(root):
    """The hard test file must match its "Frozen files" entry. Checked only through `judge.frozen`
    (load_frozen stops on any change); returns the frozen fingerprint for the record."""
    root = Path(root)
    load_frozen(root / DATA_FILES_M2[FROZEN_M2], root / "STATE.md")
    return frozen_entries(root / "STATE.md")[DATA_FILES_M2[FROZEN_M2]]


def upload_dataset_m2(api, root=REPO, CommitOperationAdd=None):
    if CommitOperationAdd is None:
        from huggingface_hub import CommitOperationAdd

    frozen_sha = check_frozen_m2(root)
    api.create_repo(DATASET_M2, repo_type="dataset", private=True, exist_ok=True)
    require_private(api, DATASET_M2, "dataset")
    files = {**DATA_FILES_M2, "README.md": DATASET_M2_CARD}
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(Path(root) / local))
                  for name, local in files.items()]
    commit = api.create_commit(DATASET_M2, operations, repo_type="dataset",
                               commit_message="M2 hard data: benchmark, training, validation and RL pool, with the dataset card")
    return [entry(DATASET_M2, "dataset", commit, files, root, known={FROZEN_M2: frozen_sha})]


def upload_card_m2(api, root=REPO):
    require_private(api, LORA_M2, "model")
    commit = api.upload_file(path_or_fileobj=str(Path(root) / LORA_M2_CARD), path_in_repo="README.md",
                             repo_id=LORA_M2, repo_type="model",
                             commit_message="Model card for M2 judge v2 (missed its bar by one pair)")
    return [entry(LORA_M2, "model", commit, {"README.md": LORA_M2_CARD}, root)]


COMMANDS = {"dataset": upload_dataset, "cards": upload_cards,
            "dataset-m2": upload_dataset_m2, "card-m2": upload_card_m2}


def record(entries, root=REPO):
    path = Path(root) / RECORD
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(old + entries, indent=2) + "\n")


def main(argv):
    if len(argv) != 2 or argv[1] not in COMMANDS:
        sys.exit("usage: python scripts/publish_hf.py " + "|".join(COMMANDS))
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        entries = COMMANDS[argv[1]](api)
    except (NotPrivate, FrozenFileError) as error:
        sys.exit(f"STOP: {error}")
    record(entries)
    for e in entries:
        print(f"{e['repo']} ({e['repo_type']}, private): commit {e['commit']}, files {', '.join(e['files'])}")


if __name__ == "__main__":
    main(sys.argv)
