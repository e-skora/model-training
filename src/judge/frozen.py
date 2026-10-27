"""Load a frozen file only if its SHA-256 matches the "Frozen files" entry in STATE.md.

Every scoring run and model run loads the test set through `load_frozen`, so
an edited or swapped test file stops the run before any answer is scored.
"""

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# A line in STATE.md such as: - `data/m1/test.jsonl`, 200 pairs, SHA-256 `6e2a...`, frozen ...
ENTRY = re.compile(r"`([^`]+)`.*?SHA-256 `([0-9a-f]{64})`")


class FrozenFileError(Exception):
    """The file is not listed as frozen, or its bytes changed."""


def frozen_entries(state_path):
    """Read the "Frozen files" section of STATE.md. Returns {repo path: SHA-256}."""
    entries = {}
    in_section = False
    for line in Path(state_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Frozen files"
            continue
        match = ENTRY.search(line) if in_section else None
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def frozen_key(path, state_path=None):
    """The repo path STATE.md would list `path` under, such as "data/m2/test-hard.jsonl"."""
    state_path = Path(state_path) if state_path else REPO / "STATE.md"
    try:
        return Path(path).resolve().relative_to(state_path.resolve().parent).as_posix()
    except ValueError:
        raise FrozenFileError(f"{path} is outside the repo folder of {state_path}") from None


def load_frozen(path, state_path=None):
    """Check the file against STATE.md, then read it as JSONL from the same bytes."""
    state_path = Path(state_path) if state_path else REPO / "STATE.md"
    if not state_path.is_file():
        raise FrozenFileError(f"{state_path} not found, so the frozen fingerprint cannot be checked")
    key = frozen_key(path, state_path)
    expected = frozen_entries(state_path).get(key)
    if expected is None:
        raise FrozenFileError(f"{key} is not listed under 'Frozen files' in {state_path}")
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise FrozenFileError(f"{key} SHA-256 is {digest}, not the frozen {expected}")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
