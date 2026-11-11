"""Check everything that will go public, and report. It changes nothing (M1 Step 9).

    uv run --with huggingface_hub==1.33.0 python scripts/prepublish_check.py
    uv run python scripts/prepublish_check.py --local     (skip the Hugging Face repos)

What it checks, at git HEAD (and, for secrets and model files, all of git history)
and in the five Hugging Face repos (M1's three, public since D-034; M2's two, which must
be private until Claude in Cowork's check, D-062). M2-08 added the M2 write-up
`docs/m2-what-we-learned.md`, the two M2 cards and the M2 data; the frozen hard test
file is read only through `judge.frozen`.
1. No secrets: no tokens or keys in any file, no `.env` or key files.
2. No model files in git.
3. No em dashes in public text: every README in git, the cards in `hf/`, the post drafts,
   and the cards on Hugging Face.
4. No real personal data and no Aeverum text: no e-mail addresses or phone numbers, no
   Mara Lindqvist, and the word "Aeverum" nowhere in the cards, the README, the posts or
   the data. Copied text cannot be proved absent without reading Aeverum's files, which
   rule 6 keeps this repo away from, so that part is checked by name only.
5. Every number in the cards and the README traces to a results file: score tables are
   recomputed from the saved answers; fingerprints, counts, times and costs come from the
   data, the `.meta.json` records and `runs/RUNLOG.md`; example pairs and answers must be
   real ones; the prompt and the Modelfile must match the repo's.
On Hugging Face it also checks that each repo is private, that its card is the one in `hf/`,
and that the dataset files are the ones in `data/m1/`.

Each line starts PASS, FAIL or NOTE. NOTE is something to know that does not block. The exit
code is 1 if anything failed.
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judge.frozen import frozen_entries, load_frozen  # noqa: E402
from judge.prompt import SYSTEM_MESSAGE, prompt_sha256  # noqa: E402
from judge.reward import REWARD  # noqa: E402
from judge.score import VERSIONS, load_jsonl, score, trap_key  # noqa: E402

EM_DASH = "—"
CARDS = {
    "eskora/memory-contradiction-pairs": ("dataset", "hf/memory-contradiction-pairs/README.md"),
    "eskora/m1-contradiction-judge-qwen3-4b-gguf": ("model", "hf/m1-contradiction-judge-qwen3-4b-gguf/README.md"),
    "eskora/m1-contradiction-judge-qwen3-4b-lora": ("model", "hf/m1-contradiction-judge-qwen3-4b-lora/README.md"),
}
DATA_FILES = {"train.jsonl": "data/m1/train.jsonl", "valid.jsonl": "data/m1/valid.jsonl", "test.jsonl": "data/m1/test.jsonl"}
# Texts whose numbers must trace: the cards and the GitHub front page.
NUMBERED_TEXT = [card for _, card in CARDS.values()] + ["README.md"]

# Saved answers behind each row name used in a score table.
SCORE_SOURCES = {
    "gguf": "results/m1/gguf-v1/m1-judge-v1",
    "adapter": "results/m1/trained-v1/qwen3-4b-trained-v1",
    "qwen3-4b": "results/m1/baseline-v2/qwen3-4b-instruct-2507",
    "qwen3-8b": "results/m1/baseline-v2/qwen3-8b",
}
ROW_NAMES = [  # (start of the row's first cell, source)
    ("This file (GGUF", "gguf"), ("GGUF version", "gguf"), ("Trained judge (GGUF", "gguf"),
    ("This adapter", "adapter"), ("Trained adapter", "adapter"),
    ("Qwen3-4B-Instruct-2507, untrained", "qwen3-4b"), ("Qwen3-8B, untrained", "qwen3-8b"),
]
VERSION_NAMES = {"as is": "as_is", "swapped": "swapped", "no dates": "no_dates"}
METRIC_COLUMNS = {"precision": "precision", "recall": "recall", "accuracy": "accuracy",
                  "unclear answers": "unclear", "invalid answers": "invalid"}
# Numbers from prose records rather than from a results file: (number, file, phrase it must contain).
TRACED = [
    ("86", "STATE.md", "rewrote 86 pairs"),
    ("20", "DECISIONS.md", "spot-checked 20 of the 200 test pairs"),
    ("60", "data/m1/README.md", "more than 60% shared words"),
    ("0.85", "milestones/M1-contradiction-judge/SPEC.md", "precision 0.85"),
    ("0.80", "milestones/M1-contradiction-judge/SPEC.md", "recall 0.80"),
]
L4_PRICE = ("0.000222", "results/m1/trained-v1/summary.md", "L4 $0.000222 a second")

# ---------- M2 (M2-08): the write-up, the two new repos ----------
# Hugging Face repos and whether each must be private. M1's were made public on D-034;
# the M2 repos stay private until Claude in Cowork checks them (D-062).
VISIBILITY = {
    "eskora/memory-contradiction-pairs": False,
    "eskora/m1-contradiction-judge-qwen3-4b-gguf": False,
    "eskora/m1-contradiction-judge-qwen3-4b-lora": False,
    "eskora/memory-contradiction-pairs-hard": True,
    "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora": True,
}
CARDS_M2 = {
    "eskora/memory-contradiction-pairs-hard": ("dataset", "hf/memory-contradiction-pairs-hard/README.md"),
    "eskora/m2-contradiction-judge-qwen3-4b-grpo-lora": ("model", "hf/m2-contradiction-judge-qwen3-4b-grpo-lora/README.md"),
}
WRITE_UP = "docs/m2-what-we-learned.md"
DATA_FILES_M2 = {"test-hard.jsonl": "data/m2/test-hard.jsonl", "train-hard.jsonl": "data/m2/train-hard.jsonl",
                 "valid-hard.jsonl": "data/m2/valid-hard.jsonl", "rl-pool.jsonl": "data/m2/rl-pool.jsonl"}
FROZEN_M2 = "data/m2/test-hard.jsonl"  # read only through judge.frozen.load_frozen
# Texts whose numbers must trace to M2's records.
NUMBERED_TEXT_M2 = [WRITE_UP] + [card for _, card in CARDS_M2.values()]
# Saved answers behind each (model, test set) row of an M2 score table.
M2_SOURCES = {
    ("v1", "hard"): "results/m2/baseline/trained-v1/qwen3-4b-trained-v1",
    ("v2", "hard"): "results/m2/v2/hard/qwen3-4b-grpo-v2",
    ("v1", "m1"): "results/m1/trained-v1/qwen3-4b-trained-v1",
    ("v2", "m1"): "results/m2/v2/m1/qwen3-4b-grpo-v2",
}
M2_TESTS = {"hard": FROZEN_M2, "m1": "data/m1/test.jsonl"}
M2_COLUMNS = {"caught / said": "caught_said", "precision": "precision", "recall": "recall",
              "unclear answers": "unclear", "invalid answers": "invalid"}
# RUNLOG rows of the write-up's cost table, each a sum of these RUNLOG lines (matched by text).
M2_COST_GROUPS = {
    "smoke": ["Smoke test, v1 adapter, 8 validation pairs", "Smoke test, untrained Qwen3-4B-Instruct-2507 and Qwen3-8B"],
    "hard baselines": ["Untrained Qwen3-8B on the frozen hard test set", "v1 adapter on the frozen hard test set",
                       "Untrained Qwen3-4B-Instruct-2507 on the frozen hard test set"],
    "v2 scoring": ["Smoke run of v2", "v2 answers the frozen hard test set once", "v2 answers the frozen M1 test set once"],
}
POOL_SMOKE_BILLS = "the two smoke runs above $0.02 and $0.03"
# Numbers in the M2 texts that come from prose records: (number, file, phrase it must contain).
TRACED_M2 = [
    ("8", "LESSONS.md", "8 \"vegetarian orders meat\" pairs"),
    ("7", "LESSONS.md", "7 \"never owned a car, drives one\""),
    ("11", "LESSONS.md", "Eleven helper agents wrote the M2 training pairs"),
    ("33", "reviews/M2-03-blind-review.md", "the 33 `unclear` pairs"),
    ("32", "reviews/M2-03-blind-review.md", "The 32 disputed `unclear` pairs are dropped"),
    ("296", "reviews/M2-01-blind-review.md", "same as the file on 296 of 300"),
    ("330", "reviews/M2-03-blind-review.md", "Blind judging of 330 pairs"),
    ("199", "reviews/M2-03-blind-review.md", "| Training, not `unclear` | 200 | 199 |"),
    ("96", "reviews/M2-03-blind-review.md", "| Validation, not `unclear` | 97 | 96 |"),
    ("14", "results/m2/baseline/summary.md", "14 of those are `unclear` answers"),
    ("16", "results/m2/baseline/summary.md", "At its worst the adapter misses 16 contradictions"),
    ("60", "docs/m2-lessons-and-fixes.md", "The last 60 or so steps taught almost nothing"),
    ("0.45", "results/m2/grpo/full-run-summary.md", "rose from 0.45 in the first 40 steps to about 0.8"),
    ("0.8", "results/m2/grpo/full-run-summary.md", "rose from 0.45 in the first 40 steps to about 0.8"),
    ("40", "results/m2/grpo/full-run-summary.md", "rose from 0.45 in the first 40 steps to about 0.8"),
    ("0.7", "milestones/M3-rl-judge-attempt-2/SPEC.md", "stays above 0.7 for 40 steps"),
    ("0.02", "docs/m2-lessons-and-fixes.md", "one false alarm moves precision by about 0.02"),
    ("0.03", "milestones/M2-rl-judge/SPEC.md", "at least 0.03 (5 pairs)"),
    ("0.01", "milestones/M2-rl-judge/SPEC.md", "no more than 0.01 lower"),
    ("600", "milestones/M3-rl-judge-attempt-2/SPEC.md", "600 pairs, 600 different people, 300 contradictions"),
    ("300", "milestones/M3-rl-judge-attempt-2/SPEC.md", "600 pairs, 600 different people, 300 contradictions"),
    ("200", "milestones/M3-rl-judge-attempt-2/SPEC.md", "200 pairs, 100 contradictions"),
    ("100", "milestones/M3-rl-judge-attempt-2/SPEC.md", "200 pairs, 100 contradictions"),
    ("45", "milestones/M3-rl-judge-attempt-2/SPEC.md", "writes 45 candidates"),
    ("15", "milestones/M3-rl-judge-attempt-2/SPEC.md", "about 15 minutes"),
    ("1,018", "results/m2/pool/summary.md", "1,018"),
    ("1.02", "results/m2/grpo/full-run-summary.md", "estimate $1.02"),
    ("0.71", "results/m2/grpo/full-run-summary.md", "estimate $0.71"),
    ("71.3", "results/m2/grpo/full-run-summary.md", "314 steps in 71.3 minutes"),
    ("250", "results/m2/grpo/full-run-summary.md", "Steps 250, 300 and 314 tie on every count"),
    ("50", "results/m2/grpo/full-run-summary.md", "the checkpoints at 50, 100, 150"),
    ("262", "results/m2/pool/summary.md", "262"),
    ("52", "results/m2/pool/summary.md", "52"),
    ("2", "milestones/M2-rl-judge/SPEC.md", "About $2 to $8 for all of M2"),
    ("8", "milestones/M2-rl-judge/SPEC.md", "About $2 to $8 for all of M2"),
]

SECRET_PATTERNS = {
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "Modal token": r"\b(?:ak|as)-[A-Za-z0-9]{20,}",
    "OpenAI or Anthropic key": r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}",
    "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "AWS key": r"\bAKIA[0-9A-Z]{16}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "password or key in a setting": r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]",
}
SECRET_FILE_NAMES = re.compile(r"(^|/)(\.env(\..*)?|[^/]*\.env|[^/]*\.pem|[^/]*\.key|id_rsa[^/]*|credentials[^/]*)$")
MODEL_EXTENSIONS = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".pkl", ".npz")
BIG_FILE_BYTES = 5_000_000
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+\d[\d ()-]{6,}\d")  # an international number: + then at least 8 digits (counted below)
BANNED_NAMES = re.compile(r"\b(?:Mara|Lindqvist)\b")
HOME_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+")
HEX = re.compile(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{64}\b")

# Names that contain digits but are not claims (model names, formats, addresses, labels).
NAME_PATTERNS = [
    r"https?://\S+", r"\]\([^)]*\)", r"\b\d{4}-\d{2}-\d{2}\b", r"Qwen3-\d+B(?:-Instruct-\d+)?", r"SHA-256",
    r"\b\d+-bit\b", r"\b4 bits\b", r"\b4 billion\b", r"\bM4 Pro\b", r"\bL4\b", r"\b(?:t|tr|v)-\d+\b",
    r"127\.0\.0\.1:\d+", r"Apache 2\.0", r"CC BY 4\.0", r"\bM\d\b", r"^\s*\d+\.\s", r"shasum -a 256",
    r"\bD-\d{3}\b",
]
SPECIAL_NUMBERS = r"\d+e-\d+|\d+\.\d+\.\d+"  # 2e-4, version numbers such as 0.24.0


class Report:
    def __init__(self):
        self.lines = []

    def add(self, status, text):
        self.lines.append(f"{status:<5} {text}")

    @property
    def failed(self):
        return any(line.startswith("FAIL") for line in self.lines)


def rounded(value, places):
    """Round half up, the way the results summaries do (0.965 becomes 0.97)."""
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def two_places(top, bottom):
    """A ratio rounded from exact counts, so 193/200 is 0.97 and not the float's 0.96."""
    return str((Decimal(top) / Decimal(bottom)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------- reading git ----------

def git(*args, root=REPO):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def head_files(root=REPO):
    """{path: size in bytes} for every file at HEAD."""
    files = {}
    for line in git("ls-tree", "-r", "-l", "-z", "HEAD", root=root).split(b"\0"):
        if line:
            meta, path = line.split(b"\t", 1)
            files[path.decode()] = int(meta.split()[3])
    return files


def head_text(path, root=REPO):
    return git("show", f"HEAD:{path}", root=root).decode("utf-8", errors="replace")


def history_blobs(root=REPO):
    """{blob id: (path, size)} for every file in every commit of every branch."""
    named = {}
    for line in git("rev-list", "--objects", "--all", root=root).decode().splitlines():
        sha, _, path = line.partition(" ")
        if path:
            named[sha] = path
    checked = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                             cwd=root, input="\n".join(named).encode(), capture_output=True, check=True).stdout
    blobs = {}
    for line in checked.decode().splitlines():
        sha, kind, size = line.split()
        if kind == "blob":
            blobs[sha] = (named[sha], int(size))
    return blobs


def blob_text(sha, root=REPO):
    return git("cat-file", "blob", sha, root=root).decode("utf-8", errors="replace")


# ---------- text checks ----------

def find_secrets(text):
    """Names of the secret patterns found in the text."""
    return sorted(name for name, pattern in SECRET_PATTERNS.items() if re.search(pattern, text))


def is_secret_file(path):
    return bool(SECRET_FILE_NAMES.search(path))


def is_model_file(path, size):
    return path.lower().endswith(MODEL_EXTENSIONS) or size > BIG_FILE_BYTES


def em_dash_lines(text):
    return [n for n, line in enumerate(text.splitlines(), 1) if EM_DASH in line]


def personal_data(text):
    phones = [p for p in PHONE.findall(text) if sum(c.isdigit() for c in p) >= 8]
    return sorted(set(EMAIL.findall(text) + phones))


def split_card(text):
    """(prose, code blocks): the text without its YAML header and fenced code, and the code blocks."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    code = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    prose = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.S)
    return prose, code


def numbers_in(prose):
    """Every number written in the prose, after names, links, dates and fingerprints are removed."""
    text = HEX.sub(" ", prose)
    for pattern in NAME_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.M)
    found = re.findall(SPECIAL_NUMBERS, text)
    text = re.sub(SPECIAL_NUMBERS, " ", text)
    found += [n.replace(",", "") for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)", text)]
    return found


def untraced_numbers(prose, facts):
    """Numbers in the prose that no fact accounts for. Whole numbers up to 10 are allowed
    (rule numbers, epochs, steps)."""
    return sorted({n for n in numbers_in(prose) if n not in facts and not (n.isdigit() and int(n) <= 10)})


def table_rows(text):
    """Each markdown table as (header cells, [row cells]), with bold marks removed."""
    tables, current = [], None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
            if current is None:
                current = (cells, [])
                tables.append(current)
            elif not set("".join(cells)) <= set("-: "):
                current[1].append(cells)
        else:
            current = None
    return tables


def check_score_tables(text, scores):
    """Problems in tables with Precision or Recall columns. `scores` is {(source, version): {metric: str}}."""
    problems = []
    for header, rows in table_rows(text):
        lower = [h.lower() for h in header]
        if "precision" not in lower and "recall" not in lower:
            continue
        version_col = lower.index("test version") if "test version" in lower else None
        for row in rows:
            source = next((s for start, s in ROW_NAMES if row[0].startswith(start)), None)
            version = VERSION_NAMES.get(row[version_col]) if version_col is not None else None
            if source is None or version is None:
                problems.append(f"row not traced to a results file: {' | '.join(row)}")
                continue
            for col, name in enumerate(lower):
                metric = METRIC_COLUMNS.get(name)
                if metric and row[col] != scores[(source, version)][metric]:
                    problems.append(f"{row[0]}, {row[version_col]}, {header[col]}: card says {row[col]}, "
                                    f"results say {scores[(source, version)][metric]}")
    return problems


def check_count_tables(text, counts):
    """Problems in the dataset card's label-count table. `counts` is {split: {(label, type): n}}."""
    problems = []
    for header, rows in table_rows(text):
        lower = [h.lower() for h in header]
        if lower[:2] != ["label", "type"] or "train" not in lower:
            continue
        for row in rows:
            for split, column in (("train", "train"), ("valid", "validation"), ("test", "test")):
                col = lower.index(column)
                if row[0].lower() == "total":
                    want = sum(counts[split].values())
                else:
                    want = counts[split].get((row[0], row[1] or None), 0)
                if row[col].replace(",", "") != str(want):
                    problems.append(f"{row[0]} {row[1]} {column}: card says {row[col]}, data says {want}")
    return problems


def check_examples(text, answers, data_lines, test_pairs):
    """Example pairs, inputs and answers in a card must be copied from the data and the saved answers."""
    problems = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('{"id": ') and json.loads(stripped) not in data_lines:
            problems.append(f"example pair not found in the data: {stripped[:60]}")
        if stripped.startswith('{"label": ') and " or " not in stripped and stripped not in answers:
            problems.append(f"example answer is not a saved answer: {stripped[:60]}")
    for person, date_a, text_a, date_b, text_b in re.findall(
            r"Person: (.+)\nMemory A \(written (\S+)\): (.+)\nMemory B \(written (\S+)\): (.+)", text):
        if not any(p["person"] == person and p["a"] == {"text": text_a, "written_at": date_a}
                   and p["b"] == {"text": text_b, "written_at": date_b} for p in test_pairs):
            problems.append(f"example input for {person} is not a test pair")
    return problems


def modelfile_settings(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def check_prompt_and_modelfile(text, modelfile):
    problems = []
    if "You judge memories." in text and SYSTEM_MESSAGE not in text:
        problems.append("the system message differs from src/judge/prompt.py")
    for block in split_card(text)[1]:
        if block.lstrip().startswith("FROM ") and modelfile_settings(block) != modelfile_settings(modelfile):
            problems.append("the Modelfile differs from ollama/Modelfile")
    return problems


# ---------- facts from the results ----------

def score_facts(root=REPO):
    """{(source, version): {metric: text}}, recomputed from the saved answers."""
    pairs = load_jsonl(Path(root) / DATA_FILES["test.jsonl"])
    scores = {}
    for source, prefix in SCORE_SOURCES.items():
        for version in VERSIONS:
            _, rows = score(pairs, load_jsonl(Path(root) / f"{prefix}.{version}.jsonl"))
            said = [r for r in rows if r["label"] == "contradiction"]
            real = [r for r in rows if r["gold_label"] == "contradiction"]
            caught = [r for r in real if r["label"] == "contradiction"]
            right = [r for r in rows if r["label"] == r["gold_label"]]
            scores[(source, version)] = {
                "precision": two_places(len(caught), len(said)),
                "recall": two_places(len(caught), len(real)),
                "accuracy": two_places(len(right), len(rows)),
                "unclear": str(sum(r["label"] == "unclear" for r in rows)),
                "invalid": str(sum(bool(r["problem"]) for r in rows)),
            }
    return scores


def data_counts(root=REPO):
    """{split: {(label, type): pairs}}."""
    counts = {}
    for split, name in (("train", "train.jsonl"), ("valid", "valid.jsonl"), ("test", "test.jsonl")):
        tally = {}
        for pair in load_jsonl(Path(root) / DATA_FILES[name]):
            key = (pair["label"], pair["type"])
            tally[key] = tally.get(key, 0) + 1
        counts[split] = tally
    return counts


def runlog_m1_cost(root=REPO):
    """Sum of the 'about $' costs of every M1 line in runs/RUNLOG.md."""
    total = Decimal(0)
    for line in (Path(root) / "runs/RUNLOG.md").read_text().splitlines():
        if line.startswith("| 20") and "| M1-" in line:
            match = re.search(r"about \$(\d+\.\d+)", line.split("|")[-2])
            if match:
                total += Decimal(match.group(1))
    return str(total.quantize(Decimal("0.01")))


def known_hex(root=REPO):
    """Every fingerprint and commit a card may name, read from the files and records."""
    root = Path(root)
    export = json.loads((root / "results/m1/gguf-v1/export.meta.json").read_text())
    train = json.loads((root / "results/m1/trained-v1/train.meta.json").read_text())
    values = {hashlib.sha256((root / p).read_bytes()).hexdigest() for p in DATA_FILES.values()}
    return values | {prompt_sha256(), export["sha256"], export["commit"], train["upload"]["commit"], train["base_revision"]}


def all_facts(root=REPO):
    """Every number the cards may state, as text, each worked out from a file in the repo."""
    root = Path(root)
    facts = set()
    scores = score_facts(root)
    for metrics in scores.values():
        facts |= set(metrics.values())
    for source in SCORE_SOURCES:
        for metric in ("unclear", "invalid"):
            facts.add(str(sum(int(scores[(source, v)][metric]) for v in VERSIONS)))
    for tally in data_counts(root).values():
        facts |= {str(n) for n in tally.values()} | {str(sum(tally.values()))}
    test = load_jsonl(root / DATA_FILES["test.jsonl"])
    real = sum(p["label"] == "contradiction" for p in test)
    facts |= {str(real), str(len(test) * len(VERSIONS)), two_places(1, real)}
    for path in DATA_FILES.values():
        facts.add(str(len({p["person"] for p in load_jsonl(root / path)})))

    train = json.loads((root / "results/m1/trained-v1/train.meta.json").read_text())
    lora, training = train["lora"], train["training"]
    chosen = train["choice"]["chosen"].split("-")[1]
    facts |= {str(lora["r"]), str(lora["lora_alpha"]), str(len(lora["target_modules"])), str(train["seed"]),
              f"{training['learning_rate']:.0e}".replace("e-0", "e-"), str(training["num_train_epochs"]),
              str(training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]),
              str(train["rows"]["train_swapped"]), str(train["rows"]["train_no_dates"]), chosen,
              train["versions"]["unsloth"], rounded(train["train_seconds"] / 60, 1)}
    facts.add(rounded(next(e["valid_loss_end"] for e in train["per_epoch"] if e["epoch"] == int(chosen)), 2))
    price, price_file, phrase = L4_PRICE
    if phrase in (root / price_file).read_text():
        facts.add(rounded(Decimal(str(train["wall_seconds"])) * Decimal(price), 2))
    facts.add(runlog_m1_cost(root))

    export = json.loads((root / "results/m1/gguf-v1/export.meta.json").read_text())
    facts |= {str(export["bytes"]), rounded(Decimal(export["bytes"]) / Decimal(10**9), 1)}
    run = json.loads((root / "results/m1/gguf-v1/m1-judge-v1.meta.json").read_text())
    facts |= {rounded(run["seconds_per_answer_mean"], 2), run["ollama_version"]}
    compare = json.loads((root / "results/m1/gguf-v1/compare-test-adapter.json").read_text())
    facts.add(str(sum(round(compare[v]["label_agreement"] * compare[v]["pairs"]) for v in VERSIONS)))

    for number, path, phrase in TRACED:
        if phrase in (root / path).read_text():
            facts.add(number)
    return facts


# ---------- M2 facts ----------

def three_places(top, bottom):
    return str((Decimal(top) / Decimal(bottom)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def m2_test_pairs(root=REPO):
    """{test set: pairs}. Both frozen files are read only through judge.frozen."""
    root = Path(root)
    return {name: load_frozen(root / path, root / "STATE.md") for name, path in M2_TESTS.items()}


def m2_rows(root=REPO, tests=None):
    """{(model, test set, version): scored rows}, from the saved answers."""
    root = Path(root)
    tests = tests or m2_test_pairs(root)
    return {(model, test, version): score(tests[test], load_jsonl(root / f"{prefix}.{version}.jsonl"))[1]
            for (model, test), prefix in M2_SOURCES.items() for version in VERSIONS}


def m2_score_facts(rows):
    """{(model, test set, version): {column: text}} for the M2 score tables."""
    facts = {}
    for key, scored in rows.items():
        said = sum(r["label"] == "contradiction" for r in scored)
        real = sum(r["gold_label"] == "contradiction" for r in scored)
        caught = sum(r["label"] == "contradiction" and r["gold_label"] == "contradiction" for r in scored)
        facts[key] = {"caught": caught, "said": said, "real": real,
                      "caught_said": f"{caught} / {said}",
                      "precision": three_places(caught, said), "recall": three_places(caught, real),
                      "unclear": str(sum(r["label"] == "unclear" for r in scored)),
                      "invalid": str(sum(bool(r["problem"]) for r in scored))}
    return facts


def m2_trap_facts(rows):
    """{(trap, model): "as is / swapped / no dates" right labels} on the hard set, plus {(trap, "pairs"): n}."""
    facts = {}
    for model in ("v1", "v2"):
        right = {}
        for version in VERSIONS:
            for r in rows[(model, "hard", version)]:
                key = trap_key({"hard": r["hard"], "type": r["gold_type"], "label": r["gold_label"]})
                right.setdefault(key, {v: 0 for v in VERSIONS})[version] += r["label"] == r["gold_label"]
        for key, counts in right.items():
            facts[(key, model)] = " / ".join(str(counts[v]) for v in VERSIONS)
    for r in rows[("v1", "hard", VERSIONS[0])]:
        key = (trap_key({"hard": r["hard"], "type": r["gold_type"], "label": r["gold_label"]}), "pairs")
        facts[key] = facts.get(key, 0) + 1
    return facts


def check_m2_score_tables(text, scores):
    """Problems in tables with Model, Test set and Version columns. `scores` is m2_score_facts()."""
    problems = []
    versions = {"as is": "as_is", "swapped": "swapped", "no dates": "no_dates"}
    for header, rows in table_rows(text):
        lower = [h.lower() for h in header]
        if not {"model", "test set", "version"} <= set(lower):
            continue
        for row in rows:
            key = (row[lower.index("model")], row[lower.index("test set")].lower(), versions.get(row[lower.index("version")]))
            if key not in scores:
                problems.append(f"row not traced to saved answers: {' | '.join(row)}")
                continue
            for col, name in enumerate(lower):
                column = M2_COLUMNS.get(name)
                if column and row[col] != scores[key][column]:
                    problems.append(f"{' '.join(row[:3])}, {header[col]}: text says {row[col]}, answers say {scores[key][column]}")
    return problems


def check_m2_trap_tables(text, traps):
    """Problems in per-trap tables: "Trap (right answer) | Pairs | v1 | v2"."""
    problems = []
    for header, rows in table_rows(text):
        lower = [h.lower() for h in header]
        if not lower or lower[0] != "trap (right answer)":
            continue
        for row in rows:
            for col, name in enumerate(lower[1:], 1):
                want = traps.get((row[0], name))
                if want is None or row[col] != str(want):
                    problems.append(f"trap {row[0]}, {header[col]}: text says {row[col]}, answers say {want}")
    return problems


def m2_data(root=REPO):
    """{file name: rows} of the four M2 data files (the frozen one through judge.frozen)."""
    root = Path(root)
    return {name: (load_frozen(root / path, root / "STATE.md") if path == FROZEN_M2 else load_jsonl(root / path))
            for name, path in DATA_FILES_M2.items()}


def m2_fingerprints(root=REPO):
    """{file name: SHA-256}: the frozen file's from STATE.md (checked by load_frozen in m2_data), the rest computed."""
    root = Path(root)
    frozen = frozen_entries(root / "STATE.md")
    return {name: frozen[path] if path == FROZEN_M2 else hashlib.sha256((root / path).read_bytes()).hexdigest()
            for name, path in DATA_FILES_M2.items()}


def check_m2_dataset_tables(text, data, fingerprints):
    """The dataset card's file table (pairs, SHA-256) and trap table (pairs per trap and right answer)."""
    problems = []
    test = data["test-hard.jsonl"]
    for header, rows in table_rows(text):
        lower = [h.lower() for h in header]
        if lower[:1] == ["file"] and "pairs" in lower:
            for row in rows:
                name = row[0].strip("`")
                if name not in data:
                    problems.append(f"file table row not an M2 data file: {row[0]}")
                    continue
                if row[lower.index("pairs")].replace(",", "") != str(len(data[name])):
                    problems.append(f"{name} pairs: card says {row[lower.index('pairs')]}, file has {len(data[name])}")
                if "sha-256" in lower and row[lower.index("sha-256")].strip("`") != fingerprints[name]:
                    problems.append(f"{name} SHA-256 differs from the file")
        if lower[:1] == ["trap"] and "right answer" in lower and "pairs" in lower:
            for row in rows:
                trap, answer = row[0].strip("`"), row[lower.index("right answer")]
                n = sum(p["hard"] == trap and (p["type"] or p["label"]) == answer for p in test)
                if row[lower.index("pairs")] != str(n):
                    problems.append(f"trap {trap} ({answer}): card says {row[lower.index('pairs')]}, test file has {n}")
    return problems


def runlog_m2_lines(root=REPO):
    return [line for line in (Path(root) / "runs/RUNLOG.md").read_text().splitlines()
            if line.startswith("| 20") and "| M2-" in line]


def m2_costs(root=REPO):
    """Numbers of the write-up's cost section, worked out from runs/RUNLOG.md."""
    lines = runlog_m2_lines(root)
    facts = set()
    for line in lines:  # every number written in an M2 line (minutes, costs)
        facts |= {n.replace(",", "") for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\w)", line)}

    def find(phrase):
        found = [line for line in lines if phrase in line]
        if len(found) != 1:
            raise SystemExit(f"STOP: RUNLOG line '{phrase}' found {len(found)} times")
        return found[0]

    def money(line):
        return [Decimal(x) for x in re.findall(r"\$(\d+\.\d+)", line.split("|")[-2])]

    def minutes(line):
        return Decimal(re.search(r"(\d+\.\d+) inside the function", line.split("|")[-3]).group(1))

    estimated_low = estimated_high = Decimal(0)
    for group, phrases in M2_COST_GROUPS.items():
        picked = [find(p) for p in phrases]
        if group == "v2 scoring":
            facts.add(str(sum(money(l)[0] for l in picked)))
            facts.add(str(sum(minutes(l) for l in picked)))
        else:
            low, high = sum(money(l)[0] for l in picked), sum(money(l)[1] for l in picked)
            estimated_low += low
            estimated_high += high
            facts |= {str(low), str(high)}
    billed = [money(line)[0] for line in lines if "billed" in line.split("|")[-2]]
    pool_smoke = [Decimal(x) for x in re.findall(r"\$(\d+\.\d+)", POOL_SMOKE_BILLS)]
    if not any(POOL_SMOKE_BILLS in line for line in lines):
        raise SystemExit("STOP: the pool smoke bills are no longer in RUNLOG")
    facts.add(str(sum(pool_smoke)))
    total_billed = (sum(billed) + sum(pool_smoke)).quantize(Decimal("0.01"))
    facts |= {str(total_billed), str(total_billed + estimated_low), str(total_billed + estimated_high)}
    return facts


def m2_facts(root=REPO, rows=None, data=None):
    """Every number the M2 texts may state, each worked out from a file in the repo."""
    root = Path(root)
    rows = rows or m2_rows(root)
    data = data or m2_data(root)
    facts = set()
    scores = m2_score_facts(rows)
    for (model, test, version), s in scores.items():
        facts |= {str(s["caught"]), str(s["said"]), str(s["real"]), s["precision"], s["recall"], s["unclear"],
                  s["invalid"], str(s["said"] - s["caught"]), str(len(rows[(model, test, version)]))}
    for model in ("v1", "v2"):
        for test in M2_TESTS:
            facts.add(str(sum(int(scores[(model, test, v)]["unclear"]) for v in VERSIONS)))
    # The bar (D-060): v1's worst-case recall + 0.03 of the real contradictions, rounded up; precision - 0.01.
    v1_hard = [scores[("v1", "hard", v)] for v in VERSIONS]
    worst = min(v1_hard, key=lambda s: Decimal(s["caught"]) / s["real"])
    bar = (Decimal(worst["caught"]) + Decimal("0.03") * worst["real"]).quantize(Decimal(1), rounding=ROUND_CEILING)
    worst_p = min(Decimal(s["caught"]) / s["said"] for s in v1_hard)
    facts |= {str(bar), str((worst_p - Decimal("0.01")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))}
    # Pair by pair: contradictions one model caught and the other missed, per version.
    for test in M2_TESTS:
        for v in VERSIONS:
            one = {r["id"]: r for r in rows[("v1", test, v)]}
            two = {r["id"]: r for r in rows[("v2", test, v)]}
            real = [i for i in one if one[i]["gold_label"] == "contradiction"]
            facts.add(str(sum(one[i]["label"] != "contradiction" and two[i]["label"] == "contradiction" for i in real)))
            facts.add(str(sum(one[i]["label"] == "contradiction" and two[i]["label"] != "contradiction" for i in real)))
    traps = m2_trap_facts(rows)
    facts |= {str(v) for k, v in traps.items() if k[1] == "pairs"}
    facts |= {part for k, v in traps.items() if k[1] != "pairs" for part in str(v).split(" / ")}
    facts.add(str(len({k[0].split(" (")[0] for k in traps if k[1] == "pairs"}) + 1))  # 11 traps (near_identical has two)
    for rows_ in data.values():
        facts |= {str(len(rows_)), str(len({p["person"] for p in rows_}))}
        tally = {}
        for p in rows_:
            for key in ((p["label"], p["type"]), p.get("hard"), p.get("outcome")):
                tally[key] = tally.get(key, 0) + 1
        facts |= {str(n) for n in tally.values()}
    facts |= m2_costs(root)
    meta = json.loads((root / "results/m2/grpo/train.meta.json").read_text())
    grpo = meta["grpo"]
    facts |= {str(meta["planned_steps"]), str(grpo["num_generations"]), rounded(meta["wall_seconds"] / 60, 1),
              f"{grpo['learning_rate']:.0e}".replace("e-0", "e-"), str(grpo["epsilon"]), str(grpo["epsilon_high"]),
              str(grpo["beta"]), str(grpo["temperature"]), str(grpo["max_completion_length"]),
              meta["versions"]["trl"], meta["versions"]["unsloth"]}
    facts |= {str(abs(value)) for value in REWARD.values()}  # the card states them with their sign in words
    for number, path, phrase in TRACED_M2:
        if phrase in (root / path).read_text():
            facts.add(number)
    return facts


def known_hex_m2(root=REPO):
    """Fingerprints and commits the M2 texts may name."""
    root = Path(root)
    meta = json.loads((root / "results/m2/grpo/train.meta.json").read_text())
    return set(m2_fingerprints(root).values()) | {meta["prompt_sha256"], meta["reward_sha256"], meta["base_revision"],
                                                   meta["adapter_revision"], meta["upload"]["commit"]}


def check_numbers_m2(report, texts, root=REPO, where="git"):
    """The M2 texts: score and trap tables recomputed from the saved answers, dataset tables from the
    files, and every other number traced to a record."""
    root = Path(root)
    rows, data = m2_rows(root), m2_data(root)
    facts, scores, traps = m2_facts(root, rows, data), m2_score_facts(rows), m2_trap_facts(rows)
    fingerprints, hexes = m2_fingerprints(root), known_hex_m2(root)
    tests = m2_test_pairs(root)
    data_lines = [p for rows_ in data.values() for p in rows_] + list(tests["m1"])
    answers = {a["raw"] for prefix in M2_SOURCES.values() for v in VERSIONS for a in load_jsonl(root / f"{prefix}.{v}.jsonl")}
    failures = 0
    for name, text in texts.items():
        problems = check_m2_score_tables(text, scores) + check_m2_trap_tables(text, traps)
        problems += check_m2_dataset_tables(text, data, fingerprints)
        problems += check_examples(text, answers, data_lines, list(tests["m1"]) + list(tests["hard"]))
        problems += [f"fingerprint not in any record: {h}" for h in sorted(set(HEX.findall(text)) - hexes)]
        loose = untraced_numbers(split_card(text)[0], facts)
        if loose:
            problems.append(f"numbers not traced to a results file: {loose}")
        for problem in problems:
            report.add("FAIL", f"{name} ({where}): {problem}")
        failures += len(problems)
    if not failures:
        report.add("PASS", f"every number, table, fingerprint and example in the M2 write-up and cards traces to "
                           f"the saved answers, data and records ({where}): {len(texts)} files")


# ---------- the checks ----------

def check_git(report, root=REPO):
    files = head_files(root)
    blobs = history_blobs(root)

    secret_files = sorted({p for p in files if is_secret_file(p)} | {p for p, _ in blobs.values() if is_secret_file(p)})
    leaks = sorted({f"{path} ({name})" for sha, (path, size) in blobs.items() if size <= BIG_FILE_BYTES
                    for name in find_secrets(blob_text(sha, root))})
    if secret_files or leaks:
        report.add("FAIL", f"secrets in git: files {secret_files}, contents {leaks}")
    else:
        report.add("PASS", f"no secrets, tokens or .env files in git ({len(files)} files at HEAD, "
                           f"{len(blobs)} file versions in history)")

    models = sorted({p for p, s in files.items() if is_model_file(p, s)} | {p for p, s in blobs.values() if is_model_file(p, s)})
    if models:
        report.add("FAIL", f"model or large files in git: {models}")
    else:
        report.add("PASS", f"no model files in git, at HEAD or in history (largest file {max(files.values()):,} bytes)")
    return files


def public_texts(files, root=REPO):
    """{path: text} of the public prose at HEAD: every README, the hf/ cards and the post drafts."""
    paths = [p for p in files if p.endswith("README.md") or (p.startswith("hf/") and p.endswith(".md"))
             or p.startswith("docs/posts/") or p == WRITE_UP]
    return {p: head_text(p, root) for p in sorted(paths)}


def public_facing(name):
    """The cards, the GitHub front page and the posts: these must not name Aeverum or Mara
    Lindqvist at all. Other files that do are listed in the notes."""
    return name in NUMBERED_TEXT or name in NUMBERED_TEXT_M2 or name.startswith(("hf/", "huggingface:", "docs/posts/"))


def check_texts(report, texts, where="git"):
    dashes = {name: em_dash_lines(text) for name, text in texts.items() if em_dash_lines(text)}
    if dashes:
        report.add("FAIL", f"em dashes in public text ({where}), file: line numbers: {dashes}")
    else:
        report.add("PASS", f"no em dashes in public text ({where}): {len(texts)} files")

    personal = {name: personal_data(text) for name, text in texts.items() if personal_data(text)}
    named = sorted(name for name, text in texts.items() if public_facing(name) and BANNED_NAMES.search(text))
    aeverum = sorted(name for name, text in texts.items() if public_facing(name) and "aeverum" in text.lower())
    if personal or named or aeverum:
        report.add("FAIL", f"personal data or Aeverum in public text ({where}): contacts {personal}, "
                           f"Mara Lindqvist {named}, Aeverum named {aeverum}")
    else:
        report.add("PASS", f"no e-mail addresses or phone numbers in public text, and no Aeverum or "
                           f"Mara Lindqvist in the cards, README or posts ({where})")


def check_data(report, root=REPO):
    problems = []
    texts = {path: (Path(root) / path).read_text() for path in DATA_FILES.values()}
    for name, rows in m2_data(root).items():  # the frozen M2 file through judge.frozen
        texts[DATA_FILES_M2[name]] = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    for path, text in texts.items():
        if BANNED_NAMES.search(text):
            problems.append(f"{path}: Mara or Lindqvist")
        if "aeverum" in text.lower():
            problems.append(f"{path}: Aeverum")
        if personal_data(text):
            problems.append(f"{path}: {personal_data(text)[:3]}")
    if problems:
        report.add("FAIL", f"data files: {problems}")
    else:
        report.add("PASS", "data files: no e-mail addresses, phone numbers, Mara Lindqvist or Aeverum "
                           "(the people are made up; no check can prove a made-up name is not also a real one)")


def check_numbers(report, texts, root=REPO, where="git", extra_facts=frozenset()):
    root = Path(root)
    facts, scores, counts, hexes = all_facts(root) | set(extra_facts), score_facts(root), data_counts(root), known_hex(root)
    test_pairs = load_jsonl(root / DATA_FILES["test.jsonl"])
    data_lines = [p for path in DATA_FILES.values() for p in load_jsonl(root / path)]
    answers = {a["raw"] for source in ("gguf", "adapter") for v in VERSIONS
               for a in load_jsonl(root / f"{SCORE_SOURCES[source]}.{v}.jsonl")}
    modelfile = (root / "ollama/Modelfile").read_text()
    failures = 0
    for name, text in texts.items():
        problems = check_score_tables(text, scores) + check_count_tables(text, counts)
        problems += check_examples(text, answers, data_lines, test_pairs)
        problems += check_prompt_and_modelfile(text, modelfile)
        problems += [f"fingerprint not in any record: {h}" for h in sorted(set(HEX.findall(text)) - hexes)]
        loose = untraced_numbers(split_card(text)[0], facts)
        if loose:
            problems.append(f"numbers not traced to a results file: {loose}")
        for problem in problems:
            report.add("FAIL", f"{name} ({where}): {problem}")
        failures += len(problems)
    if not failures:
        report.add("PASS", f"every number, fingerprint, example, prompt and Modelfile in the cards and README "
                           f"traces to the repo's results and data ({where}): {len(texts)} files")


def check_hugging_face(report, root=REPO):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    texts = {}
    frozen = frozen_entries(Path(root) / "STATE.md")
    m2_data(root)  # stops here if the frozen hard test file changed (judge.frozen)
    with tempfile.TemporaryDirectory() as tmp:
        for repo_id, (repo_type, card) in {**CARDS, **CARDS_M2}.items():
            info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)

            def download(filename):
                return Path(hf_hub_download(repo_id, filename, repo_type=repo_type, revision=info.sha,
                                            local_dir=f"{tmp}/{repo_id}"))

            names = [s.rfilename for s in info.siblings]
            problems = [f"secret-looking file {n}" for n in names if is_secret_file(n)]
            remote_card = download("README.md").read_text()
            texts[f"huggingface:{repo_id}"] = remote_card
            if remote_card != (Path(root) / card).read_text():
                problems.append(f"its card differs from {card}; upload it with scripts/publish_hf.py")
            small = [s.rfilename for s in info.siblings
                     if s.size is not None and s.size <= BIG_FILE_BYTES and not s.rfilename.endswith(MODEL_EXTENSIONS)]
            problems += [f"secret in {n} ({kind})" for n in small for kind in find_secrets(download(n).read_text(errors="replace"))]
            if repo_id == "eskora/memory-contradiction-pairs":
                problems += [f"{name} differs from {path}" for name, path in DATA_FILES.items()
                             if download(name).read_bytes() != (Path(root) / path).read_bytes()]
            if repo_id in CARDS_M2 and repo_type == "dataset":
                problems += m2_dataset_problems(names, download, frozen, root)
            problems += visibility_problems(repo_id, info.private)
            for problem in problems:
                report.add("FAIL", f"{repo_id}: {problem}")
            big = [n for n in names if n not in small]
            report.add("PASS" if not problems else "FAIL",
                       f"{repo_id}: private={info.private}, commit {info.sha}, {len(names)} files, "
                       f"{len(small)} scanned for secrets" + (f" (not scanned, too big or model files: {big})" if big else ""))
    check_texts(report, texts, where="Hugging Face")
    check_numbers(report, {k: v for k, v in texts.items() if k.removeprefix("huggingface:") in CARDS}, root,
                  where="Hugging Face")
    check_numbers_m2(report, {k: v for k, v in texts.items() if k.removeprefix("huggingface:") in CARDS_M2}, root,
                     where="Hugging Face")


def visibility_problems(repo_id, private):
    """A repo must be as VISIBILITY says: the M2 repos private until Claude in Cowork's check,
    the M1 repos public since D-034."""
    want = VISIBILITY[repo_id]
    if want and not private:
        return ["NOT PRIVATE"]
    if not want and private:
        return ["private, but it was made public on D-034"]
    return []


def m2_dataset_problems(names, download, frozen, root=REPO):
    """The M2 dataset repo must hold the four data files: the frozen one equal to its STATE.md
    fingerprint (the local copy is checked by judge.frozen), the others byte for byte the repo's."""
    problems = [f"{name} missing" for name in DATA_FILES_M2 if name not in names]
    for name, path in DATA_FILES_M2.items():
        if name not in names:
            continue
        remote = download(name).read_bytes()
        if path == FROZEN_M2:
            if hashlib.sha256(remote).hexdigest() != frozen[path]:
                problems.append(f"{name} is not the frozen file (SHA-256 differs from STATE.md)")
        elif remote != (Path(root) / path).read_bytes():
            problems.append(f"{name} differs from {path}")
    return problems


def notes(report, files, root=REPO):
    """Things that go public with the GitHub repo, for Elias to know. They do not fail the check."""
    own_files = ("scripts/prepublish_check.py", "hf/prepublish-check.txt")  # they name what is looked for
    text_files = {p: head_text(p, root) for p, size in files.items()
                  if size <= BIG_FILE_BYTES and not p.endswith((".jsonl", ".lock")) and p not in own_files}
    aeverum = sorted(p for p, t in text_files.items() if "aeverum" in t.lower())
    named = sorted(p for p, t in text_files.items() if BANNED_NAMES.search(t))
    homes = sorted(p for p, t in text_files.items() if HOME_PATH.search(t))
    if aeverum:
        report.add("NOTE", f"{len(aeverum)} files in git name Aeverum (plans, decisions, reviews, docs). "
                           f"They go public with the GitHub repo: {aeverum}")
    if named:
        report.add("NOTE", f"the name Mara Lindqvist appears, as a rule not to use it, in: {named}")
    if homes:
        report.add("NOTE", f"a Mac home folder path (/Users/...) appears in: {homes}")
    if "LICENSE" not in files:
        report.add("NOTE", "the GitHub repo has no LICENSE file: once public, its code can be read but not reused")


def main(argv):
    report = Report()
    files = check_git(report)
    texts = public_texts(files)
    check_texts(report, texts)
    check_data(report)
    check_numbers(report, {name: texts[name] for name in NUMBERED_TEXT if name in texts and name != "README.md"})
    # The README carries M1 and M2 numbers.
    check_numbers(report, {"README.md": texts["README.md"]}, extra_facts=m2_facts())
    check_numbers_m2(report, {name: texts[name] for name in NUMBERED_TEXT_M2 if name in texts})
    if "--local" in argv:
        report.add("NOTE", "Hugging Face repos not checked (--local)")
    else:
        check_hugging_face(report)
    notes(report, files)
    print("\n".join(report.lines))
    print("RESULT: " + ("FAIL" if report.failed else "PASS"))
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main(sys.argv)
