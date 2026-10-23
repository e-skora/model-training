"""Tests for scripts/prepublish_check.py.

The real repo is only read (the cards, the data and the saved answers). Each test
breaks a copy of a card in memory and checks that the problem is caught. Git checks
run on a throwaway repo in a temporary folder. No network, no Hugging Face.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepublish_check_script", REPO / "scripts" / "prepublish_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

GGUF_CARD = (REPO / "hf/m1-contradiction-judge-qwen3-4b-gguf/README.md").read_text()
DATASET_CARD = (REPO / "hf/memory-contradiction-pairs/README.md").read_text()
# Built at run time, so this file itself holds nothing that looks like a token.
FAKE_HF_TOKEN = "hf_" + "Ab3" * 12
FAKE_MODAL_TOKEN = "ak-" + "x9Y" * 8


@pytest.fixture(scope="module")
def facts():
    return check.all_facts(REPO)


@pytest.fixture(scope="module")
def scores():
    return check.score_facts(REPO)


def test_rounding_follows_the_results_files():
    assert check.two_places(193, 200) == "0.97"  # a float would round 0.965 down to 0.96
    assert check.two_places(191, 200) == "0.96"
    assert check.rounded(0.914, 2) == "0.91"


def test_secret_patterns():
    assert check.find_secrets(f"token = {FAKE_HF_TOKEN}") == ["Hugging Face token"]
    assert check.find_secrets(f"MODAL_TOKEN_ID={FAKE_MODAL_TOKEN}") == ["Modal token"]
    assert check.find_secrets("SHA-256 6e2a08323f445477f9600e371f1074ad3e2b41ef97d7072d6734ef481f6e574b, as-is") == []
    assert check.is_secret_file(".env") and check.is_secret_file("config/prod.env") and check.is_secret_file(".env.local")
    assert not check.is_secret_file("docs/environment.md")
    assert check.is_model_file("x/model.gguf", 10) and check.is_model_file("big.json", 6_000_000)
    assert not check.is_model_file("results/m1/a.jsonl", 70_000)


def test_em_dashes_and_contacts():
    assert check.em_dash_lines("fine\nnot fine — here\n") == [2]
    assert check.personal_data("write to someone@example.com or call +47 912 34 567") == ["+47 912 34 567", "someone@example.com"]
    assert check.personal_data(GGUF_CARD) == []


def test_real_cards_pass(facts, scores):
    counts = check.data_counts(REPO)
    for card in (GGUF_CARD, DATASET_CARD):
        assert check.check_score_tables(card, scores) == []
        assert check.check_count_tables(card, counts) == []
        assert check.untraced_numbers(check.split_card(card)[0], facts) == []


def test_wrong_score_in_a_table_is_caught(scores):
    broken = GGUF_CARD.replace("| as is | **0.98** | **0.96** |", "| as is | **0.97** | **0.96** |", 1)
    assert broken != GGUF_CARD
    problems = check.check_score_tables(broken, scores)
    assert len(problems) == 1 and "card says 0.97, results say 0.98" in problems[0]


def test_unknown_row_in_a_score_table_is_caught(scores):
    broken = GGUF_CARD.replace("| Qwen3-8B, untrained | as is |", "| Some other model | as is |", 1)
    assert any("row not traced" in p for p in check.check_score_tables(broken, scores))


def test_wrong_count_is_caught():
    broken = DATASET_CARD.replace("| contradiction | | 697 |", "| contradiction | | 698 |", 1)
    assert broken != DATASET_CARD
    assert check.check_count_tables(broken, check.data_counts(REPO)) == ["contradiction  train: card says 698, data says 697"]


def test_untraced_number_in_prose_is_caught(facts):
    broken = GGUF_CARD.replace("about $0.45 of GPU time", "about $0.40 of GPU time")
    assert broken != GGUF_CARD
    assert check.untraced_numbers(check.split_card(broken)[0], facts) == ["0.40"]


def test_made_up_example_answer_is_caught():
    broken = GGUF_CARD.replace("lives there now.\"}", "lives there.\"}")
    answers = {"something else"}
    assert any("not a saved answer" in p for p in check.check_examples(broken, answers, [], [{"person": "x"}]))


def test_changed_prompt_and_modelfile_are_caught():
    modelfile = (REPO / "ollama/Modelfile").read_text()
    assert check.check_prompt_and_modelfile(GGUF_CARD, modelfile) == []
    assert check.check_prompt_and_modelfile(GGUF_CARD.replace("You decide whether", "You say whether"), modelfile) == \
        ["the system message differs from src/judge/prompt.py"]
    assert check.check_prompt_and_modelfile(GGUF_CARD.replace("PARAMETER num_ctx 2048", "PARAMETER num_ctx 4096"), modelfile) == \
        ["the Modelfile differs from ollama/Modelfile"]


def test_every_fingerprint_in_the_cards_is_known():
    known = check.known_hex(REPO)
    for card in (GGUF_CARD, DATASET_CARD):
        assert set(check.HEX.findall(card)) <= known


@pytest.fixture
def throwaway_repo(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@invalid", *args],
                       cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "README.md").write_text("fine\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    return tmp_path, git


def test_clean_git_passes(throwaway_repo):
    root, _ = throwaway_repo
    report = check.Report()
    check.check_git(report, root)
    assert not report.failed


def test_secrets_and_model_files_are_caught_even_after_deletion(throwaway_repo):
    root, git = throwaway_repo
    (root / ".env").write_text("X=1\n")
    (root / "notes.md").write_text(f"my token {FAKE_HF_TOKEN}\n")
    (root / "judge.gguf").write_bytes(b"\0" * 10)
    git("add", ".")
    git("commit", "-q", "-m", "oops")
    git("rm", "-q", ".env", "notes.md", "judge.gguf")
    git("commit", "-q", "-m", "removed again")
    report = check.Report()
    check.check_git(report, root)
    failed = "\n".join(line for line in report.lines if line.startswith("FAIL"))
    assert ".env" in failed and "notes.md (Hugging Face token)" in failed and "judge.gguf" in failed
