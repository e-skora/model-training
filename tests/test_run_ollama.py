"""Tests for scripts/run_ollama.py against a fake Ollama on a local port.

No real Ollama, no model, no network beyond 127.0.0.1: a small web server in
the test answers like Ollama's API and records what it was asked. Every file
the script writes goes to a temporary folder; the real repo is only read.
"""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from judge.generate import GENERATION
from judge.prompt import build_messages, prompt_sha256
from judge.score import VERSIONS, derive
from judge.train_data import load_pairs

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_ollama_script", REPO / "scripts" / "run_ollama.py")
run_ollama = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_ollama)

SHA = "ab" * 32
TEMPLATE = ("{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n"
            "{{ end }}<|im_start|>assistant\n")
MODELFILE = (f'FROM ./judge.gguf\nTEMPLATE """{TEMPLATE}"""\nPARAMETER temperature 0\n'
             'PARAMETER num_predict 256\nPARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n')
SHOWN_PARAMETERS = ('num_predict                    256\nstop                           "<|im_end|>"\n'
                    'stop                           "<|endoftext|>"\ntemperature                    0')
SECRET = "Quentin keeps a purple heron"  # memory text that must never reach the screen or the settings file


def pair(i):
    return {
        "id": f"v-{i:03d}",
        "person": "Test Person",
        "a": {"text": f"{SECRET} number {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person keeps no birds {i}.", "written_at": "2025-01-02"},
        "label": "contradiction" if i % 2 else "non_contradiction",
        "type": None if i % 2 else "merely_related",
        "reason": "They disagree.",
    }


PAIRS = [pair(i) for i in range(3)]


class FakeOllama:
    """A fake Ollama API. `shown` is what /api/show returns; every request is kept in `requests`."""

    def __init__(self, sha=SHA, template=TEMPLATE, parameters=SHOWN_PARAMETERS):
        self.requests = []
        self.shown = {"modelfile": f"FROM /models/blobs/sha256-{sha}\n", "template": template,
                      "parameters": parameters}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests.append((self.path, None))
                self.reply({"version": "0.24.0"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, body))
                answer = json.dumps({"label": "contradiction", "type": None, "reason": "They disagree."})
                if self.path == "/api/show":
                    self.reply(fake.shown)
                elif self.path == "/api/chat":
                    self.reply({"message": {"role": "assistant", "content": answer}, "load_duration": 5e8,
                                "prompt_eval_count": 500, "eval_count": 30, "done_reason": "stop"})
                else:
                    self.reply({"response": answer, "load_duration": 0, "prompt_eval_count": 500,
                                "eval_count": 30, "done_reason": "stop"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self):
        return [path for path, _ in self.requests]


@pytest.fixture
def setup(tmp_path):
    fake = FakeOllama()
    data = tmp_path / "valid.jsonl"
    data.write_text("".join(json.dumps(p) + "\n" for p in PAIRS), encoding="utf-8")
    (tmp_path / "Modelfile").write_text(MODELFILE, encoding="utf-8")
    (tmp_path / "export.meta.json").write_text(json.dumps({"sha256": SHA}), encoding="utf-8")
    (tmp_path / "STATE.md").write_text("# STATE\n\n## Frozen files\n", encoding="utf-8")
    yield fake, tmp_path
    fake.server.shutdown()


def run(fake, tmp_path, data=None, out=None, **kwargs):
    kwargs = {"host": fake.host, "modelfile": tmp_path / "Modelfile", "export_meta": tmp_path / "export.meta.json",
              "state_path": tmp_path / "STATE.md", **kwargs}
    return run_ollama.answer(data or tmp_path / "valid.jsonl", out or tmp_path / "out", **kwargs)


def test_render_prompt_matches_the_prompt_training_recorded():
    # Step 7 recorded the SHA-256 of the first validation prompt exactly as the training template wrote it.
    recorded = (REPO / "results/m1/trained-v1/train.meta.json").read_text(encoding="utf-8")
    first = derive(load_pairs(REPO / "data/m1/valid.jsonl"), "as_is")[0]
    digest = hashlib.sha256(run_ollama.render_prompt(build_messages(first)).encode("utf-8")).hexdigest()
    assert digest == "da74aae49d9a7dc9f9c1e3d28ebc0de9a4fa83613396017aeea5f9e312775fd9"
    assert f'"first_prompt_sha256": "{digest}"' in recorded


def test_the_repo_modelfile_uses_the_training_template_and_greedy_settings():
    modelfile = run_ollama.parse_modelfile((REPO / "ollama" / "Modelfile").read_text(encoding="utf-8"))
    assert modelfile["template"] == TEMPLATE
    assert modelfile["parameters"]["temperature"] == ["0"]
    assert modelfile["parameters"]["num_predict"] == [str(GENERATION["max_new_tokens"])]
    assert set(modelfile["parameters"]["stop"]) == {"<|im_end|>", "<|endoftext|>"}
    assert modelfile["from"].endswith(".gguf")


def test_answers_every_pair_in_every_version_with_the_shared_prompt(setup):
    fake, tmp_path = setup
    meta = run(fake, tmp_path)
    chats = [body for path, body in fake.requests if path == "/api/chat"]
    expected = [build_messages(p) for version in VERSIONS for p in derive(PAIRS, version)]
    assert [c["messages"] for c in chats] == expected
    assert all(c["stream"] is False and c["model"] == "m1-judge:v1" and "options" not in c for c in chats)
    for version in VERSIONS:
        lines = [json.loads(l) for l in (tmp_path / "out" / f"m1-judge-v1.{version}.jsonl").read_text().splitlines()]
        assert [l["id"] for l in lines] == [p["id"] for p in PAIRS]
        assert all(json.loads(l["raw"])["label"] == "contradiction" for l in lines)
    assert meta["model_sha256"] == SHA and meta["prompt_sha256"] == prompt_sha256()
    assert meta["ollama_version"] == "0.24.0" and meta["route"] == "/api/chat"
    assert len(meta["timings"]) == 9 and meta["pairs"] == 3


def test_memory_text_reaches_only_the_answers_file(setup, capsys):
    fake, tmp_path = setup
    run(fake, tmp_path)
    printed = capsys.readouterr()
    assert SECRET not in printed.out + printed.err
    assert SECRET not in (tmp_path / "out" / "m1-judge-v1.meta.json").read_text()


def test_raw_mode_sends_the_training_template_text(setup):
    fake, tmp_path = setup
    meta = run(fake, tmp_path, raw=True)
    sent = [body for path, body in fake.requests if path == "/api/generate"]
    first = derive(PAIRS, "as_is")[0]
    assert sent[0]["raw"] is True and sent[0]["prompt"] == run_ollama.render_prompt(build_messages(first))
    assert "/api/chat" not in fake.paths()
    assert (tmp_path / "out" / "m1-judge-v1-raw.as_is.jsonl").exists() and meta["route"].startswith("/api/generate")


@pytest.mark.parametrize("change, message", [
    ({"sha": "cd" * 32}, "does not run the exported GGUF"),
    ({"template": "{{ .Prompt }}"}, "different chat template"),
    ({"parameters": SHOWN_PARAMETERS.replace("temperature                    0", "temperature 0.8")}, "differ"),
])
def test_a_different_model_stops_before_any_answer(setup, change, message):
    _, folder = setup
    fake = FakeOllama(**change)
    with pytest.raises(SystemExit, match=message):
        run(fake, folder)
    assert "/api/chat" not in fake.paths()
    assert not (folder / "out").exists()
    fake.server.shutdown()


def test_an_existing_answers_file_stops_before_any_request(setup):
    fake, tmp_path = setup
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "m1-judge-v1.swapped.jsonl").write_text("keep me\n")
    with pytest.raises(SystemExit, match="already exist"):
        run(fake, tmp_path)
    assert fake.requests == []
    assert (tmp_path / "out" / "m1-judge-v1.swapped.jsonl").read_text() == "keep me\n"


def frozen(tmp_path):
    test = tmp_path / "test.jsonl"
    test.write_text("".join(json.dumps(p) + "\n" for p in PAIRS), encoding="utf-8")
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    (tmp_path / "STATE.md").write_text(f"# STATE\n\n## Frozen files\n- `test.jsonl`, 3 pairs, SHA-256 `{digest}`\n")
    return test


def test_a_frozen_file_needs_final_and_its_fingerprint(setup):
    fake, tmp_path = setup
    test = frozen(tmp_path)
    with pytest.raises(SystemExit, match="frozen"):
        run(fake, tmp_path, data=test)
    assert fake.requests == []
    meta = run(fake, tmp_path, data=test, final=True)
    assert meta["data"] == "test.jsonl" and meta["data_sha256"] == hashlib.sha256(test.read_bytes()).hexdigest()


def test_an_edited_frozen_file_stops_before_any_request(setup):
    fake, tmp_path = setup
    test = frozen(tmp_path)
    test.write_text(test.read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the frozen"):
        run(fake, tmp_path, data=test, final=True)
    assert fake.requests == []


def test_final_is_refused_for_a_file_that_is_not_frozen(setup):
    fake, tmp_path = setup
    with pytest.raises(SystemExit, match="only for a frozen file"):
        run(fake, tmp_path, final=True)


def test_a_missing_ollama_stops_with_a_plain_message(setup):
    fake, tmp_path = setup
    with pytest.raises(SystemExit, match="cannot reach Ollama"):
        run(fake, tmp_path, host="http://127.0.0.1:9")


def write_answers(prefix, labels):
    for version in VERSIONS:
        with open(f"{prefix}.{version}.jsonl", "w", encoding="utf-8") as f:
            for p, label in zip(PAIRS, labels[version]):
                raw = "not json" if label is None else json.dumps(
                    {"label": label, "type": "merely_related" if label == "non_contradiction" else None, "reason": "r"})
                f.write(json.dumps({"id": p["id"], "raw": raw}) + "\n")


def test_compare_counts_label_agreement_and_invalid_answers(setup):
    _, tmp_path = setup
    same = ["non_contradiction", "contradiction", "non_contradiction"]
    write_answers(tmp_path / "theirs", {v: same for v in VERSIONS})
    write_answers(tmp_path / "ours", {"as_is": same, "swapped": ["contradiction", "contradiction", "non_contradiction"],
                                      "no_dates": [None, "contradiction", "non_contradiction"]})
    result = run_ollama.compare(tmp_path / "valid.jsonl", tmp_path / "ours", tmp_path / "theirs",
                                state_path=tmp_path / "STATE.md")
    assert result["as_is"]["label_agreement"] == 1 and result["as_is"]["same_raw_text"] == 3
    assert result["swapped"]["label_differs"] == ["v-000"]
    assert result["no_dates"]["label_differs"] == ["v-000"] and result["no_dates"]["ours"]["invalid"] == 1
    assert result["all"]["label_agreement"] == pytest.approx(7 / 9) and result["all"]["invalid_ours"] == 1
    assert result["as_is"]["type_agreement"] == 1 and result["as_is"]["type_agreement_of"] == 2
