"""Tests for the RL pool rule (judge.pool), the sampler (judge.generate.sample_items) and
scripts/sample_pool.py, with made-up pairs in a temporary folder and a fake GPU job.
No GPU, no network, nothing written to the working checkout."""

import importlib.util
import json
from pathlib import Path

import pytest

from judge.generate import SAMPLING, sample_items
from judge.pool import ALWAYS_RIGHT_SHARE, M1_UNCLEAR_FIELDS, candidates, choose, judge_samples, show, version_name
from judge.prompt import build_messages

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # builds Modal objects lazily; no network, no GPU
    return module


script = load_script("sample_pool")


def ans(label, type_=None):
    return json.dumps({"label": label, "type": type_, "reason": "A short reason."})


def pair(i, label="contradiction", type_=None, hard="paraphrase", prefix="tr2"):
    return {
        "id": f"{prefix}-{i:04d}", "person": "Test Person",
        "a": {"text": f"Test Person likes colour {i}.", "written_at": "2025-01-01"},
        "b": {"text": f"Test Person likes colour {i + 1}.", "written_at": "2025-01-02"},
        "label": label, "type": type_, "hard": hard, "reason": "Made up for a test.",
        "source": "synthetic", "label_by": "claude-code",
    }


def m1_pair(i, label):
    row = pair(i, label, None if label != "non_contradiction" else "update", prefix="tr")
    for key in ("hard", "source", "label_by"):
        del row[key]
    return row


def test_candidates_add_m1_unclear_with_its_fields_only():
    m1 = [m1_pair(1, "unclear"), m1_pair(2, "contradiction"), m1_pair(3, "unclear")]
    pairs = candidates([pair(1)], m1)
    assert [p["id"] for p in pairs] == ["tr2-0001", "tr-0001", "tr-0003"]
    assert all(pairs[i][k] == v for i in (1, 2) for k, v in M1_UNCLEAR_FIELDS.items())
    assert "hard" not in m1[0]  # the M1 rows themselves are not changed
    with pytest.raises(ValueError):
        candidates([pair(1), pair(1)], [])


def test_show_follows_format_rules_and_is_fixed():
    pairs = [pair(i) for i in range(30)]
    shown = show(pairs)
    assert shown == show(pairs)
    assert sum(s["swapped"] for s in shown) == 15 and sum(not s["dates"] for s in shown) == 10
    for s, p in zip(shown, pairs):
        assert s["version"] == version_name(s["swapped"], s["dates"])
        visible = {**p, "a": p["b"], "b": p["a"]} if s["swapped"] else p
        assert s["messages"] == build_messages(visible, dates=s["dates"])  # the prompt only, no answer


def test_version_names():
    assert [version_name(s, d) for s, d in ((False, True), (True, True), (False, False), (True, False))] == \
        ["as_is", "swapped", "no_dates", "swapped_no_dates"]


def test_outcomes():
    c = pair(1)
    assert judge_samples(c, [ans("contradiction")] * 8)["outcome"] == "always_right"
    assert judge_samples(c, [ans("unclear")] * 8)["outcome"] == "always_wrong"
    assert judge_samples(c, ["nonsense"] * 8)["outcome"] == "always_wrong"
    mixed = judge_samples(c, [ans("contradiction")] * 5 + [ans("unclear")] * 2 + ["x"])
    assert mixed["outcome"] == "mixed"
    assert mixed["rewards"] == [1.0] * 5 + [-0.5] * 2 + [-2.0]
    assert mixed["answer_counts"] == {"contradiction": 5, "non_contradiction": 0, "unclear": 2, "invalid": 1}
    # Right label every time but the type varies: the rewards differ, so it is mixed.
    u = pair(2, "non_contradiction", "update")
    assert judge_samples(u, [ans("non_contradiction", "update")] * 7 + [ans("non_contradiction", "merely_related")])["outcome"] == "mixed"
    assert judge_samples(u, [ans("non_contradiction", "merely_related")] * 8)["outcome"] == "always_right"
    # On an unclear pair, unclear every time is always right.
    assert judge_samples(pair(3, "unclear"), [ans("unclear")] * 8)["outcome"] == "always_right"


def test_choose_takes_all_mixed_and_a_fixed_share_of_always_right():
    judged = ([{"id": f"m{i}", "outcome": "mixed"} for i in range(20)]
              + [{"id": f"r{i:02d}", "outcome": "always_right"} for i in range(30)]
              + [{"id": f"w{i}", "outcome": "always_wrong"} for i in range(3)])
    pool, wrong = choose(judged)
    assert [i for i in pool if i.startswith("m")] == [f"m{i}" for i in range(20)]
    assert sum(i.startswith("r") for i in pool) == round(ALWAYS_RIGHT_SHARE * 20) == 4
    assert wrong == ["w0", "w1", "w2"] and not any(i.startswith("w") for i in pool)
    assert choose(judged) == (pool, wrong)  # the same seed gives the same pool
    assert choose(list(reversed(judged)))[0] == list(reversed(pool))  # the draw does not depend on file order
    # Fewer always-right pairs than the share: all of them.
    few = [{"id": f"m{i}", "outcome": "mixed"} for i in range(10)] + [{"id": "r", "outcome": "always_right"}]
    assert choose(few)[0][-1] == "r"


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeIds:
    def __init__(self, rows, length):
        self.shape = (rows, length)


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, texts, **kwargs):
        self.texts = texts
        return FakeInputs(input_ids=FakeIds(len(texts), 2))

    def decode(self, tokens, skip_special_tokens):
        return "".join(tokens)


class FakeModel:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs["num_return_sequences"]
        # Rows come prompt by prompt: n answers for the first prompt, then n for the next.
        return [["p", "p", f"{text[-1]}#{j}"] for text in self.tokenizer.texts for j in range(n)]


def test_sample_items_groups_n_answers_per_prompt_in_one_call():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    items = [{"id": f"x{i}", "messages": [{"role": "user", "content": f"pair {i}"}]} for i in range(5)]
    results, settings = sample_items(model, tokenizer, items, {}, batch_size=2, n=3, device="cpu")
    assert len(model.calls) == 3  # one generate call per batch of prompts
    assert all(call["num_return_sequences"] == 3 and call["do_sample"] and call["temperature"] == 1.0
               and call["top_p"] == 1.0 and call["top_k"] == 0 for call in model.calls)
    assert results[0] == {"id": "x0", "raw": ["0#0", "0#1", "0#2"]}
    assert results[4] == {"id": "x4", "raw": ["4#0", "4#1", "4#2"]}
    assert settings["generation"] == {**SAMPLING, "num_return_sequences": 3}


def test_sample_items_stops_on_a_short_answer_list():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    model.generate = lambda **kwargs: [["p", "p", "x"]]
    with pytest.raises(ValueError):
        sample_items(model, tokenizer, [{"id": "a", "messages": [{"role": "user", "content": "q"}]}] * 2,
                     {}, batch_size=2, n=2, device="cpu")


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A tiny repo in a temporary folder: candidate files, STATE.md and v1's training record."""
    from judge.prompt import prompt_sha256

    (tmp_path / "data/m2").mkdir(parents=True)
    (tmp_path / "data/m1").mkdir(parents=True)
    (tmp_path / "results/m1/trained-v1").mkdir(parents=True)
    hard = [pair(i) for i in range(1, 11)] + [pair(i, "non_contradiction", "update", "indirect_update")
                                             for i in range(11, 21)]
    m1 = [m1_pair(1, "unclear"), m1_pair(2, "contradiction")]
    for path, rows in (("data/m2/train-hard.jsonl", hard), ("data/m1/train.jsonl", m1)):
        (tmp_path / path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "STATE.md").write_text("## Frozen files\n- `data/m2/test-hard.jsonl`, SHA-256 `" + "0" * 64 + "`\n")
    repo, commit, checkpoint = script.V1_ADAPTER
    meta = {"base_model": "base/repo", "base_revision": "b" * 40, "prompt_sha256": prompt_sha256(),
            "choice": {"chosen": checkpoint}, "upload": {"repo": repo, "commit": commit, "checkpoint": checkpoint,
                                                         "private": True}}
    (tmp_path / script.TRAIN_META).write_text(json.dumps(meta))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fake_run(answers_for):
    calls = []

    def run(base, base_revision, adapter, adapter_revision, items, n, seed):
        calls.append({"adapter": adapter, "adapter_revision": adapter_revision, "n": n, "seed": seed,
                      "ids": [item["id"] for item in items]})
        return {"results": [{"id": item["id"], "raw": answers_for(item["id"])} for item in items],
                "loaded_model": "base/repo", "load_seconds": 60.0, "answer_seconds": 120.0, "gpu": "NVIDIA L4",
                "peak_vram_gib": 5.0, "unsloth": "x", "transformers": "x", "peft": "x", "torch_seed": seed,
                "generation": {**SAMPLING, "num_return_sequences": n}, "chat_template_options": {},
                "first_prompt_sha256": "f" * 64}
    return run, calls


def test_smoke_then_full_run_and_choose(fake_repo):
    # Pair 1 and M1's contradiction always right; 2 to 10 always unclear (wrong); the updates mixed;
    # M1's unclear pair always unclear (right).
    def answers_for(pair_id):
        if pair_id in ("tr2-0001", "tr-0002"):
            return [ans("contradiction")] * 8
        if pair_id.startswith("tr2-") and int(pair_id[4:]) <= 10:
            return [ans("unclear")] * 8
        if pair_id == "tr-0001":
            return [ans("unclear")] * 8
        return [ans("non_contradiction", "update")] * 6 + [ans("contradiction")] * 2

    run, calls = fake_run(answers_for)
    script.sample(True, "scratch", run)
    assert len(calls[0]["ids"]) == script.SMOKE_PAIRS and calls[0]["n"] == 8 and calls[0]["seed"] == script.SAMPLE_SEED
    assert calls[0]["adapter_revision"] == script.V1_ADAPTER[1]
    smoke_meta = json.loads((fake_repo / "scratch/v1-samples-smoke.meta.json").read_text())
    assert smoke_meta["projection"]["answer_seconds_per_pair"] == 7.5
    assert smoke_meta["candidates"] == 21  # 20 hard pairs and 1 M1 unclear pair

    with pytest.raises(SystemExit):  # never replaces a saved run
        script.sample(True, "scratch", run)

    script.sample(False, "results/m2/pool", run)
    assert len(calls) == 2 and len(calls[1]["ids"]) == 21
    script.choose_pool("results/m2/pool/v1-samples.jsonl", "out/rl-pool.jsonl", "out/wrong.jsonl", "out/summary.md",
                       billed={"usd": 0.8, "l4_usd": 0.666})
    pool = [json.loads(line) for line in (fake_repo / "out/rl-pool.jsonl").read_text().splitlines()]
    wrong = [json.loads(line) for line in (fake_repo / "out/wrong.jsonl").read_text().splitlines()]
    # 10 mixed updates; 2 always right (tr2-0001 and M1's unclear), round(0.2 * 10) = 2 join; 9 always wrong.
    assert len([r for r in pool if r["outcome"] == "mixed"]) == 10
    assert len([r for r in pool if r["outcome"] == "always_right"]) == 2
    assert [r["id"] for r in wrong] == [f"tr2-{i:04d}" for i in range(2, 11)]
    assert all(len(r["answers"]) == 8 for r in wrong)
    row = next(r for r in pool if r["outcome"] == "mixed")
    assert row["rewards"] == [1.25] * 6 + [-1.5] * 2 and row["version"] in ("as_is", "swapped", "no_dates", "swapped_no_dates")
    assert row["answer_counts"] == {"contradiction": 2, "non_contradiction": 6, "unclear": 0, "invalid": 0}
    summary = (fake_repo / "out/summary.md").read_text()
    assert "Pool size: 12 pairs" in summary and "Always wrong the same way:** 9" in summary
    assert "m1_unclear (unclear)" in summary
    assert "Billed by Modal** (`modal billing report`): $0.80 in all, of which $0.67 is the L4 (about 50.0 L4 minutes)" in summary

    with pytest.raises(SystemExit):  # the choose stage never replaces its outputs either
        script.choose_pool("results/m2/pool/v1-samples.jsonl", "out/rl-pool.jsonl", "out/w2.jsonl", "out/s2.md")


def test_sample_stops_before_any_run_when_v1_is_not_the_record(fake_repo):
    meta_path = fake_repo / script.TRAIN_META
    meta = json.loads(meta_path.read_text())
    meta["upload"]["commit"] = "c" * 40
    meta_path.write_text(json.dumps(meta))
    run, calls = fake_run(lambda pair_id: [ans("unclear")] * 8)
    with pytest.raises(SystemExit):
        script.sample(True, "scratch", run)
    assert calls == []


def test_sample_refuses_a_frozen_candidate_file(fake_repo):
    (fake_repo / "STATE.md").write_text("## Frozen files\n- `data/m2/train-hard.jsonl`, SHA-256 `" + "0" * 64 + "`\n")
    run, calls = fake_run(lambda pair_id: [ans("unclear")] * 8)
    with pytest.raises(SystemExit):
        script.sample(True, "scratch", run)
    assert calls == []


def test_choose_stops_if_a_candidate_file_changed(fake_repo):
    run, _ = fake_run(lambda pair_id: [ans("unclear")] * 8)
    script.sample(False, "results/m2/pool", run)
    with open(fake_repo / "data/m2/train-hard.jsonl", "a") as f:
        f.write(json.dumps(pair(99)) + "\n")
    with pytest.raises(SystemExit):
        script.choose_pool("results/m2/pool/v1-samples.jsonl", "out/p.jsonl", "out/w.jsonl", "out/s.md")
    assert not (fake_repo / "out/p.jsonl").exists()


def test_split_and_merge_parts_keep_every_answer_in_order():
    assert script.split(list(range(10)), 4) == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert script.split([1], 4) == [[1]]
    seen = []

    def starmap(args):
        seen.extend(args)
        return [{"results": [{"id": item["id"], "raw": [str(seed)]} for item in items], "load_seconds": 75.0,
                 "answer_seconds": 100.0, "peak_vram_gib": 10.0 + seed % 10, "torch_seed": seed, "gpu": "NVIDIA L4"}
                for (_, _, _, _, items, n, seed) in args]

    items = [{"id": f"x{i}"} for i in range(9)]
    result = script.run_in_parts(starmap, parts=4)("b", "r", "a", "ar", items, 8, 3407)
    assert [line["id"] for line in result["results"]] == [f"x{i}" for i in range(9)]
    assert [s for (*_, s) in seen] == [3407, 3408, 3409, 3410] and result["torch_seed"] == [3407, 3408, 3409, 3410]
    assert result["load_seconds"] == 300.0 and result["answer_seconds"] == 400.0 and result["parts"] == 4
    assert result["peak_vram_gib"] == 19.0  # seed 3409 gives 10 + 9


def sampled_repo(fake_repo):
    """The fake repo with a saved full run, as choose_pool reads it."""
    run, _ = fake_run(lambda pair_id: [ans("contradiction")] * 4 + [ans("unclear")] * 4)
    script.sample(False, "results/m2/pool", run)
    return fake_repo / "results/m2/pool/v1-samples.jsonl"


def choose_stops(fake_repo, samples):
    with pytest.raises(SystemExit) as stop:
        script.choose_pool(samples, "out/p.jsonl", "out/w.jsonl", "out/s.md")
    assert not (fake_repo / "out").exists()  # stopped before writing anything
    return str(stop.value)


# Codex review M2-04, finding 2: samples made with another prompt or other pool settings are refused.
def test_choose_stops_if_the_prompt_changed(fake_repo, monkeypatch):
    import judge.prompt

    samples = sampled_repo(fake_repo)
    monkeypatch.setattr(judge.prompt, "SYSTEM_MESSAGE", judge.prompt.SYSTEM_MESSAGE + " Be brief.")
    assert "prompt changed" in choose_stops(fake_repo, samples)


@pytest.mark.parametrize("name", ["SHOW_SEED", "KEEP_SEED", "SAMPLES"])
def test_choose_stops_if_a_pool_setting_changed(fake_repo, monkeypatch, name):
    import judge.pool

    samples = sampled_repo(fake_repo)
    monkeypatch.setattr(judge.pool, name, getattr(judge.pool, name) + 1)
    assert "changed since sampling" in choose_stops(fake_repo, samples)


# Codex review M2-04, finding 3: every candidate sampled exactly once, nothing else.
def rewrite_samples(samples, change):
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    samples.write_text("".join(json.dumps(row) + "\n" for row in change(rows)))


def test_choose_stops_on_a_repeated_id_that_hides_a_missing_one(fake_repo):
    samples = sampled_repo(fake_repo)
    rewrite_samples(samples, lambda rows: [rows[0], rows[0]] + rows[2:])  # same row count as before
    message = choose_stops(fake_repo, samples)
    assert "repeated ['tr2-0001']" in message and "missing ['tr2-0002']" in message


def test_choose_stops_on_a_missing_or_unknown_id(fake_repo):
    samples = sampled_repo(fake_repo)
    rewrite_samples(samples, lambda rows: rows[:-1])
    assert "missing ['tr-0001']" in choose_stops(fake_repo, samples)
    rewrite_samples(samples, lambda rows: rows + [{**rows[0], "id": "tr2-9999"}])
    assert "not expected ['tr2-9999']" in choose_stops(fake_repo, samples)


def test_a_smoke_run_must_hold_exactly_the_smoke_ids(fake_repo):
    run, _ = fake_run(lambda pair_id: [ans("contradiction")] * 4 + [ans("unclear")] * 4)
    script.sample(True, "scratch", run)
    samples = fake_repo / "scratch/v1-samples-smoke.jsonl"
    rewrite_samples(samples, lambda rows: rows[1:] + [{**rows[0], "id": "tr2-0020"}])
    assert "missing" in choose_stops(fake_repo, samples)
