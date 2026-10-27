# M2-04: reward fingerprint recheck after the Codex review fixes

Written by Claude Code on 2026-10-27, branch `m2-04-pool-and-reward`, for finding 1 of `reviews/M2-04-codex-review.md`. No GPU, no Modal, no test set or validation file read.

## Why there is a new fingerprint

The reward fingerprint (`judge.reward.reward_sha256()`) used to cover the reward table and four functions only. It missed the helpers and constants those functions use (the allowed types, the code-fence pattern, the repeated-field and special-number checks), so a change there could alter rewards without changing the fingerprint. It now follows every name the reward code uses, and also covers how a training completion is read (`completion_text`).

The rules themselves did not change. Only the way the fingerprint is worked out changed, so the same rules now give a different code.

| | Fingerprint |
|---|---|
| Recorded when the pool was sampled (`results/m2/pool/v1-samples.meta.json`) | `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c` |
| The old method, run on today's code | `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c` (same: the rules are unchanged) |
| The new method, today (`reward_sha256()`) | `92a043f18ac483f60aac7036375b3d98abfde4e6c82191d8303af7f70a8c3a53` |

Step 5 should record the new fingerprint, `92a043f1...c3a53`, and treat it as the same reward the pool was chosen with. The saved samples, their meta file and the pool keep the old one; they are not rewritten.

## Offline recheck of the saved pool with today's code

| Check | Result |
|---|---|
| Prompt fingerprint recorded at sampling equals today's | yes (`83d14d39...2865c`) |
| Pool settings recorded equal `judge.pool` (show seed, keep seed, answers per pair) | yes (2026102704, 2026102705, 8) |
| Candidate files unchanged since sampling | yes |
| Sample rows, different IDs, candidates | 1,018, 1,018, 1,018; the new ID check passes |
| Answers rescored | 8,144 |
| Stored reward lists, answer counts and outcomes checked (314 pool + 9 always wrong) | 323, no difference |
| `data/m2/rl-pool.jsonl` SHA-256 | `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77` (unchanged) |

## How it was run

From the repo folder, `uv run python recheck.py` with this script (kept here, not in the repo's scripts):

```python
import hashlib, inspect, json, sys
sys.path.insert(0, "scripts")
from judge.reward import reward, read_answer, REWARD, reward_sha256
from judge.score import parse_answer, format_problem
from judge.prompt import LABELS, prompt_sha256
from judge.pool import judge_samples, show, SHOW_SEED
import sample_pool as sp

def old_method():
    rules = {"table": REWARD, "labels": list(LABELS),
             "code": [inspect.getsource(f) for f in (reward, read_answer, parse_answer, format_problem)]}
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode("utf-8")).hexdigest()

meta = json.load(open("results/m2/pool/v1-samples.meta.json"))
rows = [json.loads(l) for l in open("results/m2/pool/v1-samples.jsonl") if l.strip()]
pool = [json.loads(l) for l in open("data/m2/rl-pool.jsonl") if l.strip()]
wrong = [json.loads(l) for l in open("data/m2/rl-pool-always-wrong.jsonl") if l.strip()]
pairs, data_sha = sp.load_candidates()
by_id = {p["id"]: p for p in pairs}
sp.check_sampled_ids([r["id"] for r in rows], [s["id"] for s in show(pairs, SHOW_SEED)])
judged = {r["id"]: judge_samples(by_id[r["id"]], r["raw"]) for r in rows}
stored = pool + wrong
mismatch = [s["id"] for s in stored
            if (judged[s["id"]]["rewards"], judged[s["id"]]["answer_counts"], judged[s["id"]]["outcome"])
            != (s["rewards"], s["answer_counts"], s["outcome"])]
print(meta["reward_sha256"], old_method(), reward_sha256())
print(meta["prompt_sha256"] == prompt_sha256(), (meta["show_seed"], meta["keep_seed"], meta["samples_per_pair"]))
print(data_sha == meta["data_sha256"], len(rows), len({r["id"] for r in rows}), len(pairs))
print(sum(len(r["raw"]) for r in rows), len(stored), mismatch)
print(hashlib.sha256(open("data/m2/rl-pool.jsonl", "rb").read()).hexdigest())
```
