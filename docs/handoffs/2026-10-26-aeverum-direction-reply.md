# Aeverum direction reply to the "Fine Tuning and RL training" project (Cowork, 2026-10-26)

Written by Cowork (Claude, Aeverum's operator), answering `.handoffs/2026-10-26-model-training-update-2.md` (SHA-256 `f04139b716e650bde8fc78851cb9041d497760b5c71de4bf8832338424ffe31b`, verified). **Information only:** nothing here changes Aeverum's code, Sources, decisions, step plan or procedures, nothing here approves any model for Aeverum, and nothing here sets a rule for your project. No benchmark, calibration or holdout pair text, ids or counts are shared, and no Source text is copied for the reading rules.

**Where the answers come from.** Aeverum's `PRODUCT.md` and `DECISIONS.md` on branch `step-10-dashboard` at `bc2835239581c2315af611ed3772637bd2929b8c` (the same text as `main` for every rule cited here). Your side was read only, at `~/code/model-training` `main` `92a78dc734f8eefebd30f14430c7eba86c386a29` (`milestones/M1-contradiction-judge/SPEC.md` reading rules 1 to 5, and `docs/run-judge-on-mac.md`, whose three pins match your update).

**Three labels used below.** "Set" means an Aeverum rule says it. "Not set yet" means nothing says it; steps 12 and 13 will decide it, through decisions Elias approves. "Cowork's view" means the operator's expectation or advice, which binds nobody.

## Short version

- Most of what M2 asks is **not set yet**. Step 11 (notifications) comes next, then step 12 (judge benchmark), then step 13 (night pass).
- **Set:** the bar (contradiction precision at least 0.85, recall at least 0.80, on Aeverum's benchmark, Elias's labels decide); unclear pairs become `needs_manual_review` when the local judge misses the bar; every proposal records the judge's identity and, where a judge was used, its reason; the judge only proposes, and Elias decides every proposal.
- **Cowork's view for M2:** keep Style A as the main line; keep testing all three conditions (as written, swapped, no dates); treat your 1.5-to-1 cost ratio as your own assumption; freeze v2 before step 12 measures anything, and do not train a later version on step-12 results unless Elias decides that (point 8).

## What sets M2's training

**1. `unclear`.**
- Set: if the local judge misses the bar, unclear pairs are staged as `needs_manual_review` (unless Elias picks a cloud judge for that pass). A proposal can carry a `needs_manual_review` flag.
- Not set yet: how `unclear` counts in step-12 precision and recall; whether `unclear` also becomes `needs_manual_review` when the judge meets the bar; whether a flagged pair is staged as a `contradiction` proposal (which would also raise both memories' "heat", Aeverum's measure of how much attention a memory is getting).
- Cowork's view: a `needs_manual_review` item still lands in front of Elias, so it costs him about what a flagged pair costs, but it does not claim to be sure. Your scoring (`unclear` is never a "yes") is the careful choice. A small cost for `unclear` on decidable pairs fits; pushing `unclear` to zero would not, because Aeverum's rules prefer "not sure" over false certainty.

**2. Dates and order.**
- Set: each memory has a write time in its record. A stored contradiction proposal lists its two memories in id order, and that order means nothing. This is about the stored proposal, not about what the judge is shown.
- Not set yet: whether steps 12 and 13 send write dates; which date counts for a memory that was later corrected.
- Cowork's view: the judge is told Aeverum's update reading, and that reading rests on the two claims being written at different times, with the later one being the update. Being told the rule is not the same as being given dates, so a harness that sends no dates would need another way to show which came later, or the judge would rely on wording alone. Sending dates seems likely for that reason, but it is not promised. Keep training and testing with dates, without dates, and swapped.

**3. Answer format and speed.**
- Set: one judge interface; each proposal names the memory ids, the run id, the judge identity and the judge's stated reason. The model only has to supply its answer.
- Not set yet: the exact answer format; whether a `steps` field is accepted; any time budget per pair or per night pass.
- Cowork's view: Style A is the safer main line, because the harness will be written against one format and v1's is already pinned. A `steps` field could probably be read as an extra field, but it is model-written text about the memories: it must never reach a log (point 8), and whether it is stored would be a step-13 decision. Nothing in Aeverum's rules today makes 0.91 seconds a pair a problem.

**4. False alarm against miss.**
- Set: the bar is unchanged, precision 0.85 and recall 0.80.
- Not set yet: any cost ratio between a false alarm and a miss. Only Elias can set one.
- Cowork's view: the bar asks slightly more of precision than of recall, which fits your choice to make a false alarm cost more.
- The facts either way: a false alarm puts a wrong item in front of Elias and raises both memories' heat; a miss leaves a real clash unflagged until something else brings it up (a client's correction, or Elias reading the memories). The 1.5-to-1 ratio is your assumption, not Aeverum's.

**5. `type`.**
- Set: step 12 scores the `contradiction` label. Cowork found no Aeverum rule that uses `type`. Aeverum's hard non-contradictions come in the same three kinds as your types (updates, compatible details, merely related claims).
- Not set yet: whether step 12 reports results per kind. Cowork's view: keep `type`; it costs little and helps explain misses.

## What keeps the two sides in step

**6. Reading rules (in Cowork's words).** Aeverum has two written reading rules, both about labelled pairs Elias accepted, and the judge built in step 12 is told both:
- **Same moment.** For pairs about a favourite or a set choice, both claims are read as true at the same moment, so different answers clash. This agrees with your rule 1. Difference: Aeverum's rule covers the pairs Elias accepted under that reading (and later ones he accepts the same way); it is not written as a general test, while yours is.
- **Different times.** For pairs Elias accepted as updates, the two claims were written at different times, the second later, so they do not clash. This agrees with your rule 2 in outcome. Differences: Aeverum's reason is the different write times, not change words; your rule needs both. Aeverum's rule does not say that any two claims written at different times are an update.
- **Your rules 3, 4 and 5** have no written Aeverum counterpart. They conflict with nothing written. Rule 3 fits Aeverum's "compatible details" kind. Rules 4 and 5 are yours alone; Aeverum's labels were not written to them, and checking them against Aeverum's pairs would mean sharing pairs, so it is not done. Aeverum's records require a label review before step 12 that settles unclear wording pair by pair with Elias, which is the same kind of question your rules 4 and 5 answer.

**7. Which pairs reach the judge.**
- Set: the night pass looks at memories with new activity since the last run, the partner memories that hints and pairs point to, and memories whose expiry review is due. (The "gate" is the check every new memory passes when it is written; when a new memory is very close to an existing one, the gate records a "pair hint" naming it.)
- Not set yet: exactly which pairs go to the contradiction judge.
- Cowork's view: mostly pairs the gate found similar, since pair hints are how partner memories enter the night pass. In general terms: memories are proposed by AI clients about one person (in real use, Elias; in Aeverum's test corpus, one made-up person); the test corpus holds plausible everyday facts and preferences, with no health, financial or identity data. Cowork found no rule that limits a memory to a single fact. Step 12 scores a set of deliberately hard pairs, not the gate's output.

**8. How candidates enter step 12.**
- Set: every proposal records the judge's identity (provider, model, policy or prompt hash). Aeverum already pins its embedding model by a fixed file with a known SHA-256, kept outside Git.
- Cowork's view on pins: expect the same for a judge. the GGUF SHA-256, the Ollama digest and the prompt SHA-256 are a good basis. Please also state the Ollama version with each candidate (you do for v1), and a download link where the file's SHA-256 can be checked. One mismatch to fix on your side: `docs/run-judge-on-mac.md` says the Hugging Face repo is private, your update says public.
- Not set yet: whether one step-12 run tests several candidates side by side. Cowork's view: Cowork expects to propose that (the untrained Qwen3-8B, v1, v2) when it writes the step-12 brief; Elias approves the brief.
- **Keeping the benchmark honest (Cowork's view, important):** Aeverum's benchmark is an exam, so a candidate must be frozen before step 12 measures it, and step-12 results should not steer the training of a later version. If they did, the benchmark would quietly become training data. Whether step-12 results are ever shared back, and in what form, is Elias's decision.
- Still needed from you: the Ollama server-log check (your open item 3). Aeverum's logs and traces must hold no memory text, so this must be settled before the judge runs on real memories.

**9. Invalid or slow answers.**
- Not set yet. Cowork's view: your assumption fits Aeverum's rules: stage the pair as `needs_manual_review` with the cause recorded (invalid answer or timeout), never treat it as a "yes", and count it as a miss when the pair is a real contradiction. Keep zero invalid answers as a hard bar.

**10. The learned model after v1.**
- Set: after v1, a local model trained on Elias's verdict events may gradually take over proposing; v1 only collects the verdicts. Any future model trains on verdict events and never on the holdout. Every proposal records the judge's identity, and every verdict is an event with its inputs.
- Cowork's view on what helps now: keep your label words the same as Aeverum's (`contradiction`, `non_contradiction`, `unclear`); publish every model with its pins and Ollama version; and shape your data so each example can also say where it came from (made-up or from a verdict) and which judge identity produced the label. A verdict-based example would then be: two claims, their dates, the judge's label and identity, and Elias's verdict.
- Where the open question is decided: whether reviewed made-up labels may be mixed with verdict events is Elias's decision, and it would also need a matching change to Aeverum's product rules and Sources, which he ratifies.

## What Cowork records

This file only (`.handoffs/` is not in Git), plus one line in Aeverum's `STATE.md` at its next ordinary record. Nothing else changes.
