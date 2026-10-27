# M2 Step 3 part B: the RL pool

v1 (the M1 judge, epoch-2 adapter) answered every training candidate 8 times, with some randomness (temperature 1.0). Each answer got points from the reward function. GRPO, the RL method, can only learn from a pair where those points differ, so those pairs make up the pool, plus a few pairs v1 always gets right.

## The result

- **Candidates:** 1018 pairs (971 M2 hard training pairs and 47 M1 `unclear` training pairs).
- **Mixed** (the 8 rewards differ): 262. All of them are in the pool.
- **Always right:** 747. The pool takes 52 of them (20% of the mixed count, drawn with seed 2026102705), so v1 does not forget them.
- **Always wrong the same way:** 9. Kept out and listed in `data/m2/rl-pool-always-wrong.jsonl` for Claude in Cowork to check for a wrong label.
- **Always `unclear`** (all 8 answers `unclear`, counted inside the groups above): 22.
- **Pool size: 314 pairs** (`data/m2/rl-pool.jsonl`).
- **v1's sampled accuracy:** 90% of 8144 answers had the right label.
- **`unclear` on pairs that can be decided:** 452 of 7760 answers (6%).
- **Invalid answers:** 9 of 8144.

Note: "always right" means the right label all 8 times. On a non-contradiction, a mix of right and wrong types gives different rewards (1.25 and 1.0), so that pair counts as mixed.

## Per trap

Accuracy is the share of the 8 answers with the right label. "Unclear" is the share of answers that said `unclear` on a pair that can be decided.

| Trap (right answer) | Pairs | Mixed | Always right | Always wrong | Always unclear | In pool | Accuracy | Unclear |
|---|---|---|---|---|---|---|---|---|
| change_word_contradiction (contradiction) | 77 | 20 | 57 | 0 | 0 | 25 | 94% | 4% |
| explain_away (contradiction) | 125 | 45 | 79 | 1 | 0 | 53 | 88% | 10% |
| far_apart_no_change (contradiction) | 78 | 14 | 64 | 0 | 0 | 17 | 97% | 3% |
| indirect_update (update) | 80 | 16 | 64 | 0 | 0 | 21 | 92% | 1% |
| looks_like_clash (compatible_detail) | 100 | 26 | 72 | 2 | 0 | 34 | 88% | 3% |
| m1_unclear (unclear) | 47 | 26 | 20 | 1 | 20 | 26 | 80% | - |
| near_identical (compatible_detail) | 70 | 3 | 67 | 0 | 0 | 8 | 100% | 0% |
| near_identical (merely_related) | 85 | 19 | 66 | 0 | 0 | 25 | 92% | 3% |
| paraphrase (contradiction) | 95 | 58 | 35 | 2 | 2 | 59 | 70% | 27% |
| rule5_bait (merely_related) | 85 | 13 | 72 | 0 | 0 | 18 | 94% | 3% |
| small_difference (contradiction) | 95 | 18 | 76 | 1 | 0 | 21 | 94% | 3% |
| true_unclear (unclear) | 1 | 0 | 0 | 1 | 0 | 0 | 0% | - |
| update_with_overlap (update) | 80 | 4 | 75 | 1 | 0 | 7 | 97% | 0% |
| **all** | 1018 | 262 | 747 | 9 | 22 | 314 | 90% | 6% |

## Five mixed pairs and their 8 answers (drawn with seed 5)

### tr2-0430 (paraphrase (contradiction)), shown no dates

- Memory a: Piret has never learned to drive. (2026-06-06)
- Memory b: Piret's weekly trip to the wholesale market means two hours at the wheel of her van. (2026-06-06)
- Right answer: contradiction

| # | Answer | Points |
|---|---|---|
| 1 | `{"label": "unclear", "type": null, "reason": "Piret may drive the van herself or just use someone else's key."}` | -0.50 |
| 2 | `{"label": "unclear", "type": null, "reason": "Wheeling the van could mean either that Piret has driven it or that she rides with others who do."}` | -0.50 |
| 3 | `{"label": "unclear", "type": null, "reason": "Van trips to the market can count as driving or merely as driving the van."}` | -0.50 |
| 4 | `{"label": "unclear", "type": null, "reason": "Piret may be an ordinary passenger or a van driver, and that turns the pair into a contradiction or a compatible stay-at-home fact."}` | -0.50 |
| 5 | `{"label": "unclear", "type": null, "reason": "Whichever van is mentioned, Piret might not need a licence to drive it or needs one to drive in the city."}` | -0.50 |
| 6 | `{"label": "unclear", "type": null, "reason": "Vans can be rented or borrowed, and Piret might still never have driven one herself."}` | -0.50 |
| 7 | `{"label": "unclear", "type": null, "reason": "For a trip to the market, Piret might well hold the wheel of the van herself, or someone else might drive."}` | -0.50 |
| 8 | `{"label": "contradiction", "type": null, "reason": "One says Piret can't drive, the other that she spends many weeks behind the wheel of her van."}` | +1.00 |

### tr2-0646 (indirect_update (update)), shown no dates

- Memory a: Chenai drinks sweet tea at her desk all afternoon. (2023-07-10)
- Memory b: Chenai gave up tea and keeps a bottle of water at her desk instead. (2023-10-16)
- Right answer: non_contradiction / update

| # | Answer | Points |
|---|---|---|
| 1 | `{"label": "contradiction", "type": null, "reason": "One says Chenai gave up tea at her desk, the other that she drinks sweet tea there all day."}` | -1.50 |
| 2 | `{"label": "contradiction", "type": null, "reason": "One says Chenai drinks sweet tea at her desk, the other that she gave up tea there."}` | -1.50 |
| 3 | `{"label": "contradiction", "type": null, "reason": "One says Chenai gave up tea and keeps water at her desk, the other that she drinks sweet tea all afternoon."}` | -1.50 |
| 4 | `{"label": "non_contradiction", "type": "update", "reason": "The newer memory says Chenai gave up tea and drinks water at her desk."}` | +1.25 |
| 5 | `{"label": "contradiction", "type": null, "reason": "One says Chenai gave up tea, the other that she drinks sweet tea at her desk all afternoon."}` | -1.50 |
| 6 | `{"label": "non_contradiction", "type": "update", "reason": "The newer memory says Chenai gave up tea at her desk."}` | +1.25 |
| 7 | `{"label": "non_contradiction", "type": "update", "reason": "The newer memory says Chenai gave up tea and drinks only water at her desk."}` | +1.25 |
| 8 | `{"label": "contradiction", "type": null, "reason": "One says Chenai gave up tea altogether, the other that she drinks sweet tea at her desk."}` | -1.50 |

### tr-1506 (m1_unclear (unclear)), shown swapped no dates

- Memory a: Paulina never takes taxis. (2023-04-22)
- Memory b: Paulina books a ride-hailing car to get to the airport. (2023-05-30)
- Right answer: unclear

| # | Answer | Points |
|---|---|---|
| 1 | `{"label": "contradiction", "type": null, "reason": "One says Paulina never takes taxis, the other that she booked a ride-hailing car."}` | -1.50 |
| 2 | `{"label": "unclear", "type": null, "reason": "Ride-hailing cars are taxis to some people and their own kind to others, and Paulina's rule says nothing about that."}` | +1.00 |
| 3 | `{"label": "unclear", "type": null, "reason": "Ride-hailing cars are taxis or independent taxis, and Paulina's view of them is not given."}` | +1.00 |
| 4 | `{"label": "unclear", "type": null, "reason": "A book-fleet car could be what Paulina takes to the airport, or her car could have broken down and she shouldn't have used a ride-hailing car."}` | +1.00 |
| 5 | `{"label": "contradiction", "type": null, "reason": "One says Paulina never takes taxis, the other that she has booked a ride-hailing car."}` | -1.50 |
| 6 | `{"label": "unclear", "type": null, "reason": "Ride-hailing cars are taxis or something different, and the memory leaves out which."}` | +1.00 |
| 7 | `{"label": "unclear", "type": null, "reason": "The memories disagree on whether Paulina ever takes taxis, and rides from a driver are named a taxi in some places and ride-hailing in others."}` | +1.00 |
| 8 | `{"label": "unclear", "type": null, "reason": "A ride-hailing car counts as a taxi to some people and as different to others, which the memories do not tell us."}` | +1.00 |

### tr2-0888 (near_identical (merely_related)), shown swapped

- Memory a: Ameyalli's grandmother gave her a ceramic owl. (2024-12-28)
- Memory b: Ameyalli's grandmother gave her a ceramic frog, which sits on the windowsill. (2025-01-19)
- Right answer: non_contradiction / merely_related

| # | Answer | Points |
|---|---|---|
| 1 | `{"label": "contradiction", "type": null, "reason": "One says Ameyalli's grandmother gave her a ceramic owl, the other a ceramic frog."}` | -1.50 |
| 2 | `{"label": "contradiction", "type": null, "reason": "They give different gifts from Ameyalli's grandmother, a ceramic owl in one and a ceramic frog in the other."}` | -1.50 |
| 3 | `{"label": "contradiction", "type": null, "reason": "They name different ceramic gifts from Ameyalli's grandmother."}` | -1.50 |
| 4 | `{"label": "contradiction", "type": null, "reason": "They name different animals as gifts from Ameyalli's grandmother."}` | -1.50 |
| 5 | `{"label": "contradiction", "type": null, "reason": "They name different ceramic animals as gifts from Ameyalli's grandmother."}` | -1.50 |
| 6 | `{"label": "unclear", "type": null, "reason": "Both describe a ceramic gift from Ameyalli's grandmother, and either could be right."}` | -0.50 |
| 7 | `{"label": "contradiction", "type": null, "reason": "One says Ameyalli's grandmother gave her a ceramic owl, the other a ceramic frog."}` | -1.50 |
| 8 | `{"label": "contradiction", "type": null, "reason": "They name different items as the gift that Ameyalli's grandmother gave her."}` | -1.50 |

### tr2-0418 (paraphrase (contradiction)), shown swapped no dates

- Memory a: Chuan refuses to carry an umbrella and has never owned one, even after he moved to the coast. (2023-10-17)
- Memory b: Chuan shakes out his folding brolly in the office doorway on wet mornings. (2023-10-17)
- Right answer: contradiction

| # | Answer | Points |
|---|---|---|
| 1 | `{"label": "contradiction", "type": null, "reason": "One says Chuan uses a folding brolly every wet morning, the other that he has never carried an umbrella anywhere."}` | +1.00 |
| 2 | `{"label": "unclear", "type": null, "reason": "The brolly might be Chuan's umbrella and a replacement, or a toy he is allowed to keep."}` | -0.50 |
| 3 | `{"label": "contradiction", "type": null, "reason": "One says Chuan has never owned an umbrella, the other that he has his folding brolly in the office doorway."}` | +1.00 |
| 4 | `{"label": "non_contradiction", "type": "compatible_detail", "reason": "Shaking out the brolly at the office doorway is the detail of Chuan carrying a collapsible brolly rather than a traditional umbrella."}` | -1.00 |
| 5 | `{"label": "contradiction", "type": null, "reason": "One says Chuan carries an umbrella, the other that he carries none and has never owned one."}` | +1.00 |
| 6 | `{"label": "unclear", "type": null, "reason": "A folding brolly is an umbrella-like thing, and Chuan's rule about carrying one is not told to be absolute."}` | -0.50 |
| 7 | `{"label": "contradiction", "type": null, "reason": "One says Chuan keeps his own folding brolly, the other that he carries none and has never owned an umbrella."}` | +1.00 |
| 8 | `{"label": "unclear", "type": null, "reason": "Shaking the brolly out on wet mornings may or may not be Chuan's usual coat for rain, since no brolly is described."}` | -0.50 |

## GPU time and cost

- One run in 4 part(s), each on a Modal NVIDIA L4: 51.9 GPU minutes inside the functions, added up (load 2.4, answering 49.6), 72.1 minutes from start to finish (this includes any wait for a free GPU).
- Estimate before the bill: about $0.69 to $3.84 (L4 at $0.000222 a second, https://modal.com/pricing; the high end assumes every part's GPU ran for the whole time).
- **Billed by Modal** (`modal billing report`): $0.80 in all, of which $0.74 is the L4 (about 55.2 L4 minutes); the rest is CPU, memory and network.

## Fingerprints and settings

- `data/m2/rl-pool.jsonl` SHA-256: `2c5c3bf2591c8376b72f15e0a34d223ec24fc366e45b781e4706dcaebdac4e77`
- Reward fingerprint (`judge.reward.reward_sha256()`): `a369e730de69a4305f079697c647c0bc25332d6bf5510eae10207bef65b0c70c`
- Prompt fingerprint: `83d14d39a2115eb1f4d2d461581673090ec25d0a6f05d444cb2da8b756a2865c`
- Candidate files: `data/m2/train-hard.jsonl` `8c2abddb93b21a0850c000c2919ee0bb11277ecf3b9e272d55cf9703110ca856`, `data/m1/train.jsonl` `a41de541d2a25c564009e232b2d75ccb714c5a668dcd30ef73004fae778231b1`
- v1: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` at `7744afa8566e264af1a92a806d8d9aae00cc7c78` plus `eskora/m1-contradiction-judge-qwen3-4b-lora` at `1d9ebbf66e1a2abcad9932932bb7d810a38c59b9` (epoch-2); loaded `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.
- Sampling: {'max_new_tokens': 256, 'do_sample': True, 'num_beams': 1, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'num_return_sequences': 8}, 8 answers per pair in one generate call, torch seed [3407, 3408, 3409, 3410].
- Versions shown (judge.format's rules, seed 2026102704): as_is 335, swapped 344, no_dates 174, swapped_no_dates 165.
- Each pool row keeps its version, its 8 rewards and its answer counts. The 8 raw answers are in `results/m2/pool/v1-samples.jsonl`.
