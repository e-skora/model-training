# Spot-check: M1 training set

**Optional.** Elias decided not to spot-check the training set (D-028: "skip the spot checking step"). Instead, Claude in Cowork has a separate agent judge a sample blind before Step 6 merges. This sheet is ready in the same format as before, for that reviewer or for Elias if he ever wants to look. Every `Elias: OK` line below is a blank form line, not an answer: nobody has checked these pairs yet.

**How to use it:** read each of the 20 pairs. If you agree with the answer, leave `Elias: OK`. If not, change it to `Elias: WRONG, <why>`.

How to judge: two memories that say different things are true at the same moment are a contradiction, even if they were written on different dates. If one memory was written later and says something changed, it is an update. The change can be said plainly ("moved", "now", "switched") or indirectly ("her new flat", "these days ... instead of"). A change word alone does not make an update. If both can be true together, it is not a contradiction. New in Step 6: `unclear` means the text alone cannot settle it, because the answer depends on a fact neither memory gives (D-025).

These 20 pairs come from `data/m1/train.jsonl`, picked with seed 11 (Python: `rng = random.Random(11)`, then `rng.sample(unclear_pairs, 2)`, `rng.sample(hard_cases, 5)` and `rng.sample(all_other_pairs, 13)`, shown in id order). A hard case here is a contradiction or a compatible detail or merely related pair that contains a change word (now, moved, switched, since, stopped, later, replaced), or an update that uses none of them. The sheet has 2 `unclear` pairs and 11 hard cases, marked below.

## 1. tr-0186: Ziad Matar

- Memory A (written 2025-08-17): Ziad has painted seascapes since his teens.
- Memory B (written 2025-08-17): Ziad frames other people's paintings at a shop in Byblos.
- Label: non_contradiction
- Type: merely_related
- Reason: Painting seascapes and framing others' paintings are different facts.
- Hard case: a merely related pair that contains a change word.

Elias: OK

## 2. tr-0193: Andry Rakoto

- Memory A (written 2026-03-15): Since January Andry has been training for the Antananarivo 10 km race.
- Memory B (written 2026-03-15): Andry is training for a 10 km race in Antananarivo, running five days a week.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Both say Andry is training for the 10 km race in Antananarivo, and one adds how often he runs.
- Hard case: a compatible detail that contains a change word.

Elias: OK

## 3. tr-0291: Krissia Alfaro

- Memory A (written 2024-01-05): Krissia flew to Montréal in December 2023 to visit her aunt.
- Memory B (written 2024-04-18): Krissia's December 2023 visit to her aunt was a bus trip to Guatemala City.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: They give two different destinations for Krissia's December 2023 visit to her aunt.

Elias: OK

## 4. tr-0379: Dilnoza Yusupova

- Memory A (written 2025-01-27): Dilnoza now knits only socks, nothing bigger.
- Memory B (written 2025-01-27): Dilnoza is now knitting a large blanket for her sister's sofa.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: One says Dilnoza knits only socks, the other that she is knitting a large blanket.
- Hard case: a contradiction that contains a change word.

Elias: OK

## 5. tr-0382: Salome Javakhishvili

- Memory A (written 2023-02-02): Salome plays the piano at home most evenings.
- Memory B (written 2023-12-12): Salome sold her piano when she moved into a smaller flat and now plays a keyboard with headphones.
- Label: non_contradiction
- Type: update
- Reason: The newer memory says Salome sold the piano and plays a keyboard instead.

Elias: OK

## 6. tr-0389: Eun-ji Shin

- Memory A (written 2024-12-03): Eun-ji's commute to the publishing house takes a little under an hour.
- Memory B (written 2024-12-03): Eun-ji spends about fifty minutes on the subway each way to work.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Fifty minutes on the subway fits a commute of a little under an hour.

Elias: OK

## 7. tr-0622: Lipika Behera

- Memory A (written 2024-09-10): Lipika painted her whole flat herself during the long weekend in August 2024.
- Memory B (written 2025-03-15): Lipika hired two painters for her flat in August 2024 and did none of the work herself.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: One says Lipika painted her flat herself that August, the other that she hired painters and did none of it.

Elias: OK

## 8. tr-0750: Mariam Fofana

- Memory A (written 2025-12-06): Mariam has eaten breakfast at the same café since arriving in Bamako.
- Memory B (written 2025-12-06): Mariam has a coffee and a buttered baguette at Café du Fleuve each morning.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Both say Mariam eats breakfast at a café each day, and one names the café and what she has.
- Hard case: a compatible detail that contains a change word.

Elias: OK

## 9. tr-0770: Rajarshi Bhattacharya

- Memory A (written 2024-05-12): Rajarshi has kept the same red armchair by the window since his student days.
- Memory B (written 2024-05-12): Rajarshi's flat has no armchairs, only floor cushions.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: One says Rajarshi has a red armchair by the window, the other that his flat has no armchairs.
- Hard case: a contradiction that contains a change word.

Elias: OK

## 10. tr-0836: Mugisha Tumwine

- Memory A (written 2025-03-02): Mugisha stopped eating sugar during his marathon training.
- Memory B (written 2025-04-18): Mugisha's goal for the marathon is to finish in under three and a half hours.
- Label: non_contradiction
- Type: merely_related
- Reason: Giving up sugar and a finishing goal are separate facts about Mugisha's marathon.
- Hard case: a merely related pair that contains a change word.

Elias: OK

## 11. tr-0918: Ziyanda Mtshali

- Memory A (written 2025-03-23): Ziyanda's puppy is ten weeks old.
- Memory B (written 2025-03-23): Ziyanda brought home a puppy that is about two and a half months old.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Ten weeks is about two and a half months, so both give the same age for Ziyanda's puppy.

Elias: OK

## 12. tr-0978: Kinan Shami

- Memory A (written 2024-03-12): Kinan watches old Syrian TV comedies online.
- Memory B (written 2024-04-01): Kinan has a framed poster of a Syrian TV comedy in his hallway.
- Label: non_contradiction
- Type: merely_related
- Reason: Watching the comedies and owning a poster are different facts.

Elias: OK

## 13. tr-1052: Florica Căldărar

- Memory A (written 2025-12-08): Florica replaced the buttons on her winter coat with old brass ones.
- Memory B (written 2025-12-08): Florica's winter coat has six brass buttons from her grandmother's tin.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Both say Florica's coat has brass buttons, and one adds how many and where they came from.
- Hard case: a compatible detail that contains a change word.

Elias: OK

## 14. tr-1205: Kaylee Sutherland

- Memory A (written 2024-03-11): Kaylee doesn't drink coffee.
- Memory B (written 2024-06-20): Kaylee meets her sister at a café for coffee every Friday.
- Label: unclear
- Type: none (only non-contradictions have a type)
- Reason: It depends on whether Kaylee actually drinks coffee at those Friday meetups or just goes along.

Elias: OK

## 15. tr-1207: Rayén Painemal

- Memory A (written 2024-04-10): Rayén has sung in the Temuco community choir since 2019.
- Memory B (written 2024-07-02): Rayén has never been a member of any choir.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: One says Rayén has sung in a community choir for years, the other that she has never been in a choir.
- Hard case: a contradiction that contains a change word.

Elias: OK

## 16. tr-1262: Adaeze Onuoha

- Memory A (written 2026-02-14): Adaeze sings in the shower every morning.
- Memory B (written 2026-02-14): Adaeze sings in the car every morning.
- Label: non_contradiction
- Type: merely_related
- Reason: Adaeze can sing in the shower and in the car on the same mornings, so the two habits fit together.

Elias: OK

## 17. tr-1282: Wouter Mulder

- Memory A (written 2025-01-14): Wouter switched to wearing a flat cap on his bike rides.
- Memory B (written 2025-01-30): The flat cap Wouter cycles in is grey herringbone tweed.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Both are about the flat cap Wouter wears cycling, and one adds its fabric and colour.
- Hard case: a compatible detail that contains a change word.

Elias: OK

## 18. tr-1295: Flomo Wesseh

- Memory A (written 2024-09-21): Flomo has played the saxophone since he was twelve.
- Memory B (written 2024-09-21): Flomo plays tenor saxophone.
- Label: non_contradiction
- Type: compatible_detail
- Reason: Both say Flomo plays the saxophone, and one adds that it is a tenor.
- Hard case: a compatible detail that contains a change word.

Elias: OK

## 19. tr-1401: Sangay Penjor

- Memory A (written 2025-06-28): Sangay switched his house from firewood to an electric heater two winters ago.
- Memory B (written 2025-06-28): Sangay switched his house from firewood to a gas heater two winters ago.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both describe the same change from firewood in Sangay's house but name different heaters, electric and gas.
- Hard case: a contradiction that contains a change word.

Elias: OK

## 20. tr-1450: Tsion Getachew

- Memory A (written 2025-08-16): Tsion plays tennis left-handed.
- Memory B (written 2025-08-16): Tsion is right-handed.
- Label: unclear
- Type: none (only non-contradictions have a type)
- Reason: It depends on whether Tsion plays tennis with her weaker hand, which neither says.

Elias: OK
