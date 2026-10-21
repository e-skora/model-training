# Spot-check: M1 test set

**What to do (about 10 minutes):** read each of the 20 pairs below. Each one shows two memories about one person, the answer Claude Code gave, and why.

- If you agree with the answer, leave the line `Elias: OK` as it is.
- If you disagree, change that line to `Elias: WRONG, <why>`, for example `Elias: WRONG, these can both be true`.

Edit this file in place and save it. You do not need to commit anything. Then tell Claude in the Cowork project "spot-check done".

How to judge: two memories that say different things are true at the same moment are a contradiction, even if they were written on different dates. If the later memory says something changed ("moved", "now", "switched"), it is an update, not a contradiction. If both can be true together, it is not a contradiction.

These 20 pairs were picked at random from `data/m1/test.jsonl` with seed 42 (Python `random.Random(42).sample(pairs, 20)`): 12 contradictions and 8 non-contradictions.

## 1. t-007: Thabo Molefe

- Memory A (written 2026-04-28): Thabo has a garden.
- Memory B (written 2026-04-28): Thabo's garden has a lemon tree.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds a detail about Thabo's garden, and both can be true.

Elias: OK

## 2. t-008: Wanjiru Kamau

- Memory A (written 2025-03-27): Wanjiru has never learned to swim.
- Memory B (written 2025-05-05): Wanjiru swims thirty lengths at the town pool every Saturday.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Someone who has never learned to swim cannot swim thirty lengths every week.

Elias: OK

## 3. t-009: Nour El-Amin

- Memory A (written 2025-06-21): Nour's Saturday shift at the café ends at 4 p.m.
- Memory B (written 2025-08-02): Nour's Saturday shift at the café ends at 6 p.m.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Neither memory describes a change, so they give two different end times for Nour's Saturday shift.

Elias: OK

## 4. t-023: Uchenna Obi

- Memory A (written 2025-09-18): Uchenna plays five-a-side football on Thursdays.
- Memory B (written 2026-01-27): Uchenna now plays five-a-side football on Tuesdays, since the league switched nights.
- Label: non_contradiction
- Type: update
- Reason: The later memory says the league switched nights, so the new day replaces the earlier one.

Elias: OK

## 5. t-024: Chidi Okeke

- Memory A (written 2025-09-25): Chidi's front door key has a green rubber cover.
- Memory B (written 2025-09-25): Chidi's front door key has a red rubber cover.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both describe the same key on the same day and give different cover colours.

Elias: OK

## 6. t-027: Tariq Rahman

- Memory A (written 2025-03-08): Tariq's only goldfish is named Captain.
- Memory B (written 2025-05-24): Tariq's only goldfish is named Admiral.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both name Tariq's only goldfish, with no change described, and give different names.

Elias: OK

## 7. t-029: Obinna Chukwu

- Memory A (written 2025-01-21): Obinna's favourite football club is Arsenal.
- Memory B (written 2025-08-17): Obinna stopped following Arsenal and now supports Chelsea.
- Label: non_contradiction
- Type: update
- Reason: The later memory says Obinna switched clubs, so it replaces the earlier one.

Elias: OK

## 8. t-036: Yuki Mori

- Memory A (written 2025-10-01): Yuki works from home on Fridays.
- Memory B (written 2025-10-01): Yuki works from the office on Thursdays.
- Label: non_contradiction
- Type: merely_related
- Reason: Working from home on Fridays and from the office on Thursdays are about different days, so Yuki's memories do not clash.

Elias: OK

## 9. t-056: Elif Yılmaz

- Memory A (written 2025-08-14): Elif's front door is painted yellow.
- Memory B (written 2025-11-02): Elif's front door is painted orange.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Neither memory mentions repainting, so they give two different colours for Elif's front door.

Elias: OK

## 10. t-058: Omar Farouk

- Memory A (written 2026-03-28): Omar's favourite football club is Al Ahly.
- Memory B (written 2026-03-28): Omar's favourite football club is Zamalek.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both name Omar's favourite football club on the same day and pick different clubs.

Elias: OK

## 11. t-060: Astrid Nilsen

- Memory A (written 2025-06-11): Astrid's house has a red roof.
- Memory B (written 2025-08-30): Astrid's house has a grey roof.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Neither memory mentions a new roof, so they give two different roof colours for Astrid's house.

Elias: OK

## 12. t-063: Liam O'Connor

- Memory A (written 2026-02-27): Liam's favourite film is Jaws.
- Memory B (written 2026-02-27): Liam's favourite film is Alien.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both name Liam's favourite film on the same day and pick different films.

Elias: OK

## 13. t-071: Tenzin Dorje

- Memory A (written 2025-10-21): Tenzin has visited Paris.
- Memory B (written 2025-10-28): Tenzin visited Paris twice, once in 2019 and once in 2024.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds when Tenzin visited Paris, and both can be true.

Elias: OK

## 14. t-109: Aleksander Wiśniewski

- Memory A (written 2026-03-09): Aleksander's hallway rug is grey with white stripes.
- Memory B (written 2026-03-09): Aleksander's hallway rug is grey with blue stripes.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Both describe the same rug on the same day and give different stripe colours.

Elias: OK

## 15. t-140: Siddharth Rao

- Memory A (written 2026-01-09): Siddharth's brother lives in Chennai.
- Memory B (written 2026-01-09): Siddharth lives in Hyderabad.
- Label: non_contradiction
- Type: merely_related
- Reason: One is about where Siddharth's brother lives and the other about where Siddharth lives, so they do not clash.

Elias: OK

## 16. t-152: Hye-jin Lee

- Memory A (written 2025-07-16): Hye-jin's cat Dubu is all white.
- Memory B (written 2025-07-16): Hye-jin's cat Dubu is white with black spots.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: The same cat cannot be all white and also have black spots.

Elias: OK

## 17. t-164: Emil Hansen

- Memory A (written 2025-03-19): Emil lives at number 14 on Nørrebrogade.
- Memory B (written 2025-05-31): Emil lives at number 41 on Nørrebrogade.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Neither memory describes a move, so they give two different house numbers for Emil.

Elias: OK

## 18. t-174: Aylin Kaya

- Memory A (written 2025-06-24): Aylin's brother lives in Izmir.
- Memory B (written 2025-06-24): Aylin's best friend lives in Ankara.
- Label: non_contradiction
- Type: merely_related
- Reason: One is about Aylin's brother and the other about Aylin's best friend, so they do not clash.

Elias: OK

## 19. t-189: Fatima Benali

- Memory A (written 2026-04-12): Fatima keeps her spare house key under the blue flowerpot by the door.
- Memory B (written 2026-06-01): Fatima keeps her spare house key under the red flowerpot by the door.
- Label: contradiction
- Type: none (only non-contradictions have a type)
- Reason: Neither memory says the key was moved, so they give two different hiding places for Fatima's spare key.

Elias: OK

## 20. t-190: Ingrid Solberg

- Memory A (written 2025-06-06): Ingrid likes to swim.
- Memory B (written 2025-06-20): Ingrid swims in the lake near her cabin every summer morning.
- Label: non_contradiction
- Type: compatible_detail
- Reason: The second memory adds where and when Ingrid swims, and both can be true.

Elias: OK
