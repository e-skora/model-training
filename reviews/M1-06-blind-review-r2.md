# M1-06 blind review, round 2 (2026-10-22)

**Reviewer:** the same separate Claude agent as round 1 (D-028). It wrote its own answers before seeing the labels.
**Sample:** 140 pairs: every `unclear` pair in train and valid (80), plus 60 pairs not seen in round 1 (seed 77).

## Result
- **Non-unclear pairs:** every label agreed. One type was a coin flip (tr-1177).
- **`unclear` pairs:**
  - The reviewer agreed on 49 of 80.
  - It gave a definite label to the other 31. It judged the key wrong on about 25 of them and called 6 genuinely ambiguous. Most of the 31 were new pairs written in M1-06b. Their reasons turn on what a word means ("may or may not count as exercise"), which is the kind of escape rule 4 rules out.

## What Claude in Cowork did (D-030)
- It **dropped all 31 disputed pairs** from the data and did not relabel them. The dropped ids are tr-0032, tr-1453, tr-1501, tr-1505, tr-1509, tr-1511, tr-1512, tr-1515, tr-1519, tr-1523, tr-1525, tr-1526, tr-1528, tr-1531, tr-1532, tr-1533, tr-1534, tr-1535, tr-1538, tr-1539, tr-1540, tr-1542, tr-1544, tr-1545, tr-1546, tr-1547, tr-1548, tr-1549, v-013, v-052 and v-102.
- It kept only the `unclear` pairs that two separate judges agreed on.
- **New counts:**
  - Train: 1,522 pairs (697 contradiction, 778 non-contradiction, 47 unclear, 3%). SHA-256 `a41de541d2a25c564009e232b2d75ccb714c5a668dcd30ef73004fae778231b1`.
  - Valid: 99 pairs (46, 51, 2). SHA-256 `0476bc57e19460a10f1a3299f17e570b0d7926cb893d6ce5d57d49f356e042d3`.
- `scripts/check_dataset.py` passes on both files.
- Left as it is: 17 `unclear` reasons still use "may or may not". That is harmless for the label, because the model writes the label before the reason.
