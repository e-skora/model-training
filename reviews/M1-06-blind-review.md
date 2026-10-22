# M1-06 blind review of the training set (2026-10-22)

**Reviewer:** a separate Claude agent (Sonnet), set up by Claude in Cowork under D-028. It wrote its own answers before seeing the labels.
**Sample:** 200 pairs from `data/m1/train.jsonl`: all 74 `unclear` pairs plus 126 others (seed 2026). Because every `unclear` pair is included, they make up 37% of the sample but only 5% of the file.

## Result
- **Contradiction and non-contradiction pairs:** clean. Of the 126 non-unclear pairs, the reviewer agreed on every label except 5 (below). Type-only disagreement: 1 (tr-0256, a coin flip). All 22 updates in the sample matched.
- **`unclear` pairs:** about half are not truly undecidable. Training on them would teach the model to dodge real contradictions, which would hurt recall and break reading rule 1.
  - **The reviewer says contradiction (22 pairs).** The key's reasons invent an escape, such as "maybe two dogs" or "yellow inside, white outside": tr-0472, tr-0279, tr-1110, tr-0371, tr-0182, tr-0871, tr-1441, tr-0953, tr-0430, tr-0948, tr-0122, tr-0408, tr-0363, tr-0714, tr-0808, tr-1353, tr-1285, tr-1019, tr-0607, tr-0324, tr-1309, tr-0663.
  - **The reviewer says compatible or merely related (12 pairs):** tr-0002, tr-1060, tr-0071, tr-0169, tr-0236, tr-0102, tr-1434, tr-0698, tr-0247, tr-1014, tr-0108, tr-0501.
  - **The reviewer agreed these are truly undecidable (examples):** tr-0742, tr-1213, tr-0309, tr-1471, tr-0989, tr-0914, tr-1093, tr-0067, tr-0203, tr-0875, tr-1216, tr-1450.
- **The same pattern is labelled two ways.** Pairs like "does X on day 1" and "does X on day 2", with no "only" or "every", are `merely_related` in tr-1179, tr-1126, tr-1287, tr-1011 and tr-1057, but `unclear` in tr-0102, tr-1434 and tr-0698. One rule is needed.
- **The reasons follow a template.** 73 of the 74 `unclear` reasons start "It depends on whether..." or "Whether they clash depends on...".

## Not problems (by design)
- In the file, `a` is always the earlier memory, and same-day pairs are never updates. The training formatter swaps the order in half the examples and removes dates from a third (D-026), so the model does not see this pattern.
