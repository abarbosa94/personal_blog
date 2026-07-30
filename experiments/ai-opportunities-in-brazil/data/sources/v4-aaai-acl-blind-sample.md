# V4 full run and AAAI/ACL blind sample

Date: 2026-07-28.

V4 was run over the four existing 50-paper formal venue samples: AAAI, ACL,
ICML, and NeurIPS. The resulting population contains 200 unique venue-paper
records.

The original ICML/NeurIPS population has no untouched records: its 100 papers
were covered by the completed 30 + 35 + 35 manual-review rounds. To preserve a
genuine out-of-sample evaluation, the new blind sample was drawn only from the
previously unreviewed AAAI and ACL populations.

## Full V4 population

| Venue | Records | SHA-256 |
| --- | ---: | --- |
| AAAI | 50 | `4057177f51441b8efc321c1782dee6e317f9f153769714f29457f9a7172564cb` |
| ACL | 50 | `1778290e1e853bec5fe281f065bcdd5b82b5e52d4e741f98a3f73a7673c79368` |
| ICML | 50 | `cd4e0765dd85e57333a7c4fb8435aeed0d68fe0be262efea43950c36f7d36ac6` |
| NeurIPS | 50 | `66a612bc3c38ece096e30f98f7ea6055984d3cd40228ac38a93c5f86513ecbe9` |

The full signal inventory contains two unreconciled records, four records with
an unresolved multinational branch, 143 mixed-source records, and 51
fallback-only records.

## Multinational branch signal

The review queue now assigns high priority to unresolved branch-level names
such as Google DeepMind, Google Research, Meta, Meta AI, Meta
Superintelligence Labs, and Microsoft Research. This signal applies even when
another affiliation already supplies a country.

Four records in the full population receive this signal:

- `icml:castro25a`;
- `neurips:27e7f21d16fb840f3720ac87ad896220`;
- `neurips:5f7a70bed3eb5ca4e3f6944af73f2819`;
- `neurips:61432775ec681f7ac6e7d335fa933e4f`.

## Blind sample

The blind sample contains 25 AAAI and 25 ACL records. Sampling occurred before
review-priority sorting, with seeds `20260728` and `20260729`, respectively.

Validation:

- 50 rows;
- 50 unique review IDs;
- zero overlap with the 100 previously reviewed ICML/NeurIPS IDs;
- 49 medium-priority mixed-source cases;
- one high-priority unreconciled case.

Queue:
`data/processed/2025-v4-aaai-acl-blind-review-50.csv`

SHA-256:
`c191a82b1911a3fad4b8ec87ad36c0d505b50676a754e5579b405728eb2cae43`.

