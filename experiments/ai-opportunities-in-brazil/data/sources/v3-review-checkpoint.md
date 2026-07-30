# V3 affiliation-resolution checkpoint

Date: 2026-07-28.

This checkpoint evaluates the second improvement round against the 35 manually
reviewed records sampled from the 70 records that were untouched in the first
review round. The reviewed country sets are frozen in
`tests/fixtures/v2_review_sample_35_expected.csv`.

## Changes

- PDF/ROR fallback affiliations are rebuilt instead of carrying old fallback
  matches into a new run.
- Compact numbered affiliation lines such as `1Org A2Org B` are split before
  organization resolution.
- A small reviewed alias table covers unambiguous organizations missed by ROR.
- ROR `chosen:true` results also require lexical support from the source
  affiliation.
- Known ambiguous truncated organization names are rejected.
- Unnumbered company affiliations in pre-abstract front matter are considered.
- Empty affiliation sets can no longer receive the `automatic_pass` signal.
- Results augmented through PDF NER are assigned `mixed_sources`, not
  `automatic_pass`.

## Frozen gate

| Measure | V2 reviewed output | V3 |
| --- | ---: | ---: |
| Exact country sets | 24/35 | 35/35 |
| Cases with an extra country | 6 | 0 |
| Cases with a missing country | 7 | 0 |

The v2 error counts overlap: two records had both missing and extra countries.
The V3 gate contains no missing or extra country.

## Test result

The complete local suite passes: `37 passed`.

## SHA-256

- frozen expectations:
  `47b00485aba4ad9d2ea0ce11e62edbdb12e4a6c6069bccd06c714939b7f58f98`;
- V3 35-record output:
  `2a0f162b2296a3f0d11fa0c14eb42d566e9bdd000cbf98c82018995ad7ba6b0b`;
- detailed gate report:
  `b3a68b34cebeec9e77b5da0254f9b3f9fe506567ee26e876dd79dcc913969ade`.

V3 was also applied to the remaining unsampled 35 records. The resulting blind
review queue is
`data/processed/2025-unreviewed-v3-unsampled-review-35.csv`, with SHA-256
`21c00d250090cae3f92251449875c5e5bccc95f7e3c0679ddc5364d3c37838b3`.
It contains 35 unique records and has zero overlap with the reviewed sample.
Those records must not be used to add aliases or tune thresholds before their
manual decisions are frozen.
