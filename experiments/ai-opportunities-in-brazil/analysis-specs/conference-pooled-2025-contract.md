# Frozen contract — pooled 2025 conference-country indicator

Version: 3.0  
Frozen: 2026-07-29

## Objective

Produce a pooled 2025 conference-country indicator that is not dependent on
AAAI alone and is not silently dominated by venue size or unequal affiliation
coverage.

This contract supersedes treating the completed four-venue table as the final
conference indicator for `H-BASE-001`.

## Frozen venue scope

The version-2 pool contains exactly seven 2025 conference/track universes:

| Venue | Official universe | Inclusion rule |
| --- | --- | --- |
| AAAI | AAAI volume 39 | Archival research papers |
| ACL | `2025.acl-long` | Main long papers |
| ICML | PMLR volume 267 | Main proceedings |
| NeurIPS | Official 2025 proceedings | Conference, datasets/benchmarks and position tracks |
| EMNLP | `2025.emnlp-main` | Main proceedings; exclude Findings, demos, industry and workshops |
| ICLR | OpenReview `ICLR.cc/2025/Conference` | Accepted Oral, Spotlight and Poster papers; exclude rejected, withdrawn and workshop submissions |
| KDD ADS | Official KDD 2025 Applied Data Science Track paper list | Both 2025 cycles; exclude Research, Datasets/Benchmarks, workshops and non-archival items |

Adding or removing a venue requires a new contract version before recomputing
the pooled result.

## Observation contract

Each venue must use:

1. an official source to define the complete paper universe;
2. the same OpenAlex exact-title/DOI reconciliation policy;
3. the same official-PDF country extraction policy;
4. the same frozen V9 affiliation reconciliation rules;
5. a unique-paper check and an auditable missing-country count.

V9 is frozen as the accepted country-reconciliation method. Its existing blind
and regression gates qualify the method for the seven-venue census; no additional
blind country-label gate will be run for EMNLP, ICLR, KDD ADS or the combined pool.

New venues still require source-integrity validation:

- agreement with the official proceedings scope;
- exclusion of front matter, rejected, withdrawn and workshop records;
- unique paper identifiers;
- stable rerun checksum;
- valid paper, author and URL schemas;
- explicit coverage and API-error reports.

These checks validate enumeration and pipeline execution. They do not constitute
a new manual accuracy estimate for V9.

## Coverage completion gate

The pooled indicator is eligible for the main post only when:

- every venue has at least 90% of papers with a resolved country;
- the difference between the highest- and lowest-coverage venue is at most
  15 percentage points;
- no venue has unresolved systematic API failures above 0.5% of its universe;
- coverage and missingness are published for every venue.

Until all conditions pass, the seven-venue output is a diagnostic artifact, not
the final pooled estimate. Failing the gate triggers additional recovery or a
new contract version; it does not permit silently dropping a venue.

## Pooled estimators

Two pooled estimates will be reported.

### Primary: equal-venue observed share

For each country:

1. calculate its within-venue fractional share among papers with a resolved
   country;
2. average the seven venue shares with weight `1/7`.

This prevents AAAI, NeurIPS, or any other large venue from dominating merely
because it publishes more papers.

### Secondary: paper-weighted pooled share

Pool fractional paper counts and divide by all papers with a resolved country.
This estimates representation across the observed paper universe, but larger
venues receive more weight.

Full paper counts remain descriptive. They are not the primary pooled
estimator because multinational papers contribute to multiple countries.

## Sensitivity requirements

The final output must include:

- full and fractional counting;
- equal-venue and paper-weighted pooling;
- leave-one-venue-out estimates;
- observed-share and missing-at-random scenarios;
- per-venue coverage;
- Brazil's rank under every estimator;
- an explicit flag when a conclusion changes after excluding one venue.

## Interpretation

Passing this contract falsifies the operational claim that the pooled result
cannot be treated as comparably observed under the frozen gate. It does not
prove that affiliation missingness is statistically random. The post must
retain that distinction.

## Current status — completed 2026-07-29

| Venue | Enumeration | Country recovery | Gate |
| --- | --- | --- | --- |
| AAAI | complete | 91.48% | coverage passes |
| ACL | complete | 96.19% | coverage passes |
| ICML | complete | 95.56% | coverage passes |
| NeurIPS | complete | 91.38% | coverage passes |
| EMNLP | complete: 1,809 research papers | 92.65% | coverage passes |
| ICLR | 3,702 of 3,703 accepted papers enumerated | 97.38% | coverage passes |
| KDD ADS | complete: 155 papers (92 February; 63 August) | 149 with country (96.13%) | coverage passes |

The official universe contains 19,908 papers. The final dataset contains
19,907 records because one accepted ICLR paper was deliberately left
unresolved; this omission is 0.027% of the ICLR universe. Every venue passes
the 90% floor, and the 6.00 percentage-point coverage spread passes the
15-point limit.

ACL Anthology lists 1,810 entries for `2025.emnlp-main`; the enumerated research
universe is 1,809 after excluding the `.0` proceedings front matter. The raw
dataset is `data/raw/emnlp-2025.jsonl`, SHA-256
`0f5c59ef2d98945491db208ee523d858c9196e1f48031298bc452bc26734bb62`.

The ICLR raw dataset is `data/raw/iclr-2025.jsonl`, SHA-256
`411e8e6f7fc5461534811f29db9189a4a6a426c739a55d904d410f1c1e343896`.
The KDD ADS raw dataset is `data/raw/kdd-ads-2025.jsonl`, SHA-256
`bec3f202a20f1b557316206b5afcf64004bdb86c401af18ca8425028d21b7cb8`.

The compact final outputs, coverage metadata, sensitivity results and
checksums are tracked in [`../publication`](../publication). Large raw,
processed and cache files remain local and are not required to inspect the
published estimates.
