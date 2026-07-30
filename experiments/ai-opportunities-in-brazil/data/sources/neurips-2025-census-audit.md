# NeurIPS 2025 census audit

Date: 2026-07-29.

## Final pipeline status

The official universe contains 5,823 papers. The PDF country pass processed all
5,823 records. After funded OpenAlex recovery and a final deterministic merge,
the pipeline produced 5,823 unique records.

| Measure | Value |
| --- | ---: |
| Official papers | 5,823 |
| Unique merged records | 5,823 |
| OpenAlex works matched | 2,970 |
| Papers with country evidence | 2,425 |
| Papers without country evidence | 3,398 |
| Final pipeline country coverage | 41.65% |
| Residual OpenAlex API errors | 3 |
| PDF country successes | 2,362 |
| PDF country unresolved | 3,460 |
| PDF country API errors | 1 |

The three residual OpenAlex errors are HTTP 429 responses after the funded
budget was consumed. They represent 0.05% of the paper universe and retain
their completed PDF evidence. They are recorded as an explicit limitation
rather than a reason to rerun the full census.

The 4,853-record retry cohort has already been materialized at
`data/processed/neurips-2025-v9-openalex-retry-papers.jsonl`. Its SHA-256 is
`05409f85e295b6ce3587efd2bc753575e84ea819455cd00473015d535251377f`.

The final merged dataset is
`data/processed/neurips-2025-v9-census.jsonl`, with SHA-256
`19d1dc4fd4af0157758dfb0f72bbf578ff2862e4b8b0ca76575f5e4c12265077`.

## Recovery history

- The initial bulk retry exposed that OpenAlex candidates may contain a null
  title. That exception terminated concurrent runs before a checkpoint.
- Title reconciliation now tolerates null candidate titles.
- Concurrent reconciliation now uses a bounded in-flight queue.
- The retry uses 25-record checkpoints and persistent stdout/stderr logs.
- The HTTP 400 title-query issue was resolved by using `search.exact=`.
- All 94 tests pass after these changes.

## Interpretation rule

- Treat `conference-presence-2025-neurips.csv` as the final output of this
  pipeline version, not as complete observation of all affiliations.
- Always disclose the 41.65% country coverage.
- Do not compare raw NeurIPS country counts directly with venues that have
  materially higher coverage without sensitivity analysis.
- Prefer within-venue country shares over pooled raw counts, while explaining
  the missingness assumption.
- Retain the three HTTP 429 cases and 3,398 missing-country papers as explicit
  limitations.
