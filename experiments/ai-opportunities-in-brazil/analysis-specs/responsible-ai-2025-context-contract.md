# Responsible AI 2025 abstract-context contract

This contract replaces title-only classification after contextual review of the
frozen 86-paper sample found six positives among 30 title-screen negatives.

## Scope

The unit remains an accepted/published paper in AAAI, AIES, FAccT, ICLR, ICML,
or NeurIPS 2025. The dimensions remain:

1. privacy and data governance;
2. transparency and explainability;
3. security and safety;
4. fairness.

The classifier operates on the official title and abstract. PDF text is used
only to recover the abstract when the official HTML/API record does not expose
it. A paper is positive when at least one frozen dimension has contextual
support. Generic ethics, ordinary statistical/optimization bias, geometric
alignment, generic robustness, and mathematical feasibility called “safety”
are excluded.

## Calibration gate

The deterministic rule must achieve at least 90% precision and 90% recall
against the 86 manually reviewed papers. Because this sample was used to refine
the rule, its measurements are calibration performance rather than an
independent generalization estimate. The final result must retain this
limitation and include a separate post-classification audit.

The frozen calibration result is 48 true positives, 2 false positives, no
false negatives, and 36 true negatives: 96.0% precision, 100% recall, and
94.7% specificity. The two false positives conservatively include papers where
interpretability/fairness language is present but not the primary contribution.

## Corpus and quality gate

All six official venue universes must be classified. Abstract coverage must be
at least 95% in every venue. Fetch errors and blank abstracts remain explicit;
they are never silently treated as negatives.

After classification, review:

- every positive triggered only by a term newly added after title screening;
- a deterministic sample of at least 15 predicted positives per venue;
- a deterministic sample of at least 10 predicted negatives per venue.

Country aggregation may proceed only after that audit. The denominator and
missing-country treatment must be reported separately from abstract coverage.

