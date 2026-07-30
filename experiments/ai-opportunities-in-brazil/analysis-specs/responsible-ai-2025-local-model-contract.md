# Responsible AI 2025 local-classifier contract

Date: 2026-07-30

This contract supersedes the proposed API/model-assisted production workflow.
Codex reviewers may calibrate or audit the codebook, but the 2025 production
classification must be local, deterministic, reproducible, and independent of
an external language model.

## Frozen target

The unit is an accepted/published 2025 paper in AAAI, AIES, FAccT, ICLR, ICML,
or NeurIPS with an observed title and abstract. A paper is positive only when
its contribution substantively studies at least one frozen dimension:

1. privacy and data governance;
2. transparency and explainability;
3. security and safety;
4. fairness.

The ten NeurIPS papers without abstracts remain explicitly unclassified.
Venue affiliation is not semantic evidence and must not be an input feature.
It may be used only for splitting, diagnostics, and sensitivity analysis.

## Reviewed-label ledger

The three review artifacts contain 304 judgments but only 298 unique papers:

| Review stage | Judgments |
| --- | ---: |
| Initial contextual review | 86 |
| Post-classification audit | 158 |
| Confirmation review | 60 |
| Total | 304 |

Six papers occur in both the 86- and 158-paper sets. All six agree exactly on
binary label and dimensions. They must nevertheless be collapsed before
training or cross-validation; otherwise a duplicate can appear in both train
and test folds and inflate performance. The 60-paper set has no paper overlap
with either earlier set.

Seventeen unique judgments are explicitly marked scope-borderline. They are
not reviewer conflicts and should not be silently converted into core examples.
Retain a `scope_status` field with values `core` or `borderline`, and report
the final country result both excluding and including borderline positives.

## Adjudicated boundary rules

The reviewed records support four stable distinctions.

### Privacy and data governance

Include privacy/data governance as the studied problem, mechanism, audit
criterion, or outcome. Include contextual-integrity analysis of algorithmic
surveillance and community-centered data-governance studies. Exclude work that
mentions privacy only as motivation or as an incidental possible use.

### Transparency and explainability

Include explanations of AI-system decisions or behavior, algorithmic recourse,
model interpretability, governance disclosure when disclosure is the measured
mechanism, and interfaces that expose predictive uncertainty in AI decision
aids. Exclude mathematical/scientific explanations, feature discrimination,
code-task explanation, and generic procedural transparency.

### Security and safety

Include concrete threats, attacks, misuse, incidents, red-teaming, harmful
outputs or interactions, AI safety engineering, and audits of algorithm-driven
risks. Exclude ordinary robustness, domain-generalization augmentation,
functional correctness, geometric alignment, and verification unrelated to a
specified AI harm or threat.

### Fairness

Include group or demographic disparities, bias audits, stereotypes,
representational or quality-of-service harms, and reviewed sociocultural harms
such as linguistic colonialism or cultural erasure caused by AI systems.
Exclude optimization/inductive bias, contrastive discrimination, fairness
datasets used only as convenient benchmarks, and a marginalized population
that is merely the deployment context.

The broadest fairness and governance inclusions remain scope-borderline. Their
country effect must be exposed through sensitivity analysis.

## Local production architecture

Use the local TF-IDF logistic-regression implementation in
`scripts/train_responsible_ai_logreg.py` as the binary paper classifier:

- word unigrams/bigrams plus character 3--5 grams;
- the deterministic sentence-context rule result as a feature;
- balanced logistic regression with a fixed random seed;
- a threshold selected only inside the training partition.

Do not include `VENUE_*` tokens. Venue is correlated with the sampling design
and target prevalence; using it as evidence can mechanically raise AIES/FAccT
predictions and bias the cross-venue country comparison.

Use the deterministic sentence-context classifier for dimension evidence until
a dimension-specific local model passes the dimension gate. Every production
row must retain the binary probability, binary decision, dimensions, matched
evidence/rules, classifier version, and abstract status.

## Leakage-safe evaluation

1. Collapse the 304 judgments to 298 unique `(venue, paper_id)` records.
2. Group any title/ID aliases for the same paper into the same fold.
3. Use deterministic outer folds stratified jointly by venue and binary label.
4. Select hyperparameters and the probability threshold using inner folds of
   the outer-training partition only.
5. Report pooled out-of-fold metrics, per-venue metrics, and leave-one-venue-out
   sensitivity.
6. Do not call the existing 60-paper set an untouched confirmation set after
   its failure patterns have informed rule changes. It is now development
   evidence.
7. After all rules, features, thresholds, and scope decisions are frozen,
   draw a new 120-paper audit: ten predicted positives and ten predicted
   negatives per venue, augmented by all sampled low-margin/borderline cases.
   Reviewers must not see predictions until their judgments are frozen.

The release gate is:

- 100% structurally valid outputs;
- paper-level precision >= 90% and recall >= 90% in nested out-of-fold
  evaluation;
- paper-level precision and recall >= 90% on the new post-freeze audit;
- no venue with precision or recall below 80% when its relevant denominator is
  at least ten;
- each reported dimension has precision and recall >= 90%, or the dimension
  breakdown is explicitly labeled exploratory and excluded from inferential
  claims.

## Current evidence and limitations

The current hybrid run over 304 judgments reports:

- stratified five-fold precision 90.50%, recall 93.64%;
- leave-one-venue-out precision 88.04%, recall 93.64%;
- stage-60 precision 97.14%, recall 100%.

These are promising, not a release result. The stratified calculation includes
six duplicate judgments, the implementation currently includes venue tokens,
and the stage-60 failure patterns have now influenced the sentence rules.

After collapsing duplicates, the current deterministic dimension rules on 298
unique papers give:

| Dimension | Precision | Recall |
| --- | ---: | ---: |
| Privacy/data governance | 96.30% | 89.66% |
| Transparency/explainability | 96.67% | 98.31% |
| Security/safety | 100.00% | 92.00% |
| Fairness | 83.93% | 94.00% |

Exact dimension-set agreement is 277/298 (92.95%). Privacy recall and fairness
precision do not pass the dimension gate. Therefore the binary classifier may
advance to a post-freeze audit after leakage corrections, but the four-way
breakdown cannot yet be presented as a validated result.

## Immediate deterministic next step

Patch the local training script to:

1. collapse the six duplicate judgments;
2. remove venue from `feature_text`;
3. use nested venue/label-stratified folds for threshold selection;
4. emit per-fold and per-venue predictions;
5. freeze a versioned model plus the 120-paper post-freeze audit sample.

Do not classify the full corpus or aggregate countries until that audit passes.
