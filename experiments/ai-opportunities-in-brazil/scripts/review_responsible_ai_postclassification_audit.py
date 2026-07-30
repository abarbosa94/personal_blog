"""Materialize the independent manual review of the frozen RAI audit sample."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/analysis/responsible-ai-2025-postclassification-audit.csv"
OUTPUT = ROOT / "artifacts/analysis/responsible-ai-2025-postclassification-audit-reviewed.csv"
REPORT = ROOT / "artifacts/analysis/responsible-ai-2025-postclassification-audit-report.md"

DIMENSION_LABELS = {
    "privacy_data_governance": "privacy/data governance",
    "transparency_explainability": "transparency/explainability",
    "security_safety": "security/safety",
    "fairness": "fairness",
}

# Rows not listed retain the predicted dimensions after manual title/abstract review.
# An empty tuple is a manual negative.
OVERRIDES: dict[int, tuple[str, ...]] = {
    5: (),
    18: ("security_safety",),
    24: (),
    38: ("privacy_data_governance", "transparency_explainability"),
    50: ("security_safety",),
    54: ("fairness",),
    58: (),
    63: ("security_safety",),
    67: ("privacy_data_governance",),
    68: ("transparency_explainability",),
    69: (),
    73: ("fairness",),
    76: ("fairness",),
    77: (),
    87: (),
    92: (),
    93: (),
    96: (),
    102: (),
    104: (),
    109: ("fairness",),
    110: (),
    138: (),
    140: ("security_safety",),
    156: (),
}

CUSTOM_NOTES = {
    5: "Negative: explainability appears only as an asserted comparative benefit; the contribution is faster graph retrieval/reasoning, not explanation or transparency.",
    15: "Positive, scope-borderline: the method explicitly exposes inner model knowledge as an explanation and intervention space, although explainability is not its sole objective.",
    18: "Positive for security/safety: adversarial perturbations are the studied threat. Exclude privacy; hiding sensitive data is only an incidental possible use.",
    24: "Negative: the deployed RL system targets oral-health adherence; 'marginalized individuals' describes the population and is not a fairness analysis.",
    35: "Positive, scope-borderline: the synthesis explicitly studies trust-transparency relationships in public-sector RAI governance rather than model-level explainability.",
    38: "Positive for data governance and transparency: the study elicits community preferences for speech-data governance, privacy, and disclosure. Marginalization is context, not a fairness analysis.",
    41: "Positive, scope-borderline: public disclosure is the empirical mechanism used to assess whether firms honor AI commitments.",
    50: "Positive: the abstract studies harmful AI-agent incidents, including prompt injection, private-information exfiltration, and unauthorized actions.",
    54: "Positive: the paper directly audits legally mandated employment bias audits and their treatment of sex/race disparities.",
    55: "Positive, scope-borderline: the review examines algorithm-audit coverage of algorithmic bias and discrimination, though much of its contribution concerns research geography.",
    56: "Positive, scope-borderline: transparency and privacy protections are substantive criteria in the paper's evaluation of federal AI grant governance.",
    58: "Negative: 'fair and transparent' describes democratic procedures; the paper does not study AI-system transparency or explanations.",
    61: "Positive, scope-borderline: fairness and transparency are evaluated as governance properties of FOSS stewardship, not properties of a learned model.",
    62: "Positive, scope-borderline: the system studies perceived fairness and equitable work allocation, outside protected-class group fairness.",
    63: "Positive: the empirical target is accessible generation of non-consensual deepfakes, a concrete harmful-content and misuse risk.",
    67: "Positive, scope-borderline: the study applies contextual-integrity privacy analysis to algorithmic travel surveillance and its rights impacts.",
    68: "Positive only for transparency/explainability: it evaluates interpretability of ML-assisted hate-crime classification; 'discriminatory actions' describes offenses, not model discrimination.",
    69: "Negative: fairness is only named as one possible motivation for multi-distribution learning; the contribution is technical calibration.",
    73: "Positive: the abstract explicitly assesses stereotypes, social-category bias, and representational harms in LLM outputs.",
    76: "Positive: demographic scenario testing identifies performance disparities in deployed biometric systems.",
    77: "Negative: the contribution concerns performative predictions and generic moral responsibility; fairness is only listed as adjacent prior work.",
    87: "Negative: 'explanation' denotes a mathematical account of substructure-counting behavior, not model explanations for users.",
    92: "Negative: 'discrimination' denotes contrastive feature discrimination in anomaly detection, not unfair treatment.",
    93: "Negative: the paper provides a mathematical explanation of covariance normalization, not AI explainability or transparency.",
    96: "Negative: 'theoretical explanation' is an account of DPO optimization limitations, not model interpretability.",
    102: "Negative: 'explanation' is theoretical and 'instance discrimination' is a self-supervised objective; neither is a frozen RAI dimension.",
    104: "Negative: textual point explanations specify pose keypoints as model input; they are not explanations of model decisions.",
    109: "Positive for fairness: the paper benchmarks group fairness with uncertainty. Generic claims of transparency/accountability do not justify a second dimension.",
    110: "Negative: fairness appears only in a parenthetical example of properties made cheaper to verify by tree compression.",
    130: "Positive, scope-borderline: auditing fixed predictions supports actionable recourse, retained under the frozen transparency/explainability interpretation.",
    138: "Negative: implicit optimization bias and 'transparent framework' are mathematical concepts, not fairness or AI transparency.",
    140: "Positive: the paper red-teams text-to-image systems to elicit inappropriate or harmful images and evade defenses.",
    146: "Positive for safety and transparency: analog models are proposed specifically to enable frontier-model safety verification and interpretability research.",
    147: "Positive, scope-borderline: economic equity for data generators is the paper's central RAI distributional concern, though it is not protected-class fairness.",
    156: "Negative: 'biased toward spatial modeling' describes architectural specialization, while 'discrimination' is a contrastive-learning operation.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def default_note(row: dict[str, str], dimensions: tuple[str, ...]) -> str:
    if dimensions:
        labels = ", ".join(DIMENSION_LABELS[value] for value in dimensions)
        return f"Positive: title/abstract substantively studies {labels}; the dimension is part of the paper's stated problem, method, or evaluation."
    return f"Negative: the abstract's contribution ({row['title']}) does not substantively study any frozen dimension."


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 158:
        raise ValueError(f"expected frozen 158-row sample, found {len(rows)}")

    reviewed: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        predicted = tuple(filter(None, row["predicted_dimensions"].split("|")))
        dimensions = OVERRIDES.get(index, predicted)
        label = "positive" if dimensions else "negative"
        reviewed.append({
            **row,
            "audit_manual_label": label,
            "audit_manual_dimensions": "|".join(dimensions),
            "audit_notes": CUSTOM_NOTES.get(index, default_note(row, dimensions)),
        })

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)

    confusion = Counter(
        (row["audit_manual_label"], row["predicted_label"]) for row in reviewed
    )
    by_venue: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in reviewed:
        by_venue[row["venue"]][
            (row["audit_manual_label"], row["predicted_label"])
        ] += 1
    tp = confusion[("positive", "positive")]
    fp = confusion[("negative", "positive")]
    fn = confusion[("positive", "negative")]
    tn = confusion[("negative", "negative")]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    borderline = [
        row for row in reviewed if "scope-borderline" in row["audit_notes"].casefold()
    ]
    errors = [
        row for row in reviewed
        if row["audit_manual_label"] != row["predicted_label"]
    ]
    dimension_corrections = [
        row for row in reviewed
        if row["audit_manual_dimensions"] != row["predicted_dimensions"]
        and row["audit_manual_label"] == row["predicted_label"]
    ]

    report = [
        "# Responsible AI 2025 — independent post-classification audit",
        "",
        "Date: 2026-07-30",
        "",
        "This audit manually applies the frozen four-dimension contracts to all "
        "158 sampled titles and official abstracts. It does not alter the classifier "
        "or compute country indicators.",
        "",
        "## Outcome",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| True positives | {tp} |",
        f"| False positives | {fp} |",
        f"| False negatives | {fn} |",
        f"| True negatives | {tn} |",
        f"| Precision | {precision:.1%} |",
        f"| Recall | {recall:.1%} |",
        f"| Specificity | {specificity:.1%} |",
        "",
        "## Per-venue results",
        "",
        "| Venue | TP | FP | FN | TN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for venue in sorted(by_venue):
        counts = by_venue[venue]
        vtp = counts[("positive", "positive")]
        vfp = counts[("negative", "positive")]
        vfn = counts[("positive", "negative")]
        vtn = counts[("negative", "negative")]
        vp = vtp / (vtp + vfp) if vtp + vfp else 0
        vr = vtp / (vtp + vfn) if vtp + vfn else 0
        report.append(
            f"| {venue.upper()} | {vtp} | {vfp} | {vfn} | {vtn} | "
            f"{vp:.1%} | {vr:.1%} |"
        )

    report += [
        "",
        "## Label errors",
        "",
    ]
    for row in errors:
        report.append(
            f"- **{row['venue'].upper()} — {row['title']}**: predicted "
            f"{row['predicted_label']}; reviewed {row['audit_manual_label']}. "
            f"{row['audit_notes']}"
        )
    report += [
        "",
        "## Dimension-only corrections",
        "",
    ]
    for row in dimension_corrections:
        report.append(
            f"- **{row['venue'].upper()} — {row['title']}**: "
            f"`{row['predicted_dimensions']}` → `{row['audit_manual_dimensions']}`. "
            f"{row['audit_notes']}"
        )
    report += [
        "",
        "## Scope-borderline cases",
        "",
    ]
    for row in borderline:
        report.append(f"- **{row['venue'].upper()} — {row['title']}**: {row['audit_notes']}")
    report += [
        "",
        "## Gate recommendation",
        "",
        "The classifier does **not** pass the independent post-classification gate "
        "unchanged. Although overall precision and recall are reported above, the "
        "audit finds systematic false positives from incidental or technical uses of "
        "dimension terms and false negatives for concrete harms not covered by the "
        "current patterns. The frozen classifier should be revised and re-evaluated "
        "on this independent sample before country aggregation. Borderline inclusion "
        "choices should remain explicit in sensitivity analysis.",
        "",
        "## Reproducibility",
        "",
        f"- Frozen input SHA-256: `{sha256(SOURCE)}`",
        f"- Reviewed CSV SHA-256: `{sha256(OUTPUT)}`",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print({
        "rows": len(reviewed),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "borderline": len(borderline),
        "dimension_corrections": len(dimension_corrections),
        "reviewed_sha256": sha256(OUTPUT),
        "report_sha256": sha256(REPORT),
    })


if __name__ == "__main__":
    main()
