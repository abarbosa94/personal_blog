"""Unblind the frozen LightGBM audit and apply the release contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DIMENSIONS = (
    "privacy_data_governance",
    "transparency_explainability",
    "security_safety",
    "fairness",
)
DIMENSION_ALIASES = {
    "privacy_and_data_governance": "privacy_data_governance",
    "transparency_and_explainability": "transparency_explainability",
    "security_and_safety": "security_safety",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def dimension_set(value: str) -> set[str]:
    return {
        DIMENSION_ALIASES.get(item, item)
        for item in value.split("|")
        if item and item.casefold() != "none"
    }


def binary_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    tp = sum(r["manual_label"] == "positive" and r["binary_prediction"] == "positive" for r in rows)
    fp = sum(r["manual_label"] == "negative" and r["binary_prediction"] == "positive" for r in rows)
    fn = sum(r["manual_label"] == "positive" and r["binary_prediction"] == "negative" for r in rows)
    tn = len(rows) - tp - fp - fn
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "predicted_positive": tp + fp, "actual_positive": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "accuracy": (tp + tn) / len(rows) if rows else 0,
    }


def dimension_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    result = {}
    for dimension in DIMENSIONS:
        tp = fp = fn = tn = 0
        for row in rows:
            truth = dimension in dimension_set(row["manual_dimensions"])
            predicted = dimension in dimension_set(row["dimensions_exploratory"])
            if truth and predicted:
                tp += 1
            elif predicted:
                fp += 1
            elif truth:
                fn += 1
            else:
                tn += 1
        result[dimension] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
        }
    return result


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.reviewed.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    with args.key.open(encoding="utf-8-sig", newline="") as handle:
        key_rows = list(csv.DictReader(handle))
    reviewed_by_key = {(r["venue"], r["paper_id"]): r for r in reviewed}
    predictions_by_key = {(r["venue"], r["paper_id"]): r for r in key_rows}
    if len(reviewed) != 120 or len(key_rows) != 120:
        raise ValueError("Both blind artifacts must contain exactly 120 rows")
    if len(reviewed_by_key) != 120 or len(predictions_by_key) != 120:
        raise ValueError("Duplicate venue/paper IDs in blind artifacts")
    if reviewed_by_key.keys() != predictions_by_key.keys():
        raise ValueError("Reviewed and prediction keys are not identical")
    joined = []
    for key in sorted(reviewed_by_key):
        review = reviewed_by_key[key]
        prediction = predictions_by_key[key]
        if review["title"] != prediction["title"]:
            raise ValueError(f"Title mismatch: {key}")
        joined.append({
            **review,
            "binary_probability": prediction["binary_probability"],
            "binary_prediction": prediction["binary_prediction"],
            "dimensions_exploratory": prediction["dimensions_exploratory"],
            "binary_correct": review["manual_label"] == prediction["binary_prediction"],
        })

    overall = binary_metrics(joined)
    per_venue = {
        venue: binary_metrics([r for r in joined if r["venue"] == venue])
        for venue in sorted({r["venue"] for r in joined})
    }
    core = binary_metrics([r for r in joined if r["scope_status"] == "core"])
    dimensions = dimension_metrics(joined)
    venue_gate = all(
        (m["predicted_positive"] < 10 or m["precision"] >= 0.8)
        and (m["actual_positive"] < 10 or m["recall"] >= 0.8)
        for m in per_venue.values()
    )
    binary_release = (
        overall["precision"] >= 0.9
        and overall["recall"] >= 0.9
        and venue_gate
    )
    validated_dimensions = [
        name for name, value in dimensions.items()
        if value["precision"] >= 0.9 and value["recall"] >= 0.9
    ]
    metrics = {
        "reviewed_sha256": sha256(args.reviewed),
        "key_sha256": sha256(args.key),
        "identity_valid": True,
        "overall": overall,
        "core_only": core,
        "per_venue": per_venue,
        "dimensions": dimensions,
        "validated_dimensions": validated_dimensions,
        "dimension_breakdown_release": len(validated_dimensions) == len(DIMENSIONS),
        "venue_floor_pass": venue_gate,
        "binary_release_pass": binary_release,
    }
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with args.joined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    lines = [
        "# Responsible AI 2025 — LightGBM blind-120 unblinding",
        "",
        "The 120 manual labels were frozen before the prediction key was opened.",
        "The reviewed and key artifacts match one-to-one on all venue/paper IDs.",
        "",
        "## Binary release gate",
        "",
        "| Scope | N | TP | FP | FN | TN | Precision | Recall | Specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| All reviewed | {overall['n']} | {overall['tp']} | {overall['fp']} | "
            f"{overall['fn']} | {overall['tn']} | {pct(overall['precision'])} | "
            f"{pct(overall['recall'])} | {pct(overall['specificity'])} |"
        ),
        (
            f"| Core only | {core['n']} | {core['tp']} | {core['fp']} | "
            f"{core['fn']} | {core['tn']} | {pct(core['precision'])} | "
            f"{pct(core['recall'])} | {pct(core['specificity'])} |"
        ),
        "",
        f"**Binary release gate: {'PASS' if binary_release else 'FAIL'}.**",
        "",
        "## Per venue",
        "",
        "| Venue | TP | FP | FN | TN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for venue, value in per_venue.items():
        lines.append(
            f"| {venue.upper()} | {value['tp']} | {value['fp']} | "
            f"{value['fn']} | {value['tn']} | {pct(value['precision'])} | "
            f"{pct(value['recall'])} |"
        )
    lines.extend([
        "",
        "## Exploratory dimension diagnostics",
        "",
        "| Dimension | TP | FP | FN | Precision | Recall | Release status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for name, value in dimensions.items():
        passed = name in validated_dimensions
        lines.append(
            f"| {name} | {value['tp']} | {value['fp']} | {value['fn']} | "
            f"{pct(value['precision'])} | {pct(value['recall'])} | "
            f"{'validated' if passed else 'exploratory'} |"
        )
    lines.extend([
        "",
        "Dimensions that do not independently pass 90% precision and recall remain",
        "exploratory and are excluded from inferential country claims.",
        "",
        "## Reproducibility",
        "",
        f"- Reviewed SHA-256: `{metrics['reviewed_sha256']}`",
        f"- Prediction-key SHA-256: `{metrics['key_sha256']}`",
    ])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
