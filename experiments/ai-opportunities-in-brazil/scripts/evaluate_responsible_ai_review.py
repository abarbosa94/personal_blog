"""Evaluate the current context classifier on a reviewed CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from conference_pipeline.responsible_ai_context import classify_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-field", default="audit_manual_label")
    parser.add_argument("--dimensions-field", default="audit_manual_dimensions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    evaluated = []
    for row in rows:
        predicted = classify_context(row["title"], row["abstract"])
        evaluated.append({
            **row,
            "revised_label": "positive" if predicted else "negative",
            "revised_dimensions": "|".join(predicted),
            "revised_label_correct": (
                ("positive" if predicted else "negative") == row[args.label_field]
            ),
        })
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)
    tp = sum(r["revised_label"] == "positive" and r[args.label_field] == "positive" for r in evaluated)
    fp = sum(r["revised_label"] == "positive" and r[args.label_field] == "negative" for r in evaluated)
    fn = sum(r["revised_label"] == "negative" and r[args.label_field] == "positive" for r in evaluated)
    tn = len(evaluated) - tp - fp - fn
    print(json.dumps({
        "n": len(evaluated), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "specificity": tn / (tn + fp) if tn + fp else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
