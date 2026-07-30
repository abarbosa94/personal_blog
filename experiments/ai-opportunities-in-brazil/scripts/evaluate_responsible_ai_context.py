"""Evaluate the deterministic context classifier on the reviewed sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from conference_pipeline.responsible_ai_context import classify_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.validation.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    contexts = {}
    with args.context.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            contexts[(row["venue"], row["paper_id"])] = row

    rows = []
    for row in reviewed:
        key = (row["venue"], row["paper_id"])
        context = contexts[key]
        predicted = classify_context(row["title"], context["abstract"])
        truth = row["manual_label"] == "positive"
        rows.append(
            {
                **row,
                "predicted_label": "positive" if predicted else "negative",
                "predicted_dimensions": "|".join(predicted),
                "label_correct": truth == bool(predicted),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tp = sum(r["manual_label"] == "positive" and r["predicted_label"] == "positive" for r in rows)
    fp = sum(r["manual_label"] == "negative" and r["predicted_label"] == "positive" for r in rows)
    fn = sum(r["manual_label"] == "positive" and r["predicted_label"] == "negative" for r in rows)
    tn = len(rows) - tp - fp - fn
    print(json.dumps({
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "specificity": tn / (tn + fp) if tn + fp else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
