"""Apply frozen LightGBM locally and create a blinded post-freeze audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from conference_pipeline.responsible_ai_context import classify_context


def feature_text(row: dict[str, object]) -> str:
    dimensions = classify_context(str(row["title"]), str(row["abstract"]))
    rule_tokens = " ".join(f"RULE_DIM_{value}" for value in dimensions)
    return (
        f"{'RULE_POSITIVE' if dimensions else 'RULE_NEGATIVE'} {rule_tokens}\n"
        f"TITLE: {row['title']}\nABSTRACT: {row['abstract']}"
    )


def stable(row: dict[str, object]) -> str:
    return hashlib.sha256(
        f"{row['venue']}|{row['paper_id']}|{row['title']}".encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--classified", type=Path, required=True)
    parser.add_argument("--reviewed-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-blind", type=Path, required=True)
    parser.add_argument("--audit-key", type=Path, required=True)
    args = parser.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]
    threshold = float(payload["threshold"])
    with args.classified.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    observed = [row for row in rows if row["abstract_status"] == "observed"]
    probabilities = model.predict_proba(
        np.array([feature_text(row) for row in observed])
    )[:, 1]
    by_key = {}
    for row, probability in zip(observed, probabilities):
        dimensions = classify_context(row["title"], row["abstract"])
        by_key[(row["venue"], row["paper_id"] or row["title"])] = {
            **row,
            "binary_probability": float(probability),
            "binary_prediction": "positive" if probability >= threshold else "negative",
            "dimensions_exploratory": "|".join(dimensions),
            "classifier_version": payload["classifier"],
            "classifier_threshold": threshold,
            "scope_status": "unreviewed",
        }
    output = []
    for row in rows:
        key = (row["venue"], row["paper_id"] or row["title"])
        if key in by_key:
            output.append(by_key[key])
        else:
            output.append({
                **row, "binary_probability": None,
                "binary_prediction": "unclassified",
                "dimensions_exploratory": "",
                "classifier_version": payload["classifier"],
                "classifier_threshold": threshold,
                "scope_status": "missing_abstract",
            })
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    for venue in sorted({row["venue"] for row in output}):
        venue_rows = [row for row in output if row["venue"] == venue]
        classified = [row for row in venue_rows if row["binary_prediction"] != "unclassified"]
        positives = [row for row in classified if row["binary_prediction"] == "positive"]
        summary.append({
            "venue": venue, "papers": len(venue_rows),
            "classified": len(classified), "positive": len(positives),
            "positive_share_classified": len(positives) / len(classified),
            "missing_abstracts": len(venue_rows) - len(classified),
        })
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    with args.reviewed_ledger.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = {
            (row["venue"], row["paper_id"] or row["title"])
            for row in csv.DictReader(handle)
        }
    eligible = [
        row for row in output
        if row["binary_prediction"] in {"positive", "negative"}
        and (row["venue"], row["paper_id"] or row["title"]) not in reviewed
    ]
    selected = {}
    for venue in sorted({row["venue"] for row in eligible}):
        for label in ("positive", "negative"):
            candidates = [
                row for row in eligible
                if row["venue"] == venue and row["binary_prediction"] == label
            ]
            for row in sorted(candidates, key=stable)[:5]:
                selected[(row["venue"], row["paper_id"] or row["title"])] = row
            remaining = [
                row for row in candidates
                if (row["venue"], row["paper_id"] or row["title"]) not in selected
            ]
            for row in sorted(
                remaining,
                key=lambda value: (
                    abs(float(value["binary_probability"]) - threshold),
                    stable(value),
                ),
            )[:5]:
                selected[(row["venue"], row["paper_id"] or row["title"])] = row
    sample = sorted(selected.values(), key=lambda row: (row["venue"], stable(row)))
    if len(sample) != 120:
        raise ValueError(f"Expected 120 blind audit records, got {len(sample)}")
    blind = [{
        "venue": row["venue"], "paper_id": row["paper_id"],
        "title": row["title"], "official_url": row["official_url"],
        "source_url": row["source_url"], "abstract": row["abstract"],
        "manual_label": "", "manual_dimensions": "",
        "scope_status": "", "review_notes": "",
    } for row in sample]
    with args.audit_blind.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind[0]))
        writer.writeheader()
        writer.writerows(blind)
    key_rows = [{
        "venue": row["venue"], "paper_id": row["paper_id"],
        "title": row["title"],
        "binary_probability": row["binary_probability"],
        "binary_prediction": row["binary_prediction"],
        "dimensions_exploratory": row["dimensions_exploratory"],
    } for row in sample]
    with args.audit_key.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    print(json.dumps({
        "papers": len(output), "classified": len(observed),
        "positive": sum(row["binary_prediction"] == "positive" for row in output),
        "unclassified": len(output) - len(observed),
        "blind_audit": len(blind), "threshold": threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
