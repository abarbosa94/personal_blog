"""Consolidate contexts, classify RAI papers, and create the audit sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from conference_pipeline.responsible_ai import screen_title
from conference_pipeline.responsible_ai_context import classify_context


def stable(row: dict[str, object]) -> str:
    return hashlib.sha256(f"{row['venue']}|{row['paper_id']}|{row['title']}".encode()).hexdigest()


def paper_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["venue"]), str(row["paper_id"] or row["title"])


def read_universe(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                paper = record.get("paper", record)
                rows.append({
                    "venue": paper["venue_key"],
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper["title"],
                    "official_url": paper.get("official_url", ""),
                })
    return rows


def read_contexts(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    values: dict[tuple[str, str], dict[str, object]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = paper_key(row)
                current = values.get(key)
                # Prefer a successful observation; among equal states, the last
                # record is the latest resumable retry.
                if current is None or current.get("error") or not row.get("error"):
                    values[key] = row
    return values


def audit_sample(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: dict[tuple[str, str], dict[str, object]] = {}
    venues = sorted({str(row["venue"]) for row in rows})
    observed = [row for row in rows if row["abstract_status"] == "observed"]
    for venue in venues:
        venue_rows = [row for row in observed if row["venue"] == venue]
        for label, count in (("positive", 15), ("negative", 10)):
            eligible = [row for row in venue_rows if row["predicted_label"] == label]
            for row in sorted(eligible, key=stable)[:count]:
                selected[paper_key(row)] = row
    for row in observed:
        if row["audit_new_title_trigger"]:
            selected[paper_key(row)] = row
    result = []
    for row in sorted(selected.values(), key=lambda value: (str(value["venue"]), stable(value))):
        result.append({
            **row,
            "audit_manual_label": "",
            "audit_manual_dimensions": "",
            "audit_notes": "",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--context", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-sample", type=Path, required=True)
    args = parser.parse_args()

    universe = read_universe(args.input)
    contexts = read_contexts(args.context)
    rows = []
    for paper in universe:
        context = contexts.get(paper_key(paper), {})
        abstract = str(context.get("abstract") or "")
        error = str(context.get("error") or "")
        dimensions = classify_context(str(paper["title"]), abstract) if abstract else ()
        abstract_dimensions = classify_context("", abstract) if abstract else ()
        title_dimensions, _ = screen_title(str(paper["title"]))
        title_only_context_dimensions = classify_context(str(paper["title"]), "")
        rows.append({
            **paper,
            "source_url": context.get("source_url", ""),
            "abstract": abstract,
            "abstract_status": "observed" if abstract else "missing",
            "context_error": error,
            "predicted_label": "positive" if dimensions else "negative" if abstract else "unclassified",
            "predicted_dimensions": "|".join(dimensions),
            "abstract_dimensions": "|".join(abstract_dimensions),
            "title_screen_dimensions": "|".join(title_dimensions),
            "audit_new_title_trigger": bool(
                title_only_context_dimensions
                and not title_dimensions
                and not abstract_dimensions
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    for venue in sorted({str(row["venue"]) for row in rows}):
        venue_rows = [row for row in rows if row["venue"] == venue]
        observed = [row for row in venue_rows if row["abstract_status"] == "observed"]
        positives = [row for row in observed if row["predicted_label"] == "positive"]
        summary.append({
            "venue": venue,
            "papers": len(venue_rows),
            "abstracts_observed": len(observed),
            "abstract_coverage": len(observed) / len(venue_rows),
            "rai_positive": len(positives),
            "rai_share_observed": len(positives) / len(observed) if observed else "",
            "missing_abstracts": len(venue_rows) - len(observed),
        })
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    sample = audit_sample(rows)
    with args.audit_sample.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    print(json.dumps({
        "papers": len(rows),
        "observed": sum(row["abstract_status"] == "observed" for row in rows),
        "positive": sum(row["predicted_label"] == "positive" for row in rows),
        "missing": sum(row["abstract_status"] == "missing" for row in rows),
        "audit_sample": len(sample),
        "new_title_triggers": sum(bool(row["audit_new_title_trigger"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
