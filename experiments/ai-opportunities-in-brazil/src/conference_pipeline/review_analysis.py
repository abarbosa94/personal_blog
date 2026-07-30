"""Freeze review checkpoints, build targeted reruns, and compare versions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .io import read_reconciled, write_jsonl
from .models import Paper
from .reconcile import CountryMentionExtractor


ISO_CODES = set(CountryMentionExtractor.COUNTRY_ALIASES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def freeze(queue: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_queue(queue)
    reviewed = [row for row in rows if row["review_status"] in {"pass", "fail"}]
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = output_dir / "review-queue.csv"
    frozen.write_bytes(queue.read_bytes())
    manifest = {
        "schema_version": 1,
        "source": str(queue),
        "queue_sha256": sha256(frozen),
        "total": len(rows),
        "reviewed": len(reviewed),
        "pass": sum(row["review_status"] == "pass" for row in reviewed),
        "fail": sum(row["review_status"] == "fail" for row in reviewed),
        "defer": sum(row["review_status"] == "defer" for row in rows),
        "review_ids": [row["review_id"] for row in reviewed],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _paper_from_record(value: dict[str, Any]) -> Paper:
    paper = dict(value["paper"])
    paper["authors"] = tuple(paper.get("authors") or [])
    return Paper(**paper)


def build_targets(
    queue: Path,
    sources: list[Path],
    paper_output: Path,
    expectation_output: Path,
) -> int:
    reviewed_failures = {
        row["review_id"]: row
        for row in read_queue(queue)
        if row["review_status"] == "fail"
    }
    papers: dict[str, Paper] = {}
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                paper = _paper_from_record(value)
                review_id = f"{paper.venue_key}:{paper.paper_id}"
                if review_id in reviewed_failures:
                    queue_authors = tuple(
                        author.strip()
                        for author in reviewed_failures[review_id]
                        .get("authors", "")
                        .split("|")
                        if author.strip()
                    )
                    if queue_authors and not paper.authors:
                        paper = replace(paper, authors=queue_authors)
                    papers[review_id] = paper
    missing = set(reviewed_failures) - set(papers)
    if missing:
        raise ValueError(f"Reviewed failures missing from sources: {sorted(missing)}")
    ordered_ids = [row["review_id"] for row in read_queue(queue) if row["review_id"] in papers]
    write_jsonl((papers[review_id] for review_id in ordered_ids), paper_output)
    expectation_output.parent.mkdir(parents=True, exist_ok=True)
    with expectation_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "expected_countries",
                "confidence",
                "review_note",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for review_id in ordered_ids:
            row = reviewed_failures[review_id]
            writer.writerow(
                {
                    "review_id": review_id,
                    "expected_countries": "|".join(
                        expected_country_codes(row["review_note"])
                    ),
                    "confidence": row["confidence"],
                    "review_note": row["review_note"],
                }
            )
    return len(ordered_ids)


def build_reviewed_expectations(queue: Path, output: Path) -> int:
    rows = [
        row
        for row in read_queue(queue)
        if row["review_status"] in {"pass", "fail"}
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "expected_countries", "confidence", "review_note"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if row["review_status"] == "fail":
                expected = expected_country_codes(row["review_note"])
            else:
                affiliations = json.loads(row["affiliations_json"] or "[]")
                expected = tuple(
                    sorted(
                        {
                            item["country_code"]
                            for item in affiliations
                            if item.get("country_code")
                        }
                    )
                )
            writer.writerow(
                {
                    "review_id": row["review_id"],
                    "expected_countries": "|".join(expected),
                    "confidence": row["confidence"],
                    "review_note": row["review_note"],
                }
            )
    return len(rows)


def expected_country_codes(note: str) -> tuple[str, ...]:
    """Extract country names plus deliberately upper-case ISO codes from notes."""

    expected_lines = re.findall(
        r"(?im)expected\s+countr(?:y|ies)(?:\s+set)?\s*(?::|is)\s*([^\r\n]+)",
        note,
    )
    # Review notes conventionally state the gold set first and may later repeat
    # rejected pipeline countries in explanatory prose.
    evidence = expected_lines[0] if expected_lines else note
    evidence = re.split(r"(?i)\s*;\s*pipeline\b", evidence, maxsplit=1)[0]
    values = set(CountryMentionExtractor.country_codes(evidence))
    for code in re.findall(r"(?<![A-Z])(?:[A-Z]{2})(?![A-Z])", evidence):
        normalized = "GB" if code == "UK" else code
        if normalized in ISO_CODES:
            values.add(normalized)
    return tuple(sorted(values))


def compare(
    expectations_path: Path,
    v1_queue_path: Path,
    v2_path: Path,
    output: Path,
) -> dict[str, Any]:
    expectations = {
        row["review_id"]: set(filter(None, row["expected_countries"].split("|")))
        for row in read_queue(expectations_path)
    }
    v1 = {
        row["review_id"]: {
            item.get("country_code")
            for item in json.loads(row["affiliations_json"])
            if item.get("country_code")
        }
        for row in read_queue(v1_queue_path)
        if row["review_id"] in expectations
    }
    v2 = {
        f"{record.paper.venue_key}:{record.paper.paper_id}": set(record.countries)
        for record in read_reconciled(v2_path)
    }
    rows = []
    for review_id, expected in expectations.items():
        actual_v1 = v1.get(review_id, set())
        actual_v2 = v2.get(review_id, set())
        rows.append(
            {
                "review_id": review_id,
                "expected": sorted(expected),
                "v1": sorted(actual_v1),
                "v2": sorted(actual_v2),
                "v1_missing": sorted(expected - actual_v1),
                "v2_missing": sorted(expected - actual_v2),
                "v2_extra": sorted(actual_v2 - expected),
                "v2_exact": actual_v2 == expected,
            }
        )
    summary = {
        "total": len(rows),
        "v1_exact": sum(not row["v1_missing"] and set(row["v1"]) == set(row["expected"]) for row in rows),
        "v2_exact": sum(row["v2_exact"] for row in rows),
        "v1_country_recall": _recall(rows, "v1"),
        "v2_country_recall": _recall(rows, "v2"),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _recall(rows: list[dict[str, Any]], field: str) -> float:
    expected = sum(len(row["expected"]) for row in rows)
    recovered = sum(len(set(row[field]) & set(row["expected"])) for row in rows)
    return recovered / expected if expected else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("queue", type=Path)
    freeze_parser.add_argument("--output-dir", required=True, type=Path)
    targets = commands.add_parser("targets")
    targets.add_argument("queue", type=Path)
    targets.add_argument("sources", nargs="+", type=Path)
    targets.add_argument("--paper-output", required=True, type=Path)
    targets.add_argument("--expectation-output", required=True, type=Path)
    expectations = commands.add_parser("expectations")
    expectations.add_argument("queue", type=Path)
    expectations.add_argument("--output", required=True, type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("expectations", type=Path)
    comparison.add_argument("v1_queue", type=Path)
    comparison.add_argument("v2", type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.queue, args.output_dir), indent=2))
    elif args.command == "targets":
        count = build_targets(
            args.queue,
            args.sources,
            args.paper_output,
            args.expectation_output,
        )
        print(f"Wrote {count} reviewed failure targets")
    elif args.command == "expectations":
        count = build_reviewed_expectations(args.queue, args.output)
        print(f"Wrote {count} reviewed expectations")
    else:
        print(
            json.dumps(
                compare(args.expectations, args.v1_queue, args.v2, args.output),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
