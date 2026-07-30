"""Create a fresh deterministic confirmation sample with no reviewed overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def stable(row: dict[str, str]) -> str:
    return hashlib.sha256(
        f"{row['venue']}|{row['paper_id']}|{row['title']}".encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    excluded = set()
    for path in args.exclude:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                excluded.add((row["venue"], row["paper_id"]))
    with args.classified.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows = [
        row for row in rows
        if row["abstract_status"] == "observed"
        and (row["venue"], row["paper_id"]) not in excluded
    ]
    selected = []
    for venue in sorted({row["venue"] for row in rows}):
        venue_rows = [row for row in rows if row["venue"] == venue]
        for label in ("positive", "negative"):
            eligible = [row for row in venue_rows if row["predicted_label"] == label]
            selected.extend(sorted(eligible, key=stable)[:5])
    output = [{
        **row,
        "confirmation_manual_label": "",
        "confirmation_manual_dimensions": "",
        "confirmation_notes": "",
    } for row in sorted(selected, key=lambda row: (row["venue"], stable(row)))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({
        "rows": len(output),
        "excluded_reviewed": len(excluded),
        "venues": sorted({row["venue"] for row in output}),
    }, indent=2))


if __name__ == "__main__":
    main()
