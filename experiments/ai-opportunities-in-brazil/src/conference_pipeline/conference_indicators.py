"""Aggregate reconciled conference papers into auditable country indicators."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .io import read_reconciled
from .models import ReconciledPaper


def country_rows(
    records: Iterable[ReconciledPaper],
    panel: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    values = list(records)
    by_venue: dict[str, list[ReconciledPaper]] = defaultdict(list)
    for record in values:
        by_venue[record.paper.venue_key].append(record)

    rows: list[dict[str, object]] = []
    scopes = [("all", values), *sorted(by_venue.items())]
    for venue, venue_records in scopes:
        for code, country in panel.items():
            full = 0
            fractional = 0.0
            for record in venue_records:
                countries = set(record.countries)
                if code not in countries:
                    continue
                full += 1
                fractional += 1.0 / len(countries)
            rows.append(
                {
                    "venue": venue,
                    "country_code": code,
                    "country_name": country["country_name"],
                    "comparison_group": country["comparison_group"],
                    "papers_full_count": full,
                    "papers_fractional_count": round(fractional, 6),
                    "paper_universe": len(venue_records),
                }
            )

    with_country = sum(bool(record.countries) for record in values)
    metadata = {
        "paper_universe": len(values),
        "papers_with_country": with_country,
        "papers_without_country": len(values) - with_country,
        "country_coverage": with_country / len(values) if values else 0.0,
        "venues": {
            venue: len(records) for venue, records in sorted(by_venue.items())
        },
        "counting": {
            "full": "One paper for every represented country",
            "fractional": "Each paper contributes 1/N to each of its N countries",
        },
    }
    return rows, metadata


def read_panel(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    panel = {row["country_code"]: row for row in rows}
    if len(panel) != len(rows):
        raise ValueError("Country comparison panel contains duplicate codes")
    return panel


def write_outputs(
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    output: Path,
    metadata_output: Path,
) -> None:
    if not rows:
        raise ValueError("No indicator rows were generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()
    records = [
        record for path in args.inputs for record in read_reconciled(path)
    ]
    rows, metadata = country_rows(records, read_panel(args.panel))
    write_outputs(rows, metadata, args.output, args.metadata_output)
    print(f"Wrote {len(rows)} country indicator rows for {len(records)} papers")


if __name__ == "__main__":
    main()
